"""Scheduler integration tests with real block pools and synthetic completions."""

import unittest
import random
from types import SimpleNamespace
from unittest.mock import patch

from .block_pool import BlockPool, Device
from .kv_cache_manager import TieredKVCacheManager
from .request import RequestStatus
from .scheduler import Scheduler


class OffloadSchedulerTests(unittest.TestCase):
    def scheduler(self, gpu_blocks=4, cxl_blocks=4, max_seqs=4):
        gpu = BlockPool(Device.NPU, gpu_blocks, 16, 1024, enable_caching=False)
        cxl = BlockPool(Device.CXL, cxl_blocks, 16, 1024, enable_caching=False)
        kv = TieredKVCacheManager(16, gpu, enable_caching=False,
                                 kv_offload_mode='explicit', cxl_execution_pools={0: cxl})
        with patch('serving.core.scheduler.get_config', return_value={'max_position_embeddings': 256}), \
                patch('serving.core.scheduler.MemoryModel', return_value=SimpleNamespace(kv=kv)):
            return Scheduler('test', 0, 0, max_seqs, 128, 1, 1, 1, 1, 0, 0,
                             None, 16, 16, 2, False, False, None, None,
                             kv_offload_mode='explicit', cxl_execution_pools={0: cxl})

    def finish(self, scheduler, batch, tick):
        if batch.kv_transfers:
            scheduler.submit_transfers(batch)
        return scheduler.add_done(batch.batch_id + 1, 0, tick)

    def drain(self, s, tick=100):
        for _ in range(300):
            if s.is_request_empty():
                return
            batch = s.schedule(tick, 0)
            self.assertIsNotNone(batch, 'Pending requests must make progress')
            tick += 10
            self.finish(s, batch, tick)
        self.fail('Scheduler did not drain')

    def test_offload_completion_admits_new_request_then_restores(self):
        s = self.scheduler()
        s.add_request((0, 'test', 32, 35, 0, 0))
        first = s.schedule(0, 0)
        self.finish(s, first, 10)
        old = s.running[0]
        self.assertEqual(old.ttft, 10)
        s.add_request((1, 'test', 48, 50, 10, 0))
        move = s.schedule(10, 0)
        self.assertEqual(move.kv_transfers[0].direction, 'gpu_to_cxl')
        self.assertEqual(s.kv.npu_pool.used_blocks, 2)
        self.assertEqual(s.kv.cxl_execution_pools[0].used_blocks, 2)
        self.assertIsNone(s.schedule(15, 0))
        self.assertEqual(self.finish(s, move, 20), (0, 0, []))
        self.assertEqual(old.ttft, 10)
        self.assertEqual(old.num_computed_tokens, 32)
        self.assertEqual(old.status, RequestStatus.OFFLOADED)
        self.assertEqual(s.kv.npu_pool.used_blocks, 0)
        self.assertEqual(s.add_done(move.batch_id + 1, 0, 21), (0, 0, []))
        admitted = s.schedule(20, 0)
        self.assertEqual([r.id for r in admitted.requests], [1])
        self.finish(s, admitted, 30)
        self.finish(s, s.schedule(30, 0), 40)
        restore = s.schedule(40, 0)
        self.assertEqual(restore.kv_transfers[0].direction, 'cxl_to_gpu')
        self.assertEqual(s.kv.npu_pool.used_blocks, 2)
        with self.assertRaises(RuntimeError):
            s.kv.acquire_gpu(old.id)
        self.assertEqual(self.finish(s, restore, 50), (0, 0, []))
        self.assertEqual(old.ttft, 10)
        compute = s.schedule(50, 0)
        self.assertEqual(compute.scheduled_tokens[old.id], 1)
        self.finish(s, compute, 60)
        self.assertEqual(old.itl, [50])
        self.drain(s, 60)
        self.assertEqual(len(s.done), 2)
        self.assertEqual(s.recompute_tokens, 0)
        self.assertEqual(s.offload_stats['gpu_to_cxl_bytes'], 2048)
        self.assertEqual(s.offload_stats['cxl_to_gpu_bytes'], 2048)
        self.assertTrue(s.kv.is_free())
        self.assertEqual(s.kv.cxl_execution_pools[0].used_blocks, 0)

    def test_full_cxl_keeps_resident_compute_progressing(self):
        s = self.scheduler(cxl_blocks=1)
        s.add_request((0, 'test', 32, 35, 0, 0))
        self.finish(s, s.schedule(0, 0), 10)
        s.add_request((1, 'test', 48, 50, 10, 0))
        batch = s.schedule(10, 0)
        self.assertFalse(batch.kv_transfers)
        self.assertEqual([r.id for r in batch.requests], [0])
        self.finish(s, batch, 20)
        self.drain(s)
        self.assertEqual(s.offload_stats['gpu_to_cxl_count'], 0)
        self.assertTrue(s.kv.is_free())

    def test_decode_growth_swaps_instead_of_recomputing(self):
        s = self.scheduler(gpu_blocks=3)
        for req_id in range(2):
            s.add_request((req_id, 'test', 16, 33, 0, 0))
        self.finish(s, s.schedule(0, 0), 10)
        batch = s.schedule(10, 0)
        self.assertTrue(batch.kv_transfers)
        self.finish(s, batch, 20)
        self.drain(s)
        self.assertEqual(len(s.done), 2)
        self.assertEqual(s.recompute_tokens, 0)
        self.assertGreater(s.offload_stats['cxl_to_gpu_count'], 0)
        self.assertTrue(s.kv.is_free())

    def test_sequence_limit_and_oversize_request(self):
        s = self.scheduler(max_seqs=1)
        for req_id in range(2):
            s.add_request((req_id, 'test', 32, 35, 0, 0))
        self.drain(s)
        self.assertEqual(s.offload_stats['gpu_to_cxl_count'], 0)
        with self.assertRaises(ValueError):
            s.add_request((3, 'test', 64, 66, 0, 0))

    def test_chunked_workloads_conserve_tokens_and_release_pools(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                s = self.scheduler(cxl_blocks=40)
                s.max_num_batched_tokens = rng.choice([8, 16, 32, 128])
                s.enable_chunked_prefill = True
                s.reserve_full_isl = rng.choice([True, False])
                for req_id in range(6):
                    prompt = rng.randint(1, 48)
                    output = prompt + rng.randint(1, 65 - prompt)
                    s.add_request((req_id, 'test', prompt, output, 0, 0))
                self.drain(s)
                self.assertEqual(len(s.done), 6)
                self.assertTrue(s.kv.is_free())
                self.assertEqual(s.kv.cxl_execution_pools[0].used_blocks, 0)
                for req in s.done:
                    self.assertEqual(len(req.itl), req.output - req.input - 1)
                    self.assertEqual(sum(req.itl), req.latency - req.ttft)


if __name__ == '__main__':
    unittest.main()
