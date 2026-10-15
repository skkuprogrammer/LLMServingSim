"""Ledger-only tests: python3 -m unittest serving.core.test_kv_transfer.

Completion calls here are synthetic; these tests do not validate ASTRA timing.
"""

import unittest
from types import SimpleNamespace

from .block_pool import BlockPool, Device
from .kv_cache_manager import TieredKVCacheManager
from .kv_transfer import TransferState


class KVTransferTests(unittest.TestCase):
    def make_manager(self, gpu_blocks=4, cxl_blocks=4, pools=None):
        gpu = BlockPool(Device.NPU, gpu_blocks, 16, 1024, enable_caching=False)
        if pools is None:
            pools = {0: BlockPool(Device.CXL, cxl_blocks, 16, 1024, enable_caching=False)}
        return TieredKVCacheManager(16, gpu, enable_caching=False,
                                    kv_offload_mode='explicit', cxl_execution_pools=pools)

    def seed(self, m, req_id=1, tokens=17):
        blocks = m.npu_pool.get_new_blocks((tokens + 15) // 16)
        m.register_gpu_blocks(req_id, blocks, tokens)
        return SimpleNamespace(id=req_id)

    def finish(self, m, plans):
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        m.mark_transfer_submitted(plan.transfer_id)
        self.assertTrue(m.complete_transfer(plan.transfer_id))
        return plan

    def usage(self, m):
        return (m.npu_used_bytes(), m.cxl_execution_pools[0].used_bytes())

    def test_round_trip_capacity_validity_and_tail_padding(self):
        m = self.make_manager()
        req = self.seed(m)
        e = m.transfer_ledger.entries[req.id]
        self.assertIs(e.gpu_blocks, m.req_to_blocks[req.id])
        self.assertEqual(self.usage(m), (2048, 0))
        plans = m.plan_offload(req.id, 10)
        self.assertEqual(plans[0].nbytes, 2048)
        self.assertEqual(self.usage(m), (2048, 2048))
        self.assertTrue(e.gpu_valid)
        self.assertFalse(e.cxl_valid)
        self.assertEqual([b.ref_cnt for b in e.gpu_blocks], [2, 2])
        self.finish(m, plans)
        self.assertEqual(self.usage(m), (0, 2048))
        self.assertEqual(m.req_to_blocks[req.id], [])
        self.assertFalse(e.gpu_valid)
        plans = m.plan_restore(req.id, 11)
        self.assertEqual(self.usage(m), (2048, 2048))
        with self.assertRaises(RuntimeError):
            m.acquire_gpu(req.id)
        self.finish(m, plans)
        self.assertTrue(e.gpu_valid and e.cxl_valid)
        self.assertEqual(m.plan_restore(req.id, 12), [])
        m.free(req)
        self.assertEqual(self.usage(m), (0, 0))
        self.assertTrue(m.is_free())

    def test_offload_waits_for_consumers_in_both_completion_orders(self):
        for consumer_first in (False, True):
            with self.subTest(consumer_first=consumer_first):
                m = self.make_manager()
                req = self.seed(m)
                m.acquire_gpu(req.id)
                plans = m.plan_offload(req.id, 1)
                if consumer_first:
                    m.release_gpu(req.id)
                    self.assertEqual(self.usage(m), (2048, 2048))
                self.finish(m, plans)
                if not consumer_first:
                    self.assertEqual(self.usage(m), (2048, 2048))
                    with self.assertRaises(RuntimeError):
                        m.acquire_gpu(req.id)
                    m.release_gpu(req.id)
                self.assertEqual(self.usage(m), (0, 2048))
                m.free(req)
                self.assertTrue(m.is_free())

    def test_insufficient_cxl_capacity_does_not_mutate(self):
        m = self.make_manager(cxl_blocks=1)
        req = self.seed(m)
        self.assertIsNone(m.plan_offload(req.id, 1))
        self.assertEqual(self.usage(m), (2048, 0))
        self.assertEqual([b.ref_cnt for b in m.req_to_blocks[req.id]], [1, 1])
        self.assertEqual(m.transfer_ledger.records, {})
        m.free(req)

    def test_insufficient_gpu_capacity_then_restore(self):
        m = self.make_manager(gpu_blocks=2)
        req = self.seed(m)
        self.finish(m, m.plan_offload(req.id, 1))
        other = self.seed(m, req_id=2)
        self.assertIsNone(m.plan_restore(req.id, 2))
        self.assertEqual([b.ref_cnt for b in m.transfer_ledger.entries[1].cxl_blocks], [1, 1])
        m.free(other)
        self.finish(m, m.plan_restore(req.id, 3, keep_cxl_copy=False))
        self.assertEqual(self.usage(m), (2048, 0))
        m.free(req)

    def test_rollback_both_directions_and_completion_guards(self):
        m = self.make_manager()
        req = self.seed(m)
        plan = m.plan_offload(req.id, 1)[0]
        with self.assertRaises(RuntimeError):
            m.complete_transfer(plan.transfer_id)
        self.assertTrue(m.rollback_transfer(plan.transfer_id))
        self.assertFalse(m.rollback_transfer(plan.transfer_id))
        self.assertEqual(self.usage(m), (2048, 0))
        self.assertEqual(m.transfer_ledger.records[plan.transfer_id].state, TransferState.CANCELLED)
        with self.assertRaises(RuntimeError):
            m.mark_transfer_submitted(plan.transfer_id)
        plan = m.plan_offload(req.id, 2)[0]
        m.mark_transfer_submitted(plan.transfer_id)
        with self.assertRaises(RuntimeError):
            m.rollback_transfer(plan.transfer_id)
        m.complete_transfer(plan.transfer_id)
        self.assertFalse(m.complete_transfer(plan.transfer_id))
        restore = m.plan_restore(req.id, 3)[0]
        m.rollback_transfer(restore.transfer_id)
        self.assertEqual(self.usage(m), (0, 2048))
        self.assertTrue(m.transfer_ledger.entries[req.id].cxl_valid)
        m.free(req)

    def test_valid_copy_reuse_and_dirty_invalidation(self):
        m = self.make_manager()
        req = self.seed(m)
        self.finish(m, m.plan_offload(req.id, 1, release_gpu=False))
        self.assertEqual(self.usage(m), (2048, 2048))
        self.assertEqual(m.plan_offload(req.id, 2, release_gpu=False), [])
        m.mark_gpu_modified(req.id)
        self.assertEqual(self.usage(m), (2048, 0))
        plans = m.plan_offload(req.id, 3)
        self.assertEqual(plans[0].version, 1)
        with self.assertRaises(RuntimeError):
            m.mark_gpu_modified(req.id)
        with self.assertRaises(RuntimeError):
            m.plan_offload(req.id, 4)
        self.finish(m, plans)
        self.finish(m, m.plan_restore(req.id, 5))
        m.acquire_gpu(req.id)
        self.assertEqual(m.plan_offload(req.id, 6), [])
        self.assertEqual(self.usage(m), (2048, 2048))
        m.release_gpu(req.id)
        self.assertEqual(self.usage(m), (0, 2048))
        m.free(req)

    def test_request_finish_waits_for_transfer_and_consumer(self):
        m = self.make_manager()
        req = self.seed(m)
        m.acquire_gpu(req.id)
        plan = m.plan_offload(req.id, 1)[0]
        m.mark_transfer_submitted(plan.transfer_id)
        m.free(req)
        with self.assertRaises(RuntimeError):
            m.acquire_gpu(req.id)
        self.assertFalse(m.is_free())
        m.complete_transfer(plan.transfer_id)
        self.assertEqual(self.usage(m), (2048, 2048))
        m.release_gpu(req.id)
        self.assertEqual(self.usage(m), (0, 0))
        self.assertTrue(m.is_free())
        self.assertFalse(m.complete_transfer(plan.transfer_id))
        m.free(req)

    def test_finish_during_restore_and_reserved_cancellation(self):
        for submitted in (False, True):
            m = self.make_manager()
            req = self.seed(m)
            self.finish(m, m.plan_offload(req.id, 1))
            plan = m.plan_restore(req.id, 2)[0]
            if submitted:
                m.mark_transfer_submitted(plan.transfer_id)
            m.free(req)
            if submitted:
                m.complete_transfer(plan.transfer_id)
            else:
                m.rollback_transfer(plan.transfer_id)
            self.assertEqual(self.usage(m), (0, 0))
            self.assertTrue(m.is_free())

    def test_shared_capacity_and_manager_local_ownership(self):
        first = self.make_manager(cxl_blocks=2)
        second = self.make_manager(pools=first.cxl_execution_pools)
        r1 = self.seed(first)
        r2 = self.seed(second)
        self.finish(first, first.plan_offload(r1.id, 1))
        self.assertIsNone(second.plan_offload(r2.id, 1))
        second.free(r2)
        self.assertTrue(second.is_free())
        self.assertFalse(first.is_free())
        first.free(r1)
        self.assertTrue(first.cxl_execution_pools[0].is_free())

    def test_registration_rejects_aliases_and_foreign_blocks(self):
        m = self.make_manager()
        req = self.seed(m)
        with self.assertRaises(ValueError):
            m.register_gpu_blocks(2, m.req_to_blocks[req.id], 17)
        foreign = self.make_manager().npu_pool.get_new_blocks(1)
        with self.assertRaises(ValueError):
            m.register_gpu_blocks(2, foreign, 1)
        with self.assertRaises(ValueError):
            m.plan_offload(req.id, 1, cxl_device_id=3)
        self.assertEqual(self.usage(m), (2048, 0))
        m.free(req)

    def test_disabled_and_recompute_preemption_stay_guarded(self):
        gpu = BlockPool(Device.NPU, 4, 16, 1024, enable_caching=False)
        old = TieredKVCacheManager(16, gpu, enable_caching=False)
        self.assertIsNone(old.transfer_ledger)
        with self.assertRaises(RuntimeError):
            old.plan_offload(1, 1)
        m = self.make_manager()
        req = self.seed(m)
        with self.assertRaises(NotImplementedError):
            m.preempt(req)
        self.assertEqual(self.usage(m), (2048, 0))
        m.free(req)

    def test_compute_growth_is_not_valid_until_completion(self):
        m = self.make_manager(gpu_blocks=2)
        req = SimpleNamespace(id=1, num_computed_tokens=0)
        self.assertIsNotNone(m.allocate_slots(req, 16))
        with self.assertRaises(RuntimeError):
            m.plan_offload(req.id, 0)
        with self.assertRaises(RuntimeError):
            m.acquire_gpu(req.id)
        m.transfer_ledger.complete_compute(req.id)
        self.finish(m, m.plan_offload(req.id, 1, release_gpu=False))
        req.num_computed_tokens = 16
        self.assertIsNone(m.allocate_slots(req, 17))
        self.assertTrue(m.transfer_ledger.entries[1].cxl_valid)
        self.assertIsNotNone(m.allocate_slots(req, 1))
        self.assertFalse(m.transfer_ledger.entries[1].cxl_valid)
        self.assertEqual(self.usage(m), (2048, 0))
        m.free(req)
        self.assertFalse(m.is_free())
        m.transfer_ledger.complete_compute(req.id)
        self.assertTrue(m.is_free())


if __name__ == '__main__':
    unittest.main()
