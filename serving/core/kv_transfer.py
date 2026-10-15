"""Request-sized KV transfer bookkeeping, independent of simulation timing.

Only the explicit mode uses this ledger. Callers deliver compute and transfer
completion events. Pools remain the sole authority for physical capacity.
"""

from dataclasses import dataclass, field
from enum import Enum


class TransferState(Enum):
    RESERVED = 'reserved'
    SUBMITTED = 'submitted'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True)
class TransferPlan:
    # IDs are local to one manager. External events must also identify it.
    transfer_id: int
    batch_id: int
    req_id: int
    direction: str
    cxl_device_id: int
    src_block_ids: tuple
    dst_block_ids: tuple
    nbytes: int
    version: int
    release_source: bool


@dataclass
class TransferRecord:
    plan: TransferPlan
    state: TransferState = TransferState.RESERVED


@dataclass
class KVSnapshot:
    num_tokens: int
    gpu_blocks: list
    cxl_blocks: list = field(default_factory=list)
    cxl_device_id: int = None
    gpu_valid: bool = True
    cxl_valid: bool = False
    version: int = 0
    pending_transfer: int = None
    gpu_consumers: int = 0
    release_gpu_when_idle: bool = False
    release_requested: bool = False
    computing_tokens: int = None


class KVTransferLedger:
    def __init__(self, npu_pool, cxl_pools, req_to_blocks):
        self.npu_pool = npu_pool
        self.cxl_pools = cxl_pools
        self.req_to_blocks = req_to_blocks
        self.entries = {}
        self.records = {}
        self._next_transfer_id = 0

    def register_gpu_blocks(self, req_id, blocks, num_tokens):
        """Adopt exclusive, already-valid GPU blocks without allocating again.

        The caller transfers its ownership reference, not a borrowed reference.
        Initial snapshots only; growing/shrinking a request is a later step.
        """
        blocks = list(blocks)
        if req_id in self.entries or req_id in self.req_to_blocks:
            raise ValueError(f'Request {req_id} is already registered')
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError('Snapshot token count must be a positive integer')
        expected = (num_tokens + self.npu_pool.block_size - 1) // self.npu_pool.block_size
        if len(blocks) != expected or len({id(b) for b in blocks}) != len(blocks):
            raise ValueError('Snapshot blocks must uniquely cover the token count')
        owned = {id(b) for e in self.entries.values() for b in e.gpu_blocks}
        for block in blocks:
            if (not 0 <= block.block_id < self.npu_pool.num_blocks
                    or self.npu_pool.blocks[block.block_id] is not block
                    or block.ref_cnt != 1 or block.block_hash is not None
                    or id(block) in owned):
                raise ValueError('Snapshot requires exclusive non-cached blocks from the GPU pool')
        entry = KVSnapshot(num_tokens, blocks)
        self.entries[req_id] = entry
        # One shared list, not an independently updated GPU ownership table.
        self.req_to_blocks[req_id] = entry.gpu_blocks

    def _entry(self, req_id):
        entry = self.entries[req_id]
        if entry.release_requested:
            raise RuntimeError('Request release is pending')
        return entry

    def _idle_entry(self, req_id):
        entry = self._entry(req_id)
        if entry.computing_tokens is not None:
            raise RuntimeError('KV computation is still in flight')
        if entry.pending_transfer is not None:
            raise RuntimeError('A transfer is already pending for this request')
        return entry

    def allocate_compute(self, req_id, num_tokens):
        """Reserve a growing GPU snapshot; data becomes valid on completion."""
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError('Compute token count must be positive')
        entry = self._idle_entry(req_id) if req_id in self.entries else None
        if entry is None and req_id in self.req_to_blocks:
            raise RuntimeError('GPU blocks exist without a snapshot')
        if entry is not None and (not entry.gpu_valid or entry.gpu_consumers
                                  or entry.release_gpu_when_idle
                                  or num_tokens <= entry.num_tokens):
            raise RuntimeError('Compute requires a restored, idle, growing GPU snapshot')
        required = (num_tokens + self.npu_pool.block_size - 1) // self.npu_pool.block_size
        count = required - (len(entry.gpu_blocks) if entry else 0)
        if count > self.npu_pool.get_num_free_blocks():
            return None
        blocks = self.npu_pool.get_new_blocks(count)
        if entry is None:
            entry = KVSnapshot(0, [])
            self.entries[req_id] = entry
            self.req_to_blocks[req_id] = entry.gpu_blocks
        entry.gpu_blocks.extend(blocks)
        self._drop_cxl(entry)
        entry.gpu_valid = False
        entry.computing_tokens = num_tokens
        return blocks

    def complete_compute(self, req_id):
        entry = self.entries[req_id]
        if entry.computing_tokens is None:
            raise RuntimeError('No KV computation in flight')
        entry.num_tokens = entry.computing_tokens
        entry.computing_tokens = None
        entry.gpu_valid = True
        entry.version += 1
        self._settle(req_id)

    def _reserve(self, req_id, batch_id, direction, device_id, release_source):
        entry = self.entries[req_id]
        cxl_pool = self.cxl_pools[device_id]
        if direction == 'gpu_to_cxl':
            source_pool, dest_pool = self.npu_pool, cxl_pool
            source, dest = entry.gpu_blocks, entry.cxl_blocks
        else:
            source_pool, dest_pool = cxl_pool, self.npu_pool
            source, dest = entry.cxl_blocks, entry.gpu_blocks
        if dest:
            raise RuntimeError('Transfer destination already holds blocks')
        if len(source) > dest_pool.get_num_free_blocks():
            return None
        reserved = dest_pool.get_new_blocks(len(source))
        source_pool.touch(source)
        dest.extend(reserved)
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1
        # Transport whole allocated blocks, including the partial tail padding.
        plan = TransferPlan(transfer_id, batch_id, req_id, direction, device_id,
                            tuple(b.block_id for b in source),
                            tuple(b.block_id for b in dest),
                            len(source) * source_pool.bytes_per_block,
                            entry.version, release_source)
        entry.cxl_device_id = device_id
        entry.pending_transfer = transfer_id
        self.records[transfer_id] = TransferRecord(plan)
        return [plan]

    def plan_offload(self, req_id, batch_id, cxl_device_id=0, *, release_gpu=True):
        """Return [plan], [] for a valid existing copy, or None for no capacity."""
        entry = self._idle_entry(req_id)
        if cxl_device_id not in self.cxl_pools:
            raise ValueError(f'Unknown CXL device: {cxl_device_id}')
        if entry.release_gpu_when_idle:
            raise RuntimeError('GPU copy release is already pending')
        if entry.cxl_valid:
            if entry.cxl_device_id != cxl_device_id:
                raise ValueError('Moving between CXL devices is not implemented')
            if release_gpu:
                entry.release_gpu_when_idle = True
                self._settle(req_id)
            return []
        if not entry.gpu_valid:
            raise RuntimeError('Offload requires valid GPU data')
        return self._reserve(req_id, batch_id, 'gpu_to_cxl', cxl_device_id, release_gpu)

    def plan_restore(self, req_id, batch_id, *, keep_cxl_copy=True):
        """Reserve GPU destination; existing valid GPU data needs no transfer."""
        entry = self._idle_entry(req_id)
        if entry.release_gpu_when_idle:
            raise RuntimeError('GPU copy release is already pending')
        if entry.gpu_valid:
            if not keep_cxl_copy and entry.cxl_valid:
                self._drop_cxl(entry)
            return []
        if not entry.cxl_valid:
            raise RuntimeError('Restore requires valid CXL data')
        return self._reserve(req_id, batch_id, 'cxl_to_gpu', entry.cxl_device_id,
                             not keep_cxl_copy)

    def mark_transfer_submitted(self, transfer_id):
        record = self.records[transfer_id]
        if record.state != TransferState.RESERVED:
            raise RuntimeError('Only a reserved transfer can be submitted')
        record.state = TransferState.SUBMITTED

    def complete_transfer(self, transfer_id):
        """Called after actual completion; duplicate completion is a no-op."""
        record = self.records[transfer_id]
        if record.state == TransferState.COMPLETED:
            return False
        if record.state != TransferState.SUBMITTED:
            raise RuntimeError('Only a submitted transfer can complete')
        plan = record.plan
        entry = self.entries[plan.req_id]
        if entry.pending_transfer != transfer_id or entry.version != plan.version:
            raise RuntimeError('Transfer does not match the current KV snapshot')
        if plan.direction == 'gpu_to_cxl':
            entry.cxl_valid = True
            self.npu_pool.free_blocks(entry.gpu_blocks)  # transfer reference
            entry.release_gpu_when_idle = plan.release_source
        else:
            entry.gpu_valid = True
            self.cxl_pools[plan.cxl_device_id].free_blocks(entry.cxl_blocks)
            if plan.release_source:
                self._drop_cxl(entry)
        entry.pending_transfer = None
        record.state = TransferState.COMPLETED
        self._settle(plan.req_id)
        return True

    def rollback_transfer(self, transfer_id):
        """Cancel a reservation only; submitted events must complete normally."""
        record = self.records[transfer_id]
        if record.state == TransferState.CANCELLED:
            return False
        if record.state != TransferState.RESERVED:
            raise RuntimeError('Cannot roll back a submitted or completed transfer')
        plan = record.plan
        entry = self.entries[plan.req_id]
        if plan.direction == 'gpu_to_cxl':
            self._drop_cxl(entry)
            self.npu_pool.free_blocks(entry.gpu_blocks)
        else:
            self._drop_gpu(entry)
            self.cxl_pools[plan.cxl_device_id].free_blocks(entry.cxl_blocks)
        entry.pending_transfer = None
        record.state = TransferState.CANCELLED
        self._settle(plan.req_id)
        return True

    def acquire_gpu(self, req_id):
        """Pin a read consumer. Writing requires mark_gpu_modified instead."""
        entry = self._entry(req_id)
        if not entry.gpu_valid or entry.release_gpu_when_idle:
            raise RuntimeError('GPU data is not available for consumption')
        self.npu_pool.touch(entry.gpu_blocks)
        entry.gpu_consumers += 1
        return tuple(entry.gpu_blocks)

    def release_gpu(self, req_id):
        entry = self.entries[req_id]
        if entry.gpu_consumers <= 0:
            raise RuntimeError('No GPU consumer reference to release')
        self.npu_pool.free_blocks(entry.gpu_blocks)
        entry.gpu_consumers -= 1
        self._settle(req_id)

    def mark_gpu_modified(self, req_id):
        """Record completed in-place changes to a quiescent fixed-size snapshot.

        Token-count growth and overlapping compute writes are not supported.
        Invalidate and release the old CXL copy so it cannot satisfy a restore.
        """
        entry = self._idle_entry(req_id)
        if not entry.gpu_valid or entry.gpu_consumers or entry.release_gpu_when_idle:
            raise RuntimeError('KV modification requires an idle valid GPU snapshot')
        self._drop_cxl(entry)
        entry.version += 1

    def release_request(self, req_id):
        """Defer destruction until outstanding transfer and consumers finish."""
        if req_id not in self.entries:
            return
        self.entries[req_id].release_requested = True
        self._settle(req_id)

    def _drop_gpu(self, entry):
        if entry.gpu_blocks:
            self.npu_pool.free_blocks(reversed(entry.gpu_blocks))
            entry.gpu_blocks.clear()
        entry.gpu_valid = False
        entry.release_gpu_when_idle = False

    def _drop_cxl(self, entry):
        if entry.cxl_blocks:
            self.cxl_pools[entry.cxl_device_id].free_blocks(reversed(entry.cxl_blocks))
            entry.cxl_blocks.clear()
        entry.cxl_valid = False
        entry.cxl_device_id = None

    def _settle(self, req_id):
        entry = self.entries[req_id]
        if (entry.pending_transfer is not None or entry.gpu_consumers
                or entry.computing_tokens is not None):
            return
        if entry.release_requested:
            self._drop_gpu(entry)
            self._drop_cxl(entry)
            del self.entries[req_id]
            del self.req_to_blocks[req_id]
        elif entry.release_gpu_when_idle:
            if not entry.cxl_valid:
                raise RuntimeError('Cannot discard the only valid KV copy')
            self._drop_gpu(entry)

    def is_empty(self):
        # Shared CXL pools can still hold blocks belonging to other managers.
        return not self.entries
