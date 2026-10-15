---
title: Experimental KV Offload
---

# Experimental KV Offload

`--offload` enables request-level GPU/CXL KV migration. A transfer-only Chakra
graph executes a CXL STORE before the scheduler releases GPU blocks. A parked
request executes a CXL LOAD before it becomes runnable again. Transfer batches
do not produce tokens or update TTFT. Their time is included in subsequent token
completion times and therefore in TPOT and ITL.

```bash
python3 -m serving \
  --cluster-config configs/cluster/melody_cxl_b_kv_only.json \
  --dataset workloads/example_trace.jsonl \
  --no-enable-prefix-caching \
  --offload \
  --output outputs/offload.csv
```

The supplied example may have enough GPU memory to run without migrations.
To exercise offloading, choose a workload and GPU KV capacity that create memory
pressure. Each request must still fit on the GPU when running alone.

`--kv-offload-mode explicit` selects the same path. Omitting both options or using
`--kv-offload-mode disabled` preserves the existing allocator, prefix cache and
placement behavior. Combining `--offload` with `--kv-offload-mode disabled` is an
error. See `python3 -m serving --help` for the option descriptions.

## Scheduling and accounting

The experimental policy serves resident requests first. If their KV growth does
not fit, it tries to migrate a resident starting from the youngest. It can also
migrate a resident to admit a waiting request when that request will fit after a
single migration. This admission swap is allowed only when no request is already
parked, to avoid repeatedly swapping residents just to admit newcomers.

After a STORE completes, the request moves from running to waiting and its GPU
blocks become available to admission. Its computed-token progress is retained in
the CXL snapshot; this does not use the default recompute-preemption path.
Parked requests are restored when space exists for both their snapshot and the
next computation. Restore releases the CXL copy on completion. Decode growth is
reserved before computation, and the updated GPU snapshot becomes valid only
when that computation completes.

The GPU and CXL pools both account for a transfer's space while it is in flight.
If there is insufficient CXL space for an admission swap, resident computation
continues. If resident growth cannot proceed and no migration can be reserved,
the run fails with an explicit capacity error rather than silently dropping KV.

## Scope

The CLI currently requires one independent, single-rank instance with prefix
caching disabled. DP, PD, PIM attention offloading, sub-batch interleaving, power
modeling and the ns-3 backend are not supported by this mode. The configuration
validation and `_build_cxl_execution_pools()` in `serving/__main__.py` enforce
the supported combinations.

In this mode, attention consumes restored GPU KV. The frontend overrides KV read
placement to LOCAL so an existing `kv_loc=CXL` setting does not charge a second
read after restoration. Weight placement is preserved. Migration destinations
are tried in CXL device-ID order, and capacity is tracked in non-caching execution
pools separate from prefix storage.

Transfers are serialized with compute at batch boundaries. This is not layer
prefetch, partial-request staging, or asynchronous copy/compute overlap. The CXL
backend retains its existing per-device FIFO and latency/bandwidth model; PCIe
copy engines and HBM contention are not added. Completed transfer records remain
in memory for duplicate-completion protection.

## Inspecting results

Use `--log-level INFO` to record transfer reservations, completion timestamps,
freed GPU blocks and new admissions. The end-of-run summary reports completed
transfer counts and bytes in each direction. `--keep-inputs` retains the generated
graphs, including transfer-only graphs; `--save-trace-text` retains text traces
for compute batches. Transfer-only batches have no layer text trace.

Compare request CSVs under identical arrivals, token lengths, capacity and batching
settings. For a fair no-offload baseline, set KV read placement to GPU (`npu`) on
both sides: legacy `kv_loc=CXL` represents a different access policy. Measure TTFT,
TPOT, ITL and throughput together. Admission can improve a waiting request's TTFT
while delaying the migrated request and increasing total runtime.

Run the ledger and scheduler tests from the repository root:

```bash
python3 -m unittest serving.core.test_kv_transfer serving.core.test_offload_scheduler
./serving/validate.sh
```
