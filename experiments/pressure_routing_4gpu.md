# Four-GPU pressure-routing check (2026-10-14)

Single simulator run per policy; these are diagnostic results, not statistically
robust performance claims or real-traffic measurements.

## Setup

- Cluster: `configs/cluster/pressure_routing_4_instance.json` (4 independent
  RTX 4090 instances, 24 GiB HBM each, `mem_util=0.9`, Llama-3.1-8B, TP=1).
- Workload: `workloads/pressure_routing_96_requests.jsonl` (96 requests,
  2,048 input and 512 output tokens each, 20 ms between arrivals).
- Simulator: bfloat16, block size 16, prefix caching and chunked prefill enabled,
  no CXL or CPU KV offload. RAND uses the router's default seed 42.
- Commands: `./serving/run_instance.sh 4 RR 1 96 pressure`, with `RAND` or
  `CUSTOM` replacing `RR`; the lowered-threshold run used
  `./serving/run_instance.sh 4 CUSTOM 1 96 pressure 0.7`.

## Results

| Policy | CUSTOM threshold | Memory-mode selections | Total latency (s) | Requests/s | Mean request latency (ms) | Mean TTFT (ms) | Preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RR | — | — | 24.410 | 3.93 | 18,401.12 | 1,903.38 | 12 |
| RAND | — | — | 29.895 | 3.21 | 18,887.18 | 2,322.07 | 13 |
| CUSTOM | 0.80 | 0 / 96 | 24.410 | 3.93 | 18,401.12 | 1,903.38 | 12 |
| CUSTOM | 0.70 | 24 / 96 | 26.594 | 3.61 | 18,452.61 | 1,911.27 | 12 |

The 0.80 CUSTOM run's maximum **routing-time** HBM pressure was 72.79%, so
its memory branch never activated and the result matched RR. Runtime HBM
occupancy later reached about 89.8%, after the relevant routing decisions.
Lowering the threshold to 0.70 activated the branch for 24 requests, but this
single run completed 2.184 s later than RR. A lower threshold is useful for
exercising the branch, not yet supported as a performance improvement. Weight
memory alone occupies roughly 63.5% of the 24 GiB HBM in this setup, so very
low total-HBM thresholds would activate almost immediately.

Raw per-request CSVs and full logs, including `CUSTOM_ROUTING_METADATA`, are
under `outputs/real/pressure_4inst_*_run1.{csv,log}` (gitignored). The table
uses simulator summary latency/throughput/preemptions and arithmetic means of
CSV request latency and TTFT (nanoseconds converted to milliseconds).
