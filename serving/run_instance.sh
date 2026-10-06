#!/bin/bash
# Usage: ./serving/run_instance.sh <num_instances> <policy> <repeats> [num_reqs]
#   e.g. ./serving/run_instance.sh 4 LOAD 3
set -e

N=${1:-2}              # number of instances (2 or 4)
POLICY=${2:-RR}        # RR / LOAD / CUSTOM
K=${3:-1}              # how many times to repeat the same workload
NUM_REQS=${4:-100}     # number of requests to load from the dataset (first N lines)

CONFIG="configs/cluster/single_node_${N}_instance.json"
#DATASET="workloads/example_trace.jsonl"
DATASET="workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"
OUT_DIR="outputs/real"
mkdir -p "$OUT_DIR"

for k in $(seq 1 "$K"); do
    echo "=== ${N} instances / ${POLICY} / run ${k}/${K} ==="
    python3 -m serving --cluster-config "$CONFIG" \
        --request-routing-policy "$POLICY" \
        --dtype bfloat16 --block-size 16 \
        --dataset "$DATASET" --num-reqs "$NUM_REQS" \
        --output "${OUT_DIR}/${N}inst_${POLICY}_run${k}.csv"
done