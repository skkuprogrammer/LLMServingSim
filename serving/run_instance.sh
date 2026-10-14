#!/bin/bash
# Usage: ./serving/run_instance.sh <2|4> <RR|RAND|LOAD|CUSTOM> <repeats> [num_reqs] [baseline|pressure] [custom_threshold]
#   e.g. ./serving/run_instance.sh 4 CUSTOM 1 96 pressure 0.7
set -euo pipefail

N=${1:-2}
POLICY=${2:-RR}
K=${3:-1}
NUM_REQS=${4:-}
SCENARIO=${5:-baseline}
CUSTOM_THRESHOLD=${6:-}

case "$N" in
    2|4) ;;
    *) echo "num_instances must be 2 or 4" >&2; exit 2 ;;
esac

case "$SCENARIO" in
    baseline)
        CONFIG="configs/cluster/single_node_${N}_instance.json"
        DATASET="workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"
        NUM_REQS=${NUM_REQS:-100}
        ;;
    pressure)
        CONFIG="configs/cluster/pressure_routing_${N}_instance.json"
        DATASET="workloads/pressure_routing_96_requests.jsonl"
        NUM_REQS=${NUM_REQS:-96}
        ;;
    *) echo "scenario must be baseline or pressure" >&2; exit 2 ;;
esac

OUT_DIR="outputs/real"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
SUFFIX=""
if [[ "$POLICY" == "CUSTOM" && -n "$CUSTOM_THRESHOLD" ]]; then
    EXTRA_ARGS=(--custom-memory-threshold "$CUSTOM_THRESHOLD")
    SUFFIX="_t${CUSTOM_THRESHOLD}"
fi

for ((k = 1; k <= K; k++)); do
    OUT_BASE="${OUT_DIR}/${SCENARIO}_${N}inst_${POLICY}${SUFFIX}_run${k}"
    echo "=== ${SCENARIO} / ${N} instances / ${POLICY} / run ${k}/${K} ==="
    python3 -m serving --cluster-config "$CONFIG" \
        --request-routing-policy "$POLICY" \
        --dtype bfloat16 --block-size 16 \
        --dataset "$DATASET" --num-reqs "$NUM_REQS" "${EXTRA_ARGS[@]}" \
        --output "${OUT_BASE}.csv" 2>&1 | tee "${OUT_BASE}.log"
done
