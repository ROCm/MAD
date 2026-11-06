#!/bin/bash
# =============================================================================
# Throughput benchmark for the single-node vLLM+MoRIIO P/D deployment.
# Runs `vllm bench serve` through the router across a concurrency sweep, for one
# or more ISL/OSL shapes. Assumes run_pd_singlenode.sh is already up (router :30000).
#
# Usage:
#   ./bench_pd.sh                       # default: 1k/1k @ conc 1,8,32
#   SHAPES="8192,1024" CONCS="8 16"     ./bench_pd.sh
#   MODEL=/models/MiniMax-M3-MXFP4 ROUTER_PORT=30000 ./bench_pd.sh
# =============================================================================
set -uo pipefail
IMG="${IMG:-rocm/vllm-dev:vllm-0.23.1-rocm723-mi35x-mori-0625}"
MODEL="${MODEL:-/models/MiniMax-M3-MXFP4}"
MOUNT="${MOUNT:-/models}"
LOG="${LOG:-$PWD/logs/bench}"; mkdir -p "$LOG"; LOG="$(cd "$LOG" && pwd)"   # absolute (docker -v needs it)
ROUTER_PORT="${ROUTER_PORT:-30000}"
SHAPES="${SHAPES:-1024,1024}"     # comma-separated "isl,osl"; space-separate multiple shapes
CONCS="${CONCS:-1 8 32}"
HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{print $7}')

echo "[bench] router=$HOST_IP:$ROUTER_PORT shapes='$SHAPES' concs='$CONCS' $(date)" | tee "$LOG/bench.log"
for shape in $SHAPES; do
  ISL="${shape%,*}"; OSL="${shape#*,}"
  for c in $CONCS; do
    echo "[bench] --- isl=$ISL osl=$OSL conc=$c ---" | tee -a "$LOG/bench.log"
    docker run --rm --entrypoint bash --network host -v "$MOUNT":"$MOUNT" -v "$LOG":"$LOG" "$IMG" -c "
      vllm bench serve --backend openai-chat \
        --base-url http://$HOST_IP:$ROUTER_PORT --endpoint /v1/chat/completions \
        --model $MODEL --served-model-name minimaxm3 --dataset-name random \
        --random-input-len $ISL --random-output-len $OSL \
        --num-prompts \$(( c > 4 ? c*2 : 8 )) --max-concurrency $c \
        --percentile-metrics ttft,tpot,e2el --save-result \
        --result-dir $LOG --result-filename bench_isl${ISL}_osl${OSL}_conc${c}.json 2>&1" \
      | grep -iE "Successful|throughput|Mean TTFT|Median TTFT|Mean TPOT" | tee -a "$LOG/bench.log"
  done
done
echo "[bench] DONE $(date)  results in $LOG" | tee -a "$LOG/bench.log"
