#!/bin/bash
# bench.sh — driven by server_atom.sh on node 0 (router node), AFTER atomesh is /v1/models-ready.
# Reproduces the InferenceX call signature exactly:
#   bash bench.sh <xP> <yD> <PF_total_gpus> <DEC_total_gpus> <model_dir> <model_name> \
#                 <log_dir> <isl> <osl> "<conc_list>" <req_rate> <range_ratio> <prompts_mult>
#
# Drives the OpenAI-compatible ATOM router at localhost:$ROUTER_PORT with the vendored
# InferenceX benchmark_serving client (same one used for the Qwen sweep) — one result JSON
# per concurrency. This is a TRUE aggregate (one endpoint, real PD + mooncake KV), no x N.
set -uo pipefail
xP="$1"; yD="$2"; PF_GPUS="$3"; DEC_GPUS="$4"
MODEL_DIR="$5"; MODEL_NAME="$6"; LOG_DIR="$7"
ISL="$8"; OSL="$9"; CONC_LIST="${10}"; REQ_RATE="${11}"; RANGE_RATIO="${12}"; PROMPTS_MULT="${13}"
# server_atom.sh passes the conc list 'x'-separated (e.g. 4x8x16). Normalise to spaces.
CONC_LIST=$(echo "$CONC_LIST" | tr 'x' ' ')

ROUTER_PORT="${ROUTER_PORT:-8000}"
BENCH_CLIENT="${BENCH_CLIENT:-/scripts/utils/bench_serving/benchmark_serving.py}"
OUT="${LOG_DIR}"; mkdir -p "$OUT" 2>/dev/null || true

echo "[bench] router=localhost:${ROUTER_PORT} model=${MODEL_NAME} isl=${ISL} osl=${OSL} conc='${CONC_LIST}' (xP=${xP} yD=${yD})"

# --- ACCURACY GATE (context-varied) before perf -------------------------------------------
# Validate the FULL PD pipeline (prefill+mooncake KV+decode) returns correct output at varied
# context sizes. Runs ONCE per launch (guarded by a marker), not per-shape. Non-fatal: logs
# result but does not abort the perf sweep (so we still get numbers even if a long-ctx probe fails).
ACC_OUT="${OUT}/accuracy_ctx.json"
if [[ ! -f "$ACC_OUT" ]]; then
  echo "[accuracy] context-varied probe via router (sizes: ${CTX_SIZES:-512 2048 8192 16384})"
  ROUTER_URL="http://localhost:${ROUTER_PORT}" MODEL="${MODEL_DIR}/${MODEL_NAME}" \
    OUT="$ACC_OUT" CTX_SIZES="${CTX_SIZES:-512 2048 8192 16384}" \
    python3 /ws/accuracy_ctx.py || echo "[accuracy] WARN: some context sizes failed (see $ACC_OUT)"
fi

for c in $CONC_LIST; do
  echo "[bench] --- concurrency $c ---"
  python3 "$BENCH_CLIENT" \
    --model "${MODEL_DIR}/${MODEL_NAME}" --backend vllm \
    --base-url "http://localhost:${ROUTER_PORT}" --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio "$RANGE_RATIO" \
    --num-prompts "$(( c * PROMPTS_MULT ))" --num-warmups "$(( c * 2 ))" \
    --max-concurrency "$c" --request-rate "$REQ_RATE" --ignore-eos --save-result \
    --percentile-metrics ttft,tpot,itl,e2el --result-dir "$OUT" \
    --result-filename "disagg_${xP}p${yD}d_isl${ISL}_osl${OSL}_c${c}.json" \
    || echo "[bench] FAILED isl=$ISL osl=$OSL conc=$c"
done
echo "[bench] done -> $OUT"
