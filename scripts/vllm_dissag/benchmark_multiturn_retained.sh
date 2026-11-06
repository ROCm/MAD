#!/bin/bash
# =============================================================================
# benchmark_multiturn_retained.sh - warm multi-turn benchmark with an exact retained
# context, exact new tokens per turn and an exact output length.
# Selected via BENCHMARK_SCRIPT=multiturn_retained; also runs standalone against any
# OpenAI-compatible endpoint (e.g. an llm-d gateway) with MT_BASE_URL set.
# =============================================================================
# Per round: fresh sessions from benchmark_multiturn_retained.py (turn 1 = retained context
# + new tokens, identical across rounds, so only round 1 prefills it cold), then vLLM's
# benchmarks/multi_turn client with --warmup-step (turn 1 not measured), MT_T_MEAS measured
# turns per session, min = max output tokens = MT_OSL, X-Session-ID per conversation.
# Before the rounds: one global warmup (16 x 32/32, con 1) and one shape warmup (8 prompts at
# the real ISL/OSL, con 4), as benchmark_xPyD.sh / SHAPE_WARMUP do.
#
# Env:
#   MT_PRESET            R128-10 | R128-30 | R128-70 | H256-30 | H256-70 | M512-30:
#                        sets the next three (concurrency = total over a 1P1D pair)
#   MT_RETAINED_TOKENS   retained context per session      (H256-30: 256000)
#   MT_NEW_TOKENS        new tokens per turn               (H256-30: 30000)
#   BENCHMARK_CON        concurrent sessions (total)       (H256-30: 16)
#   MT_OSL               exact output tokens per turn      (default 1000)
#   MT_T_MEAS            measured turns per session        (default 2)
#   MT_ROUNDS            rounds                            (default 6)
#   MT_STICKY_MOD        DP ranks behind the sticky router (default 8; 0 = plain ids)
#   MT_BASE_URL          endpoint (default http://127.0.0.1:${BENCHMARK_PORT})
#   SERVED_MODEL_NAME    model name the server serves (default ${MODEL_PATH})
#   TOKENIZER            tokenizer dir (default ${MODEL_PATH})
#   MT_GPUS              GPUs for tok/s/GPU (default (xP + yD) * GPUS_PER_NODE)
#   MT_LAYOUT            pd | coloc (default pd)
#   MT_TTFT_SLO_MS       TTFT p90 target for goodput (default max(10 s, retained / 10k tok/s))
#   METRICS_ENDPOINTS    "role=host:port ..." engine /metrics (default: from
#                        AGENTIC_SERVER_METRICS, set by the connector when ENABLE_SERVER_METRICS=1)
#   MT_CORPUS            corpus path (default /tmp/pg1184.txt; downloaded and SHA256-checked)
#   MT_CLIENT_DIR        vLLM benchmarks/multi_turn dir (default /app/vllm/benchmarks/multi_turn)
#   ROUND_TIMEOUT        seconds per round (default 1800)
#   DRY_RUN=1            print the resolved plan and per-round commands, send nothing
# =============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp=$(date "+%Y%m%d_%H%M%S")

case "${MT_PRESET:-H256-30}" in
  R128-10) _r=128000; _n=10000; _c=32 ;;
  R128-30) _r=128000; _n=30000; _c=32 ;;
  R128-70) _r=128000; _n=70000; _c=16 ;;
  H256-30) _r=256000; _n=30000; _c=16 ;;
  H256-70) _r=256000; _n=70000; _c=16 ;;
  M512-30) _r=512000; _n=30000; _c=8 ;;
  *) echo "[mt] unknown MT_PRESET='${MT_PRESET}' (R128-10, R128-30, R128-70, H256-30, H256-70, M512-30)" >&2; exit 2 ;;
esac
RETAINED="${MT_RETAINED_TOKENS:-$_r}"
NEW="${MT_NEW_TOKENS:-$_n}"
CONC="${BENCHMARK_CON:-$_c}"
OSL="${MT_OSL:-1000}"
T_MEAS="${MT_T_MEAS:-2}"
ROUNDS="${MT_ROUNDS:-6}"
MOD="${MT_STICKY_MOD:-8}"
ISL=$(( RETAINED + NEW ))
# TTFT SLO: retained context at 10k tokens/s, at least 10 s (128k: 12.8 s, 256k: 25.6 s, 512k: 51.2 s).
TTFT_SLO_MS="${MT_TTFT_SLO_MS:-$(( RETAINED / 10 > 10000 ? RETAINED / 10 : 10000 ))}"
BASE_URL="${MT_BASE_URL:-http://127.0.0.1:${BENCHMARK_PORT:-30000}}"
BASE_URL="${BASE_URL%/}"
MODEL="${SERVED_MODEL_NAME:-${MODEL_PATH:-}}"
TOK="${TOKENIZER:-${MODEL_PATH:-}}"
GPUS="${MT_GPUS:-$(( (${xP:-1} + ${yD:-1}) * ${GPUS_PER_NODE:-8} ))}"
LAYOUT="${MT_LAYOUT:-pd}"
CORPUS="${MT_CORPUS:-/tmp/pg1184.txt}"
CORPUS_URL="https://www.gutenberg.org/ebooks/1184.txt.utf-8"
CORPUS_SHA256="99245b8f3f6b637a3670d37305bb1ba52003f05797b6638571562999fd623f9c"
CLIENT_DIR="${MT_CLIENT_DIR:-/app/vllm/benchmarks/multi_turn}"
if [[ -z "${METRICS_ENDPOINTS:-}" && -n "${AGENTIC_SERVER_METRICS:-}" ]]; then
  read -r _p _d _ <<< "${AGENTIC_SERVER_METRICS}"
  METRICS_ENDPOINTS="prefill=${_p}${_d:+ decode=${_d}}"
fi
export METRICS_ENDPOINTS="${METRICS_ENDPOINTS:-}"
NAME="${MODEL_NAME:-model}"
TAG="${MT_PRESET:-H256-30}"
[[ -n "${MT_RETAINED_TOKENS:-}${MT_NEW_TOKENS:-}" ]] && TAG="R${RETAINED}-N${NEW}"
RUN_DIR="${MT_RUN_DIR:-/run_logs/${SLURM_JOB_ID:-local}/multiturn_${TAG}_${timestamp}_xP${xP:-1}_yD${yD:-1}_${NAME}}"

echo "==== Warm multi-turn benchmark: retained=${RETAINED} new=${NEW} isl=${ISL} osl=${OSL} con=${CONC} t_meas=${T_MEAS} rounds=${ROUNDS} sticky_mod=${MOD} gpus=${GPUS} layout=${LAYOUT} ===="
echo "url=${BASE_URL} model=${MODEL} metrics='${METRICS_ENDPOINTS}' run_dir=${RUN_DIR}"
[[ -n "$MODEL" && -n "$TOK" ]] || { echo "[mt] set MODEL_PATH (or SERVED_MODEL_NAME and TOKENIZER)" >&2; exit 2; }

wu=(--model "$MODEL" --tokenizer "$TOK" --trust-remote-code --backend vllm --base-url "$BASE_URL"
    --endpoint /v1/completions --dataset-name random --random-prefix-len 0 --random-range-ratio 0
    --ignore-eos --request-rate inf --disable-tqdm)
mt_argv() { # round -> MT_ARGV
  MT_ARGV=(python3 "$RUN_DIR/client/benchmark_serving_multi_turn.py" -i "$RUN_DIR/round-$1/sessions.json"
    -m "$TOK" --served-model-name "$MODEL" --trust-remote-code -u "$BASE_URL"
    -p "$CONC" -k "$CONC" --warmup-step --max-turns $(( 2 * (T_MEAS + 1) ))
    --limit-min-tokens "$OSL" --limit-max-tokens "$OSL" --send-conversation-id
    --request-timeout-sec 1800 --conversation-sampling round_robin --seed "$1"
    --stats-json-output "$RUN_DIR/round-$1/stats.json")
}
gen_argv() { # round -> GEN_ARGV
  GEN_ARGV=(python3 "$DIR/benchmark_multiturn_retained.py" --tokenizer "$TOK" --corpus "$CORPUS"
    --isl "$ISL" --new-tokens "$NEW" --turns $(( T_MEAS + 1 )) --conversations "$CONC"
    --round "$1" --osl "$OSL" --sticky-mod "$MOD" --out "$RUN_DIR/round-$1/sessions.json")
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] corpus: ${CORPUS_URL} sha256=${CORPUS_SHA256}"
  echo "[dry-run] warmup: vllm bench serve ${wu[*]} --random-input-len 32 --random-output-len 32 --num-prompts 16 --max-concurrency 1"
  echo "[dry-run] warmup: vllm bench serve ${wu[*]} --random-input-len ${ISL} --random-output-len ${OSL} --num-prompts 8 --max-concurrency 4"
  for r in $(seq 1 "$ROUNDS"); do
    gen_argv "$r"; mt_argv "$r"
    echo "[dry-run] round $r: ${GEN_ARGV[*]}"; echo "[dry-run] round $r: ${MT_ARGV[*]}"
  done
  echo "[dry-run] reduce: python3 $DIR/multiturn_metrics.py $RUN_DIR --mode multiturn --gpus $GPUS --isl $ISL --osl $OSL --t-meas $T_MEAS --layout $LAYOUT --ttft-slo-ms $TTFT_SLO_MS"
  exit 0
fi

mkdir -p "$RUN_DIR/client"
exec > >(tee -a "$RUN_DIR/bench.log") 2>&1
# Project Gutenberg text used by vLLM's benchmarks/multi_turn README; fetched, never vendored.
[[ -s "$CORPUS" ]] || curl -fsSL --retry 3 "$CORPUS_URL" -o "$CORPUS" || { echo "[mt] corpus download failed" >&2; exit 1; }
echo "${CORPUS_SHA256}  ${CORPUS}" | sha256sum -c - || { echo "[mt] corpus SHA256 mismatch: ${CORPUS}" >&2; exit 1; }
# The client needs per-chunk gaps, reasoning tokens counted as output, and min_tokens sent;
# apply those changes to a private copy of the image's client.
cp "$CLIENT_DIR"/*.py "$RUN_DIR/client/" \
  && git -C "$RUN_DIR/client" apply "$DIR/multiturn/benchmark_serving_multi_turn.diff" \
  || { echo "[mt] cannot prepare the multi-turn client from ${CLIENT_DIR}" >&2; exit 1; }
printf 'retained=%s\nnew=%s\nisl=%s\nosl=%s\nconc=%s\nt_meas=%s\nrounds=%s\nsticky_mod=%s\ngpus=%s\nlayout=%s\nurl=%s\n' \
  "$RETAINED" "$NEW" "$ISL" "$OSL" "$CONC" "$T_MEAS" "$ROUNDS" "$MOD" "$GPUS" "$LAYOUT" "$BASE_URL" > "$RUN_DIR/point.env"

timeout 900 vllm bench serve "${wu[@]}" --random-input-len 32 --random-output-len 32 --num-prompts 16 \
  --max-concurrency 1 --seed 7 > "$RUN_DIR/warmup-global.log" 2>&1 || echo "[STALL] global warmup"
timeout "${SHAPE_WARMUP_TIMEOUT:-2400}" vllm bench serve "${wu[@]}" --random-input-len "$ISL" \
  --random-output-len "$OSL" --num-prompts 8 --max-concurrency 4 --seed 8 \
  > "$RUN_DIR/warmup-shape.log" 2>&1 || echo "[STALL] shape warmup isl=$ISL osl=$OSL"

bash "$DIR/scrape_engine_metrics.sh" sample "$RUN_DIR" & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
for r in $(seq 1 "$ROUNDS"); do
  rd="$RUN_DIR/round-$r"; mkdir -p "$rd"
  gen_argv "$r"; mt_argv "$r"
  "${GEN_ARGV[@]}" > "$rd/sessions.meta.json" || { echo "[mt] session generation failed (round $r)" >&2; exit 1; }
  printf '%q ' "${MT_ARGV[@]}" > "$rd/benchmark_command.txt"; echo >> "$rd/benchmark_command.txt"
  bash "$DIR/scrape_engine_metrics.sh" snapshot "$rd" before
  echo "[RUNNING] round $r isl $ISL new $NEW osl $OSL con $CONC"
  # The client can hang at exit (vllm#42226): bound each round.
  MT_ITL_JSONL="$rd/itl.jsonl" timeout "${ROUND_TIMEOUT:-1800}" "${MT_ARGV[@]}" > "$rd/multi_turn.stdout" 2>&1
  rc=$?
  date -u +%s.%N > "$rd/t_end"
  [[ $rc == 124 ]] && echo "[STALL] round $r timed out after ${ROUND_TIMEOUT:-1800}s" | tee -a "$RUN_DIR/STALLS.log"
  bash "$DIR/scrape_engine_metrics.sh" snapshot "$rd" after
  rm -f "$rd/sessions.json"
done
kill $SAMPLER 2>/dev/null || true

python3 "$DIR/multiturn_metrics.py" "$RUN_DIR" --mode multiturn --gpus "$GPUS" --isl "$ISL" --osl "$OSL" \
  --t-meas "$T_MEAS" --layout "$LAYOUT" --ttft-slo-ms "$TTFT_SLO_MS" | tee "$RUN_DIR/result.line"
python3 "$DIR/parse_to_csv.py" "$RUN_DIR/result.json" --result-json \
  --perf-csv "$(dirname "$RUN_DIR")/perf.csv" --model-name "${NAME}" \
  --workload "mt-R${RETAINED}-N${NEW}"
echo "Warm multi-turn results -> ${RUN_DIR}"
