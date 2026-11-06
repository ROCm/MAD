#!/bin/bash
# Drop-in replacement for benchmark_xPyD.sh that runs the NIAH long-context
# retrieval test (issue vllm-project/vllm#47042) against the live disagg server,
# instead of the throughput sweep. Selected via BENCHMARK_SCRIPT_FILE=benchmark_niah.sh.
#
# Reads (from the launcher env): BENCHMARK_PORT (router/proxy port), MODEL_PATH,
#   MODEL_NAME, SLURM_JOB_ID, xP, yD. NIAH_WORDS overridable.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/run_logs/${SLURM_JOB_ID}/niah_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}.log"

echo "==== NIAH long-context retrieval test ===="
echo "port=${BENCHMARK_PORT}  model=${MODEL_PATH}  sizes=${NIAH_WORDS:-2000,8000,20000,35000}"

# Wait until the router actually serves before starting (replaces a blind sleep). On a
# fresh boot the router may register a few seconds after the workers report ready; poll
# /v1/models until it answers, up to ~5 min. Non-fatal: fall through if the probe can't
# confirm (the harness's own warmup + timeout still protect the run).
_ready=0
for _i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null | grep -q '^200$'; then
        _ready=1; echo "[niah] router ready after ~$((_i*5))s"; break
    fi
    sleep 5
done
[ "$_ready" = 1 ] || echo "[niah] WARN: router readiness not confirmed in 300s; proceeding (warmup will absorb)"

# NIAH_MODE=corpus | multiturn: token-exact needle tests on corpus text (benchmark_niah_corpus.py)
# instead of the word-list test. corpus: NIAH_LENGTHS (tokens, default 65536,262144,524288) x
# NIAH_DEPTHS (0,0.5,1.0) x NIAH_NEEDLES (2) requests. multiturn: NIAH_MT_START, a comma list of
# turn-1 context sizes in tokens (default 246000,490000).
if [[ "${NIAH_MODE:-single}" == "corpus" || "${NIAH_MODE:-single}" == "multiturn" ]]; then
    CORPUS="${MT_CORPUS:-/tmp/pg1184.txt}"
    [[ -s "$CORPUS" ]] || curl -fsSL --retry 3 https://www.gutenberg.org/ebooks/1184.txt.utf-8 -o "$CORPUS"
    echo "99245b8f3f6b637a3670d37305bb1ba52003f05797b6638571562999fd623f9c  $CORPUS" | sha256sum -c - \
        || { echo "[niah] corpus SHA256 mismatch: $CORPUS" >&2; exit 1; }
    _common=(--url "http://127.0.0.1:${BENCHMARK_PORT}" --model "${MODEL_PATH}"
             --tokenizer "${TOKENIZER:-${MODEL_PATH}}" --corpus "$CORPUS")
    if [[ "${NIAH_MODE}" == "corpus" ]]; then
        python3 "${DIR}/benchmark_niah_corpus.py" single "${_common[@]}" --out "${LOG%.log}_single" \
            --lengths "${NIAH_LENGTHS:-65536,262144,524288}" --depths "${NIAH_DEPTHS:-0,0.5,1.0}" \
            --needles "${NIAH_NEEDLES:-2}" --concurrency "${NIAH_CONCURRENCY:-6}" 2>&1 | tee -a "${LOG}"
    else
        IFS=',' read -ra _starts <<< "${NIAH_MT_START:-246000,490000}"
        for _s in "${_starts[@]}"; do
            python3 "${DIR}/benchmark_niah_corpus.py" multi "${_common[@]}" --out "${LOG%.log}_mt${_s}" \
                --start "${_s}" --turns "${NIAH_MT_TURNS:-3}" --new-tokens "${NIAH_MT_NEW_TOKENS:-10000}" \
                --k "${NIAH_MT_K:-3}" --sessions "${NIAH_MT_SESSIONS:-4}" --tag "mt${_s}" 2>&1 | tee -a "${LOG}"
        done
    fi
    python3 "$NIXL_COOKBOOK_PATH/parse_to_csv.py" "${LOG}" --niah-mt \
        --perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv --model-name "${MODEL_NAME}" 2>&1 | tee -a "${LOG}"
    echo "NIAH (${NIAH_MODE}) results -> ${LOG}"
    exit 0
fi

# The server registers the model under its path (served_model_name = MODEL_PATH).
# NIAH_WARMUP=1 (harness default): first-hit JIT compiles off the scored path so a cold
# boot does not produce false 0/10 or timeouts on the first scored request.
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_WORDS="${NIAH_WORDS:-2000,8000,20000,35000}" \
NIAH_SEEDS="${NIAH_SEEDS:-0,1,2}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}" \
NIAH_WARMUP="${NIAH_WARMUP:-1}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${LOG}"

# Generate madengine perf.csv rows from NIAH results (mirrors benchmark_xPyD.sh)
python3 "$NIXL_COOKBOOK_PATH/parse_to_csv.py" "${LOG}" --niah \
    --perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv --model-name "${MODEL_NAME}" \
    2>&1 | tee -a "${LOG}"

echo "NIAH results -> ${LOG}"
