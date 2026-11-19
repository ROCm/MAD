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

# Prefer /ready (200). vllm-router answers GET 405
# "Only POST requests are supported for transparent proxy" — the process is up.
# /ready is the stronger signal (Hypercorn up, ZMQ dead → /v1/models 200
# but completions 503), so a proxy that serves it is accepted on that alone.
# Proxies that do not serve /ready still fall back to the /v1/models probe, which
# is what every non-DSV4 model used before this path existed.
_ready=0
for _i in $(seq 1 60); do
    _code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${BENCHMARK_PORT}/ready" 2>/dev/null || echo 000)
    if [ "$_code" = "200" ] || [ "$_code" = "405" ]; then
        _ready=1
        echo "[niah] proxy /ready HTTP ${_code} after ~$((_i*5))s"
        break
    fi
    _mcode=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null || echo 000)
    if [ "$_mcode" = "200" ]; then
        _ready=1
        echo "[niah] no /ready (HTTP ${_code}); /v1/models 200 after ~$((_i*5))s"
        break
    fi
    sleep 5
done
if [ "$_ready" != 1 ]; then
    echo "[niah] WARN: neither /ready (last HTTP ${_code:-none}) nor /v1/models (last HTTP ${_mcode:-none})" \
         "confirmed in 300s; proceeding (warmup + per-request timeout still protect the run)" >&2
    curl -s -D - "http://127.0.0.1:${BENCHMARK_PORT}/ready" >&2 || true
fi

# The server registers the model under its path (served_model_name = MODEL_PATH).
# Uses /v1/completions with stream + x-request-id (disagg protocol).
# NIAH_WARMUP=1 (harness default): first-hit JIT compiles off the scored path so a cold
# boot does not produce false 0/10 or timeouts on the first scored request.
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_WORDS="${NIAH_WORDS:-2000,8000,20000,35000}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}" \
NIAH_WARMUP="${NIAH_WARMUP:-1}" \
NIAH_SEEDS="${NIAH_SEEDS:-0,1,2}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${LOG}"
# Includes harness failure: python exits 1 when every request is NO-RESULT.
niah_rc=${PIPESTATUS[0]}

# Generate madengine perf.csv rows from NIAH results (mirrors benchmark_xPyD.sh)
python3 "$NIXL_COOKBOOK_PATH/parse_to_csv.py" "${LOG}" --niah \
    --perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv --model-name "${MODEL_NAME}" \
    2>&1 | tee -a "${LOG}"
parse_rc=${PIPESTATUS[0]}

echo "NIAH results -> ${LOG}"
if [ "$niah_rc" -ne 0 ]; then
    exit "$niah_rc"
fi
exit "$parse_rc"
