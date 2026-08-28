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

# Give the router a moment to be fully ready for chat completions.
sleep 10

# The model tag to request MUST equal the server's served_model_name, or every
# request 404s and the whole sweep is recorded as FAILURE.
#   * vllm_dissag passes no --served-model-name, so vLLM defaults it to MODEL_PATH.
#   * vllm_multinode passes --served-model-name "$MODEL_NAME" and exports NIAH_MODEL
#     to match.
# Hence: honour NIAH_MODEL when the launcher sets it, else fall back to MODEL_PATH.
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${NIAH_MODEL:-${MODEL_PATH}}" \
NIAH_WORDS="${NIAH_WORDS:-2000,8000,20000,35000}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-8192}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${LOG}"

# Emit the madengine perf.csv so an accuracy run reports a metric like a
# throughput run does (one row per context size, needles found /10). Without
# this, BENCHMARK_SCRIPT=niah produces logs only and the model's
# multiple_results CSV is never written.
python3 "${DIR}/parse_to_csv.py" "${LOG}" --niah \
        --perf-csv "/run_logs/${SLURM_JOB_ID}/perf.csv" \
        --model-name "${MODEL_NAME}" 2>&1 | tee -a "${LOG}"

echo "NIAH results -> ${LOG}"
