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

# The server registers the model under its path (served_model_name = MODEL_PATH).
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_WORDS="${NIAH_WORDS:-2000,8000,20000,35000}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${LOG}"

echo "NIAH results -> ${LOG}"
