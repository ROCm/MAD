#!/bin/bash
# Accuracy of the served model (BENCHMARK_SCRIPT=accuracy): benchmark_accuracy.py over the task
# sets in ACC_TASKS (default: the arithmetic smoke set beside this script), then one perf.csv row
# per set (parse_to_csv.py --accuracy). Selected via BENCHMARK_SCRIPT_FILE=benchmark_accuracy.sh.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/run_logs/${SLURM_JOB_ID}/accuracy_${SLURM_JOB_ID}_${timestamp}_xP${xP:-1}_yD${yD:-1}_${MODEL_NAME}.log"

echo "==== accuracy: ${ACC_TASKS:-${DIR}/accuracy_tasks/arith_smoke.jsonl} ===="
_ready=0
for _i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null | grep -q '^200$'; then
        _ready=1; echo "[accuracy] endpoint ready after ~$((_i*5))s"; break
    fi
    sleep 5
done
[ "$_ready" = 1 ] || echo "[accuracy] WARN: endpoint readiness not confirmed in 300s; proceeding"

ACC_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
ACC_MODEL="${ACC_MODEL:-${SERVED_MODEL_NAME:-${MODEL_PATH}}}" \
ACC_TASKS="${ACC_TASKS:-${DIR}/accuracy_tasks/arith_smoke.jsonl}" \
  python3 "${DIR}/benchmark_accuracy.py" 2>&1 | tee -a "${LOG}"

python3 "${DIR}/parse_to_csv.py" "${LOG}" --accuracy \
        --perf-csv "/run_logs/${SLURM_JOB_ID}/perf.csv" \
        --model-name "${MODEL_NAME}" 2>&1 | tee -a "${LOG}"
echo "accuracy results -> ${LOG}"
