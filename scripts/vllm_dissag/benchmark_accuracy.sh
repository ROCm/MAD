#!/bin/bash
# Drop-in replacement for benchmark_xPyD.sh that runs the 20-prompt
# correctness suite (benchmark_accuracy.py) against the live disagg server.
# Selected via BENCHMARK_SCRIPT=accuracy. No NIAH, no concurrency sweep.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEM="/run_logs/${SLURM_JOB_ID}/accuracy_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"
LOG="${STEM}.log"

echo "==== accuracy suite ===="
echo "port=${BENCHMARK_PORT}  model=${MODEL_PATH}  thinking=${ACC_THINKING:-chat}  maxtok=${ACC_MAXTOK:-auto}"

_ready=0
for _i in $(seq 1 60); do
    _code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${BENCHMARK_PORT}/ready" 2>/dev/null || echo 000)
    if [ "$_code" = "200" ] || [ "$_code" = "405" ]; then
        _ready=1
        echo "[acc] proxy /ready HTTP ${_code} after ~$((_i*5))s"
        break
    fi
    sleep 5
done
if [ "$_ready" != 1 ]; then
    echo "[acc] Error: /ready not 200/405 in 300s (last HTTP ${_code:-none}). Abort." >&2
    exit 1
fi

ACC_URL="http://127.0.0.1:${BENCHMARK_PORT}" \
ACC_MODEL="${MODEL_PATH}" \
ACC_THINKING="${ACC_THINKING:-chat}" \
ACC_MAXTOK="${ACC_MAXTOK:-}" \
ACC_TIMEOUT="${ACC_TIMEOUT:-900}" \
ACC_JSONL="${STEM}.jsonl" \
  python3 "${DIR}/benchmark_accuracy.py" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "accuracy results -> ${LOG} (${STEM}.jsonl)"
exit "$rc"
