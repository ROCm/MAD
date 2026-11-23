#!/bin/bash
# Multi-LoRA serving (BENCHMARK_SCRIPT=lora): benchmark_lora.py over the adapters in LORA_ADAPTERS,
# one perf.csv row per concurrency (parse_to_csv.py --lora). The adapters come from the role's
# lora: in the recipe (scripts/common/recipe), which the engine loads beside the base model.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/run_logs/${SLURM_JOB_ID}/lora_${SLURM_JOB_ID}_${timestamp}_${MODEL_NAME}.log"
_ready=0
for _i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null | grep -q '^200$'; then
        _ready=1; break
    fi
    sleep 5
done
[ "$_ready" = 1 ] || echo "[lora] WARN: endpoint readiness not confirmed in 300s; proceeding"
LORA_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
  python3 "${DIR}/benchmark_lora.py" 2>&1 | tee -a "${LOG}"
python3 "${DIR}/parse_to_csv.py" "${LOG}" --lora \
        --perf-csv "/run_logs/${SLURM_JOB_ID}/perf.csv" \
        --model-name "${MODEL_NAME}" 2>&1 | tee -a "${LOG}"
echo "lora results -> ${LOG}"
