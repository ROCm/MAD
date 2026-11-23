#!/bin/bash
# Embeddings serving (BENCHMARK_SCRIPT=embeddings): benchmark_embeddings.py against the endpoint,
# one perf.csv row per concurrency (parse_to_csv.py --embeddings). The server runs as an
# embedding model through its flags (e.g. COLOCATED_EXTRA_ARGS="--runner pooling" on vLLM).
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/run_logs/${SLURM_JOB_ID}/embeddings_${SLURM_JOB_ID}_${timestamp}_${MODEL_NAME}.log"
_ready=0
for _i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null | grep -q '^200$'; then
        _ready=1; break
    fi
    sleep 5
done
[ "$_ready" = 1 ] || echo "[embeddings] WARN: endpoint readiness not confirmed in 300s; proceeding"
EMB_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/embeddings" \
EMB_MODEL="${EMB_MODEL:-${SERVED_MODEL_NAME:-${MODEL_PATH}}}" \
  python3 "${DIR}/benchmark_embeddings.py" 2>&1 | tee -a "${LOG}"
python3 "${DIR}/parse_to_csv.py" "${LOG}" --embeddings \
        --perf-csv "/run_logs/${SLURM_JOB_ID}/perf.csv" \
        --model-name "${MODEL_NAME}" 2>&1 | tee -a "${LOG}"
echo "embeddings results -> ${LOG}"
