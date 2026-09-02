#!/bin/bash
# Prefix-reuse benchmark — exercises tiered prefix caching (KV_OFFLOAD).
# Sends a large SHARED prefix with a FIXED seed, then runs the same request set
# twice: cold (cache empty) then warm (prefix reused). The cold->warm TTFT drop
# is the tiered-cache signal. Select via BENCHMARK_SCRIPT=prefix_cache.
#
# Env knobs (defaults):
#   PC_PREFIX_LEN shared prefix tokens (4096)   PC_NUM_PROMPTS prompts/pass (64)
#   PC_ISL        unique input tokens (1024)    PC_CON         max concurrency (32)
#   PC_OSL        output tokens (128)           PC_SEED        RNG seed (12345)

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
LOG="/run_logs/${SLURM_JOB_ID}/prefixcache_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"

PC_PREFIX_LEN="${PC_PREFIX_LEN:-4096}"
PC_ISL="${PC_ISL:-1024}"
PC_OSL="${PC_OSL:-128}"
PC_NUM_PROMPTS="${PC_NUM_PROMPTS:-64}"
PC_CON="${PC_CON:-32}"
PC_SEED="${PC_SEED:-12345}"

echo "==== Prefix-cache benchmark ${LOG} =====" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "Port ${BENCHMARK_PORT}  prefix_len=${PC_PREFIX_LEN} isl=${PC_ISL} osl=${PC_OSL} prompts=${PC_NUM_PROMPTS} con=${PC_CON} seed=${PC_SEED} KV_OFFLOAD=${KV_OFFLOAD:-none}" \
    | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null

sleep 10

# Warmup (small, does not touch the measured prefix).
vllm bench serve \
    --model $MODEL_PATH --backend vllm --host 127.0.0.1 --port $BENCHMARK_PORT \
    --dataset-name random --random-input-len 32 --random-output-len 32 --random-prefix-len 0 \
    --num-prompts 16 --request-rate inf --ignore-eos --max-concurrency 1 \
    2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null

_run_pass() {
    local label="$1"
    echo "[PASS ${label}] prefix_len=${PC_PREFIX_LEN} isl=${PC_ISL} osl=${PC_OSL} con=${PC_CON} seed=${PC_SEED}" \
        | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    vllm bench serve \
        --model $MODEL_PATH \
        --backend vllm \
        --host 127.0.0.1 \
        --port $BENCHMARK_PORT \
        --dataset-name random \
        --random-input-len $PC_ISL \
        --random-output-len $PC_OSL \
        --random-prefix-len $PC_PREFIX_LEN \
        --num-prompts $PC_NUM_PROMPTS \
        --request-rate inf \
        --ignore-eos \
        --seed $PC_SEED \
        --max-concurrency $PC_CON \
        2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
}

# Cold pass populates the cache; warm pass reuses the same shared prefix.
_run_pass cold
sleep 10
_run_pass warm

python3 $NIXL_COOKBOOK_PATH/parse_to_csv.py ${LOG}_CONCURRENCY.log -o ${LOG}_CONCURRENCY.csv \
    --perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv \
    --model-name "${MODEL_NAME}" \
    2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
