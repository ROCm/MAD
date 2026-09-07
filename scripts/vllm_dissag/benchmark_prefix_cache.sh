#!/bin/bash
# Prefix-reuse benchmark — exercises tiered prefix caching (KV_OFFLOAD).
# Sends a large SHARED prefix, then runs the same request set twice: cold (cache empty)
# then warm (prefix reused). The cold->warm TTFT drop is the tiered-cache signal.
# Sweeps concurrency (PC_CON_LIST); each point uses a fresh seed so its cold pass is
# genuinely cold even though the server/CPU tier stay warm across points.
# Select via BENCHMARK_SCRIPT=prefix_cache.
#
# Env knobs (defaults):
#   PC_PREFIX_LEN shared prefix tokens (4096)   PC_NUM_PROMPTS prompts/pass (64)
#   PC_ISL        unique input tokens (1024)    PC_CON_LIST    concurrency sweep (PC_CON or 32)
#   PC_OSL        output tokens (128)           PC_SEED        base RNG seed (12345)

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
LOG="/run_logs/${SLURM_JOB_ID}/prefixcache_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"

PC_PREFIX_LEN="${PC_PREFIX_LEN:-4096}"
PC_ISL="${PC_ISL:-1024}"
PC_OSL="${PC_OSL:-128}"
PC_NUM_PROMPTS="${PC_NUM_PROMPTS:-64}"
# PC_CON_LIST sweeps concurrency; PC_CON kept as a single-point fallback for back-compat.
PC_CON_LIST="${PC_CON_LIST:-${PC_CON:-32}}"
PC_SEED="${PC_SEED:-12345}"

# Per-(con,pass) result JSONs land next to the log so aggregate_sweep.py can read them.
RESULT_DIR="/run_logs/${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

echo "==== Prefix-cache benchmark ${LOG} =====" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "Port ${BENCHMARK_PORT}  prefix_len=${PC_PREFIX_LEN} isl=${PC_ISL} osl=${PC_OSL} prompts=${PC_NUM_PROMPTS} con_list='${PC_CON_LIST}' seed=${PC_SEED} KV_OFFLOAD=${KV_OFFLOAD:-none}" \
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
    local label="$1" con="$2" seed="$3"
    echo "[RUNNING] pass ${label} prefix_len=${PC_PREFIX_LEN} isl=${PC_ISL} osl=${PC_OSL} con ${con} seed=${seed}" \
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
        --seed $seed \
        --max-concurrency $con \
        --save-result \
        --result-filename "${RESULT_DIR}/con${con}_${label}.json" \
        2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
}

# Sweep concurrency. Each point gets a fresh seed so its cold pass is genuinely cold even
# though the server/CPU tier stay warm across points; a single working set fits the CPU
# tier, so LRU evicts the prior point's stale set before the next warm pass.
idx=0
for con in $PC_CON_LIST; do
    seed=$((PC_SEED + idx))
    echo "==== concurrency ${con} (seed ${seed}) ====" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    _run_pass cold "$con" "$seed"
    sleep 10
    _run_pass warm "$con" "$seed"
    idx=$((idx + 1))
done

python3 $NIXL_COOKBOOK_PATH/parse_to_csv.py ${LOG}_CONCURRENCY.log -o ${LOG}_CONCURRENCY.csv \
    --perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv \
    --model-name "${MODEL_NAME}" \
    2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
