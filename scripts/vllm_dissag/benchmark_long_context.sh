#!/bin/bash
# =============================================================================
# benchmark_long_context.sh - long-context / steady-state serving benchmark
# (per-shape warmup).
# =============================================================================
# Difference vs benchmark_xPyD.sh (the general harness):
#   - PER-SHAPE warmup: each (ISL/OSL, concurrency) cell runs --num-warmups warmup
#     requests of the SAME shape first (discarded), so the measured result is
#     steady-state and not contaminated by first-hit JIT/cudagraph/kernel-autotune
#     or connector-handshake costs. (The general harness does one tiny isl=32
#     warmup at the start, which does not prime the large-context paths.)
#   - c=1 FIRST: concurrency=1 is the primary latency metric, so it is
#     measured first (and cleanly warmed) before higher concurrencies.
#   - Metrics surfaced: total throughput/GPU, TTFT, ITL/TPOT.
#
# Env (same names as benchmark_xPyD.sh, plus WARMUPS):
#   BENCHMARK_CON            concurrency list           (default "1 4 8")
#   BENCHMARK_COMBINATIONS   ISL/OSL list               (default "1024/1024")
#   WARMUPS                  --num-warmups per cell      (default 2)
#   NUM_PROMPTS_FACTOR       measured prompts = factor*con (default 4, min 16)
#   STEP_TIMEOUT             base timeout (s), scaled by tokens (default 2400)
# =============================================================================

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
GPUS_TOTAL="${GPUS_TOTAL:-$(( (${xP:-1} > ${yD:-1} ? ${xP:-1} : ${yD:-1}) * ${GPUS_PER_NODE:-8} ))}"
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_long_context_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"

CON="${BENCHMARK_CON:-1 4 8}"
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024}"
WARMUPS="${WARMUPS:-2}"
NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"

echo "==== Long-context benchmark (per-shape warmup=${WARMUPS}, EP GPUs=${GPUS_TOTAL}) ${LOG} ====" \
    | tee -a "${LOG}_CONCURRENCY.log" >/dev/null
echo "Port ${BENCHMARK_PORT}  combos='${BENCHMARK_COMBINATIONS}'  con='${CON}'" \
    | tee -a "${LOG}_CONCURRENCY.log" >/dev/null

sleep 10

for combo in "${COMBINATIONS[@]}"; do
    IFS="/" read -r isl osl <<< "$combo"
    # Concurrency order follows BENCHMARK_CON as given; the default "1 4 8" lists
    # c=1 first so the primary latency metric is measured first (and cleanly warmed).
    # Measured prompts per cell = NUM_PROMPTS_FACTOR * con (default 4, min 16) --
    # differs from benchmark_xPyD.sh, which uses con*2.
    for con in $CON; do
        n_prompts=$(( con * NUM_PROMPTS_FACTOR ))
        [ "$n_prompts" -lt 16 ] && n_prompts=16
        _base_timeout="${STEP_TIMEOUT:-2400}"
        _total_tok=$(( isl + osl ))
        _scaled_timeout=$(( _base_timeout * _total_tok / 2048 ))
        [ "$_scaled_timeout" -lt "$_base_timeout" ] && _scaled_timeout=$_base_timeout

        echo "[RUNNING] isl=$isl osl=$osl con=$con warmups=$WARMUPS prompts=$n_prompts (timeout ${_scaled_timeout}s)" \
            | tee -a "${LOG}_CONCURRENCY.log" >/dev/null
        timeout "$_scaled_timeout" vllm bench serve \
            --model "$MODEL_PATH" \
            --backend vllm \
            --host 127.0.0.1 \
            --port "$BENCHMARK_PORT" \
            --dataset-name random \
            --random-input-len "$isl" \
            --random-output-len "$osl" \
            --random-prefix-len 0 \
            --num-prompts "$n_prompts" \
            --num-warmups "$WARMUPS" \
            --request-rate inf \
            --ignore-eos \
            --max-concurrency "$con" \
            2>&1 | tee -a "${LOG}_CONCURRENCY.log" >/dev/null
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 124 ]; then
            echo "[STALL] isl=$isl osl=$osl con=$con timed out after ${_scaled_timeout}s" \
                | tee -a "${LOG}_CONCURRENCY.log" "${LOG}_STALLS.log" >/dev/null
        fi
        sleep 10
    done
done
echo "==== Long-context benchmark complete ====" | tee -a "${LOG}_CONCURRENCY.log" >/dev/null
