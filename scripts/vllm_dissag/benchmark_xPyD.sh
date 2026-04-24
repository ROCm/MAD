#!/bin/bash

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"

echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ===== "
echo "Benchmark Port: ${BENCHMARK_PORT}"
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null

sleep 10
WARMUP_CON="${WARMUP_CON:-1}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-16}"
WARMUP_ISL="${WARMUP_ISL:-32}"
WARMUP_OSL="${WARMUP_OSL:-32}"
echo "Warmup run: ${WARMUP_PROMPTS} prompts @ con=${WARMUP_CON} isl=${WARMUP_ISL} osl=${WARMUP_OSL}" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
vllm bench serve \
    --model $MODEL_PATH \
    --backend vllm \
    --host 127.0.0.1 \
    --port $BENCHMARK_PORT \
    --dataset-name "random" \
    --random-input-len $WARMUP_ISL \
    --random-output-len $WARMUP_OSL \
    --random-prefix-len 0 \
    --num-prompts $WARMUP_PROMPTS \
    --request-rate "inf" \
    --ignore-eos \
    --max-concurrency $WARMUP_CON \
    2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo ""

CON="${BENCHMARK_CON:-8 16 32 64 128 256 512}"
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024 1024/8192}"

echo "Benchmarking iterations: $BENCHMARK_ITR" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
for i in $(seq 1 $BENCHMARK_ITR); do
    echo "Running the benchserving script for iter: $i"
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       for con in $CON; do
           p_con=$(($con * 2))
           if [ "$p_con" -lt 16 ]; then
               p_con=16
           fi
           _base_timeout="${STEP_TIMEOUT:-1800}"
           _total_tok=$(( isl + osl ))
           _scaled_timeout=$(( _base_timeout * _total_tok / 2048 ))
           if [ "$_scaled_timeout" -lt "$_base_timeout" ]; then
               _scaled_timeout=$_base_timeout
           fi
           echo "[RUNNING] prompts $p_con isl $isl osl $osl con $con (timeout ${_scaled_timeout}s)"
           timeout $_scaled_timeout vllm bench serve \
           --model $MODEL_PATH \
           --backend vllm \
           --host 127.0.0.1 \
           --port $BENCHMARK_PORT \
           --dataset-name "random" \
           --random-input-len $isl \
           --random-output-len $osl \
           --random-prefix-len 0 \
           --num-prompts $p_con \
           --request-rate "inf" \
           --ignore-eos \
           --max-concurrency $con \
           2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
           rc=${PIPESTATUS[0]}
           if [ $rc -eq 124 ]; then
               echo "[STALL] isl=$isl osl=$osl con=$con timed out after ${_scaled_timeout}s" \
                   | tee -a ${LOG}_CONCURRENCY.log ${LOG}_STALLS.log >/dev/null
           fi

       sleep 10
        done
    done
done
