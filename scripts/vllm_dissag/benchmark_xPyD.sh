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
echo "Warmup run:" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
vllm bench serve \
    --model $MODEL_PATH \
    --backend vllm \
    --host 127.0.0.1 \
    --port $BENCHMARK_PORT \
    --dataset-name "random" \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-prefix-len 0 \
    --num-prompts 16 \
    --request-rate "inf" \
    --ignore-eos \
    --max-concurrency 16 \
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
           echo "[RUNNING] prompts $p_con isl $isl osl $osl con $con"
           vllm bench serve \
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

       sleep 10
        done
    done
done
