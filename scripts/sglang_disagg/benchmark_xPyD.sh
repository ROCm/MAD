#!/bin/bash

timestamp=$(date "+%Y%m%d_%H%M%S")
LOG_PATH="${LOG_PATH:-/shared-inference/${USER_NAME}/model_blog_logs}"
LOG="/${LOG_PATH}/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"

### GSM8K benchmark ###
{
    echo "=== Benchmark Start Time ${LOG} ==="
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================="
    echo ""
} | tee "${LOG}_GSM8K.log"

pushd /sgl-workspace/sglang
python3 benchmark/gsm8k/bench_sglang.py --parallel 1400 --num-questions 1400 2>&1 | tee -a "${LOG}_GSM8K.log"
popd

sleep 5

### Concurrency Sweep Test ###
{
    echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ====="
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""
} | tee -a "${LOG}_CONCURRENCY.log"

CON="32 64 128 256 512 1024"
COMBINATIONS=("4096/100" "2048/100" "1024/1024" "512/1500")

for i in {1..1}; do
    echo "Running the benchserving script for iter: $i" | tee -a "${LOG}_CONCURRENCY.log"
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       for con in $CON; do
           p_con=$(($con * 2))
           if [ "$p_con" -lt 16 ]; then
               p_con=16
           fi
           echo "[RUNNING] prompts $prompts isl $isl osl $osl con $con model ${MODEL_NAME} xP=${xP} yD=${yD} job=${SLURM_JOB_ID}" | tee -a "${LOG}_CONCURRENCY.log"
           python3 -m sglang.bench_serving \
           --backend sglang \
           --host 127.0.0.1 \
           --port 30000 \
           --dataset-name generated-shared-prefix \
           --gsp-system-prompt-len 0 \
           --gsp-question-len $isl \
           --gsp-output-len $osl \
           --gsp-num-groups 1 \
           --gsp-prompts-per-group $p_con \
           --random-range-ratio 1 --pd-separated \
           --max-concurrency $con \
           2>&1 | tee -a "${LOG}_CONCURRENCY.log"

           sleep 10
       done
    done
done


### Concurrency Sweep End Time ###
{
    echo "==== Benchmark Serving Concurrency End Time ${LOG} ====="
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""
} | tee -a "${LOG}_CONCURRENCY.log"
