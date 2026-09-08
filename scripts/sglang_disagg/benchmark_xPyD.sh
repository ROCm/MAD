#!/bin/bash

timestamp=$(date "+%Y%m%d_%H%M%S")
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"
echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ===== "
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
 
sleep 60
echo "Test run:" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
python3 -m sglang.bench_serving \
           --model $MODEL_PATH \
           --backend sglang \
           --host 127.0.0.1 \
           --port 2322 \
           --dataset-name random \
           --random-input 1024 \
           --random-output 1024\
           --random-range-ratio 1.0 \
           --max-concurrency 512 \
           --num-prompt 1024 \
           --pd-separated \
	   2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo ""
CON="8 16 32 64 128 256 512"
# ISL/OSL combinations — override via BENCHMARK_COMBINATIONS env var (space-separated, e.g. "1024/1024 8192/1024")
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}"
echo "Benchmarking iterations: $BENCHMARK_ITR" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
# seq, not {1..$BENCHMARK_ITR}: brace expansion happens before parameter expansion, so the
# brace form iterates once over the literal string "{1..2}". Job 252775 ran with
# BENCHMARK_ITR=2, logged "iter: {1..2}", and swept each concurrency exactly once.
# scripts/vllm_dissag/benchmark_xPyD.sh already uses the seq form.
for i in $(seq 1 "${BENCHMARK_ITR:-1}"); do
    sleep 60
    echo "RUNNING: the benchserving script for iter: $i" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       for con in $CON; do
           p_con=$(($con * 2))
           if [ "$p_con" -lt 16 ]; then
               p_con=16
           fi
           # $prompts was never set; the prompt count for this cell is $p_con.
           echo "RUNNING: prompts $p_con isl $isl osl $osl con $con" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
           python3 -m sglang.bench_serving \
           --model $MODEL_PATH \
           --backend sglang \
           --host 127.0.0.1 \
           --port 2322 \
           --dataset-name random \
           --random-input $isl \
           --random-output $osl \
           --random-range-ratio 1.0 \
           --max-concurrency $con \
           --num-prompt $p_con \
           --pd-separated \
           2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null

	   sleep 10
        done
    done
done
python3 parse_to_csv.py ${LOG}_CONCURRENCY.log  -o ${LOG}_CONCURRENCY.csv \
	--perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv \
	--model-name "${MODEL_NAME}" \
	2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
