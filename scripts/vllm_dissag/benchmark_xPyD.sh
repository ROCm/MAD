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
    echo "Running the benchserving script for iter: $i" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       # Per-shape warmup at the REAL isl/osl, low concurrency. The global warmup above
       # is isl=osl=32/con=1, which never exercises this shape's prefill path, its Triton/
       # aiter kernel variants, or the decode cudagraph batch sizes -- so without this the
       # FIRST measured cell of each shape absorbs all the residual JIT and reports an
       # inflated TPOT. Measured cells must start from a warm graph. Skip with
       # SHAPE_WARMUP=0. (Do not confuse this with the ~302ms -> ~88ms TPOT figure in the
       # GLM-5.1-FP8 recipe: that one is the over-wide MoRI EP all2all buffer, fixed by
       # --max-num-batched-tokens, not by warmup.)
       if [[ "${SHAPE_WARMUP:-1}" == "1" ]]; then
           _w_con="${SHAPE_WARMUP_CON:-4}"
           _w_prompts="${SHAPE_WARMUP_PROMPTS:-8}"
           echo "[WARMUP] shape isl $isl osl $osl con ${_w_con} prompts ${_w_prompts}" \
               | tee -a ${LOG}_CONCURRENCY.log >/dev/null
           timeout "${SHAPE_WARMUP_TIMEOUT:-2400}" vllm bench serve \
               --model $MODEL_PATH \
               --backend vllm \
               --host 127.0.0.1 \
               --port $BENCHMARK_PORT \
               --dataset-name "random" \
               --random-input-len $isl \
               --random-output-len $osl \
               --random-prefix-len 0 \
               --num-prompts ${_w_prompts} \
               --request-rate "inf" \
               --ignore-eos \
               --max-concurrency ${_w_con} \
               2>&1 | tee -a ${LOG}_SHAPEWARMUP.log >/dev/null
       fi
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
           echo "[RUNNING] prompts $p_con isl $isl osl $osl con $con (timeout ${_scaled_timeout}s)" \
               | tee -a ${LOG}_CONCURRENCY.log >/dev/null
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
python3 $NIXL_COOKBOOK_PATH/parse_to_csv.py ${LOG}_CONCURRENCY.log -o ${LOG}_CONCURRENCY.csv \
	--perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv \
	--model-name "${MODEL_NAME}" \
	2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
