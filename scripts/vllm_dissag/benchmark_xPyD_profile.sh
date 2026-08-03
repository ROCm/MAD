#!/bin/bash
set -uo pipefail

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"
TIMING_LOG="/run_logs/${SLURM_JOB_ID}/benchmark_timing.jsonl"
TRACE_TOOLS="$NIXL_COOKBOOK_PATH/moriio_profiling/trace_tools.py"

record_benchmark_marker() {
    local event="$1" step_id="$2" iteration="$3" isl="$4" osl="$5" con="$6" prompts="$7" rc="${8:-}"
    local -a args=(benchmark-marker --out "$TIMING_LOG" --event "$event" --step-id "$step_id"
        --iteration "$iteration" --isl "$isl" --osl "$osl" --concurrency "$con" --num-prompts "$prompts")
    [[ -z "$rc" ]] || args+=(--return-code "$rc")
    PYTHONDONTWRITEBYTECODE=1 python3 "$TRACE_TOOLS" "${args[@]}" || {
        echo "ERROR: failed to record benchmark $event marker for $step_id" >&2
        return 1
    }
}

echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ===== "
echo "Benchmark Port: ${BENCHMARK_PORT}"
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null

sleep 10
if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
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
else
    echo "Warmup skipped (SKIP_WARMUP=1)" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
fi

CON="${BENCHMARK_CON:-8 16 32 64 128 256 512}"
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024 1024/8192}"

echo "Benchmarking iterations: $BENCHMARK_ITR" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
for i in $(seq 1 $BENCHMARK_ITR); do
    echo "Running the benchserving script for iter: $i" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       for con in $CON; do
           p_con=$(($con * 2))
           if [ "$p_con" -lt 16 ]; then
               p_con=16
           fi
           if [ -n "${BENCHMARK_NUM_PROMPTS:-}" ]; then
               p_con=$BENCHMARK_NUM_PROMPTS
           fi
           _base_timeout="${STEP_TIMEOUT:-1800}"
           _total_tok=$(( isl + osl ))
           _scaled_timeout=$(( _base_timeout * _total_tok / 2048 ))
           if [ "$_scaled_timeout" -lt "$_base_timeout" ]; then
               _scaled_timeout=$_base_timeout
           fi
           echo "[RUNNING] prompts $p_con isl $isl osl $osl con $con (timeout ${_scaled_timeout}s)" \
               | tee -a ${LOG}_CONCURRENCY.log >/dev/null
           _step_id="iter${i}_isl${isl}_osl${osl}_con${con}"
           record_benchmark_marker start "$_step_id" "$i" "$isl" "$osl" "$con" "$p_con" || exit 1
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
           record_benchmark_marker end "$_step_id" "$i" "$isl" "$osl" "$con" "$p_con" "$rc" || exit 1
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
