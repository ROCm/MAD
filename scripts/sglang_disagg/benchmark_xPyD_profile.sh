#!/bin/bash

timestamp=$(date "+%Y%m%d_%H%M%S")
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_$MODEL_NAME"
echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ===== "
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
 
: "${BENCHMARK_ITR:=1}"
: "${SKIP_WARMUP:=1}"
CON="8 16 32 64 128 256 512"

PROFILE_TRACE_ARGS=()
PROFILE_SWEEP_COUNT=0
if [[ "${RUN_PROFILE:-0}" == "1" ]]; then
    read -ra _profile_combinations <<< "${BENCHMARK_COMBINATIONS-1024/1024 8192/1024}"
    read -ra _profile_concurrency <<< "${BENCHMARK_CON:-}"
    _profile_inputs_valid=1
    [[ "$BENCHMARK_ITR" =~ ^[1-9][0-9]*$ && "${#_profile_combinations[@]}" -gt 0 && "${#_profile_concurrency[@]}" -gt 0 ]] || _profile_inputs_valid=0
    for profile_combo in "${_profile_combinations[@]}"; do [[ "$profile_combo" =~ ^[1-9][0-9]*/[1-9][0-9]*$ ]] || _profile_inputs_valid=0; done
    for profile_con in "${_profile_concurrency[@]}"; do [[ "$profile_con" =~ ^[1-9][0-9]*$ ]] || _profile_inputs_valid=0; done
    (( _profile_inputs_valid )) || { echo "ERROR: invalid profile sweep configuration." >&2; exit 2; }
    CON="${BENCHMARK_CON}"
    PROFILE_SWEEP_COUNT=$((BENCHMARK_ITR * ${#_profile_combinations[@]} * ${#_profile_concurrency[@]}))
    declare -A _profile_sweep_keys=()
    for ((profile_i=1; profile_i<=BENCHMARK_ITR; profile_i++)); do
        for profile_combo in "${_profile_combinations[@]}"; do
            IFS="/" read -r profile_isl profile_osl <<< "$profile_combo"
            for profile_con in "${_profile_concurrency[@]}"; do
                profile_sweep_key="i${profile_i}_isl${profile_isl}_osl${profile_osl}_c${profile_con}"
                if [[ -n "${_profile_sweep_keys[$profile_sweep_key]+x}" ]]; then
                    echo "ERROR: duplicate profile sweep key: $profile_sweep_key" >&2
                    exit 2
                fi
                _profile_sweep_keys[$profile_sweep_key]=1
            done
        done
    done
fi

echo "Benchmark config: prompts=${BENCHMARK_NUM_PROMPTS:-auto} combinations=${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024} concurrency=${CON} iterations=${BENCHMARK_ITR} skip_warmup=${SKIP_WARMUP}" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
if [[ "$SKIP_WARMUP" != "1" ]]; then
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
else
  echo "Skipping 1024-prompt warmup (SKIP_WARMUP=1)" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
fi
# ISL/OSL combinations — override via BENCHMARK_COMBINATIONS env var (space-separated, e.g. "1024/1024 8192/1024")
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}"
echo "Benchmarking iterations: $BENCHMARK_ITR" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
for ((i=1; i<=BENCHMARK_ITR; i++)); do
    sleep 60
    echo "RUNNING: the benchserving script for iter: $i" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
       for con in $CON; do
           p_con=$(($con * 2))
           if [ "$p_con" -lt 16 ]; then
               p_con=16
           fi
           if [[ "${RUN_PROFILE:-0}" == "1" ]]; then
               p_con="${BENCHMARK_NUM_PROMPTS:-$p_con}"
           fi
           PROFILE_TRACE_ARGS=()
           if [[ "${RUN_PROFILE:-0}" == "1" ]]; then
               sweep_key="i${i}_isl${isl}_osl${osl}_c${con}"
               if (( PROFILE_SWEEP_COUNT == 1 )); then
                   request_prefix="profile-${SLURM_JOB_ID}"
                   artifact_suffix=""
               else
                   request_prefix="profile-${SLURM_JOB_ID}-${sweep_key}"
                   artifact_suffix="_${sweep_key}"
               fi
               PROFILE_TRACE_ARGS=(
                   --request-id-prefix "$request_prefix"
                   --client-timing-csv "/run_logs/${SLURM_JOB_ID}/rocprof_probe_client${artifact_suffix}.csv"
                   --client-manifest "/run_logs/${SLURM_JOB_ID}/rocprof_probe_manifest${artifact_suffix}.json"
               )
           fi
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
           "${PROFILE_TRACE_ARGS[@]}" \
           2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null

	   sleep 10
        done
    done
done


python3 parse_to_csv.py ${LOG}_CONCURRENCY.log  -o ${LOG}_CONCURRENCY.csv \
	--perf-csv /run_logs/${SLURM_JOB_ID}/perf.csv \
	--model-name "${MODEL_NAME}" \
	2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null
