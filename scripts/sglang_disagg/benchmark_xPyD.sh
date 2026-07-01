#!/bin/bash
set -uo pipefail

timestamp=$(date "+%Y%m%d_%H%M%S")
RUN_LOG_JOB_ID="${SLURM_JOB_ID:-0}"
RUN_LOG_DIR="/run_logs/${RUN_LOG_JOB_ID}"
mkdir -p "$RUN_LOG_DIR" 2>/dev/null || true

LOG="${RUN_LOG_DIR}/benchmark_${RUN_LOG_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"
LOG_FILE="${LOG}_CONCURRENCY.log"

BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
if ! [[ "$BENCHMARK_ITR" =~ ^[0-9]+$ ]] || [[ "$BENCHMARK_ITR" -lt 1 ]]; then
    echo "ERROR: BENCHMARK_ITR must be a positive integer, got '${BENCHMARK_ITR}'" >&2
    exit 1
fi

CON="${BENCHMARK_CONCURRENCY_LEVELS:-8 16 32 64 128 256 512}"
# ISL/OSL combinations; override with e.g. BENCHMARK_COMBINATIONS="1024/1024 8192/1024".
IFS=' ' read -ra COMBINATIONS <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}"

# --- Per-point resilience (kept from #328) -----------------------------------
# A single transient gateway circuit-open / server recovery hiccup should not
# nuke a multi-hour sweep. Retry the whole point after a cooldown; a persistently
# failing point still fails the run (fail-fast preserved).
# Set BENCHMARK_POINT_RETRIES=0 to restore single-attempt behavior.
BENCHMARK_FAIL_FAST="${BENCHMARK_FAIL_FAST:-1}"
BENCHMARK_POINT_RETRIES="${BENCHMARK_POINT_RETRIES:-2}"
BENCHMARK_RETRY_COOLDOWN_SECONDS="${BENCHMARK_RETRY_COOLDOWN_SECONDS:-45}"

log_msg() {
    echo "$*" | tee -a "$LOG_FILE" >/dev/null
}

run_serving_bench() {
    local isl="$1"
    local osl="$2"
    local con="$3"
    local prompts="$4"

    # Extra context on a separate INFO line; the parseable marker below stays in
    # the exact shape parse_to_csv.py expects: "RUNNING: prompts isl X osl Y con Z".
    log_msg "INFO: model=${MODEL_NAME} xP=${xP} yD=${yD} job=${RUN_LOG_JOB_ID} prompts=${prompts}"
    log_msg "RUNNING: prompts isl ${isl} osl ${osl} con ${con}"
    python3 -m sglang.bench_serving \
        --model "$MODEL_PATH" \
        --backend sglang \
        --host 127.0.0.1 \
        --port 2322 \
        --dataset-name random \
        --random-input "$isl" \
        --random-output "$osl" \
        --random-range-ratio 1.0 \
        --max-concurrency "$con" \
        --num-prompt "$prompts" \
        --pd-separated \
        2>&1 | tee -a "$LOG_FILE" >/dev/null
}

log_msg "==== Benchmark Serving Concurrency Sweep Test ${LOG} ====="
log_msg "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
log_msg "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
log_msg "Benchmarking iterations: ${BENCHMARK_ITR}"
log_msg "Benchmarking combinations: ${COMBINATIONS[*]}"
log_msg "Benchmarking concurrency levels: ${CON}"

benchmark_failures=0

# Optional warmup / precheck. Runs BEFORE the "iter: 1" marker, so parse_to_csv.py
# (which ignores everything prior to that marker) never counts it as a result.
if [[ "${BENCHMARK_PRECHECK:-0}" == "1" ]]; then
    sleep "${BENCHMARK_START_DELAY_SECONDS:-10}"
    log_msg "Test run:"
    if ! run_serving_bench \
        "${BENCHMARK_PRECHECK_ISL:-128}" \
        "${BENCHMARK_PRECHECK_OSL:-16}" \
        "${BENCHMARK_PRECHECK_CONCURRENCY:-1}" \
        "${BENCHMARK_PRECHECK_PROMPTS:-1}"; then
        benchmark_failures=$((benchmark_failures + 1))
        log_msg "ERROR: benchmark precheck failed"
        [[ "$BENCHMARK_FAIL_FAST" == "1" ]] && exit 1
    fi
fi

for ((i = 1; i <= BENCHMARK_ITR; i++)); do
    sleep "${BENCHMARK_ITERATION_DELAY_SECONDS:-60}"
    log_msg "RUNNING: the benchserving script for iter: $i"
    for combo in "${COMBINATIONS[@]}"; do
        IFS="/" read -r isl osl <<< "$combo"
        for con in $CON; do
            p_con=$((con * 2))
            if [[ "$p_con" -lt 16 ]]; then
                p_con=16
            fi
            total_attempts=$((BENCHMARK_POINT_RETRIES + 1))
            point_ok=0
            for ((attempt = 1; attempt <= total_attempts; attempt++)); do
                if run_serving_bench "$isl" "$osl" "$con" "$p_con"; then
                    point_ok=1
                    break
                fi
                if [[ "$attempt" -lt "$total_attempts" ]]; then
                    log_msg "WARN: bench_serving failed (attempt ${attempt}/${total_attempts}) for iter=${i} isl=${isl} osl=${osl} con=${con}; retrying after ${BENCHMARK_RETRY_COOLDOWN_SECONDS}s (transient circuit-open / server recovery)"
                    sleep "${BENCHMARK_RETRY_COOLDOWN_SECONDS}"
                fi
            done
            if [[ "$point_ok" -ne 1 ]]; then
                benchmark_failures=$((benchmark_failures + 1))
                log_msg "ERROR: bench_serving failed after ${total_attempts} attempt(s) for iter=${i} isl=${isl} osl=${osl} con=${con}"
                [[ "$BENCHMARK_FAIL_FAST" == "1" ]] && break 3
            fi

            sleep "${BENCHMARK_BETWEEN_RUN_SECONDS:-10}"
        done
    done
done

log_msg "==== Benchmark Serving Concurrency End Time ${LOG} ====="
log_msg "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
log_msg "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"

# --- Finalization: aligned with develop (parse_to_csv.py + MAD_OUTPUT_CSV) ----
# Write the madengine multiple_results CSV to the container cwd under the name
# madengine harvests (MAD_OUTPUT_CSV, e.g. perf_sglang-disagg-<MODEL>.csv); keep
# a copy under /run_logs for post-mortem inspection.
PERF_CSV="${MAD_OUTPUT_CSV:-perf_sglang-disagg-${MODEL_NAME}.csv}"
if ! python3 parse_to_csv.py "${LOG}_CONCURRENCY.log" -o "${LOG}_CONCURRENCY.csv" \
    --perf-csv "${PERF_CSV}" \
    --model-name "${MODEL_NAME}" \
    2>&1 | tee -a "$LOG_FILE" >/dev/null; then
    benchmark_failures=$((benchmark_failures + 1))
    log_msg "ERROR: parse_to_csv.py failed to produce ${PERF_CSV}"
fi
cp -f "${PERF_CSV}" "${RUN_LOG_DIR}/" 2>/dev/null || true

if [[ "$benchmark_failures" -ne 0 ]]; then
    log_msg "ERROR: benchmark completed with ${benchmark_failures} failure(s)"
    exit 1
fi
