#!/bin/bash
set -uo pipefail

timestamp=$(date "+%Y%m%d_%H%M%S")
RUN_LOG_JOB_ID="${SLURM_JOB_ID:-0}"
# Honor RUN_LOG_DIR exported by run.sh (single source of truth for the log
# location, including its /run_logs-vs-fallback decision). Only derive the
# default when this script is invoked standalone without run.sh in the env.
RUN_LOG_DIR="${RUN_LOG_DIR:-/run_logs/${RUN_LOG_JOB_ID}}"
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

# --- Optional torch-profiler capture (PROFILE_ENABLE=1) ----------------------
# In PD mode sglang refuses to profile both roles at once: --profile-prefill-url
# and --profile-decode-url are mutually exclusive, so one dedicated point runs per
# role. The workers write the traces themselves, hence PROFILE_OUTPUT_DIR must be
# on the shared /run_logs mount for a multi-node run, and each server must have
# been launched with SGLANG_TORCH_PROFILER_DIR set or --profile is a no-op.
# The point writes to its own log: parse_to_csv.py reads ${LOG}_CONCURRENCY.log,
# so a profiled request stream never reaches the perf CSV.
run_profile_point() {
    local role="$1"
    shift
    local plog="${LOG}_PROFILE_${role}.log"
    local con="${PROFILE_CONCURRENCY:-16}"
    local extra=()
    [[ -n "${PROFILE_START_STEP:-}" ]] && extra+=(--profile-start-step "${PROFILE_START_STEP}")

    echo "INFO: profiling ${role} worker(s): $*" | tee -a "$plog" >/dev/null
    python3 -m sglang.bench_serving \
        --model "$MODEL_PATH" \
        --backend sglang \
        --host 127.0.0.1 \
        --port 2322 \
        --dataset-name random \
        --random-input "${PROFILE_ISL:-1024}" \
        --random-output "${PROFILE_OSL:-1024}" \
        --random-range-ratio 1.0 \
        --max-concurrency "$con" \
        --num-prompt "${PROFILE_PROMPTS:-$((con * 2))}" \
        --pd-separated \
        --profile \
        --profile-output-dir "${PROFILE_OUTPUT_DIR}" \
        --profile-steps "${PROFILE_STEPS:-5}" \
        "${extra[@]+"${extra[@]}"}" \
        "--profile-${role}-url" "$@" \
        2>&1 | tee -a "$plog" >/dev/null
}

profile_roles() {
    PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${RUN_LOG_DIR}/torchprof}"
    local port="${PROFILE_WORKER_PORT:-3000}"
    local ips=() prefill_urls=() decode_urls=() urls=() role n

    IFS=',' read -ra ips <<< "${IPADDRS:-127.0.0.1}"
    if [[ "${#ips[@]}" -lt $((xP + yD)) ]]; then
        log_msg "WARN: IPADDRS lists ${#ips[@]} host(s), need $((xP + yD)) for xP=${xP} yD=${yD}; profiling skipped"
        return 0
    fi
    for ((n = 0; n < xP; n++)); do prefill_urls+=("http://${ips[$n]}:${port}"); done
    for ((n = xP; n < xP + yD; n++)); do decode_urls+=("http://${ips[$n]}:${port}"); done

    mkdir -p "${PROFILE_OUTPUT_DIR}" 2>/dev/null || true
    log_msg "INFO: profiling enabled, traces -> ${PROFILE_OUTPUT_DIR}"
    for role in ${PROFILE_ROLES:-prefill decode}; do
        case "$role" in
            prefill) urls=("${prefill_urls[@]}") ;;
            decode)  urls=("${decode_urls[@]}") ;;
            *) log_msg "WARN: unknown PROFILE_ROLES entry '${role}', skipped"; continue ;;
        esac
        if ! run_profile_point "$role" "${urls[@]}"; then
            log_msg "ERROR: profiling point failed for role=${role} (see ${LOG}_PROFILE_${role}.log)"
            [[ "${PROFILE_FAIL_FAST:-0}" == "1" ]] && return 1
        fi
    done
    return 0
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

# Profiling runs last: the perf CSV is already written, so a profiler hiccup can
# no longer cost the sweep its results.
if [[ "${PROFILE_ENABLE:-0}" == "1" ]]; then
    if ! profile_roles; then
        benchmark_failures=$((benchmark_failures + 1))
    fi
fi

if [[ "$benchmark_failures" -ne 0 ]]; then
    log_msg "ERROR: benchmark completed with ${benchmark_failures} failure(s)"
    exit 1
fi
