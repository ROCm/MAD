#!/bin/bash

moriio_profiling_configure_run() {
    local run_deepep="${1:-0}"
    RUN_PROFILE="${RUN_PROFILE:-0}"
    case "$RUN_PROFILE" in
        0) return 0 ;;
        1)
            if [[ "${CONNECTOR:-}" != "moriio" ]]; then
                echo "mori must be on for profiling" >&2
                return 1
            fi
            if [[ "$run_deepep" == "1" ]]; then
                echo "Error: RUN_PROFILE=1 is incompatible with legacy RUN_DEEPEP=1; use CONNECTOR=moriio." >&2
                return 1
            fi
            ROCPROF=1
            ROCPROF_FLAGS="${ROCPROF_FLAGS:---kernel-trace --marker-trace}"
            ROCPROF_DIR_BASE="${ROCPROF_DIR_BASE:-/run_logs}"
            MORI_ROCTX_TRANSFER=1
            MORIIO_REQID_MAP="${MORIIO_REQID_MAP:-0}"
            ANALYZE_KERNELS=0
            export RUN_PROFILE CONNECTOR ROCPROF ROCPROF_FLAGS ROCPROF_DIR_BASE \
                MORI_ROCTX_TRANSFER MORIIO_REQID_MAP ANALYZE_KERNELS
            echo "RUN_PROFILE=1: moriio rocprof kernel/marker capture enabled (ANALYZE_KERNELS=0)"
            ;;
        *)
            echo "Error: RUN_PROFILE must be 0 or 1 (got '$RUN_PROFILE')." >&2
            return 1
            ;;
    esac
}

moriio_profiling_append_env_args() {
    local env_file="$1" array_name="$2" line key value
    local -n env_args="$array_name"
    if [[ ! -f "$env_file" ]]; then
        echo "WARN: connector env file not found: $env_file" >&2
        return 0
    fi
    echo "Loading connector platform env: $env_file"
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# || -z "${line// }" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        env_args+=( -e "${key}=${!key:-$value}" )
    done < "$env_file"
}

moriio_profiling_propagate_status() {
    local run_rc="$1" posthoc_rc="$2"
    [[ "${RUN_PROFILE:-0}" == "1" ]] || return 0
    if (( run_rc != 0 )); then
        echo "ERROR: profiled server run/finalization failed with status ${run_rc}" >&2
        return "$run_rc"
    fi
    if (( posthoc_rc != 0 )); then
        echo "ERROR: profiling post-processing failed with status ${posthoc_rc}" >&2
        return "$posthoc_rc"
    fi
}


_MORIIO_PROFILING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# rocprofv3 writes results below per-host subdirectories.

moriio_rocprof_prefix() {
    local role="$1"
    if [[ "${ROCPROF:-0}" != "1" ]]; then
        echo ""
        return 0
    fi
    local base="${ROCPROF_DIR_BASE:-/run_logs}"
    local rpdir="${base}/${SLURM_JOB_ID:-0}/rocprof_${role}_NODE${NODE_RANK:-0}"
    mkdir -p "$rpdir"
    local flags="${ROCPROF_FLAGS:---kernel-trace}"
    # vLLM owns graceful SIGTERM handling. rocprofiler-sdk 1.1.0 signal
    # prioritization can deadlock while finalizing a multiprocess application.
    [[ " ${flags} " == *" --disable-signal-handlers "* ]] || flags+=" --disable-signal-handlers"
    echo "rocprofv3 ${flags} --output-format pftrace csv json -d ${rpdir} -o %hostname%_%pid% -- "
}

moriio_profiling_apply_reqid_patch() {
    [[ "${ROCPROF:-0}" == "1" && "${MORIIO_REQID_MAP:-0}" == "1" ]] || return 0
    python3 "${_MORIIO_PROFILING_DIR}/patch_moriio_reqid_map.py" \
        || { echo "moriio_profiling: reqid-map patch failed" >&2; exit 1; }
}

moriio_profiling_hook_start() {
    local log_prefix="$1"
    if [[ "${DRY_RUN:-0}" != "1" && "${ROCPROF:-0}" == "1" && \
          "${MORIIO_REQID_MAP:-0}" == "1" && "${_MORIIO_REQID_PATCH_APPLIED:-0}" != "1" ]]; then
        moriio_profiling_apply_reqid_patch
        _MORIIO_REQID_PATCH_APPLIED=1
    fi
    # vLLM waits only five seconds for workers by default. Large rocprof traces
    # need longer to flush during vLLM's own parent-driven shutdown.
    if [[ "${ROCPROF:-0}" == "1" ]]; then
        export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="${VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS:-120}"
    fi
    MORIIO_PROFILING_ROLE_DIR="${ROCPROF_DIR_BASE:-/run_logs}/${SLURM_JOB_ID:-0}/rocprof_${log_prefix}_NODE${NODE_RANK:-0}"
    MORIIO_PROFILING_RUN_PREFIX="$(moriio_rocprof_prefix "${log_prefix}")"
}

moriio_profiling_verify_trace() {
    local trace_dir="${MORIIO_PROFILING_ROLE_DIR:-}"
    local trace_file found=0
    [[ -n "$trace_dir" && -d "$trace_dir" ]] || {
        echo "[moriio_profiling] ERROR: missing trace directory: ${trace_dir:-<unset>}" >&2
        return 1
    }
    shopt -s nullglob
    for trace_file in "$trace_dir"/*_kernel_trace.csv; do
        if [[ -s "$trace_file" ]] && {
            IFS= read -r _trace_header && IFS= read -r _trace_record
        } < "$trace_file"; then
            found=1
            break
        fi
    done
    shopt -u nullglob
    (( found == 1 )) || {
        echo "[moriio_profiling] ERROR: no nonempty kernel trace was finalized in ${trace_dir}" >&2
        return 1
    }
}

# Recursive shutdown is used as bounded KILL escalation if a profiled,
# parent-driven shutdown fails. `pkill -P "$pid"` reaches direct children only,
# while vLLM descendants span API server, EngineCore, and worker levels.
kill_tree() {
    local pid="$1" sig="${2:-TERM}"
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$child" "$sig"
    done
    kill -"$sig" "$pid" 2>/dev/null || true
}

stop_worker() {
    local worker_pid="$1"
    if [[ "${ROCPROF:-0}" == "1" ]] && declare -f moriio_profiling_hook_stop >/dev/null; then
        moriio_profiling_hook_stop "$worker_pid"
    else
        # Preserve the established fallback if profiling was not fully configured.
        pkill -P "$worker_pid" 2>/dev/null; kill "$worker_pid" 2>/dev/null || true
    fi
}

_moriio_profiling_wait_and_verify() {
    local worker_pid="$1" intentional_term="${2:-0}"
    local worker_rc=0 trace_rc=0

    if wait "$worker_pid" 2>/dev/null; then
        worker_rc=0
    else
        worker_rc=$?
    fi
    if (( worker_rc == 127 )); then
        echo "[moriio_profiling] ERROR: PID ${worker_pid} is not waitable by this shell" >&2
    fi
    if (( intentional_term == 1 && worker_rc == 143 )); then
        worker_rc=0
    fi
    if moriio_profiling_verify_trace; then
        trace_rc=0
    else
        trace_rc=$?
    fi
    if (( worker_rc != 0 )); then
        return "$worker_rc"
    fi
    return "$trace_rc"
}

# With rocprof signal-handler prioritization disabled, terminate only the vLLM
# application parent. vLLM then shuts down EngineCore and workers in its required
# order; each instrumented process flushes its own trace during normal exit.
# Polling checks zombie state because kill -0 succeeds for an unreaped child.
moriio_profiling_hook_stop() {
    local worker_pid="$1"
    [[ "${ROCPROF:-0}" == "1" ]] || return 0

    local grace_s="${ROCPROF_FINALIZE_TIMEOUT_S:-180}"
    local kill_grace_s="${ROCPROF_KILL_TIMEOUT_S:-10}"
    local elapsed=0 state worker_ppid shell_pid="${BASHPID:-$$}"
    local intentional_term=0 finalize_rc=0

    if [[ ! "$worker_pid" =~ ^[0-9]+$ ]] || (( worker_pid <= 1 )) || (( worker_pid == shell_pid )); then
        echo "[moriio_profiling] ERROR: refusing to manage invalid vLLM PID '${worker_pid}'" >&2
        return 1
    fi
    if [[ ! "$grace_s" =~ ^[0-9]+$ ]] || [[ ! "$kill_grace_s" =~ ^[0-9]+$ ]]; then
        echo "[moriio_profiling] ERROR: finalization timeouts must be non-negative integers" >&2
        return 1
    fi

    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "[moriio_profiling] vLLM/rocprofv3 PID ${worker_pid} already exited"
        _moriio_profiling_wait_and_verify "$worker_pid" 0
        return $?
    fi

    worker_ppid="$(ps -o ppid= -p "$worker_pid" 2>/dev/null || true)"
    worker_ppid="${worker_ppid//[[:space:]]/}"
    if [[ "$worker_ppid" != "$shell_pid" ]]; then
        echo "[moriio_profiling] ERROR: refusing to manage PID ${worker_pid}; it is not a child of this shell" >&2
        return 1
    fi

    state="$(ps -o stat= -p "$worker_pid" 2>/dev/null || true)"
    if [[ "$state" != Z* ]] && kill -TERM "$worker_pid" 2>/dev/null; then
        intentional_term=1
    fi

    while kill -0 "$worker_pid" 2>/dev/null; do
        state="$(ps -o stat= -p "$worker_pid" 2>/dev/null || true)"
        if [[ "$state" == Z* ]]; then
            echo "[moriio_profiling] vLLM/rocprofv3 PID ${worker_pid} finalized in ${elapsed}s"
            _moriio_profiling_wait_and_verify "$worker_pid" "$intentional_term"
            return $?
        fi
        if (( elapsed >= grace_s )); then
            echo "[moriio_profiling] ERROR: PID ${worker_pid} did not exit after ${grace_s}s; process snapshot follows" >&2
            ps -o pid,ppid,pgid,sid,stat,etime,comm,args -p "$worker_pid" --ppid "$worker_pid" >&2 || true
            echo "[moriio_profiling] ERROR: escalating stuck profiling tree to KILL" >&2
            if declare -f kill_tree >/dev/null; then
                kill_tree "$worker_pid" KILL
            else
                kill -KILL "$worker_pid" 2>/dev/null || true
            fi
            for ((elapsed = 0; elapsed < kill_grace_s; elapsed++)); do
                state="$(ps -o stat= -p "$worker_pid" 2>/dev/null || true)"
                [[ -z "$state" || "$state" == Z* ]] && break
                sleep 1
            done
            state="$(ps -o stat= -p "$worker_pid" 2>/dev/null || true)"
            if [[ -z "$state" || "$state" == Z* ]]; then
                if _moriio_profiling_wait_and_verify "$worker_pid" 0; then
                    finalize_rc=0
                else
                    finalize_rc=$?
                fi
            else
                echo "[moriio_profiling] ERROR: PID ${worker_pid} remains after KILL cleanup timeout" >&2
                if moriio_profiling_verify_trace; then
                    finalize_rc=0
                else
                    finalize_rc=$?
                fi
            fi
            if (( finalize_rc != 0 )); then
                return "$finalize_rc"
            fi
            return 1
        fi
        sleep 1
        ((elapsed++)) || true
    done

    echo "[moriio_profiling] vLLM/rocprofv3 PID ${worker_pid} finalized in ${elapsed}s"
    _moriio_profiling_wait_and_verify "$worker_pid" "$intentional_term"
}
