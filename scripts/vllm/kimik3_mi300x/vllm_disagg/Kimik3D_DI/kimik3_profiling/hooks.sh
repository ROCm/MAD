#!/bin/bash

_MORIIO_PROFILING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# rocprofv3 writes results below per-host subdirectories.

moriio_rocprof_prefix() {
    local role="$1"
    if [[ "${ROCPROF:-0}" != "1" ]]; then
        echo ""
        return 0
    fi
    # trace_tools.py combine keys on ^rocprof_(prefill|decode)_NODE\d+$ -- the base
    # role only. Our launcher roles are prefill_master/prefill_worker/decode_master/
    # decode_worker; collapse to the base role (NODE_RANK already disambiguates
    # master vs worker) so the combine step recognizes the shard dirs.
    local base_role="${role%%_*}"   # prefill_master -> prefill, decode_worker -> decode
    local base="${ROCPROF_DIR_BASE:-/run_logs}"
    local rpdir="${base}/${SLURM_JOB_ID:-0}/rocprof_${base_role}_NODE${NODE_RANK:-0}"
    mkdir -p "$rpdir"
    local flags="${ROCPROF_FLAGS:---kernel-trace}"
    echo "rocprofv3 ${flags} --output-format pftrace csv json -d ${rpdir} -o %hostname%_%pid% -- "
}

moriio_profiling_apply_reqid_patch() {
    [[ "${MORIIO_REQID_MAP:-1}" == "1" ]] || return 0
    python3 "${_MORIIO_PROFILING_DIR}/patch_moriio_reqid_map.py" \
        || { echo "moriio_profiling: reqid-map patch failed" >&2; exit 1; }
}

moriio_profiling_hook_start() {
    local log_prefix="$1"
    MORIIO_PROFILING_RUN_PREFIX="$(moriio_rocprof_prefix "${log_prefix}")"
}
