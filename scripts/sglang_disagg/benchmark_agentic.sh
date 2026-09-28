#!/bin/bash
# Thin shim: runs the shared scripts/common/benchmark_agentic.sh --backend sglang,
# then publishes the aggregate JSON as the perf.csv madengine collects.
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _cand in \
    "${_here}/../common/benchmark_agentic.sh" \
    "${MOONCAKE_REPO_DIR:-}/../common/benchmark_agentic.sh" \
    "${AGENTIC_LIB:+$(dirname "$AGENTIC_LIB")/benchmark_agentic.sh}"; do
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then
        bash "$_cand" --backend sglang "$@"; _rc=$?
        # Publish the run's metrics where madengine collects them, as benchmark_xPyD.sh
        # does for the sweep: the replay writes an aggregate JSON and no perf.csv, so a
        # run that measured throughput, latency and cache hit reported "0 perf files"
        # (build 145). One set of rows per workload; see parse_to_csv.py.
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
            mapfile -t _agg < <(find "/run_logs/${SLURM_JOB_ID}" -maxdepth 2 -type f \
                -name "agentic_${SLURM_JOB_ID}_*.json" 2>/dev/null | sort)
            if [ "${#_agg[@]}" -gt 0 ]; then
                python3 "${_here}/parse_to_csv.py" - --agentic-json "${_agg[@]}" \
                    --perf-csv "/run_logs/${SLURM_JOB_ID}/perf.csv" --model-name "${MODEL_NAME:-}" \
                    || echo "[agentic][WARN] could not write perf.csv from ${_agg[*]}" >&2
            else
                echo "[agentic][WARN] no agentic_${SLURM_JOB_ID}_*.json under /run_logs/${SLURM_JOB_ID}; no perf.csv" >&2
            fi
        fi
        exit "$_rc"
    fi
done
echo "[agentic][ERROR] shared benchmark_agentic.sh not found (set AGENTIC_LIB)" >&2
exit 1
