#!/bin/bash
# AgentX campaign harvester (read-only). For each job result dir, parse
# suite_summary.json and print one matrix row per workload:
#   cell  workload  error_rate  gpu_cache_hit_rate  theoretical_cache_hit_rate  verdict
#
# Verdict (same thresholds as validate_agentic_result.sh / the READMEs):
#   FAIL  suite_summary.json missing/unreadable, status INVALID, or
#         error_rate missing or > AGENTIC_MAX_ERROR_RATE (default 0.10)
#   WARN  gpu_cache_hit_rate < AGENTIC_MIN_CACHE_HIT (default 0.30)
#   PASS  otherwise
#
# Inputs: JOBIDs (or full result dirs) as args. A JOBID resolves to
# ${RESULT_ROOT:-/run_logs}/<jobid>. If a JOB_MAP tsv (JOBID<TAB>cell) is given,
# the cell label is looked up from it; otherwise the jobid/dir is the cell label.
#
# Usage:
#   bash harvest.sh 12345 12346                 # /run_logs/12345 /run_logs/12346
#   RESULT_ROOT=/run_logs bash harvest.sh 12345
#   JOB_MAP=agentx_jobmap.tsv bash harvest.sh $(cut -f1 agentx_jobmap.tsv)
#   bash harvest.sh /path/to/run_logs/12345     # explicit dir
set -uo pipefail

RESULT_ROOT="${RESULT_ROOT:-/run_logs}"
JOB_MAP="${JOB_MAP:-}"
MAX_ERROR_RATE="${AGENTIC_MAX_ERROR_RATE:-0.10}"
MIN_CACHE_HIT="${AGENTIC_MIN_CACHE_HIT:-0.30}"

if [ "$#" -eq 0 ]; then
    echo "usage: bash harvest.sh <JOBID|result_dir> [JOBID|result_dir ...]" >&2
    exit 2
fi

PY="python3"
command -v "$PY" >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 2; }

_cell_label() {  # $1=jobid/dirname -> cell label via JOB_MAP or identity
    local key="$1"
    if [ -n "$JOB_MAP" ] && [ -f "$JOB_MAP" ]; then
        local hit
        hit="$(awk -F'\t' -v k="$key" '$1==k{print $2; exit}' "$JOB_MAP")"
        [ -n "$hit" ] && { echo "$hit"; return; }
    fi
    echo "$key"
}

printf '%-28s %-18s %10s %12s %12s  %s\n' \
    "cell" "workload" "err_rate" "cache_hit" "theo_hit" "verdict"
printf '%s\n' "----------------------------------------------------------------------------------------------"

overall_fail=0
for arg in "$@"; do
    if [ -d "$arg" ]; then
        dir="$arg"; key="$(basename "$arg")"
    else
        key="$arg"; dir="${RESULT_ROOT}/${arg}"
    fi
    cell="$(_cell_label "$key")"
    summary="${dir}/suite_summary.json"

    if [ ! -f "$summary" ]; then
        printf '%-28s %-18s %10s %12s %12s  %s\n' \
            "$cell" "-" "-" "-" "-" "FAIL(no summary)"
        overall_fail=1
        continue
    fi

    # Parse + verdict per workload in python3; print TSV rows, exit 1 if any FAIL.
    rows="$(SUMMARY="$summary" CELL="$cell" MAXERR="$MAX_ERROR_RATE" MINCACHE="$MIN_CACHE_HIT" \
        "$PY" - <<'PY'
import json, os, sys
summary = os.environ["SUMMARY"]; cell = os.environ["CELL"]
maxerr = float(os.environ["MAXERR"]); mincache = float(os.environ["MINCACHE"])
try:
    d = json.load(open(summary))
except Exception as e:
    print(f"{cell}\t-\t-\t-\t-\tFAIL(bad json)")
    sys.exit(1)
wls = d.get("workloads") or []
if not wls:
    print(f"{cell}\t-\t-\t-\t-\tFAIL(empty)")
    sys.exit(1)
any_fail = 0
def pct(v): return "-" if v is None else f"{v*100:.1f}%"
for w in wls:
    name = w.get("workload", "?")
    err = w.get("error_rate"); ch = w.get("gpu_cache_hit_rate"); th = w.get("theoretical_cache_hit_rate")
    status = w.get("status")
    if status == "INVALID" or err is None or err > maxerr:
        verdict = "FAIL"; any_fail = 1
    elif ch is None or ch < mincache:
        verdict = "WARN"
    else:
        verdict = "PASS"
    print(f"{cell}\t{name}\t{pct(err)}\t{pct(ch)}\t{pct(th)}\t{verdict}")
sys.exit(any_fail)
PY
    )" || overall_fail=1

    while IFS=$'\t' read -r c wl er chit thit verdict; do
        [ -n "$c" ] || continue
        printf '%-28s %-18s %10s %12s %12s  %s\n' "$c" "$wl" "$er" "$chit" "$thit" "$verdict"
    done <<< "$rows"
done

printf '%s\n' "----------------------------------------------------------------------------------------------"
[ "$overall_fail" -eq 0 ] && echo "harvest: all cells PASS/WARN" || echo "harvest: one or more cells FAIL/INVALID"
exit 0
