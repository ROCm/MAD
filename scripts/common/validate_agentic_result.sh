#!/bin/bash
# Phase 4 post-benchmark health check: read the agentic aggregate JSON and print
# a PASS/WARN summary for error rate and prefix-cache hit rate. A low GPU cache
# hit rate on a multi-turn agentic replay usually means the disagg router is not
# keeping a conversation's turns on the worker that holds its prefix (missing
# session stickiness) - see the session-affinity note in the SGLang README.
#
# Usage:
#   scripts/common/validate_agentic_result.sh <agg_result.json>
#   # or point at a run dir and it picks the newest *.json (excluding sidecars)
#   scripts/common/validate_agentic_result.sh /run_logs/<jobid>
#
# Thresholds (override via env):
#   AGENTIC_MAX_ERROR_RATE   (default 0.10)
#   AGENTIC_MIN_CACHE_HIT    (default 0.30)  # warn-only; workload/affinity dependent
set -uo pipefail

arg="${1:-}"
[ -n "$arg" ] || { echo "[validate][ERROR] usage: validate_agentic_result.sh <json|run_dir>" >&2; exit 2; }

json="$arg"
if [ -d "$arg" ]; then
    json="$(ls -t "$arg"/*.json 2>/dev/null | grep -v -E 'RUN_INVALID|profile_export|server_metrics' | head -1)"
fi
[ -n "$json" ] && [ -f "$json" ] || { echo "[validate][ERROR] no aggregate JSON found at $arg" >&2; exit 2; }

MAX_ERR="${AGENTIC_MAX_ERROR_RATE:-0.10}" MIN_HIT="${AGENTIC_MIN_CACHE_HIT:-0.30}" \
"${AIPERF_PYTHON:-python3}" - "$json" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
max_err = float(os.environ.get("MAX_ERR", "0.10"))
min_hit = float(os.environ.get("MIN_HIT", "0.30"))

# Error rate is over *measured* requests only. Warmup records are intentionally
# dropped and must not count as failures, so prefer request_accounting: errors are
# records_error_dropped over (records_profiled + records_error_dropped).
acct = d.get("request_accounting", {})
if acct:
    ok = acct.get("records_profiled", 0)
    errs = acct.get("records_error_dropped", 0)
    total = ok + errs
else:
    total = d.get("num_requests_total") or 0
    ok = d.get("num_requests_successful") or 0
    errs = total - ok
err_rate = (errs / total) if total else 0.0

sm_cache = d.get("server_metrics", {}).get("cache", {})
hit = sm_cache.get("gpu_cache_hit_rate")
if hit is None:
    hit = d.get("request_metrics", {}).get("cache", {}).get("theoretical_cache_hit_rate")

tput = d.get("request_metrics", {}).get("throughput", {})
per_gpu = tput.get("per_gpu", {}).get("total_tput_tps")

print(f"[validate] file: {p}")
print(f"[validate] requests: {ok}/{total} ok  error_rate={err_rate:.1%}")
print(f"[validate] gpu_cache_hit_rate: {hit if hit is None else f'{hit:.1%}'}")
if per_gpu is not None:
    print(f"[validate] throughput_per_gpu: {per_gpu:.0f} tok/s")

status = 0
if total == 0:
    print("[validate][WARN] no requests recorded"); status = 1
elif err_rate > max_err:
    print(f"[validate][WARN] error_rate {err_rate:.1%} exceeds {max_err:.0%}"); status = 1
else:
    print(f"[validate][PASS] error_rate within {max_err:.0%}")

if hit is not None and hit < min_hit:
    print(f"[validate][WARN] cache hit {hit:.1%} < {min_hit:.0%} - check router session affinity (xP>1)")

sys.exit(status)
PY
