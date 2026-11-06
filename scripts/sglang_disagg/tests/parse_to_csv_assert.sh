#!/bin/bash
# Offline checks for parse_to_csv.py on a sweep log shaped like benchmark_xPyD.sh's.
# No cluster / no GPUs. Exits 0 if all assertions hold.
#
# Covers:
#   - the first measured cell (right after "iter: 1") gets its row
#   - a cell whose successful requests fall short of its prompts is a FAILURE row
#     (sglang.bench_serving prints no failed count)
#   - a cell that printed no result (a benchmark that aborted, e.g. a warmup that got
#     "Bad Gateway") is a FAILURE row at its own concurrency, not a missing row
#   - zero throughput is a FAILURE row
#   - requests the server aborted are FAILURE: bench_serving counts them successful and reports
#     the requested length as generated, so the tell is "(retokenized): 0" or a 0 ms mean TTFT
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

_cell() { # con successful throughput
  cat <<EOF
RUNNING: prompts $(( $1 * 2 )) isl 1024 osl 1024 con $1
============ Serving Benchmark Result ============
Max request concurrency:                 $1
Successful requests:                     $2
Total token throughput (tok/s):          $3
==================================================
EOF
}
{
  echo "============ Serving Benchmark Result ============"
  echo "Successful requests:                     1024"
  echo "Total token throughput (tok/s):          25288.26"
  echo "RUNNING: the benchserving script for iter: 1"
  _cell 8 16 100.00
  _cell 16 32 200.00
  _cell 32 32 300.00
  _cell 64 128 0.00
  echo "RUNNING: prompts 256 isl 1024 osl 1024 con 128"
  echo "Traceback (most recent call last):"
  echo "ValueError: Warmup failed - Error: Bad Gateway"
} > "$TMP/sweep_CONCURRENCY.log"

python3 "$DIR/parse_to_csv.py" "$TMP/sweep_CONCURRENCY.log" -o "$TMP/out.csv" \
  --perf-csv "$TMP/perf.csv" --model-name M >/dev/null 2>&1
ROWS="$(python3 -c "
import csv
for r in csv.DictReader(open('$TMP/perf.csv')): print(r['status'], r['performance'], r['metric'])")"

_has()    { grep -qxF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot() { grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }

echo "=== parse_to_csv: sweep rows and their status ==="
_has    "$ROWS" "SUCCESS 100.00 tok/s (isl=1024 osl=1024 con=8)"   "first measured cell has a row"
_has    "$ROWS" "SUCCESS 200.00 tok/s (isl=1024 osl=1024 con=16)"  "a clean cell is SUCCESS"
_has    "$ROWS" "FAILURE 300.00 tok/s (isl=1024 osl=1024 con=32)"  "32 of 64 prompts succeeded: FAILURE"
_has    "$ROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=64)"    "zero throughput is FAILURE"
_has    "$ROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=128)"   "a cell that printed no result is a FAILURE row"
_hasnot "$ROWS" "25288.26"                                        "the warmup before iter 1 is not a row"
_has    "$(wc -l <<<"$ROWS" | tr -d ' ')" "5"                     "one row per cell"

echo "=== parse_to_csv: cells whose requests returned no text ==="
# Shaped like a real run whose KV transfer failed on every request (decode aborted each one).
_full() { # con retokenized ttft_ms throughput
  cat <<EOF
RUNNING: prompts $(( $1 * 2 )) isl 1024 osl 1024 con $1
============ Serving Benchmark Result ============
Successful requests:                     $(( $1 * 2 ))
Total generated tokens:                  $(( $1 * 2048 ))
Total generated tokens (retokenized):    $2
Total token throughput (tok/s):          $4
Mean TTFT (ms):                          $3
==================================================
EOF
}
{
  echo "RUNNING: the benchserving script for iter: 1"
  _full 8 16380 412.70 2198.11
  _full 16 0 0.00 292536.37
  _full 32 0 812.40 3100.00
  _full 64 65000 0.00 5000.00
} > "$TMP/empty_CONCURRENCY.log"
python3 "$DIR/parse_to_csv.py" "$TMP/empty_CONCURRENCY.log" -o "$TMP/empty.csv" \
  --perf-csv "$TMP/empty_perf.csv" --model-name M >/dev/null 2>&1
EROWS="$(python3 -c "
import csv
for r in csv.DictReader(open('$TMP/empty_perf.csv')): print(r['status'], r['performance'], r['metric'])")"
_has "$EROWS" "SUCCESS 2198.11 tok/s (isl=1024 osl=1024 con=8)"    "text came back and TTFT measured: SUCCESS"
_has "$EROWS" "FAILURE 292536.37 tok/s (isl=1024 osl=1024 con=16)" "no text, 0 ms TTFT, impossible throughput: FAILURE"
_has "$EROWS" "FAILURE 3100.00 tok/s (isl=1024 osl=1024 con=32)"   "no text back is FAILURE on its own"
_has "$EROWS" "FAILURE 5000.00 tok/s (isl=1024 osl=1024 con=64)"   "a 0 ms mean TTFT is FAILURE on its own"
echo ""
echo "=== agentic aggregate JSON (InferenceX process_agentic_result, pinned in agentic_lib.sh) ==="
# The aggregate's own shape: request_accounting counts warmup records in records_total
# and drops them from records_profiled; throughput is request_metrics.throughput.
mkdir -p "$TMP/agentic/wl"
_agg() { # profiled error_dropped warmup with_metrics
  python3 - "$TMP/agentic/wl/agg.json" "$@" <<'PY'
import json, sys
path, prof, err, warm, metrics = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
total = prof + err + warm
d = {"num_requests_total": total, "num_requests_successful": prof,
     "request_accounting": {"records_total": total, "records_profiled": prof,
                            "records_dropped_total": total - prof, "records_warmup_dropped": warm,
                            "records_error_dropped": err, "error_categories": {}}}
if metrics:
    d["request_metrics"] = {"throughput": {"total": {"tokens_per_second": 5000.0},
                                           "output": {"tokens_per_second": 800.0},
                                           "input": {"tokens_per_second": 4200.0},
                                           "per_gpu": {"total_tput_tps": 625.0}},
                            "latency": {"ttft": {"p50": 0.25}, "tpot": {"p50": 0.02}, "e2el": {"p50": 3.0}}}
    d["server_metrics"] = {"cache": {"gpu_cache_hit_rate": 0.42}}
json.dump(d, open(path, "w"))
PY
}
_rows() { # -> "n status-set" and the error output
  python3 -c "
import sys; sys.path.insert(0, '$DIR')
import parse_to_csv as p
rows = p.agentic_perf_rows('$TMP/agentic/wl/agg.json')
print(len(rows), ','.join(sorted({r[2] for r in rows})), rows[0][1] if rows else '')"
}
_t() { [ "$1" = "$2" ] && { echo "  PASS  $3"; pass=$((pass+1)); } || { echo "  FAIL  $3 (got: $1, want: $2)"; fail=$((fail+1)); }; }

# 70 measured, 2 errors, 8 warmup: error rate 2/72 = 2.8% as the validator computes it.
_agg 70 2 8 1
r="$(_rows)"
_t "$(echo "$r" | cut -d' ' -f1-2)" "6 SUCCESS" "a normal aggregate: every metric row, SUCCESS"
_t "$(echo "$r" | grep -o '70/72 requests')" "70/72 requests" "requests counted as the validator counts them (warmup excluded)"
# 85 measured, 15 errors: 15% > 10%. Over records_total (with 50 warmup) it would read 10%.
_agg 85 15 50 1
_t "$(_rows | cut -d' ' -f2)" "FAILURE" "error rate over measured requests, not records_total"
# No metric fields at all: no rows, a reason, and a header-only CSV the verdict rejects.
_agg 10 0 0 0
_t "$(_rows | cut -d' ' -f1)" "0" "an aggregate without metrics yields no rows"
err="$(cd "$TMP" && python3 -c "
import sys; sys.path.insert(0, '$DIR')
import parse_to_csv as p
p.save_agentic_perf_csv(['$TMP/agentic/wl/agg.json'], '$TMP/agentic/perf.csv', 'M')" 2>&1 >/dev/null)"
_t "$(echo "$err" | grep -c 'ERROR: no metric rows')" "1" "...and says so"
_t "$(wc -l < "$TMP/agentic/perf.csv" | tr -d ' ')" "1" "...leaving a header-only perf.csv (cluster_job_verdict fails on it)"
# A concurrency sweep writes <workload>/conc<N>/: each row names the workload and N.
_agg 70 2 8 1
mkdir -p "$TMP/agentic/wl/conc16"; cp "$TMP/agentic/wl/agg.json" "$TMP/agentic/wl/conc16/agg.json"
_t "$(python3 -c "
import sys; sys.path.insert(0, '$DIR')
import parse_to_csv as p
print(p.agentic_perf_rows('$TMP/agentic/wl/conc16/agg.json')[0][1])")" \
   "tok/s total (agentic wl conc16, 70/72 requests)" "a sweep cell's rows name the workload and its concurrency"
_t "$(_rows | cut -d' ' -f3- | grep -o 'agentic wl,')" "agentic wl," "...a single-concurrency run's, the workload alone"

echo "======================================================"
echo "  parse_to_csv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
