#!/bin/bash
# Offline checks for parse_to_csv.py on a sweep log shaped like benchmark_xPyD.sh's.
# No cluster / no GPUs. Exits 0 if all assertions hold.
#
# Covers:
#   - the first measured cell (right after "iter: 1") gets its row
#   - a cell that stalls ([STALL], no result block) is a FAILURE row at its own
#     concurrency, and the next cell's result is filed under the next cell
#   - a cell with failed requests, or zero throughput, is a FAILURE row
#   - a cell that printed no result at all (a crashed benchmark, no [STALL]) is a FAILURE row
#   - NIAH: a length with any timed-out or errored request, or NO-RESULT, is a FAILURE row
#   - benchmark_long_context.sh's "[RUNNING] isl=... con=..." cells parse, and it publishes perf.csv
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

_cell() { # con failed throughput
  cat <<EOF
[RUNNING] prompts $(( $1 * 2 )) isl 1024 osl 1024 con $1 (timeout 1800s)
Maximum request concurrency: $1
============ Serving Benchmark Result ============
Successful requests:                     $(( $1 * 2 - $2 ))
Failed requests:                         $2
Maximum request concurrency:             $1
Total token throughput (tok/s):          $3
==================================================
EOF
}
{
  echo "Maximum request concurrency: 1"
  echo "============ Serving Benchmark Result ============"
  echo "Total token throughput (tok/s):          52.93"
  echo "Running the benchserving script for iter: 1"
  _cell 8 0 100.00
  _cell 16 0 200.00
  echo "[RUNNING] prompts 64 isl 1024 osl 1024 con 32 (timeout 1800s)"
  echo "  0%|          | 1/64 [02:16<9:39:07, 136.27s/it][STALL] isl=1024 osl=1024 con=32 timed out after 1800s"
  _cell 64 10 300.00
  _cell 128 256 0.00
  echo "[RUNNING] prompts 512 isl 1024 osl 1024 con 256 (timeout 1800s)"
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
_has    "$ROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=32)"    "a stalled cell is a FAILURE row at its own concurrency"
_has    "$ROWS" "FAILURE 300.00 tok/s (isl=1024 osl=1024 con=64)"  "the cell after a stall keeps its own concurrency; failed requests fail it"
_has    "$ROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=128)"   "zero throughput is FAILURE"
_has    "$ROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=256)"   "a cell that crashed with no result and no [STALL] is a FAILURE row"
_hasnot "$ROWS" "con=1)"                                          "the warmup before iter 1 is not a row"
_has    "$(wc -l <<<"$ROWS" | tr -d ' ')" "6"                     "one row per cell"

echo "=== parse_to_csv: benchmark_long_context.sh cells ==="
cat > "$TMP/lc_CONCURRENCY.log" <<'LC'
==== Long-context benchmark (per-shape warmup=2, EP GPUs=8) /run_logs/1/benchmark_long_context ====
[RUNNING] isl=1024 osl=1024 con=1 warmups=2 prompts=16 (timeout 2400s)
============ Serving Benchmark Result ============
Failed requests:                         0
Total token throughput (tok/s):          123.45
==================================================
[RUNNING] isl=1024 osl=1024 con=4 warmups=2 prompts=16 (timeout 2400s)
[STALL] isl=1024 osl=1024 con=4 timed out after 2400s
==== Long-context benchmark complete ====
LC
python3 "$DIR/parse_to_csv.py" "$TMP/lc_CONCURRENCY.log" -o "$TMP/lc.csv" \
  --perf-csv "$TMP/lc_perf.csv" --model-name M >/dev/null 2>&1
LROWS="$(python3 -c "
import csv
for r in csv.DictReader(open('$TMP/lc_perf.csv')): print(r['status'], r['performance'], r['metric'])")"
_has    "$LROWS" "SUCCESS 123.45 tok/s (isl=1024 osl=1024 con=1)"  "a long_context cell is read from its isl=/osl=/con= header"
_has    "$LROWS" "FAILURE 0.00 tok/s (isl=1024 osl=1024 con=4)"    "a stalled long_context cell is a FAILURE row"
_has    "$(grep -c '"$_BENCH_DIR/parse_to_csv.py"' "$DIR/benchmark_long_context.sh")" "1" \
        "benchmark_long_context.sh publishes perf.csv"

echo "=== parse_to_csv --niah: one row per context length, errors are FAILURE ==="
cat > "$TMP/niah.log" <<'NIAH'
=== NIAH summary (mean/min/max across 3 seed(s)) ===
  words= 10000  mean=10.0/10  min=10  max=10  (n=3)
  words= 50000  mean=9.5/10  min=9  max=10  (n=2)  [1 timeout/err excluded]
  words=100000  mean=7.0/10  min=6  max=8  (n=3)  (TRUNCATED in 1/3: raise NIAH_MAXTOK)
  words=200000  NO-RESULT (3/3 timed out or errored - likely cold compile; raise NIAH_TIMEOUT or keep NIAH_WARMUP=1)
NIAH
python3 "$DIR/parse_to_csv.py" "$TMP/niah.log" --niah --perf-csv "$TMP/niah.csv" --model-name M >/dev/null 2>&1
NROWS="$(python3 -c "
import csv
for r in csv.DictReader(open('$TMP/niah.csv')): print(r['status'], r['performance'], r['metric'])")"
_has    "$NROWS" "SUCCESS 10.0 retrieval/10 (niah words=10000 seeds=3)"  "a clean length is SUCCESS"
_has    "$NROWS" "FAILURE 9.5 retrieval/10 (niah words=50000 seeds=2)"   "a length with an errored request is FAILURE"
_has    "$NROWS" "SUCCESS 7.0 retrieval/10 (niah words=100000 seeds=3)"  "a low (or truncated) score is a measurement, SUCCESS"
_has    "$NROWS" "FAILURE 0.0 retrieval/10 (niah words=200000 seeds=3)"  "a NO-RESULT length is a FAILURE row, not a missing row"
_has    "$(wc -l <<<"$NROWS" | tr -d ' ')" "4"                          "one NIAH row per length"

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

echo "======================================================"
echo "  parse_to_csv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
