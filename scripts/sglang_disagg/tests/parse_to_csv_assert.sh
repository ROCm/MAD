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

echo "======================================================"
echo "  parse_to_csv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
