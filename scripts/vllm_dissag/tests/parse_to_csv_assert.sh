#!/bin/bash
# Offline checks for parse_to_csv.py on a sweep log shaped like benchmark_xPyD.sh's.
# No cluster / no GPUs. Exits 0 if all assertions hold.
#
# Covers:
#   - the first measured cell (right after "iter: 1") gets its row
#   - a cell that stalls ([STALL], no result block) is a FAILURE row at its own
#     concurrency, and the next cell's result is filed under the next cell
#   - a cell with failed requests, or zero throughput, is a FAILURE row
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
_hasnot "$ROWS" "con=1)"                                          "the warmup before iter 1 is not a row"
_has    "$(wc -l <<<"$ROWS" | tr -d ' ')" "5"                     "one row per cell"

echo "======================================================"
echo "  parse_to_csv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
