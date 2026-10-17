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
_has    "$(grep -c 'NIXL_COOKBOOK_PATH/parse_to_csv.py' "$DIR/benchmark_long_context.sh")" "1" \
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

echo "======================================================"
echo "  parse_to_csv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
