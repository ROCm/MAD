#!/bin/bash
# PR validation: NIAH then the concurrency smoke, against the same live PD
# serve. Curl already ran in connectors/moriio.sh before this file is invoked.
# NIAH failure does not skip smoke — both must run sequentially in one allocation.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_niah_rc=0
_smoke_rc=0

echo "===== validate: NIAH (after curl) ====="
bash "$DIR/benchmark_niah.sh" || _niah_rc=$?
echo "===== validate: NIAH exit=${_niah_rc}; starting smoke sweep ====="
export BENCHMARK_SCRIPT=sweep
export BENCHMARK_SCRIPT_FILE=benchmark_xPyD.sh
bash "$DIR/benchmark_xPyD.sh" || _smoke_rc=$?
echo "===== validate: smoke exit=${_smoke_rc} ====="
echo "[validate] summary niah=${_niah_rc} smoke=${_smoke_rc}"
if [ "$_niah_rc" != 0 ] || [ "$_smoke_rc" != 0 ]; then
    exit 1
fi
exit 0
