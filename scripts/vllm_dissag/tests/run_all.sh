#!/bin/bash
# Run all offline test suites (no cluster). Exit 0 iff all pass.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
for t in gate_check.sh argv_assert.sh; do
  echo "############### $t ###############"
  env -i PATH="$PATH" HOME="$HOME" bash "$DIR/$t" || rc=1
  echo ""
done
ROOT="$(cd "$DIR/.." && pwd)"
echo "############### test_niah_score.py ###############"
python3 "$ROOT/tests/test_niah_score.py" || rc=1
echo ""
echo "############### test_list_prime.py ###############"
python3 "$ROOT/tests/test_list_prime.py" || rc=1
echo ""
echo "############### test_accuracy.py ###############"
python3 "$ROOT/tests/test_accuracy.py" || rc=1
echo ""
echo "############### test_parsers.py ###############"
python3 "$ROOT/tests/test_parsers.py" || rc=1
echo ""
echo "############### test_socket_barrier.py ###############"
python3 "$ROOT/tests/test_socket_barrier.py" || rc=1
echo ""
[[ "$rc" == "0" ]] && echo "ALL OFFLINE SUITES PASSED ✅" || echo "SOME SUITES FAILED ❌"
exit $rc
