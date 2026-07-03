#!/bin/bash
# Run all offline test suites (no cluster). Exit 0 iff all pass.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
for t in gate_check.sh argv_assert.sh parity_check.sh; do
  echo "############### $t ###############"
  env -i PATH="$PATH" HOME="$HOME" bash "$DIR/$t" || rc=1
  echo ""
done
[[ "$rc" == "0" ]] && echo "ALL OFFLINE SUITES PASSED ✅" || echo "SOME SUITES FAILED ❌"
exit $rc
