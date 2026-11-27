#!/usr/bin/env bash
# Check one debug-numeric run after step 9: the result card, the report, the comparison with
# the example's expected card, and the timing table. No placeholders to fill in.
#
#   scripts/check_run.sh <artifact dir | tag> [example]
#
#   <artifact dir | tag>  a path, or a tag under artifacts/ (e.g. 2026-11-27-softmoe_train_nan-2)
#   [example]             the expected card's name under validation/expected/ (default: the example
#                         named in the run's result.json, e.g. softmoe_train_nan)
#
# Prints one line per check and ends with CHECK_RUN=PASS or CHECK_RUN=FAIL. Exit status: 0 pass,
# 1 a check failed, 2 usage or no such run. A run whose example has no expected card is reported
# as NO_CARD (not compared), which is not a pass for validation.

ND=$(cd -P "$(dirname "$0")/.." && pwd -P)

if [ $# -lt 1 ] || [ -z "$1" ]; then
    echo "usage: $(basename "$0") <artifact dir | tag> [example]" >&2
    exit 2
fi

A=""
for c in "$1" "$ND/artifacts/$1" "$ND/$1"; do
    if [ -d "$c" ]; then A=$(cd -P "$c" && pwd -P); break; fi
done
if [ -z "$A" ]; then
    echo "no such artifact dir: $1 (tried the path as given, artifacts/$1 and $1 under $ND)" >&2
    exit 2
fi

cd "$ND" || exit 2
fail=0

# 1. the result card: every field filled in
out=$(python3 scripts/result_card.py check "$A" 2>&1)
rc=$?
echo "$out" | tail -n 1
[ $rc -eq 0 ] || fail=1

# 2. the report: step 9 finished (no TODO(agent) left)
if [ -f "$A/report.md" ]; then
    todo=$(grep -c 'TODO(agent)' "$A/report.md")
    if [ "$todo" -eq 0 ]; then
        echo "REPORT_TODO=0 $A/report.md"
    else
        echo "REPORT_TODO=$todo (step 9 not finished) $A/report.md"
        fail=1
    fi
else
    echo "REPORT_TODO: no report.md in $A (step 9 did not run)"
    fail=1
fi

# 3. the comparison with the expected card: SAME or REFINED pass, DIFFERENT fails
EX="${2:-}"
if [ -z "$EX" ] && [ -f "$A/result.json" ]; then
    EX=$(python3 -c 'import json, os, sys
run = json.load(open(sys.argv[1])).get("run_script", "")
print(os.path.basename(os.path.dirname(run)))' "$A/result.json" 2>/dev/null)
fi
card="validation/expected/$EX.json"
if [ -n "$EX" ] && [ -f "$card" ]; then
    out=$(python3 scripts/compare_results.py "$card" "$A" 2>&1)
    last=$(echo "$out" | tail -n 1)
    echo "$last"
    case $last in
        RESULT_COMPARE=SAME*|RESULT_COMPARE=REFINED*) ;;
        *) fail=1 ;;
    esac
else
    echo "RESULT_COMPARE=NO_CARD (no expected card for '${EX:-?}'; not compared)"
fi

# 4. the timing table
if [ -f "$A/ledger.md" ]; then
    python3 scripts/ledger.py "$A/ledger.md" table
else
    echo "ledger: no ledger.md in $A"
    fail=1
fi

# 5. the package itself: a run should not change tracked files (informational)
changed=$(git -C "$ND/../.." status --porcelain --untracked-files=no -- "$ND" 2>/dev/null | wc -l)
if [ "$changed" -eq 0 ]; then
    echo "PACKAGE_TRACKED_CHANGES=0"
else
    echo "PACKAGE_TRACKED_CHANGES=$changed (the run should not change tracked files; see git status)"
fi

if [ $fail -eq 0 ]; then
    echo "CHECK_RUN=PASS $A"
else
    echo "CHECK_RUN=FAIL $A"
fi
exit $fail
