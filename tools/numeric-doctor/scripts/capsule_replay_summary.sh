#!/bin/bash
#
# Summarise ONE attempt of a node-wide capsule replay (debug-numeric step 5b).
#
#   bash scripts/capsule_replay_summary.sh <attempt_dir>
#
# <attempt_dir> holds one log per copy (g<gpu>_k<copy>.<host>.log), written by
# scripts/jobs/replay_ladder.sh.  Prints a line
# per copy, the [C1] walk from the first copy that has one (it is deterministic,
# so every copy prints the same table), totals, and finally exactly one
#
#   ATTEMPT_RESULT=REPRODUCED|CLEAN|NOT_SET_UP|OOM|ERROR
#
# line, which the launchers read to drive the concurrency ladder.  The same text
# is written to <attempt_dir>/summary.txt.
#
# Per copy:  NAN    the loop counted >= 1 non-finite run/launch
#            CLEAN  the loop ran and counted 0
#            OOM    the copy died out of device memory
#            ERROR  anything else (no loop line) -- read the log
# A copy is never scored from its exit code: the chain loop exits 1 when clean.
# The attempt is NOT_SET_UP when the launcher's futility stop ended it (FUTILE in
# the attempt dir) with no hit: the replay did not recreate the model's trigger.
#
set -uo pipefail

dir=${1:?usage: capsule_replay_summary.sh <attempt_dir>}
# Lines that are never part of a verdict: ROCm/aiter/Python runtime warnings.
NOISE_RE=${NOISE_RE:-'^\[hsa-hotswap|^\[aiter\] (WARNING|import)|UserWarning|^ *warnings\.warn|NUMA balancing|^rocblaslt (error|warning)|^hipModuleLoad failed| error: file not found$'}
[[ -d "$dir" ]] || { echo "no such directory: $dir" >&2; exit 2; }

{
  n_nan=0; n_clean=0; n_oom=0; n_err=0; bad=0; runs=0; c1_shown=0
  shopt -s nullglob
  logs=("$dir"/g*_k*.log)
  [[ ${#logs[@]} -gt 0 ]] || { echo "no copy logs in $dir"; echo "ATTEMPT_RESULT=ERROR"; exit 0; }

  echo "=== copies ($dir) ==="
  for f in "${logs[@]}"; do
    # "non-finite on 2/1000000 chain runs" (chain loop) or ".../1000000 launches" (single op)
    line=$(grep -oE 'non-finite on [0-9]+/[0-9]+ (chain runs|launches)' "$f" | tail -1)
    verdict=$(grep -oE '^\[verdict\] .*' "$f" | tail -1)
    if [[ -n "$line" ]]; then
      b=$(sed -E 's#non-finite on ([0-9]+)/.*#\1#' <<<"$line")
      r=$(sed -E 's#non-finite on [0-9]+/([0-9]+).*#\1#' <<<"$line")
      bad=$((bad + b)); runs=$((runs + r))
      if (( b > 0 )); then
        # a wrong-value capsule's hits are wrong finite outputs, not NaNs
        if grep -qs '\[wrong-value capsule' "$f"; then st=WRONG; else st=NAN; fi
        n_nan=$((n_nan + 1))
      elif [[ -e "$dir/STOP" ]]; then st=STOPPED   # 0 hits when the node-wide target stopped it: not a clean copy
      else st=CLEAN; n_clean=$((n_clean + 1)); fi
    elif grep -qE 'OutOfMemoryError|out of memory|hipErrorOutOfMemory' "$f"; then
      st=OOM; n_oom=$((n_oom + 1))
    else
      st=ERROR; n_err=$((n_err + 1))
    fi
    shown=$line     # a wrong-value capsule's count is of wrong finite outputs: say so (the log keeps
                    # the parsed "non-finite on" form)
    grep -qs '\[wrong-value capsule' "$f" && shown=${line/non-finite on/wrong output on}
    printf '  %-24s %-5s %-40s %s\n' "$(basename "$f")" "$st" "${shown:--}" "${verdict:--}"
  done
  # A wrong-value capsule (captured for a wrong FINITE value): the loop's "non-finite" counts
  # are runs that failed the capsule's value check again; the outputs themselves are finite.
  if grep -qsE 'returned a WRONG FINITE value|\[wrong-value capsule' "${logs[@]}"; then
    echo "  (wrong-value capsule: WRONG / non-finite above = runs whose output failed the capsule's" \
         "value check again -- the outputs are finite, the values wrong)"
  fi

  for f in "${logs[@]}"; do
    if grep -q '^\[C1\] chain walk' "$f"; then
      echo "=== [C1] walk (deterministic; from $(basename "$f")) ==="
      # Runtime chatter interleaved with the walk (a binary translator's
      # per-block warnings ran to ~800 lines on one gfx1250 host) is dropped and
      # counted: this summary is what an agent reads, and it pays per line.
      sed -n '/^\[C1\] chain walk/,/^\[verdict\]/p' "$f" |
        NOISE_RE="$NOISE_RE" awk -v src="$(basename "$f")" '   # ENVIRON: -v would eat the \[ escapes
             $0 ~ ENVIRON["NOISE_RE"] { n++; next } { print }
             END { if (n) printf "  (%d runtime-noise line(s) dropped by NOISE_RE; full text in %s)\n", n, src }'
      c1_shown=1; break
    fi
  done
  (( c1_shown )) || echo "=== [C1] walk: none (no chain in the capsule, or no copy reached it) ==="

  # One environment and build banner is enough when they agree; say so if not.
  envs=$(grep -h '^env:' "${logs[@]}" | sort | uniq -c)
  echo "=== env banners ==="; echo "${envs:-  (none)}"
  grep -h '^captured build:' "${logs[@]}" | sort -u | head -1
  # Every distinct operands line (generic SYNTHETIC, derived, or INCONSISTENT) and any
  # verdict flag -- the first one alone hid the INCONSISTENT warning.
  grep -h '^operands:' "${logs[@]}" | sort -u
  grep -h '^\[verdict-flag\]' "${logs[@]}" | sort -u

  hitlabel=NAN; grep -qsE '\[wrong-value capsule' "${logs[@]}" && hitlabel=WRONG
  echo "=== totals: copies $hitlabel $n_nan / CLEAN $n_clean / OOM $n_oom / ERROR $n_err;" \
       "non-finite $bad/$runs ==="
  if [[ -e "$dir/FUTILE" ]] && (( bad > 0 )); then
    # the stop fired on the progress files (written every 2000 runs); a copy's later counts hold hits:
    # it reproduced -- one line, not "NOT_SET_UP" beside REPRODUCED (2026-11-23).  The rate is a floor.
    echo "=== futility stop ($(cat "$dir/STOP")) fired before the first hits reached the progress files;" \
         "the copies' totals hold $bad: REPRODUCED -- the rate is a floor (re-run with" \
         "CAPSULE_FUTILITY=0 for a rate) ==="
  elif [[ -e "$dir/FUTILE" ]]; then
    echo "=== futility stop ($(cat "$dir/STOP")): 0 non-finite in the runs the model needed many times" \
         "over -- the replay is not recreating the model's trigger (NOT_SET_UP, not a negative) ==="
  elif [[ -e "$dir/STOP" ]]; then
    echo "=== stopped early ($(cat "$dir/STOP")): rate = non-finite / runs actually done;" \
         "stopping on a hit count biases it slightly high (~1/hits) ==="
  fi
  if (( n_nan > 0 )); then res=REPRODUCED
  elif (( n_oom > 0 )); then res=OOM
  elif [[ -e "$dir/FUTILE" ]] && (( n_err == 0 )); then res=NOT_SET_UP
  elif (( n_clean > 0 )); then res=CLEAN
  else res=ERROR; fi
  # The loop's first-non-finite tally and the [C1]-proposed verdict, so reading an
  # attempt needs this file only.
  verdict_py="$(dirname -- "$0")/replay_verdict.py"
  if [[ ( $res == REPRODUCED || $res == CLEAN ) && -r "$verdict_py" ]] && command -v python3 >/dev/null; then
    python3 "$verdict_py" attempt "$dir" $( [[ $res == REPRODUCED ]] && echo --reproduced )
  fi
  echo "ATTEMPT_RESULT=$res"
} | tee "$dir/summary.txt"
