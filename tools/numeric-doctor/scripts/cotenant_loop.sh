#!/bin/bash
# Run a model launcher as a co-tenant for replay_nan_op.py --contend-with
# (debug-numeric step5-pressure.md, rung 5c), restarting it whenever it exits -- a
# training run stops at its first NaN, and the load has to last as long as the
# replay.  Its output is appended to LOG, one "=== cotenant run <n>" header per start.
#
#   cotenant_loop.sh <log> <command> [args...]
#
# The replay kills this loop's whole process group when it ends (start_contenders).
LOG=${1:?usage: cotenant_loop.sh <log> <command> [args...]}
shift
[[ $# -gt 0 ]] || { echo "usage: cotenant_loop.sh <log> <command> [args...]" >&2; exit 2; }
run=0
while :; do
  echo "=== cotenant run $run $(date +%T)" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  run=$((run + 1))
  sleep 1          # a command that fails at once must not spin
done
