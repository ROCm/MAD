#!/bin/bash
# debug-numeric steps 7-8: run a COLLECTIVE reproducer (collective_r3.py replay) under several env arms,
# one fresh torchrun per arm -- a framework reads its env when it builds its communicators, so an arm
# cannot be switched inside one process.
#
#   bash collective_arms.sh <capsule dir> <arms file> [--iters N]
#
# Arms file: one arm per line, `<name>|<VAR=VALUE ...>` (`#` and blank lines skipped); an empty env
# field is the stock arm.  Put stock first and last (the control, then the undo):
#   stock|
#   candidate|<LIB_SETTING>=<value> [<a setting that keeps the same implementation>=<value>]
#   stock_again|
# Each arm's variables win over the capsules' recorded env (NAN_R3_KEEP_ENV), everything else is
# restored from the capsules.  Per arm: `ARM <name> path=<implementation> result=<n/N> <verdict>`
# -- the path line is what shows an arm that only MOVED the call (step7-rootcause.md, keep the code
# path).  Last line: ARMS_RESULT=<arms run> stock=<n/N> <name>=<n/N> ...
# Runs where the reproducer runs (inside the runtime: `nd run --gpus 0,1 -- 'bash ...'`, or from a
# package's scripts/).  RANKS (default: the distinct ranks of the capsules), R3_PORT (29771).
set -uo pipefail
dir=$1 arms=$2; shift 2
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
ranks=${RANKS:-$(ls "$dir"/rank*_*.pt 2>/dev/null | sed 's|.*/rank\([0-9]*\)_.*|\1|' | sort -u | wc -l)}
[ "$ranks" -ge 2 ] 2>/dev/null || { echo "no member-rank capsules (rank<r>_*.pt) in $dir"; exit 2; }
port=${R3_PORT:-29771}
summary=""; n=0
while IFS='|' read -r name env; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  n=$((n + 1))
  keep=$(for kv in $env; do printf '%s,' "${kv%%=*}"; done)
  out=$(env $env NAN_R3_KEEP_ENV="${keep%,}" torchrun --nproc_per_node "$ranks" --master_addr 127.0.0.1 \
          --master_port $((port + n)) "$here/collective_r3.py" replay "$dir" "$@" 2>&1)
  path=$(sed -n 's/^R3 vllm: path //p' <<<"$out" | tail -1)
  res=$(sed -n 's/^LADDER_RESULT=//p' <<<"$out" | tail -1)
  echo "ARM $name path=${path:-?} result=${res##* } ${res%% *}${env:+  env: $env}"
  [ -z "$res" ] && grep -m3 -E 'Error|Traceback|error' <<<"$out" | sed 's/^/    /'
  sig=$(grep -m1 '^SIGNATURE' <<<"$out"); [ -n "$sig" ] && echo "    $sig"
  summary+=" $name=${res##* }"
done <"$arms"
echo "ARMS_RESULT=$n${summary}"
