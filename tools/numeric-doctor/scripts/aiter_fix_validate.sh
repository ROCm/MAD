#!/bin/bash
# debug-numeric step 8: validate patched aiter .co fixes on the step-6 reproducer -- stock and each fix
# side by side, one GPU each, the chain loop at the reproducer's load, same iteration count.
#
#   bash aiter_fix_validate.sh <capsule.pt> <rel path, e.g. gfx950/mla/<name>.co> <NAME=patched.co> ... \
#        [-- extra replay args, e.g. --contend 1 --synthesize]
#
# Each fix arm runs under aiter_overlay.sh's exports (held back by --restore-env, so the capture's
# stock AITER_* do not undo them) and must show its patched .co in aiter's load log, else the arm is
# NOT_SERVED (it ran stock).  Value checks count too: a wrong-value capsule's runs are wrong finite
# outputs.  Last line FIX_VALIDATE=PASS when stock hit >= 3 and every served fix arm hit 0, at ITERS
# (default 1000) runs each -- size ITERS from power.py so stock expects >= 3.  Under <GPU_EXEC>,
# PYTHONPATH reaching scripts/ (nd run sets it); OUT (default $ND_REPO/out or ./out) holds the logs.
set -u
CAP=${1:?capsule}; REL=${2:?rel path}; shift 2
ARMS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARMS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
O=${OUT:-${ND_REPO:-.}/out}; mkdir -p "$O"; O=$(cd "$O" && pwd -P)   # physical: aiter logs resolved paths
R=(python3 -m replay_nan_op "$CAP" --chain --no-reference --restore-env --chain-iters "${ITERS:-1000}" "$@")
g=0
AITER_LOG_LEVEL=INFO HIP_VISIBLE_DEVICES=$g "${R[@]}" > "$O/stock.log" 2>&1 &
for arm in "${ARMS[@]}"; do
  g=$((g + 1)); name=${arm%%=*}; co=${arm#*=}
  ( ex=$(bash "$HERE/aiter_overlay.sh" "$co" "$REL" "$O/overlay_$name") && [ -n "$ex" ] ||
      { echo "OVERLAY_BUILD_FAILED for $name ($co): the arm would run the stock kernel"; exit 3; }
    eval "$ex"
    AITER_LOG_LEVEL=INFO HIP_VISIBLE_DEVICES=$g "${R[@]}" ) > "$O/$name.log" 2>&1 &
done
wait
hits() { grep -oE 'non-finite on [0-9]+/[0-9]+ (chain runs|launches)' "$1" | tail -n 1 | grep -oE '[0-9]+/[0-9]+'; }
s=$(hits "$O/stock.log"); echo "VALIDATE stock: ${s:-?} runs hit (wrong value or non-finite)"
ok=1; [ "${s%%/*}" -ge 3 ] 2>/dev/null || { ok=0; echo "  stock hit < 3: the control did not reproduce -- void, not a pass"; }
for arm in "${ARMS[@]}"; do
  name=${arm%%=*}; h=$(hits "$O/$name.log")
  if grep -qE "(hsaco: |hipModuleLoad: )$O/overlay_$name/[^ ]*$(basename "$REL")" "$O/$name.log"; then served=served
  else served=NOT_SERVED; ok=0; fi
  grep -q OVERLAY_BUILD_FAILED "$O/$name.log" && served="NOT BUILT ($(grep -m1 OVERLAY_BUILD_FAILED "$O/$name.log"))"
  echo "VALIDATE $name: ${h:-?} runs hit, patched .co $served"
  [ "${h%%/*}" = 0 ] || ok=0
done
[ $ok = 1 ] && echo "FIX_VALIDATE=PASS" || echo "FIX_VALIDATE=FAIL"
