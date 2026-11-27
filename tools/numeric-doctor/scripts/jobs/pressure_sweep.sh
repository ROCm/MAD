#!/bin/bash
#SBATCH --job-name=nd-pressure-sweep
#SBATCH --nodes=1
#SBATCH --time=00:30:00
# debug-numeric step5-pressure.md, rung 5d: one co-tenant variant per GPU, all replaying the same
# capsule, so one node answers "which load triggers it" at once.  Any runtime.
#
#   CAPSULE=<host .pt> OUT_DIR=<host dir> VARIANTS=<file> \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name sweep --gpus 8 --exclusive -- pressure_sweep.sh
#
# VARIANTS: one co-tenant command per line (blank lines, # comments skipped), line i on GPU i, at
# most one per GPU; write paths as $ND_REPO / $ND_ART (@REPO@ still accepted).  No single quotes.
#   python3 -m transformer_hog --mode full --procs 2 <sizes>
#   python3 -m transformer_hog --mode fwd --procs 2 <sizes>
#   bash $ND_REPO/scripts/cotenant_loop.sh <log> python3 <trainer> ...      # control
#   solo                                                                     # no co-tenant
# `solo` is the no-load arm on its own GPU (a `sleep` co-tenant was refused: CONTENTION_INVALID,
# it never loads the device, 2026-11-20).
# Each GPU runs one `nd run` of
#   replay_nan_op CAPSULE --iters N --restore-env --contend-with '<line>'
# (CAPSULE_MODE=chain: --chain --chain-fallback --chain-iters N); under apptainer with --pid, so
# the co-tenant dies with its replay.
#
# Knobs: ITERS (1000000), CAPSULE_MODE (op), SECONDS_PER_GPU (0 = off), CAPSULE_EXTRA_ARGS.
# Result: the table after '=== Results', one row per GPU -- verdict, n/N, the co-tenant's last
# line when it died at start (full: g<i>.cotenant.log), its gate line, and STARVED when that
# GPU's replay ran under 1/10 of the median launch rate (too few launches to mean a 0).
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
[ "${RUNTIME:-}" = apptainer ] || grep -qs 'RUNTIME=apptainer' "${ND_ENV:-/dev/null}" &&
  case " ${APPTAINER_FLAGS:-} " in *" --pid "*) ;; *) export APPTAINER_FLAGS="${APPTAINER_FLAGS:-} --pid" ;; esac
nd_job_init
CAPSULE=$(rt_hostpath "${CAPSULE:?CAPSULE is required}")
[ -r "$CAPSULE" ] || nd_die "CAPSULE=$CAPSULE unreadable"
[ -n "${OUT_DIR:-}" ] && [ -r "${VARIANTS:-}" ] || nd_die "OUT_DIR and a readable VARIANTS file are required"
ITERS=${ITERS:-1000000}; MODE=${CAPSULE_MODE:-op}
tag=$ND_JOB_TAG; host=$(hostname -s)
OUT=$OUT_DIR/sweep.$tag.$host; mkdir -p -- "$OUT" && chmod 777 "$OUT"
OUT=$(cd "$OUT" && pwd -P); OC=$(rt_path "$OUT")
ND_REPO_RT=$(rt_path "$ND_REPO_HOST"); ND_ART_RT=$([ -n "${ND_ART_HOST:-}" ] && rt_path "$ND_ART_HOST")
mapfile -t V < <(grep -vE '^\s*(#|$)' "$VARIANTS")
ng=$(nd_ngpus)
[ "${#V[@]}" -le "$ng" ] || nd_die "${#V[@]} variants for $ng GPUs"
for v in "${V[@]}"; do [[ $v != *"'"* ]] || nd_die "no single quotes in VARIANTS: $v"; done
cp -- "$VARIANTS" "$OUT/variants.txt"
if [ "$MODE" = op ]; then RUN="--iters $ITERS"; else RUN="--chain --chain-fallback --chain-iters $ITERS"; fi
[ "${SECONDS_PER_GPU:-0}" -gt 0 ] && RUN+=" --seconds $SECONDS_PER_GPU"
RUN+="${CAPSULE_EXTRA_ARGS:+ $CAPSULE_EXTRA_ARGS}"
echo "[$(date)] host=$host tag=$tag runtime=$RUNTIME capsule=$CAPSULE mode=$MODE iters=$ITERS out=$OUT"
"$ND" prepare >/dev/null || nd_die "image not usable on $host"
trap 'echo "[$(date)] TERM -- stopping"; kill -TERM $(jobs -p) 2>/dev/null' TERM INT
for g in "${!V[@]}"; do
  cot=$(nd_subst "${V[$g]//@REPO@/$ND_REPO_RT}")
  echo "GPU $g: $cot"
  if [ "$cot" = solo ]; then ctn=""; else ctn="--contend-with '$cot' --contend-log $OC/g$g.cotenant.log"; fi
  nd_bg_trial "$OUT/g$g.$host.log" "$ND" run --gpus "$g" --name "nd-sweep-$tag-g$g" -- \
    "python3 -m replay_nan_op $(rt_path "$CAPSULE") $RUN --restore-env $ctn --progress-file $OC/g$g.progress"
  date +%s >"$OUT/g$g.t0"
done
while [ -n "$(jobs -rp)" ]; do wait; done
echo "=== Results (non-finite / launches; $OUT) ==="
declare -A RATE
for g in "${!V[@]}"; do
  n=0; [ -r "$OUT/g$g.progress" ] && read -r n _ < "$OUT/g$g.progress"   # none: died before a launch
  dt=$(( $(stat -c %Y "$OUT/g$g.progress" 2>/dev/null || date +%s) - $(cat "$OUT/g$g.t0") ))
  RATE[$g]=$(( n / (dt > 0 ? dt : 1) ))
done
med=$(printf '%s\n' "${RATE[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR + 1) / 2)] + 0}')
for g in "${!V[@]}"; do
  f=$OUT/g$g.$host.log
  v=$(grep -m1 -oE 'non-finite on [0-9]+/[0-9]+ (launches|chain runs)' "$f")
  verdict=$(grep -oP '^\[verdict\] \K\S+' "$f" | tail -1)
  gate=$(grep -m1 -oE 'co-tenants are loading the device \([^)]*\)|never loaded the device|exited before loading' "$f")
  [ "${V[$g]}" = solo ] && gate="solo, no co-tenant"
  starved=""; (( med > 0 && RATE[$g] * 10 < med )) && starved="  STARVED (${RATE[$g]}/s vs median ${med}/s)"
  echo "  GPU $g  ${verdict:-?}  ${v:-no count}  [${gate:-no gate line}]${starved}  ${V[$g]}"
  if [[ $gate == *"exited before loading"* || $gate == *"never loaded"* ]]; then
    why=$(grep -avE '^\s*$|Warning' "$OUT/g$g.cotenant.log" 2>/dev/null | tail -1)
    echo "           co-tenant said: ${why:-(nothing; see $OUT/g$g.cotenant.log)}"
  fi
done
