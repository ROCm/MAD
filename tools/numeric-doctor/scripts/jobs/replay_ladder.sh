#!/bin/bash
#SBATCH --job-name=nd-replay-ladder
#SBATCH --nodes=1
#SBATCH --time=02:00:00
# debug-numeric STEP 5b: replay one capsule on every GPU of this node, with one more copy per GPU
# than the capture saw processes on its device (the capsule's gpu_procs x containers_per_gpu),
# so the replay is at least as loaded as the model was.  Out of device memory: one copy fewer,
# down to 1.  Stops at the first attempt that runs to completion -- reproduced, clean, or
# NOT_SET_UP.  Any runtime, any scheduler:
#
#   CAPSULE=<host path to .pt> OUT_DIR=<host dir> \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name ladder_G1 --gpus 8 --exclusive -- replay_ladder.sh
#   (or run it directly on a GPU node: bash scripts/jobs/replay_ladder.sh)
#
# Every copy is one `nd run --gpus <g>` of
#   python3 -m replay_nan_op CAPSULE --chain --chain-fallback --chain-iters N --restore-env
# i.e. the [C1] per-op walk (fp64 reference column), then the whole chain re-run N times; a
# capsule without a chain falls back to the single op x N.  The copies on one GPU are each
# other's co-tenants -- no --contend -- and wait for each other at a barrier before their
# first run, so they overlap from the start.
#
# Env knobs:
#   CAPSULE              required: the capsule, as a host path or as a runtime path recorded in
#                        groups.json (/mnt/... under apptainer); must be visible in the runtime
#   OUT_DIR              required: host dir for the logs; one subdirectory per attempt:
#                        c<concurrency>.<tag>.<host>/
#   CAPSULE_ITER_COUNT   runs per copy, default 1000000 (an upper bound: the stops end sooner)
#   CAPSULE_CONCURRENCY  the starting concurrency, default capture's processes per GPU + 1 (2 if
#                        the capsule records none); on OOM one less, down to 1
#   CAPSULE_MODE         chain (default) | op: the failing op alone x ITERS
#   CAPSULE_COTENANT     a command run as co-tenant beside ONE replay per GPU (--contend-with)
#                        instead of copies as each other's co-tenants (step5-pressure.md); write
#                        paths in it as $ND_REPO / $ND_ART (@REPO@ still accepted).  No single
#                        quotes.  Forces concurrency 1.
#   CAPSULE_EXTRA_ARGS   extra replay flags for every copy (--synthesize for a shapes-only
#                        capsule -- it also reaches the kernel-id replay; --no-hipblaslt, ...)
#   CAPSULE_STOP_HITS    stop every copy once the node has this many non-finite runs (10; 0 never)
#   CAPSULE_FUTILITY     NOT_SET_UP after this many times the capsule's op call index of clean
#                        node-wide runs -- the model needed far fewer (100; 0 off)
#   CAPSULE_FUTILITY_GRACE  seconds after an attempt starts whose runs the futility stop does not
#                        count (default 180 with CAPSULE_COTENANT, else 0): a co-tenant "loads the
#                        device" (256 MB) in seconds but computes only after its own start-up, and
#                        solo-speed runs in between ended a reproducing attempt NOT_SET_UP 20 s in
#                        (2026-11-23: a ~2e-4 rate read as 1/102000)
#   CAPSULE_SECONDS      also stop every copy this long after its loop starts (off)
#   CAPSULE_DETERMINISTIC_PROBE  runs of ONE solo copy on GPU 0 before the fan-out (100; 0 off;
#                        skipped with CAPSULE_COTENANT): every run non-finite = a deterministic
#                        fault -- the ladder ends there, REPRODUCED at concurrency 1, instead of
#                        loading the capsule into NGPUS x concurrency copies (~1 min vs ~5)
#   CAPSULE_KERNEL_ID    1 (default): after the ladder (any outcome but ERROR) identify the
#                        dispatched kernel on GPU 0 -> kernel.<tag>.<host>.txt (solution index,
#                        name, arch, injected?, pinned bench line) -- compare with the model's
#                        (kernel_match.py)
#   NGPUS                GPUs to use (default: Slurm's count, else the node's)
#   The image, mounts and env come from the profile (ND_ENV); per-copy env: its EXTRA_ENV.
#
# Last line: LADDER_RESULT=... (after KERNEL=... when identified); each attempt dir holds
# summary.txt (capsule_replay_summary.sh).
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init

[ -n "${CAPSULE:-}" ] || nd_die "CAPSULE is required"
[ -n "${OUT_DIR:-}" ] || nd_die "OUT_DIR is required"
CAPSULE=$(rt_hostpath "$CAPSULE")                  # groups.json records the runtime's path
[ -r "$CAPSULE" ] || nd_die "CAPSULE=$CAPSULE is not readable"
CAPSULE=$(cd -- "$(dirname -- "$CAPSULE")" && pwd -P)/$(basename -- "$CAPSULE")
mkdir -p -- "$OUT_DIR" && OUT_DIR=$(cd -- "$OUT_DIR" && pwd -P) && chmod 777 "$OUT_DIR" || exit 1
CAPSULE_RT=$(rt_path "$CAPSULE")

ITERS=${CAPSULE_ITER_COUNT:-1000000}
STOP_HITS=${CAPSULE_STOP_HITS:-10}
FUTILITY=${CAPSULE_FUTILITY:-100}
MODE=${CAPSULE_MODE:-chain}
COTENANT=${CAPSULE_COTENANT:-}
[[ $MODE == chain || $MODE == op ]] || nd_die "CAPSULE_MODE must be chain or op"
[[ $COTENANT != *"'"* ]] || nd_die "CAPSULE_COTENANT must not contain single quotes"
ND_REPO_RT=$(rt_path "$ND_REPO_HOST"); ND_ART_RT=$([ -n "${ND_ART_HOST:-}" ] && rt_path "$ND_ART_HOST")
COTENANT=$(nd_subst "${COTENANT//@REPO@/$ND_REPO_RT}")    # runtime paths, substituted here
NG=$(nd_ngpus)
tag=$ND_JOB_TAG
host=$(hostname -s)
DET_RUNS=${CAPSULE_DETERMINISTIC_PROBE:-100}
payload() {   # payload <iters>
  local p
  if [[ $MODE == op ]]; then
    p="python3 -m replay_nan_op $CAPSULE_RT --iters $1 --restore-env"
  else
    p="python3 -m replay_nan_op $CAPSULE_RT --chain --chain-fallback --chain-iters $1 --restore-env"
  fi
  [[ -n $COTENANT ]] && p+=" --contend-with '$COTENANT'"
  [[ ${CAPSULE_SECONDS:-0} != 0 ]] && p+=" --seconds $CAPSULE_SECONDS"
  echo "$p ${CAPSULE_EXTRA_ARGS:-}"
}
PAYLOAD=$(payload "$ITERS")

echo "[$(date)] host=$host tag=$tag runtime=$RUNTIME image=$(nd_image_desc) gpus=$NG"
"$ND" prepare >/dev/null || { echo "LADDER_RESULT=ERROR the image is not usable on $host"; exit 1; }

# The capture's processes per GPU and op call index come from the capsule (needs torch): read
# them once, in the runtime, before the fan-out.
probe=$("$ND" run --gpus 0 -- "python3 -m replay_nan_op $CAPSULE_RT --capture-procs" 2>&1)
cap_procs=$(sed -n 's/^CAPTURE_PROCS=//p' <<<"$probe" | tail -1)
cap_detail=$(sed -n 's/^CAPTURE_PROCS_DETAIL=//p' <<<"$probe" | tail -1)
op_index=$(sed -n 's/^CAPTURE_OP_INDEX=//p' <<<"$probe" | tail -1)
if [[ -z "$cap_procs" ]]; then
  echo "--capture-procs printed no CAPTURE_PROCS line:" >&2; tail -n 20 <<<"$probe" >&2
  echo "LADDER_RESULT=ERROR reading the capture's contention from the capsule"; exit 1
fi
if [[ -n $COTENANT ]]; then
  START=1; start_why="one replay per GPU beside CAPSULE_COTENANT"
elif [[ -n "${CAPSULE_CONCURRENCY:-}" ]]; then
  START=$CAPSULE_CONCURRENCY; start_why="CAPSULE_CONCURRENCY"
elif [[ $cap_procs =~ ^[0-9]+$ ]]; then
  START=$((cap_procs + 1)); start_why="capture: $cap_detail; +1"
else
  START=2; start_why="capture contention not recorded ($cap_detail); assumed 1, +1"
fi
GRACE=${CAPSULE_FUTILITY_GRACE:-$([[ -n $COTENANT ]] && echo 180 || echo 0)}
EXPECT=0
(( FUTILITY > 0 )) && [[ $op_index =~ ^[0-9]+$ ]] && EXPECT=$((FUTILITY * op_index))
echo "[$(date)] capsule=$CAPSULE mode=$MODE iters=$ITERS start=c$START ($start_why), one less per OOM out=$OUT_DIR"
echo "[$(date)] futility: $( ((EXPECT)) && echo "NOT_SET_UP after $EXPECT clean node-wide runs (CAPSULE_FUTILITY=$FUTILITY x capture op index $op_index), counted after a ${GRACE}s grace" || echo off)"
echo "[$(date)] payload: $PAYLOAD"

# Each copy's `nd run` stops its command gracefully on TERM; forward it.
trap 'echo "[$(date)] TERM -- stopping copies"; kill -TERM $(jobs -p) 2>/dev/null' TERM INT

# Stop ALL copies together once the node has STOP_HITS non-finite runs (enough to quote a
# rate), or has run EXPECT times clean (futility: NOT_SET_UP, not a negative).  Stopping one
# copy early would de-stress its co-tenants, so the stop file is node-wide.
wait_copies() {
  local dir=$1 stop_hits=${2:-$STOP_HITS} runs hits t0=$SECONDS base=0
  while [[ -n "$(jobs -rp)" ]]; do
    if [[ ! -e "$dir/STOP" ]]; then
      read -r runs hits < <(cat "$dir"/*.progress 2>/dev/null | awk '{r += $1; s += $2} END {print r + 0, s + 0}')
      # runs inside the grace do not count toward futility (hits always count)
      if (( SECONDS - t0 < GRACE )); then base=$runs; fi
      runs=$((runs - base))
      if (( stop_hits > 0 && hits >= stop_hits )); then
        echo "[$(date)] node-wide non-finite runs: $hits >= CAPSULE_STOP_HITS=$stop_hits -- stopping every copy"
        echo "hits=$hits target=$stop_hits at $(date '+%F %T')" >"$dir/STOP"
      elif (( EXPECT > 0 && hits == 0 && runs >= EXPECT )); then
        echo "[$(date)] futility: 0 non-finite in $runs node-wide runs >= $EXPECT -- stopping every copy (NOT_SET_UP)"
        echo "futility runs=$runs expect=$EXPECT at $(date '+%F %T')" >"$dir/STOP"
        echo "$runs" >"$dir/FUTILE"
      fi
    fi
    sleep 3 & wait $!     # interruptible by the TERM trap
  done
}

# The copies stay children of this shell so the TERM trap reaches them.
# run_attempt <concurrency> [<gpus> <payload> <dir suffix> <stop hits>]
run_attempt() {
  local c=$1 ng=${2:-$NG} pl=${3:-$PAYLOAD} dir="$OUT_DIR/c$1${4:-}.$tag.$host" dir_rt gpu k loop_args
  mkdir -p -- "$dir"; chmod 777 "$dir"; dir_rt=$(rt_path "$dir")
  echo "[$(date)] attempt: concurrency=$c -> $((ng * c)) copies, logs in $dir"
  for gpu in $(seq 0 $((ng - 1))); do
    for k in $(seq 0 $((c - 1))); do
      loop_args="--progress-file $dir_rt/g${gpu}_k${k}.progress --stop-file $dir_rt/STOP"
      (( c > 1 )) && loop_args+=" --start-barrier $dir_rt/barrier.g$gpu --barrier-n $c"
      nd_bg_trial "$dir/g${gpu}_k${k}.$host.log" \
        "$ND" run --gpus "$gpu" --name "nd-ladder-$tag-c$c-g$gpu-k$k" --env "NAN_REPLAY_CONCURRENCY=$c" \
          -- "$pl $loop_args"
    done
  done
  wait_copies "$dir" "${5:-$STOP_HITS}"
  while [[ -n "$(jobs -rp)" ]]; do wait; done     # a trapped TERM interrupts `wait` early
  bash "$(nd_tool capsule_replay_summary.sh)" "$dir"
  ATTEMPT_RES=$(sed -n 's/^ATTEMPT_RESULT=//p' "$dir/summary.txt")
  # the totals line of summary.txt (the copies' final counts; .progress is written only every 2000 runs)
  read -r ATTEMPT_HITS ATTEMPT_RUNS < <(sed -nE 's#^=== totals:.*non-finite ([0-9]+)/([0-9]+).*#\1 \2#p' "$dir/summary.txt" | tail -1)
  ATTEMPT_HITS=${ATTEMPT_HITS:-0} ATTEMPT_RUNS=${ATTEMPT_RUNS:-0}
}

# A deterministic fault needs no fan-out: one solo copy that NaNs on every run settles it (a
# contention-gated fault does not hit every run alone, and goes on to the ladder).
tried=(); c=$START; result=""
if (( DET_RUNS > 0 )) && [[ -z $COTENANT ]]; then
  run_attempt 1 1 "$(payload "$DET_RUNS")" det 0
  tried+=("c1det:$ATTEMPT_RES:$ATTEMPT_HITS/$ATTEMPT_RUNS")
  if [[ $ATTEMPT_RES == REPRODUCED ]] && (( ATTEMPT_RUNS > 0 && ATTEMPT_HITS == ATTEMPT_RUNS )); then
    result="REPRODUCED at concurrency 1 -- deterministic: $ATTEMPT_HITS/$ATTEMPT_RUNS solo runs on GPU 0 non-finite, no fan-out"
  elif [[ $ATTEMPT_RES == ERROR ]]; then
    result="ERROR at concurrency 1 (deterministic probe) -- read the copy logs"
  fi
  echo "[$(date)] deterministic probe: $ATTEMPT_RES $ATTEMPT_HITS/$ATTEMPT_RUNS -> ${result:-not deterministic, ladder from c$START}"
fi
# An OOM that ended the ladder is reported as such, not as a negative.
while [[ -z $result ]]; do
  run_attempt "$c"; r=$ATTEMPT_RES; tried+=("c$c:$r")
  case $r in
    REPRODUCED) result="REPRODUCED at concurrency $c" ;;
    CLEAN)      result="NOT_REPRODUCED at concurrency $c" ;;
    NOT_SET_UP) result="NOT_SET_UP at concurrency $c -- 0 in $(cat "$OUT_DIR/c$c.$tag.$host/FUTILE") runs, the model hit it within $op_index op calls: escalate (step5-pressure.md)" ;;
    OOM)        if (( c > 1 )); then c=$((c - 1)); continue; fi
                result="NOT_REPRODUCED -- ladder ended on OOM at concurrency 1" ;;
    *)          result="ERROR at concurrency $c -- read the copy logs" ;;
  esac
  break
done
echo "[$(date)] attempts: ${tried[*]}  (iters=$ITERS per copy)"

# Name the kernel while the node is still ours: one replay on GPU 0 (--list-mutations
# identifies the dispatched kernel), then the bench line pinned to its solution.
if [[ $result != ERROR* && "${CAPSULE_KERNEL_ID:-1}" == 1 ]]; then
  klog="$OUT_DIR/kernel.$tag.$host.log"; ktxt="$OUT_DIR/kernel.$tag.$host.txt"
  synth=$(grep -oE -- '--synthesize( --synthesize-seed [0-9]+)?' <<<"${CAPSULE_EXTRA_ARGS:-}")
  # a grouped GEMM: name the WRONG group's solution, not the first group's (the copies'
  # value check printed GROUPED_WRONG_ROWS=)
  gw=$(grep -hos 'GROUPED_WRONG_ROWS=[0-9,]*' "$OUT_DIR"/c*/g*_k*.log | head -n 1 | cut -d= -f2)
  RNG="python3 -m replay_nan_op $CAPSULE_RT --restore-env${synth:+ $synth}${gw:+ --grouped-wrong-rows $gw}"
  "$ND" run --gpus 0 --name "nd-ladder-$tag-kernel" -- \
    "out=\$($RNG --list-mutations 2>&1); echo \"\$out\"; idx=\$(grep -m1 -oP 'solution_index: \\K\\d+' <<<\"\$out\"); $RNG --bench \${idx:+--solution-index \$idx}" \
    >"$klog" 2>&1
  python3 "$(nd_tool replay_verdict.py)" kernel "$klog" >"$ktxt" 2>&1
  # a library solution leads with its index; anything else (an aiter / CK code object) with its name --
  # "KERNEL=? <path>" read as kernel=? in the result line (2026-11-23, aiter MLA .co)
  ksi=$(sed -n 's/^solution_index: *//p' "$ktxt"); kname=$(sed -n 's/^kernel: *//p' "$ktxt" | awk '{print $1}')
  if [[ -n $ksi && $ksi != "?" ]]; then echo "KERNEL=$ksi ${kname:0:60}...  ($(basename -- "$ktxt"))"
  else echo "KERNEL=$(basename -- "${kname:-?}") (no solution index)  ($(basename -- "$ktxt"))"; fi
fi
echo "LADDER_RESULT=$result"
