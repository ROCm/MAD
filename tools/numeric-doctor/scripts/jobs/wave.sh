#!/bin/bash
#SBATCH --job-name=nd-wave
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# One wave of model trials on one node -- the step-4 capture wave (capture + control arms,
# early stop), the 5N compiled-track C0/C1 wave, the step-8 model A/B.  Any runtime (each trial
# is one `nd run`).  Replaces model_wave_docker.sbatch and the wave half of
# the old capture-to-reproducer job.
#
#   SPEC=<spec file> NAN_SIGNAL='<regex>' [CAPTURE_ENV='NAN_CAPTURE_DATA=1 ...'] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name wave --gpus 8 --exclusive -- wave.sh
#
# Spec: one trial per line, `#` and blank lines skipped:
#   <gpu>|<arm>|<env VAR=VALUE ...>|<command, bash -c in WORKDIR>
#   0|cap||python3 /path/train.py --steps 100
#   6|ctl||python3 /path/train.py --steps 100
#   5|f2|PYTHONPATH=$ND_ART/fix/patch|python3 /path/train.py --steps 100
#   env words are shell-quoted (V='a b' or V=a, both reach the trial as the bare value); each must be VAR=VALUE
#   0,1|ctl||python3 /path/serve.py --tp 2     (a GPU list: ONE trial over those GPUs -- TP / torchrun on
#                                             one node; its names use 0-1: nd-wave-<tag>-0-1, 0-1_0-1.ctl...)
# Write paths as $ND_REPO / $ND_ART (the runtime's view; @REPO@ @ART@ @WAVE@ still accepted).
# Arm `cap` gets NAN_CAPTURE_DIR=<CAPSULE_DIR>/t<i>, NAN_CAPTURE_CONTAINERS_PER_GPU=<trials on
# that GPU>, NAN_PROGRESS_EVERY=10 and CAPTURE_ENV; LOCATE=1 gives the first cap trial the loop
# locator.  Every trial: its own name, inductor cache, HIP_VISIBLE_DEVICES=<gpu> and a graceful
# SIGTERM on stop (nd run).
#
# Multi-node (NODES > 1, default the job's node count -- nd submit --nodes N): every spec line has
# GPU field `all`, the trial of one multi-rank job; it runs as one `nd run --gpus all` per node on
# the job's first NODES nodes (srun), each told ND_NNODES, ND_NODE_RANK and ND_MASTER_ADDR (the
# first node) for its torchrun --nnodes / --node_rank / --master_addr.  Every `all` line gets them
# (1, 0, localhost on one node), so a spec written with them runs on any node count.  k `all` lines
# = k copies of the job at once on the same nodes (step3-raise.md): each gets MASTER_PORT 29500 +
# 10 x copy (write --master_port=$MASTER_PORT) and NAN_CAPTURE_CONTAINERS_PER_GPU=k.
#
# Instead of SPEC, WAVE_SCRIPT=<an existing capture launcher that honours CAPSULE_DIR> runs as
# is, in its own process group (an older example's capture launcher); then set WAVE_LOGS.
#
# Knobs: NODES, CAPSULE_DIR (default <ART>/capsules; empty when the spec has `cap` arms -- a
# control-only A/B does not write it), WAVE_DIR (<ART>/wave), WORKDIR
# (<ART>), EARLY_STOP (1 when a cap arm exists: stop every trial, controls too, once
# MIN_CAPSULES (1) capsules have settled -- none written in the last 15 s), WAIT_CONTROLS (1 =
# also wait for every control trial to end), WAIT_ARMS (the arms WAIT_CONTROLS waits for, default
# `ctl`; e.g. `ctl c0 c1` when the compiled track's C0/C1 lines ride in the capture wave:
# step4-run.md *Compiled NaN predicted*), CONTROLS=none (no control arm, with WAIT_CONTROLS),
# EARLY_STOP_CMD (stop when `bash -c` of it exits 0), NAN_SIGNAL / NAN_ERROR / WAVE_NAME_RE (for
# wave_summary.py), WAVE_SUMMARY (its output, wave_summary.<tag>.txt beside WAVE_DIR), WAVE_LOGS /
# CONTROL_LOGS (globs, WAVE_SCRIPT mode).
#
# Logs: <WAVE_DIR>/<gpu>_<i>.<arm>.<tag>.<host>.log ending TRIAL_RC=<rc> (the wave_summary.py /
# native_handoff.py shape).  Last lines: RESULT per trial, CAPSULES=<n>, WAVE_RESULT=done|stopped.
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
[ -n "${ND_ART_HOST:-}" ] || nd_die "no artifact dir (use the investigation's env.sh as ND_ENV)"
[ -n "${SPEC:-}" ] || [ -n "${WAVE_SCRIPT:-}" ] || nd_die "SPEC or WAVE_SCRIPT is required"
nd_check_capture_env
CAPS=${CAPSULE_DIR:-$ND_ART_HOST/capsules}; WAVE=${WAVE_DIR:-$ND_ART_HOST/wave}
WORKDIR=${WORKDIR:-$ND_ART_HOST}
tag=$ND_JOB_TAG; host=$(hostname -s)
mkdir -p "$WAVE" "$CAPS" && chmod 777 "$WAVE" "$CAPS"
export CAPSULE_DIR=$CAPS
ND_REPO_RT=$(rt_path "$ND_REPO_HOST"); ND_ART_RT=$(rt_path "$ND_ART_HOST")
CAPS_RT=$(rt_path "$CAPS"); WAVE_RT=$(rt_path "$WAVE")
sub() { local x=${1//@WAVE@/$WAVE_RT}; x=${x//@ART@/$ND_ART_RT}; x=${x//@REPO@/$ND_REPO_RT}; nd_subst "$x"; }
say() { echo "[$(date '+%F %T')] $*"; }
# The env words of a spec line or CAPTURE_ENV, shell-quoted as one would write them (VAR='a b(', VAR="x"):
# split by xargs, which honours quotes and runs nothing.  Split on blanks alone, a quoted value kept its
# quotes: NAN_FIX_INDUCTOR_OVERRIDES='sqrt:...' reached the trial as 'sqrt and the fix silently did not
# apply -- five "fixed" trials ran stock (2026-11-23).  Every word must be VAR=VALUE.
env_words() {
  local w out=()
  [ -n "${1//[[:space:]]/}" ] || return 0
  mapfile -d '' out < <(printf '%s' "$1" | xargs printf '%s\0') || nd_die "env field: unbalanced quotes: $1"
  for w in "${out[@]}"; do
    [[ $w =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || nd_die "env field: '$w' is not VAR=VALUE (in: $1)"
    envs+=(--env "$w")
  done
}
t0=$(date +%s); touch "$WAVE/.start.$tag"

"$ND" prepare >/dev/null || { echo "WAVE_RESULT=ERROR image not usable on $host"; exit 1; }

NODES=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
[ "$NODES" = 1 ] || [ -n "${SPEC:-}" ] || nd_die "NODES=$NODES needs SPEC (a WAVE_SCRIPT launcher runs on this node only)"
nd_multinode "$NODES"

pids=(); has_cap=0; wave_pgid=""
if [ -n "${SPEC:-}" ]; then
  mapfile -t lines < <(grep -vE '^\s*(#|$)' "$SPEC")
  [ "${#lines[@]}" -gt 0 ] || nd_die "empty spec $SPEC"
  [ "$NODES" = 1 ] || ! grep -qvE '^all\|' <(printf '%s\n' "${lines[@]}") ||
    nd_die "NODES=$NODES: every spec line must have GPU field all (a multi-rank job over the nodes)"
  declare -A per_gpu
  for l in "${lines[@]}"; do g=${l%%|*}; per_gpu[$g]=$(( ${per_gpu[$g]:-0} + 1 )); [ "$(cut -d'|' -f2 <<<"$l")" = cap ] && has_cap=1; done
  # k > 1 copies of a multi-rank job: apptainer without --pid (lib.sh nd_drop_pid_ns); the
  # declared containers per GPU stay k -- the capture's scan still records gpu_procs=1 per rank
  if [ "${per_gpu[all]:-0}" -gt 1 ] && nd_drop_pid_ns; then
    say "${per_gpu[all]} copies on the same GPUs: apptainer without --pid, NAN_CAPTURE_CONTAINERS_PER_GPU=${per_gpu[all]}"
  fi
  say "host=$host tag=$tag runtime=$RUNTIME image=$(nd_image_desc) spec=$SPEC trials=${#lines[@]} nodes=$NODES${ND_LAUNCH[*]:+ ($ND_HOSTS)}"
  located=0; copy=0
  for i in "${!lines[@]}"; do
    IFS='|' read -r g arm env cmd <<<"${lines[$i]}"
    gid=${g//,/-}                                    # 0,1 -> 0-1: docker names ([a-zA-Z0-9_.-]) and log names
    idx=$i; [ "${per_gpu[$g]}" = 1 ] && idx=$gid    # one trial per GPU: index = GPU (t<gpu>, <gpu>_<gpu>)
    log=$WAVE/${gid}_${idx}.${arm}.$tag.$host.log
    envs=()
    if [ "$arm" = cap ]; then
      mkdir -p "$CAPS/t$idx" && chmod 777 "$CAPS/t$idx"
      envs+=(--env "NAN_CAPTURE_DIR=$CAPS_RT/t$idx" --env "NAN_CAPTURE_CONTAINERS_PER_GPU=${per_gpu[$g]}" --env NAN_PROGRESS_EVERY=10)
      env_words "${CAPTURE_ENV:-}"
      if [ "${LOCATE:-0}" = 1 ] && [ "$located" = 0 ]; then
        envs+=(--env "PYTHONPATH=$(rt_path "$(nd_tool locate)")"); located=1
      fi
    fi
    # several `all` lines = copies of the job side by side on the same nodes: a port each
    [ "$g" = all ] && { envs+=(--env "MASTER_PORT=$((29500 + 10 * copy))"); copy=$((copy + 1)); }
    env_words "$(sub "$env")"
    nd_bg_trial "$log" "${ND_LAUNCH[@]}" "$ND" run --gpus "$g" --name "nd-wave-$tag-$idx" --workdir "$WORKDIR" "${envs[@]}" -- "$(sub "$cmd")"
    pids+=($!)
  done
  WAVE_LOGS=$WAVE/*.$tag.$host.log
  CONTROL_LOGS=""; for a in ${WAIT_ARMS:-ctl}; do CONTROL_LOGS+="$WAVE/*.$a.$tag.$host.log "; done
else
  [ -r "$WAVE_SCRIPT" ] || nd_die "WAVE_SCRIPT=$WAVE_SCRIPT unreadable"
  has_cap=1
  say "host=$host tag=$tag WAVE_SCRIPT=$WAVE_SCRIPT (run as is, own process group)"
  setsid bash "$WAVE_SCRIPT" >"$WAVE/wave_script.$tag.$host.out" 2>&1 &
  pids+=($!); wave_pgid=$!
  : "${WAVE_LOGS:?WAVE_LOGS (glob of the trial logs of the launcher) is required with WAVE_SCRIPT}"
  WAVE_LOGS=${WAVE_LOGS//\{job\}/$tag}; CONTROL_LOGS=${CONTROL_LOGS:-${WAVE_LOGS/\*.$tag./*.ctl.$tag.}}
fi

stop_all() {
  if [ -n "$wave_pgid" ]; then
    kill -TERM -- "-$wave_pgid" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$wave_pgid" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$wave_pgid" 2>/dev/null
  else
    kill -TERM "${pids[@]}" 2>/dev/null       # each nd run stops its trial gracefully (docker stop -t 20)
  fi
}
trap 'say "TERM -- stopping the wave"; stop_all' TERM INT

controls_done() {
  local f any=0
  # shellcheck disable=SC2086
  for f in $CONTROL_LOGS; do [ -e "$f" ] || continue; any=1; grep -q '^TRIAL_RC=' "$f" || return 1; done
  [ "$any" = 1 ] && return 0
  [ "${CONTROLS:-}" = none ]
}
capsules_settled() {
  local n; n=$(find "$CAPS" -name '*.pt' -newer "$WAVE/.start.$tag" | wc -l)
  [ "$n" -ge "${MIN_CAPSULES:-1}" ] || return 1
  [ -z "$(find "$CAPS" -name '*.pt' -newermt '15 seconds ago')" ]
}
running() { local p; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

stop=""
EARLY_STOP=${EARLY_STOP:-$has_cap}
while running; do
  sleep 10 & wait $!
  [ "$EARLY_STOP" = 1 ] || continue
  capsules_settled || continue
  if [ -n "${EARLY_STOP_CMD:-}" ]; then bash -c "$EARLY_STOP_CMD" && stop=cmd
  elif [ "${WAIT_CONTROLS:-0}" = 1 ]; then controls_done && stop=controls
  else stop=capsule; fi
  [ -n "$stop" ] && break
done
if [ -n "$stop" ]; then
  say "early stop ($stop): $(find "$CAPS" -name '*.pt' -newer "$WAVE/.start.$tag" | wc -l) capsule(s) settled; stopping the remaining trials"
  stop_all
fi
wait "${pids[@]}" 2>/dev/null
ncap=$(find "$CAPS" -name '*.pt' -newer "$WAVE/.start.$tag" | wc -l)

# shellcheck disable=SC2086
wlogs=(); for f in $WAVE_LOGS; do [ -e "$f" ] && wlogs+=("$f"); done
# beside WAVE_DIR: <ART>/ for step 4's <ART>/wave; a step-8 A/B's <ART>/fix_ab/wave keeps its summary
# in fix_ab/ (it was written to <ART>/, apart from its logs, 2026-11-20)
summary=${WAVE_SUMMARY:-$(dirname -- "$WAVE")/wave_summary.$tag.txt}
if [ ${#wlogs[@]} -gt 0 ] && [ -n "${NAN_SIGNAL:-}" ]; then
  python3 "$(nd_tool wave_summary.py)" "${wlogs[@]}" --capsules "$CAPS" --signal "$NAN_SIGNAL" \
      ${NAN_ERROR:+--error "$NAN_ERROR"} ${WAVE_NAME_RE:+--name "$WAVE_NAME_RE"} ${stop:+--stopped-early} >"$summary" 2>&1
  say "summary: $summary"
fi
for f in "${wlogs[@]}"; do echo "RESULT $(basename "$f"): $(tail -n 1 "$f")"; done
echo "CAPSULES=$ncap"
echo "WAVE_STOP=${stop:-none} WAVE_SECONDS=$(( $(date +%s) - t0 ))"
echo "WAVE_RESULT=$([ -n "$stop" ] && echo "stopped($stop)" || echo done) tag=$tag host=$host"
