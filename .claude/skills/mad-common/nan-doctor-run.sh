#!/usr/bin/env bash
# Run one nan-doctor GPU job detached, under the host GPU lock.
#
# nan-doctor replays NaN capsules without the model, in the image madengine
# built. Those jobs use the GPUs like a madengine run, so they take the same
# lock as mad-run.sh and geak-run.sh, and have the same start/wait/status/stop
# shape. mad-run.sh starts only madengine; this starts only the two nan-doctor
# launchers below. Callers cannot pass another script or a shell command.
#
# Usage (from the MAD repo root):
#   nan-doctor-run.sh start <run-id> <nan-doctor-home> <job> [KEY=VALUE ...]
#   nan-doctor-run.sh wait <run-id> [max-seconds]   # blocks up to 540s
#   nan-doctor-run.sh status <run-id>
#   nan-doctor-run.sh stop <run-id>
#
# <job>  replay  scripts/internal/docker_replay_nan_op.sh: one command (CMD) in
#                the image, the debug-nan <GPU_EXEC>.
#                Keys: CMD IMAGE IMAGE_TAR GPUS EXTRA_MOUNTS EXTRA_ENV WORKDIR LOG
#        node5b  scripts/internal/replay_capsule_node_docker.sh: debug-nan step 5b,
#                one capsule replayed under load on every GPU of this host.
#                Keys: CAPSULE OUT_DIR IMAGE IMAGE_TAR NGPUS CAPSULE_*
# KEY=VALUE pairs become the launcher's environment; any other key is refused.
# CMD runs inside the container, never on the host.
#
# Report first line:
#   NAN_DOCTOR_RUN_STATUS=queued|running|succeeded|failed exit=<code|none|stopped|unknown> dir=<dir>
# followed by the log tail. Keep calling `wait` until succeeded or failed.
#
# State lives in .mad-runs/<run-id>/ (cmd, pid, log, started, exit_code).
# MAD_GPU_LOCK overrides the lock file (default /tmp/mad-gpu.lock, shared with
# mad-run.sh and geak-run.sh).
set -euo pipefail

RUNS_DIR="${MAD_RUNS_DIR:-.mad-runs}"
LOCK="${MAD_GPU_LOCK:-/tmp/mad-gpu.lock}"
MAX_WAIT=540
USAGE="usage: nan-doctor-run.sh start <run-id> <nan-doctor-home> replay|node5b [KEY=VALUE ...] | wait <run-id> [secs] | status <run-id> | stop <run-id>"

die() { echo "[nan-doctor-run] $*" >&2; exit 2; }

run_dir() {
  [[ "${1:-}" =~ ^[A-Za-z0-9._-]+$ ]] || die "run-id must match [A-Za-z0-9._-]+, got '${1:-}'"
  echo "$RUNS_DIR/$1"
}

existing_run_dir() {
  local dir
  dir="$(run_dir "${1:-}")"
  [ -d "$dir" ] || die "no run '${1}' under $RUNS_DIR"
  echo "$dir"
}

alive() { [ -f "$1/pid" ] && kill -0 "$(cat "$1/pid")" 2>/dev/null; }

report() {
  local dir="$1" state code="none"
  if [ -f "$dir/exit_code" ]; then
    code="$(cat "$dir/exit_code")"
    if [ "$code" = 0 ]; then state=succeeded; else state=failed; fi
  elif alive "$dir"; then
    if [ -f "$dir/started" ]; then state=running; else state=queued; fi
  else
    state=failed; code=unknown
  fi
  echo "NAN_DOCTOR_RUN_STATUS=$state exit=$code dir=$dir"
  if [ "$state" = queued ]; then
    echo "Waiting for the GPU lock $LOCK (another GPU run on this host holds it)."
  fi
  echo "--- log tail ($dir/log) ---"
  tail -n 40 "$dir/log" 2>/dev/null || true
}

key_allowed() { # <job> <key>
  case "$1:$2" in
    replay:CMD|replay:IMAGE|replay:IMAGE_TAR|replay:GPUS|replay:EXTRA_MOUNTS|replay:EXTRA_ENV|replay:WORKDIR|replay:LOG) return 0 ;;
    node5b:CAPSULE|node5b:OUT_DIR|node5b:IMAGE|node5b:IMAGE_TAR|node5b:NGPUS|node5b:CAPSULE_*) return 0 ;;
  esac
  return 1
}

cmd_start() {
  [ $# -ge 3 ] || die "$USAGE"
  local id="$1" home="$2" job="$3" script
  shift 3
  case "$job" in
    replay) script=scripts/internal/docker_replay_nan_op.sh ;;
    node5b) script=scripts/internal/replay_capsule_node_docker.sh ;;
    *) die "job must be 'replay' or 'node5b' (got '$job')" ;;
  esac
  [ -f models.json ] || die "run from the MAD repo root (models.json not found in $(pwd))"
  [ -d "$home" ] || die "nan-doctor home '$home' does not exist"
  home="$(cd "$home" && pwd)"
  [ -f "$home/$script" ] || die "$home/$script not found; is '$home' a nan-doctor checkout?"
  command -v flock >/dev/null && command -v setsid >/dev/null || die "flock and setsid (util-linux) are required"

  local -a envs=()
  local kv key
  for kv in "$@"; do
    [[ "$kv" == *=* ]] || die "expected KEY=VALUE, got '$kv'"
    key="${kv%%=*}"
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "bad key '$key'"
    key_allowed "$job" "$key" || die "key '$key' is not accepted for job '$job'"
    envs+=("$kv")
  done
  if [ "$job" = replay ]; then
    printf '%s\n' "${envs[@]+"${envs[@]}"}" | grep -q '^CMD=' || die "job 'replay' needs CMD=<command>"
  fi

  local dir
  dir="$(run_dir "$id")"
  [ ! -e "$dir" ] || die "run '$id' already exists at $dir; pick a new run-id"
  mkdir -p "$dir"
  { printf '%q ' env "${envs[@]+"${envs[@]}"}" bash "$home/$script"; echo; } > "$dir/cmd"

  # setsid gives the run its own process group so `stop` can signal all of it.
  # fd 9 holds the GPU lock until the job and its children exit.
  setsid nohup bash -c '
    dir=$1 lock=$2
    shift 2
    exec 9>"$lock"
    flock 9
    date +%s > "$dir/started"
    "$@"
    echo $? > "$dir/exit_code.tmp" && mv "$dir/exit_code.tmp" "$dir/exit_code"
  ' nan-doctor-run "$dir" "$LOCK" env "${envs[@]+"${envs[@]}"}" bash "$home/$script" > "$dir/log" 2>&1 < /dev/null &
  echo $! > "$dir/pid"
  echo "[nan-doctor-run] started '$id' (pid $(cat "$dir/pid")): $(cat "$dir/cmd")"
  echo "[nan-doctor-run] poll with: bash .claude/skills/mad-common/nan-doctor-run.sh wait $id"
}

cmd_wait() {
  local dir max end
  dir="$(existing_run_dir "${1:-}")"
  max="${2:-$MAX_WAIT}"
  [[ "$max" =~ ^[0-9]+$ ]] || die "max-seconds must be a whole number, got '$max'"
  [ "$max" -le "$MAX_WAIT" ] || max=$MAX_WAIT
  end=$((SECONDS + max))
  while [ ! -f "$dir/exit_code" ] && alive "$dir" && [ "$SECONDS" -lt "$end" ]; do
    sleep $(( end - SECONDS < 10 ? end - SECONDS : 10 ))
  done
  report "$dir"
}

cmd_stop() {
  local dir pid
  dir="$(existing_run_dir "${1:-}")"
  if alive "$dir"; then
    pid="$(cat "$dir/pid")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do alive "$dir" || break; sleep 1; done
    alive "$dir" && kill -KILL -- "-$pid" 2>/dev/null || true
    [ -f "$dir/exit_code" ] || echo stopped > "$dir/exit_code"
    echo "[nan-doctor-run] stopped '$1'. Replay containers may still be running; check 'docker ps'."
  fi
  report "$dir"
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  wait)   shift; cmd_wait "$@" ;;
  status) shift; report "$(existing_run_dir "${1:-}")" ;;
  stop)   shift; cmd_stop "$@" ;;
  *) die "$USAGE" ;;
esac
