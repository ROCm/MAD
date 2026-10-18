#!/usr/bin/env bash
# Run a long madengine command detached, one GPU job at a time.
#
# A Bash tool call is capped at 10 minutes and madengine runs take hours, so
# agents start the run here and poll it with `wait`. Each run holds the GPU
# lock (flock) for its whole lifetime, so runs started by parallel agents,
# workflows, or sessions on this host execute one after another instead of
# sharing GPUs and clobbering each other's working files.
#
# Usage (from the MAD repo root):
#   mad-run.sh start <run-id> -- madengine run|build [args...]
#   mad-run.sh wait <run-id> [max-seconds]    # blocks up to 540s (the default)
#   mad-run.sh status <run-id>
#   mad-run.sh stop <run-id>
#
# Every report starts with one line:
#   MAD_RUN_STATUS=queued|running|succeeded|failed exit=<code|none|stopped|unknown> dir=<dir>
# followed by any `performance:` lines and the log tail. Keep calling `wait`
# until the status is succeeded or failed.
#
# State lives in .mad-runs/<run-id>/ (cmd, pid, log, started, exit_code).
# MAD_GPU_LOCK overrides the lock file (default /tmp/mad-gpu.lock).
set -euo pipefail

RUNS_DIR="${MAD_RUNS_DIR:-.mad-runs}"
LOCK="${MAD_GPU_LOCK:-/tmp/mad-gpu.lock}"
MAX_WAIT=540

die() { echo "[mad-run] $*" >&2; exit 2; }

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
    state=failed; code=unknown  # the process is gone without recording an exit code
  fi
  echo "MAD_RUN_STATUS=$state exit=$code dir=$dir"
  if [ "$state" = queued ]; then
    echo "Waiting for the GPU lock $LOCK (another madengine run on this host holds it)."
  fi
  grep -a 'performance:' "$dir/log" 2>/dev/null | tail -n 20 || true
  echo "--- log tail ($dir/log) ---"
  tail -n 40 "$dir/log" 2>/dev/null || true
}

cmd_start() {
  local id="${1:-}"
  [ $# -ge 2 ] && [ "$2" = "--" ] || die "usage: mad-run.sh start <run-id> -- madengine run|build [args...]"
  shift 2
  [ "${1:-}" = madengine ] || die "only madengine commands can be started (got '${1:-}')"
  case "${2:-}" in
    run|build) ;;
    *) die "only 'madengine run' and 'madengine build' can be started (got 'madengine ${2:-}')" ;;
  esac
  [ -f models.json ] || die "run from the MAD repo root (models.json not found in $(pwd))"
  command -v flock >/dev/null && command -v setsid >/dev/null || die "flock and setsid (util-linux) are required"

  local dir
  dir="$(run_dir "$id")"
  [ ! -e "$dir" ] || die "run '$id' already exists at $dir; pick a new run-id"
  mkdir -p "$dir"
  printf '%q ' "$@" > "$dir/cmd"
  echo >> "$dir/cmd"

  # setsid gives the run its own process group so `stop` can signal all of it.
  # fd 9 holds the GPU lock until the run and its children exit.
  setsid nohup bash -c '
    dir=$1 lock=$2
    shift 2
    exec 9>"$lock"
    flock 9
    date +%s > "$dir/started"
    "$@"
    echo $? > "$dir/exit_code.tmp" && mv "$dir/exit_code.tmp" "$dir/exit_code"
  ' mad-run "$dir" "$LOCK" "$@" > "$dir/log" 2>&1 < /dev/null &
  echo $! > "$dir/pid"
  echo "[mad-run] started '$id' (pid $(cat "$dir/pid")): $(cat "$dir/cmd")"
  echo "[mad-run] poll with: bash .claude/skills/mad-common/mad-run.sh wait $id"
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
    echo "[mad-run] stopped '$1'. Containers madengine started may still be running; check 'docker ps'."
  fi
  report "$dir"
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  wait)   shift; cmd_wait "$@" ;;
  status) shift; report "$(existing_run_dir "${1:-}")" ;;
  stop)   shift; cmd_stop "$@" ;;
  *) die "usage: mad-run.sh start <run-id> -- madengine run|build [args...] | wait <run-id> [secs] | status <run-id> | stop <run-id>" ;;
esac
