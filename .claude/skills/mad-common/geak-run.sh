#!/usr/bin/env bash
# Run GEAK's kernel_workflow detached, in its own top-level headless Claude session.
#
# kernel_workflow.js is itself a Workflow script. Workflows cannot nest and
# subagents have no Workflow tool, so mad-optimize cannot start it from inside
# its own run. GEAK's own driver (interface/run_e2e.py) launches it with
# `claude -p`; this helper does the same, detached and under the host GPU lock,
# with the same start/wait/status/stop shape as mad-run.sh.
#
# Usage (from the MAD repo root):
#   geak-run.sh start <run-id> <geak-home> <args.json>
#   geak-run.sh wait <run-id> [max-seconds]     # blocks up to 540s
#   geak-run.sh status <run-id>
#   geak-run.sh stop <run-id>
#
# <args.json> holds the kernel_workflow args object (kernel_path, workflow_dir,
# budget, gpu_ids, task, apply_to_original, eval_dir, ...). The command line is
# fixed here; callers cannot pass arbitrary claude flags or commands.
#
# Report first line:
#   GEAK_RUN_STATUS=queued|running|succeeded|failed exit=<code|...> dir=<dir> eval_dir=<dir|none> final_patch=present|missing
# `succeeded` means the process exited 0 AND a non-empty final_patch.diff is in
# <eval_dir>, or one directory above it. GEAK's instructions name eval_dir, but
# a finished session has written the diff beside that directory. status copies
# that file into eval_dir. `claude -p` can return while the workflow is still
# working, so a clean exit with no final patch is reported as failed.
#
# State lives in .mad-runs/<run-id>/ (cmd, pid, log, started, exit_code, args.json).
# MAD_GPU_LOCK overrides the lock file (default /tmp/mad-gpu.lock, shared with
# mad-run.sh so GEAK and madengine runs queue instead of sharing GPUs).
# GEAK_TIMEOUT_SECS bounds the session (default 14400).
set -euo pipefail

RUNS_DIR="${MAD_RUNS_DIR:-.mad-runs}"
LOCK="${MAD_GPU_LOCK:-/tmp/mad-gpu.lock}"
MAX_WAIT=540
SETTINGS='{"enableWorkflows": true, "ultracode": true}'
TOOLS='Workflow,Bash,Read,Write'

die() { echo "[geak-run] $*" >&2; exit 2; }

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

eval_dir_of() { # prints the eval_dir from the saved args, or nothing
  python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("eval_dir",""))' "$1/args.json" 2>/dev/null || true
}

# A finished GEAK session writes final_patch.diff in eval_dir, or one directory
# above it, next to the source tree. Copy the parent file into eval_dir so the
# contract path exists.
locate_patch() {
  local ev="$1" parent
  [ -n "$ev" ] || return 1
  [ -s "$ev/final_patch.diff" ] && return 0
  parent="$(dirname "$ev")"
  [ -s "$parent/final_patch.diff" ] || return 1
  cp "$parent/final_patch.diff" "$ev/final_patch.diff"
  echo "[geak-run] copied ${parent}/final_patch.diff into ${ev}/" >&2
  return 0
}

report() {
  local dir="$1" state code="none" ev patch=missing
  ev="$(eval_dir_of "$dir")"
  if locate_patch "$ev"; then patch=present; fi
  if [ -f "$dir/exit_code" ]; then
    code="$(cat "$dir/exit_code")"
    if [ "$code" = 0 ] && [ "$patch" = present ]; then state=succeeded; else state=failed; fi
  elif alive "$dir"; then
    if [ -f "$dir/started" ]; then state=running; else state=queued; fi
  else
    state=failed; code=unknown
  fi
  echo "GEAK_RUN_STATUS=$state exit=$code dir=$dir eval_dir=${ev:-none} final_patch=$patch"
  if [ "$state" = queued ]; then
    echo "Waiting for the GPU lock $LOCK (another GPU run on this host holds it)."
  fi
  if [ "$state" = failed ] && [ "$code" = 0 ]; then
    echo "claude exited 0 but ${ev:-<no eval_dir>}/final_patch.diff is missing: the session returned before the workflow finished, or the workflow failed."
  fi
  echo "--- log tail ($dir/log) ---"
  tail -n 40 "$dir/log" 2>/dev/null || true
}

cmd_start() {
  local id="${1:-}" home="${2:-}" argsfile="${3:-}"
  [ $# -eq 3 ] || die "usage: geak-run.sh start <run-id> <geak-home> <args.json>"
  [ -f models.json ] || die "run from the MAD repo root (models.json not found in $(pwd))"
  command -v flock >/dev/null && command -v setsid >/dev/null && command -v timeout >/dev/null \
    || die "flock, setsid and timeout (util-linux/coreutils) are required"
  command -v claude >/dev/null || die "claude is not on PATH"
  home="${home/#\$HOME/$HOME}"
  [ -f "$home/kernel_workflow/kernel_workflow.js" ] || die "$home/kernel_workflow/kernel_workflow.js not found"
  [ -f "$argsfile" ] || die "args file '$argsfile' not found"
  python3 -I - "$argsfile" <<'PY' || die "args.json must be an object with kernel_path, workflow_dir and eval_dir"
import json, sys
a = json.load(open(sys.argv[1]))
assert isinstance(a, dict) and all(a.get(k) for k in ("kernel_path", "workflow_dir", "eval_dir"))
PY

  local dir
  dir="$(run_dir "$id")"
  [ ! -e "$dir" ] || die "run '$id' already exists at $dir; pick a new run-id"
  mkdir -p "$dir"
  cp "$argsfile" "$dir/args.json"

  local prompt
  prompt="Invoke the Workflow tool exactly once with scriptPath \"$home/kernel_workflow/kernel_workflow.js\" and args set to the JSON object in $(pwd)/$dir/args.json (read the file, pass the parsed object, not a string). Wait for the workflow to finish and report its result. Do not start e2e_workflow. Do not run madengine. Do not edit any file outside the kernel_path and eval_dir in those args."
  printf '%q ' claude -p "$prompt" --output-format json --settings "$SETTINGS" --allowed-tools "$TOOLS" --permission-mode auto > "$dir/cmd"
  echo >> "$dir/cmd"

  # setsid gives the run its own process group so `stop` can signal all of it.
  # fd 9 holds the GPU lock until the session and its children exit.
  setsid nohup bash -c '
    dir=$1 lock=$2 secs=$3
    shift 3
    exec 9>"$lock"
    flock 9
    date +%s > "$dir/started"
    # claude -p stops background workflows ~10 min after its last turn unless this is 0;
    # the timeout below is the bound instead.
    export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0
    timeout "$secs" "$@"
    echo $? > "$dir/exit_code.tmp" && mv "$dir/exit_code.tmp" "$dir/exit_code"
  ' geak-run "$dir" "$LOCK" "${GEAK_TIMEOUT_SECS:-14400}" \
    claude -p "$prompt" --output-format json --settings "$SETTINGS" \
    --allowed-tools "$TOOLS" --permission-mode auto > "$dir/log" 2>&1 < /dev/null &
  echo $! > "$dir/pid"
  echo "[geak-run] started '$id' (pid $(cat "$dir/pid")); eval_dir=$(eval_dir_of "$dir")"
  echo "[geak-run] poll with: bash .claude/skills/mad-common/geak-run.sh wait $id"
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
    echo "[geak-run] stopped '$1'."
  fi
  report "$dir"
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  wait)   shift; cmd_wait "$@" ;;
  status) shift; report "$(existing_run_dir "${1:-}")" ;;
  stop)   shift; cmd_stop "$@" ;;
  *) die "usage: geak-run.sh start <run-id> <geak-home> <args.json> | wait <run-id> [secs] | status <run-id> | stop <run-id>" ;;
esac
