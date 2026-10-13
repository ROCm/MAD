#!/bin/bash
# Regression tests for the shutdown and readiness-barrier fixes.
#
# The launchers cannot be sourced -- they run top to bottom and need a cluster -- so each test
# extracts the function under test out of the real file by name and sources only that. The code
# under test is therefore the shipped code, not a copy of it.
#
# Four of the six fail against the pre-fix files, which is how they were checked:
#
#   SGLANG_SH=<old> VLLM_SH=<old> bash test_shutdown_and_barrier.sh   ->  passed 2, failed 4
#
# with the pre-fix run showing `exit status stays 0: expected '0', got '1'` and the vLLM group
# `before=3 after=3`, i.e. the two defects this change is about.
#
# The other two -- "a real failure is not masked" and "stale log ignored" -- are guards against
# over-correcting, not regression tests, and pass either way. They are here because both fixes
# widen something, and a widened check that swallows a real failure or resurrects a stale log
# would be a worse bug than the one being fixed.
#
# No GPUs, no SLURM, no container: `bash test_shutdown_and_barrier.sh`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable so the suite can be pointed at the pre-fix files to confirm it actually fails there.
SGLANG_SH="${SGLANG_SH:-${HERE}/sglang_disagg_mori_io_ep.sh}"
VLLM_SH="${VLLM_SH:-${HERE}/../vllm_deepep/run_vllm_deepep.sh}"

_pass=0
_fail=0

ok()   { echo "  ok   - $1"; _pass=$(( _pass + 1 )); }
bad()  { echo "  FAIL - $1" >&2; _fail=$(( _fail + 1 )); }
check() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1: expected '$3', got '$2'"; fi; }

# extract FILE NAME -- the text of one shell function, dedented to column 0 so a function defined
# inside a block can still be sourced on its own.
extract() {
    awk -v name="$2" '
        $0 ~ "^[[:space:]]*" name "\\(\\)[[:space:]]*\\{" { depth = 1; indent = match($0, /[^ ]/) - 1
                                                            print substr($0, indent + 1); next }
        depth > 0 { line = substr($0, indent + 1)
                    if (line == "") line = $0
                    print line
                    depth += gsub(/\{/, "{"); depth -= gsub(/\}/, "}")
                    if (depth <= 0) exit }
    ' "$1"
}

echo "== _group_survivors names what is still in the group =="
(
    eval "$(extract "$SGLANG_SH" _group_survivors)"
    set -m; sleep 30 & pid=$!; set +m
    out="$(_group_survivors "$pid" 2>&1)"
    kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    # The whole point of the diagnostic: the pid, its state letter and its wchan, so a reader can
    # tell "parked in the GPU driver" (D) from "alive and ignoring us" (S/R).
    case "$out" in
        *"pid=${pid}"*state=*wchan=*) echo "  ok   - reports pid, state and wchan" ;;
        *) echo "  FAIL - diagnostic missing fields: ${out}" >&2; exit 1 ;;
    esac
) && _pass=$(( _pass + 1 )) || _fail=$(( _fail + 1 ))

echo "== a surviving server does not change the exit status =="
(
    # The pre-fix _cleanup_on_exit ran `exit 1` whenever _shutdown_failures was non-zero, which
    # madengine records as Failed -- and a Failed run has no perf CSV collected.
    cat > /tmp/_t_exit.sh <<'EOS'
_shutdown_failures=1
_running_servers=()
NODE_RANK=0
_shutdown_server() { :; }
_stop_rdma_counters() { :; }
EOS
    extract "$SGLANG_SH" _cleanup_on_exit >> /tmp/_t_exit.sh
    echo 'trap _cleanup_on_exit EXIT; exit 0' >> /tmp/_t_exit.sh
    bash /tmp/_t_exit.sh >/dev/null 2>&1
    echo "$?"
) > /tmp/_t_exit_rc
check "exit status stays 0" "$(cat /tmp/_t_exit_rc)" "0"

echo "== a non-zero status is still passed through =="
(
    cat > /tmp/_t_exit2.sh <<'EOS'
_shutdown_failures=1
_running_servers=()
NODE_RANK=0
_shutdown_server() { :; }
_stop_rdma_counters() { :; }
EOS
    extract "$SGLANG_SH" _cleanup_on_exit >> /tmp/_t_exit2.sh
    echo 'trap _cleanup_on_exit EXIT; exit 3' >> /tmp/_t_exit2.sh
    bash /tmp/_t_exit2.sh >/dev/null 2>&1
    echo "$?"
) > /tmp/_t_exit2_rc
check "a real failure is not masked" "$(cat /tmp/_t_exit2_rc)" "3"

echo "== the barrier notices a death in a log it is not waiting on =="
(
    _tmp="$(mktemp -d)"
    _ATTEMPT_TOKEN="attempt=1.0"
    printf '%s\nstarting\n' "$_ATTEMPT_TOKEN"                               > "${_tmp}/prefill.log"
    printf '%s\nRank 0 scheduler died during initialization\n' "$_ATTEMPT_TOKEN" > "${_tmp}/decode.log"
    eval "$(extract "$SGLANG_SH" _log_is_this_attempt)"
    eval "$(extract "$SGLANG_SH" _server_died)"
    _SERVER_DEATH_SIGNS="scheduler died during initialization|torch.OutOfMemoryError"
    eval "$(extract "$SGLANG_SH" _any_watched_died)"
    # Role order: prefill first. Pre-fix, the barrier polled only the log it was blocked on, so a
    # decode that died during prefill bring-up went unnoticed for about an hour.
    _watched_logs=("master prefill|${_tmp}/prefill.log" "master decode|${_tmp}/decode.log")
    if _any_watched_died; then echo "${_died_label}"; else echo "NOTHING"; fi
    rm -rf "$_tmp"
) > /tmp/_t_barrier
check "names the decode while waiting on prefill" "$(cat /tmp/_t_barrier)" "master decode"

echo "== a stale log from the previous attempt is still ignored =="
(
    _tmp="$(mktemp -d)"
    _ATTEMPT_TOKEN="attempt=2.0"
    printf 'attempt=1.0\nscheduler died during initialization\n' > "${_tmp}/decode.log"
    printf '%s\nstarting\n' "$_ATTEMPT_TOKEN"                    > "${_tmp}/prefill.log"
    eval "$(extract "$SGLANG_SH" _log_is_this_attempt)"
    eval "$(extract "$SGLANG_SH" _server_died)"
    _SERVER_DEATH_SIGNS="scheduler died during initialization"
    eval "$(extract "$SGLANG_SH" _any_watched_died)"
    _watched_logs=("master prefill|${_tmp}/prefill.log" "master decode|${_tmp}/decode.log")
    if _any_watched_died; then echo "${_died_label}"; else echo "NOTHING"; fi
    rm -rf "$_tmp"
) > /tmp/_t_barrier_stale
check "widening the check did not resurrect stale logs" "$(cat /tmp/_t_barrier_stale)" "NOTHING"

echo "== the vllm_deepep shutdown reaches the whole process group =="
(
    eval "$(extract "$VLLM_SH" _server_gone)"
    eval "$(extract "$VLLM_SH" _wait_gone)"
    eval "$(extract "$VLLM_SH" _shutdown_server)"
    SHUTDOWN_GRACE_S=3
    # A parent with a child, both in one group: the pre-fix `kill ${SERVER_PID}` signalled only
    # the parent, which is how eight engine subprocesses were left holding GPUs.
    set -m
    bash -c 'sleep 120 & sleep 120' &
    SERVER_PID=$!
    set +m
    sleep 1
    _kids_before=$(pgrep -g "$SERVER_PID" 2>/dev/null | wc -l)
    _shutdown_server >/dev/null 2>&1
    sleep 1
    _kids_after=$(pgrep -g "$SERVER_PID" 2>/dev/null | wc -l)
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    echo "${_kids_before} ${_kids_after}"
) > /tmp/_t_group
_before="$(cut -d' ' -f1 /tmp/_t_group)"
_after="$(cut -d' ' -f2 /tmp/_t_group)"
if (( _before >= 2 )) && (( _after == 0 )); then
    ok "group of ${_before} fully stopped"
else
    bad "group not fully stopped: before=${_before} after=${_after}"
fi

echo
echo "passed ${_pass}, failed ${_fail}"
(( _fail == 0 ))
