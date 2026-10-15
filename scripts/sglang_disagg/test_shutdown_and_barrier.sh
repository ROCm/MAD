#!/bin/bash
# Regression tests for the shutdown and readiness-barrier fixes.
#
# The launchers cannot be sourced -- they run top to bottom and need a cluster -- so each test
# extracts the function under test out of the real file by name and sources only that. The code
# under test is therefore the shipped code, not a copy of it.
#
# Six of the nine fail against the files as they were before this work, which is how they were
# checked:
#
#   SGLANG_SH=<old> VLLM_SH=<old> bash test_shutdown_and_barrier.sh   ->  passed 3, failed 6
#
# showing `exit status stays 0: expected '0', got '1'`, the vLLM group `before=3 after=3`, and
# `zombie-only group is gone: expected 'GONE', got 'ALIVE'` -- the three defects behind this.
#
# The zombie test is also the single regression against the intermediate commit, where the
# liveness check still asked `kill -0`: that revision passes 8 of 9 and fails only that one.
#
# Three tests pass either way -- "a real failure is not masked", "stale log ignored", and
# "ignoring zombies did not make us blind to live processes". They are guards against
# over-correcting, not regression tests. Each fix widens something, and a widened check that
# swallows a real failure, resurrects a stale log, or declares a running server dead would be a
# worse bug than the one being fixed.
#
# No GPUs, no SLURM, no container: `bash test_shutdown_and_barrier.sh`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable so the suite can be pointed at the pre-fix files to confirm it actually fails there.
SGLANG_SH="${SGLANG_SH:-${HERE}/sglang_disagg_mori_io_ep.sh}"
VLLM_SH="${VLLM_SH:-${HERE}/../vllm_deepep/run_vllm_deepep.sh}"

_pass=0
_fail=0
_skip=0

ok()   { echo "  ok   - $1"; _pass=$(( _pass + 1 )); }
bad()  { echo "  FAIL - $1" >&2; _fail=$(( _fail + 1 )); }
skip() { echo "  skip - $1"; _skip=$(( _skip + 1 )); }
check() {
    # SKIP is a value, not an absence: a guard that cannot build its fixture prints it on stdout
    # so this can tell "the host reaped the zombie" from "the assertion failed".
    if [[ "$2" == SKIP* ]]; then skip "$1 (${2#SKIP })"
    elif [[ "$2" == "$3" ]]; then ok "$1"
    else bad "$1: expected '$3', got '$2'"; fi
}

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

# lone_zombie -- print the pid of a zombie that is the only member of its own process group.
#
# Monitor mode gives the short-lived child its own group; the parent then execs into something
# that never calls wait(), so nothing reaps it. That is the shape the cluster produces: sglang's
# scheduler children are orphaned when the server exits and the container's init does not reap
# them, leaving the server's group populated entirely by zombies. `kill -0` succeeds on every one.
lone_zombie() {
    local _f; _f="$(mktemp)"
    bash -c "set -m; sleep 0.2 & echo \$! > '${_f}'; exec sleep 25" >/dev/null 2>&1 &
    sleep 1
    cat "$_f"; rm -f "$_f"
}

echo "== a group containing only zombies counts as gone =="
(
    eval "$(extract "$SGLANG_SH" _proc_scan)"
    eval "$(extract "$SGLANG_SH" _group_is_gone)"
    eval "$(extract "$SGLANG_SH" _server_gone)"
    z="$(lone_zombie)"
    state="$(awk '{print $3}' "/proc/${z}/stat" 2>/dev/null)"
    # Guard the fixture itself: if the host reaped it, the assertion below would pass for the
    # wrong reason and this test would silently stop testing anything.
    if [[ "$state" != "Z" ]]; then
        echo "SKIP the host reaped the fixture (state='${state}')"; exit 0
    fi
    kill -0 "$z" 2>/dev/null || { echo "SKIP kill -0 already reports the fixture gone"; exit 0; }
    if _server_gone "$z"; then echo "GONE"; else echo "ALIVE"; fi
) > /tmp/_t_zombie
check "zombie-only group is gone" "$(cat /tmp/_t_zombie)" "GONE"

echo "== a running process in the group still counts as alive =="
(
    eval "$(extract "$SGLANG_SH" _proc_scan)"
    eval "$(extract "$SGLANG_SH" _group_is_gone)"
    eval "$(extract "$SGLANG_SH" _server_gone)"
    set -m; sleep 20 & pid=$!; set +m
    if _server_gone "$pid"; then echo "GONE"; else echo "ALIVE"; fi
    kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
) > /tmp/_t_live
check "ignoring zombies did not make us blind to live processes" "$(cat /tmp/_t_live)" "ALIVE"

echo "== _group_survivors reports the running, counts the zombies =="
(
    eval "$(extract "$SGLANG_SH" _proc_scan)"
    eval "$(extract "$SGLANG_SH" _group_survivors)"
    set -m; sleep 30 & pid=$!; set +m
    out="$(_group_survivors "$pid" 2>&1)"
    kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    # The point of the diagnostic: the pid, its state letter and its wchan, so a reader can tell
    # "parked in the GPU driver" (D) from "alive and ignoring us" (S/R).
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
    # _proc_scan too: without it _server_gone's `while read` sees nothing, concludes the group is
    # gone and the escalation stops after SIGINT. The first version of this test omitted it and
    # reported a shutdown failure that was its own.
    eval "$(extract "$VLLM_SH" _proc_scan)"
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
    # Running members only. A zombie left in the group is not a failure to stop -- it holds
    # nothing -- and counting it is exactly the mistake the liveness fix is about.
    _live_in_group() {
        local _n=0 _s _r _p _a _st _pp _pg
        for _s in /proc/[0-9]*/stat; do
            read -r _r < "$_s" 2>/dev/null || continue
            _p=${_r%% *}; _a=${_r##*) }
            read -r _st _pp _pg _ <<< "$_a"
            [[ "$_pg" == "$1" && "$_st" != "Z" ]] && _n=$(( _n + 1 ))
        done
        echo "$_n"
    }
    _kids_before=$(_live_in_group "$SERVER_PID")
    _shutdown_server >/dev/null 2>&1
    sleep 1
    _kids_after=$(_live_in_group "$SERVER_PID")
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

echo "== a survived shutdown marks the vllm_deepep counter window untrustworthy =="
(
    _tmp="$(mktemp -d)"
    eval "$(extract "$VLLM_SH" _server_gone)"
    eval "$(extract "$VLLM_SH" _wait_gone)"
    eval "$(extract "$VLLM_SH" _shutdown_server)"
    eval "$(extract "$VLLM_SH" _cleanup_on_exit)"
    _shutdown_failed=0
    _rdma_sampler_pid=""
    _rdma_counters_out="${_tmp}/counters.csv"
    : > "${_rdma_counters_out}"
    _stop_rdma_counters() { :; }
    # A server that cannot be stopped: traffic keeps landing on the adapters, so the window the
    # sampler closed is not this run's and the report has to withhold on it.
    SERVER_PID=999999
    _server_gone() { return 1; }
    _wait_gone() { return 1; }
    _shutdown_server() { _shutdown_failed=1; return 0; }
    _cleanup_on_exit >/dev/null 2>&1
    [[ -f "${_rdma_counters_out}.failed" ]] && echo "MARKED" || echo "UNMARKED"
    rm -rf "$_tmp"
) > /tmp/_t_marker
check "counter file marked .failed" "$(cat /tmp/_t_marker)" "MARKED"

echo
echo "passed ${_pass}, failed ${_fail}, skipped ${_skip}"
(( _fail == 0 ))
