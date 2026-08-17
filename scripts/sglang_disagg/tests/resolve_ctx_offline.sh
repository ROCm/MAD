#!/bin/bash
# Node-free offline test for the disagg served-context-window resolver in
# benchmark_agentic.sh. Spins up a stdlib http stub in place of the sglang
# prefill worker and exercises 4 paths: SUCCESS, FAIL-FAST, FALLTHROUGH,
# PROPAGATION. No GPU, no network, no dep install (RESOLVE_ONLY exits early).
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="${_here}/../benchmark_agentic.sh"
STUB="${_here}/_stub_server.py"
PORT=$((18000 + RANDOM % 900 + 100))

_pid=""
kill_stub() { [ -n "$_pid" ] && kill "$_pid" 2>/dev/null; _pid=""; }
cleanup() { kill_stub; }
trap cleanup EXIT

# Start the stub ("$@" -> optional "empty"), wait until it accepts connections.
start_stub() {
    kill_stub
    python3 "$STUB" "$PORT" "$@" &
    _pid=$!
    local i
    for i in $(seq 1 50); do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
    echo "stub did not come up on port $PORT" >&2
    return 1
}

_fails=0
report() {  # report <name> <pass:0/1>
    if [ "$2" = "0" ]; then echo "PASS: $1"; else echo "FAIL: $1"; _fails=$((_fails + 1)); fi
}

# (a) SUCCESS: worker advertises the window -> exported and printed.
start_stub || exit 1
out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:${PORT}" MAX_MODEL_LEN= DRY_RUN=0 \
       bash "$HOOK" 2>&1)"
echo "$out" | grep -q "MAX_MODEL_LEN=131072" && report "(a) SUCCESS resolves 131072" 0 || { report "(a) SUCCESS resolves 131072" 1; echo "$out"; }

# (b) FAIL-FAST: worker set but no window field anywhere -> non-zero exit.
start_stub empty || exit 1
AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:${PORT}" MAX_MODEL_LEN= DRY_RUN=0 \
    bash "$HOOK" >/dev/null 2>&1
rc=$?
[ "$rc" -ne 0 ] && report "(b) FAIL-FAST exits non-zero" 0 || report "(b) FAIL-FAST exits non-zero" 1

# (c) FALLTHROUGH: no worker (non-disagg) -> exit 0, empty MAX_MODEL_LEN printed.
out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS= bash "$HOOK" 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && echo "$out" | grep -q "MAX_MODEL_LEN="; then
    report "(c) FALLTHROUGH exit 0 + empty MAX_MODEL_LEN" 0
else
    report "(c) FALLTHROUGH exit 0 + empty MAX_MODEL_LEN" 1; echo "rc=$rc"; echo "$out"
fi

# (d) PROPAGATION: pinned MAX_MODEL_LEN flows through the DRY_RUN suite plan.
out="$(MAX_MODEL_LEN=131072 DRY_RUN=1 AGENTIC_WORKLOAD=small bash "$HOOK" 2>&1)"
if echo "$out" | grep -E "max_model_len" | grep -q "131072"; then
    report "(d) PROPAGATION max_model_len=131072 in suite plan" 0
else
    report "(d) PROPAGATION max_model_len=131072 in suite plan" 1; echo "$out"
fi

[ "$_fails" -eq 0 ] && echo "ALL PASS" || echo "$_fails assertion(s) FAILED"
exit "$_fails"
