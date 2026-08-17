#!/bin/bash
# Offline (node-free, no GPU) test for the disagg served-context-window resolver
# in scripts/vllm_dissag/benchmark_agentic.sh. Starts a stdlib stub of a vLLM
# `vllm serve` OpenAI server (/v1/models) and drives 4 assertions via
# AGENTIC_RESOLVE_ONLY / DRY_RUN. Prints PASS/FAIL per assertion; exits non-zero
# if any fail.
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$_here/../benchmark_agentic.sh"
STUB="$_here/_stub_server.py"

fails=0
STUB_PID=""

cleanup() { [ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; }
trap cleanup EXIT

pick_port() {
    python3 -c 'import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

start_stub() {  # $1=port  $2=optional "empty"
    STUB_PID=""
    python3 "$STUB" "$1" ${2:-} &
    STUB_PID=$!
    for _ in $(seq 1 50); do
        if curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
    echo "  stub failed to accept on port $1" >&2
    return 1
}

stop_stub() { [ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; wait "$STUB_PID" 2>/dev/null; STUB_PID=""; }

report() {  # $1=name  $2=0/1 pass
    if [ "$2" = "1" ]; then echo "PASS: $1"; else echo "FAIL: $1"; fails=$((fails+1)); fi
}

# (a) SUCCESS: resolver exports MAX_MODEL_LEN from the worker /v1/models.
PORT="$(pick_port)"
if start_stub "$PORT"; then
    out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:$PORT" MAX_MODEL_LEN= DRY_RUN=0 \
        bash "$HOOK" 2>&1)"
    echo "$out" | grep -q "MAX_MODEL_LEN=131072" && report "(a) SUCCESS resolve" 1 || { echo "$out"; report "(a) SUCCESS resolve" 0; }
else
    report "(a) SUCCESS resolve" 0
fi
stop_stub

# (b) FAIL-FAST: worker set but /v1/models omits max_model_len -> non-zero exit.
PORT="$(pick_port)"
if start_stub "$PORT" empty; then
    AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:$PORT" MAX_MODEL_LEN= DRY_RUN=0 \
        bash "$HOOK" >/dev/null 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && report "(b) FAIL-FAST unresolvable" 1 || report "(b) FAIL-FAST unresolvable" 0
else
    report "(b) FAIL-FAST unresolvable" 0
fi
stop_stub

# (c) FALLTHROUGH: no worker (non-disagg) -> exit 0, empty MAX_MODEL_LEN, no die.
out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS= bash "$HOOK" 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && echo "$out" | grep -q "MAX_MODEL_LEN="; then
    report "(c) FALLTHROUGH non-disagg" 1
else
    echo "$out"; report "(c) FALLTHROUGH non-disagg" 0
fi

# (d) PROPAGATION: pinned MAX_MODEL_LEN reaches the DRY_RUN suite plan.
out="$(MAX_MODEL_LEN=131072 DRY_RUN=1 AGENTIC_WORKLOAD=small bash "$HOOK" 2>&1)"
if echo "$out" | grep "max_model_len" | grep -q "131072"; then
    report "(d) PROPAGATION dry-run" 1
else
    echo "$out"; report "(d) PROPAGATION dry-run" 0
fi

echo "----"
if [ "$fails" -eq 0 ]; then echo "ALL PASS"; exit 0; else echo "$fails FAILED"; exit 1; fi
