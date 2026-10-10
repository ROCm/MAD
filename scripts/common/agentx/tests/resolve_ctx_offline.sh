#!/bin/bash
# Backend-agnostic OFFLINE test for the disagg served-context-window resolver in
# the per-backend benchmark_agentic.sh hooks. Spins up a stdlib http stub in
# place of the prefill worker and, for EACH backend (sglang, vllm), exercises 4
# paths: SUCCESS, FAIL-FAST, FALLTHROUGH, PROPAGATION. No GPU, no network, no
# dep install (RESOLVE_ONLY exits early). The shared stub is a superset that
# serves both /v1/models (max_model_len) and /get_server_info
# (server_args.context_length); each backend's resolver queries whichever it
# needs. Prints PASS/FAIL per assertion; exits non-zero if any fail.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tests -> agentx -> common -> scripts -> repo root
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
STUB="$HERE/_stub_server.py"

if [ ! -f "$REPO_ROOT/scripts/sglang_disagg/benchmark_agentic.sh" ]; then
    echo "could not locate repo root (missing scripts/sglang_disagg/benchmark_agentic.sh under $REPO_ROOT)" >&2
    exit 1
fi

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

run_backend() {  # $1=backend name  $2=backend dir under scripts/
    local bk="$1"
    local dir="$2"
    local HOOK="$REPO_ROOT/scripts/${dir}/benchmark_agentic.sh"
    local before=$fails

    if [ ! -f "$HOOK" ]; then
        report "[$bk] hook exists" 0
        return
    fi

    # (a) SUCCESS: resolver exports MAX_MODEL_LEN from the worker.
    local PORT out rc
    PORT="$(pick_port)"
    if start_stub "$PORT"; then
        out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:$PORT" MAX_MODEL_LEN= DRY_RUN=0 \
            bash "$HOOK" 2>&1)"
        echo "$out" | grep -q "MAX_MODEL_LEN=131072" && report "[$bk] (a) SUCCESS resolves 131072" 1 || { echo "$out"; report "[$bk] (a) SUCCESS resolves 131072" 0; }
    else
        report "[$bk] (a) SUCCESS resolves 131072" 0
    fi
    stop_stub

    # (b) FAIL-FAST: worker set but no window field anywhere -> non-zero exit.
    PORT="$(pick_port)"
    if start_stub "$PORT" empty; then
        AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS="127.0.0.1:$PORT" MAX_MODEL_LEN= DRY_RUN=0 \
            bash "$HOOK" >/dev/null 2>&1
        rc=$?
        [ "$rc" -ne 0 ] && report "[$bk] (b) FAIL-FAST exits non-zero" 1 || report "[$bk] (b) FAIL-FAST exits non-zero" 0
    else
        report "[$bk] (b) FAIL-FAST exits non-zero" 0
    fi
    stop_stub

    # (c) FALLTHROUGH: no worker (non-disagg) -> exit 0, empty MAX_MODEL_LEN.
    out="$(AGENTIC_RESOLVE_ONLY=1 AGENTIC_SERVER_METRICS= bash "$HOOK" 2>&1)"
    rc=$?
    if [ "$rc" -eq 0 ] && echo "$out" | grep -q "MAX_MODEL_LEN="; then
        report "[$bk] (c) FALLTHROUGH exit 0 + empty MAX_MODEL_LEN" 1
    else
        echo "rc=$rc"; echo "$out"; report "[$bk] (c) FALLTHROUGH exit 0 + empty MAX_MODEL_LEN" 0
    fi

    # (d) PROPAGATION: pinned MAX_MODEL_LEN flows through the DRY_RUN suite plan.
    out="$(MAX_MODEL_LEN=131072 DRY_RUN=1 AGENTIC_WORKLOAD=small bash "$HOOK" 2>&1)"
    if echo "$out" | grep -E "max_model_len" | grep -q "131072"; then
        report "[$bk] (d) PROPAGATION max_model_len=131072 in suite plan" 1
    else
        echo "$out"; report "[$bk] (d) PROPAGATION max_model_len=131072 in suite plan" 0
    fi

    local bk_fails=$((fails - before))
    echo "---- [$bk] summary: $([ "$bk_fails" -eq 0 ] && echo "ALL PASS" || echo "$bk_fails FAILED")"
}

# backend name -> its directory under scripts/ (note the vllm dir spelling).
for pair in "sglang:sglang_disagg" "vllm:vllm_dissag"; do
    bk="${pair%%:*}"; dir="${pair##*:}"
    echo "=== backend: $bk ==="
    run_backend "$bk" "$dir"
    echo ""
done

echo "======================================================"
if [ "$fails" -eq 0 ]; then
    echo "resolve_ctx_offline: ALL PASS"
    exit 0
else
    echo "resolve_ctx_offline: $fails assertion(s) FAILED"
    exit 1
fi
