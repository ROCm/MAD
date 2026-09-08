#!/bin/bash
# Shared agentic trace-replay benchmark hook for disaggregated P/D launchers.
#
# Backend-parameterized (--backend sglang|vllm, or AGENTIC_BACKEND) single source
# of truth for what used to be two byte-identical-except-three-axes hooks. Thin
# per-backend shims (scripts/<backend>_disagg/benchmark_agentic.sh) exec this
# script with the right --backend. Drop-in alternative to benchmark_xPyD.sh:
# instead of the random ISL/OSL concurrency sweep, it replays real Claude Code
# agentic traces via aiperf's inferencex-agentx-mvp scenario against the backend
# router/proxy and writes aiperf artifacts + an aggregate JSON + plots. Selected
# by the launcher via
#   export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh   (BENCHMARK_SCRIPT=agentic)
#
# The three backend axes live in the `case "$backend"` below: (1) AGENTIC_PORT
# default, (2) the ctx-window resolver endpoint list, and (3) the lib-locator
# repo-dir candidates (superset, harmless when an env var is unset).
#
# Testable standalone (Phase 1/2) against a running server, and with DRY_RUN=1
# without any server. Env knobs are documented in scripts/common/agentic_lib.sh.
set -uo pipefail

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2;;
        *) break;;
    esac
done
backend="${backend:-${AGENTIC_BACKEND:-}}"
case "$backend" in
    sglang|vllm) : ;;
    "") echo "[agentic][ERROR] --backend or AGENTIC_BACKEND required (sglang|vllm)" >&2; exit 2;;
    *) echo "[agentic][ERROR] unknown backend '$backend'" >&2; exit 2;;
esac

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Locate the shared lib. In-container the launcher dir may be mounted somewhere
# whose sibling common/ does not exist, so search a superset of candidates and
# allow AGENTIC_LIB to override. This file lives in scripts/common/, so
# agentic_lib.sh is a SIBLING (normal case); the repo-dir candidates cover the
# in-container mounts of both sglang (MOONCAKE_REPO_DIR) and vllm
# (NIXL_COOKBOOK_PATH / NIXL_REPO_DIR) launchers.
_agentic_lib=""
for _cand in \
    "${AGENTIC_LIB:-}" \
    "${_here}/agentic_lib.sh" \
    "${_here}/../common/agentic_lib.sh" \
    "${MOONCAKE_REPO_DIR:-}/../common/agentic_lib.sh" \
    "${NIXL_COOKBOOK_PATH:-}/../common/agentic_lib.sh" \
    "${NIXL_REPO_DIR:-}/../common/agentic_lib.sh"; do
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then _agentic_lib="$_cand"; break; fi
done
[ -n "$_agentic_lib" ] || { echo "[agentic][ERROR] agentic_lib.sh not found (set AGENTIC_LIB)" >&2; exit 1; }
# shellcheck source=/dev/null
source "$_agentic_lib"

# Agentic benchmarking is an explicit opt-in path, so permit the pinned uv
# install by default (overridable with AGENTIC_ALLOW_UV_INSTALL=0). Non-agentic
# launcher paths keep the gate off.
: "${AGENTIC_ALLOW_UV_INSTALL:=1}"
export AGENTIC_ALLOW_UV_INSTALL

# Backend axes: serve-port default + ctx-window resolver endpoint list
# ("path|kind"; kind drives the parser below). sglang's disagg front-end
# (sglang_router :2322) exposes the window only on the worker, sometimes only via
# /get_server_info on older builds -> two endpoints; vLLM's worker advertises it
# on /v1/models -> one endpoint.
case "$backend" in
    sglang)
        : "${AGENTIC_PORT:=2322}"                                   # sglang router port
        ctx_endpoints=("/v1/models|models" "/get_server_info|serverinfo")
        ;;
    vllm)
        : "${AGENTIC_PORT:=${BENCHMARK_PORT:-${PROXY_PORT:-8000}}}"  # vLLM router/proxy port (BENCHMARK_PORT at runtime)
        ctx_endpoints=("/v1/models|models")
        ;;
esac
RESULT_DIR="${RESULT_DIR:-/run_logs/${SLURM_JOB_ID:-0}}"
# MODEL_PREFIX feeds the trace-loader default; derive from MODEL_NAME if unset.
: "${MODEL_PREFIX:=${MODEL_NAME:-}}"

# === agentx:BEGIN resolve served context window (disagg) ===
# The disagg front-end (router/proxy/shim) does not advertise max_model_len, so
# resolve it from the prefill WORKER (first host:port in AGENTIC_SERVER_METRICS).
# Skipped when pinned (>0) or DRY_RUN; non-disagg (no AGENTIC_SERVER_METRICS)
# falls through to existing auto-detect.
if [ "${DRY_RUN:-0}" != "1" ] && ! { [ -n "${MAX_MODEL_LEN:-}" ] && [ "${MAX_MODEL_LEN}" -gt 0 ] 2>/dev/null; }; then
    _worker="${AGENTIC_SERVER_METRICS%% *}"
    if [ -z "$_worker" ]; then
        agentic_log "AGENTIC_SERVER_METRICS unset (non-disagg); skipping worker max_model_len auto-detect"
    else
        [[ "$_worker" =~ ^[^[:space:]]+:[0-9]+$ ]] \
            || agentic_die "malformed worker endpoint '$_worker' (expected host:port); pin MAX_MODEL_LEN"
        _tried=""; _mml=""; _won=""
        for _cand in "${ctx_endpoints[@]}"; do
            _path="${_cand%%|*}"; _kind="${_cand##*|}"
            _tried="${_tried:+$_tried, }${_path}"
            for _i in 1 2 3; do
                _mml="$(curl -sf "http://${_worker}${_path}" 2>/dev/null \
                        | python3 -c "$(cat <<'PY'
import sys, json
kind = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit()
v = ""
if kind == "models":
    data = d.get("data") or []
    if data:
        v = data[0].get("max_model_len") or ""
else:
    sa = d.get("server_args") or {}
    v = d.get("max_model_len") or d.get("context_length") \
        or sa.get("max_model_len") or sa.get("context_length") or ""
print(v or "")
PY
)" "$_kind")"
                if [ -n "$_mml" ] && [ "$_mml" != "0" ]; then _won="$_path"; break 2; fi
                sleep 2
            done
        done
        [ -n "$_mml" ] && [ "$_mml" != "0" ] \
            || agentic_die "could not resolve served max_model_len from ${backend} worker ${_worker} (tried ${_tried}); pin MAX_MODEL_LEN"
        export MAX_MODEL_LEN="$_mml"
        agentic_log "resolved MAX_MODEL_LEN=${MAX_MODEL_LEN} from ${backend} worker ${_worker} (${_won})"
    fi
fi
[ "${AGENTIC_RESOLVE_ONLY:-0}" = "1" ] && { echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}"; exit 0; }
# === agentx:END resolve served context window ===

# Suite mode: a workloads config (AGENTIC_CONFIG) or a single-workload shorthand
# (AGENTIC_WORKLOAD) runs the generic multi-workload driver. Without either, the
# legacy single hf/inferencex replay below runs UNCHANGED (byte-identical).
if [ -n "${AGENTIC_CONFIG:-}" ] || [ -n "${AGENTIC_WORKLOAD:-}" ]; then
    _agentic_suite="$(dirname "$_agentic_lib")/benchmark_agentic_suite.sh"
    [ -f "$_agentic_suite" ] || { echo "[agentic][ERROR] suite driver not found: $_agentic_suite" >&2; exit 1; }
    # shellcheck source=/dev/null
    source "$_agentic_suite"
    exit $?
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    agentic_dry_run "$RESULT_DIR"
    exit 0
fi

install_agentic_deps
resolve_trace_source
wait_for_router_ready
[ -n "${MODEL:-}" ] || resolve_served_model_name
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
