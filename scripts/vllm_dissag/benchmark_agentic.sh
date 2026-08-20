#!/bin/bash
# Agentic trace-replay benchmark hook for the vLLM disaggregated P/D launcher.
#
# Drop-in alternative to benchmark_xPyD.sh: instead of the random ISL/OSL
# concurrency sweep, it replays real Claude Code agentic traces via aiperf's
# inferencex-agentx-mvp scenario against the vLLM router/proxy (BENCHMARK_PORT)
# and writes aiperf artifacts + an aggregate JSON + plots. Selected by the
# launcher via
#   export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh   (BENCHMARK_SCRIPT=agentic)
#
# Testable standalone (Phase 1/2) against a running server, and with DRY_RUN=1
# without any server. Env knobs are documented in scripts/common/agentic_lib.sh.
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Locate the shared lib. In-container only scripts/vllm_dissag is mounted at
# NIXL_COOKBOOK_PATH, so "../common" relative to that mount does not exist;
# NIXL_REPO_DIR (the host repo path, visible via the $HOME bind mount) points at
# scripts/vllm_dissag whose sibling common/ does exist. Search a few candidates
# and allow AGENTIC_LIB to override.
_agentic_lib=""
for _cand in \
    "${AGENTIC_LIB:-}" \
    "${_here}/../common/agentic_lib.sh" \
    "${NIXL_COOKBOOK_PATH:-}/../common/agentic_lib.sh" \
    "${NIXL_REPO_DIR:-}/../common/agentic_lib.sh" \
    "${_here}/agentic_lib.sh"; do
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

: "${AGENTIC_PORT:=${BENCHMARK_PORT:-${PROXY_PORT:-8000}}}"  # vLLM router/proxy port (BENCHMARK_PORT at runtime)
RESULT_DIR="${RESULT_DIR:-/run_logs/${SLURM_JOB_ID:-0}}"
# MODEL_PREFIX feeds the trace-loader default; derive from MODEL_NAME if unset.
: "${MODEL_PREFIX:=${MODEL_NAME:-}}"

# === agentx:BEGIN resolve served context window (disagg) ===
# The disagg front-end (agentic_models_shim.py) does not advertise max_model_len,
# so resolve it from the real prefill WORKER (first host:port in
# AGENTIC_SERVER_METRICS, a real `vllm serve` OpenAI server whose /v1/models
# ModelCard includes max_model_len). Skipped when pinned (>0) or DRY_RUN;
# non-disagg (no AGENTIC_SERVER_METRICS) falls through to existing auto-detect.
if [ "${DRY_RUN:-0}" != "1" ] && ! { [ -n "${MAX_MODEL_LEN:-}" ] && [ "${MAX_MODEL_LEN}" -gt 0 ] 2>/dev/null; }; then
    _worker="${AGENTIC_SERVER_METRICS%% *}"
    if [ -z "$_worker" ]; then
        agentic_log "AGENTIC_SERVER_METRICS unset (non-disagg); skipping worker max_model_len auto-detect"
    else
        [[ "$_worker" =~ ^[^[:space:]]+:[0-9]+$ ]] \
            || agentic_die "malformed worker endpoint '$_worker' (expected host:port); pin MAX_MODEL_LEN"
        _mml=""
        for _i in 1 2 3; do
            _mml="$(curl -sf "http://${_worker}/v1/models" 2>/dev/null \
                    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print(""); sys.exit()
data=d.get("data") or []
print((data[0].get("max_model_len") if data else "") or "")' 2>/dev/null)"
            if [ -n "$_mml" ] && [ "$_mml" != "0" ]; then break; fi
            sleep 2
        done
        [ -n "$_mml" ] && [ "$_mml" != "0" ] \
            || agentic_die "could not resolve served max_model_len from vLLM worker ${_worker} (/v1/models); pin MAX_MODEL_LEN"
        export MAX_MODEL_LEN="$_mml"
        agentic_log "resolved MAX_MODEL_LEN=${MAX_MODEL_LEN} from vLLM worker ${_worker} (/v1/models)"
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
