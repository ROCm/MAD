#!/bin/bash
# Agentic trace-replay benchmark hook for the vLLM disaggregated P/D launcher.
#
# Drop-in alternative to benchmark_xPyD.sh: instead of the random ISL/OSL
# concurrency sweep, it replays real Claude Code agentic traces via aiperf's
# inferencex-agentx-mvp scenario against the vLLM router/proxy (BENCHMARK_PORT)
# and writes aiperf artifacts + an aggregate JSON + plots. Selected by the
# launcher via
#   export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh   (AGENTIC=1 shorthand)
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

: "${AGENTIC_PORT:=${BENCHMARK_PORT:-${PROXY_PORT:-8000}}}"  # vLLM router/proxy port (BENCHMARK_PORT at runtime)
: "${AGENTIC_CONC:=16}"
: "${DURATION:=120}"
RESULT_DIR="${RESULT_DIR:-/run_logs/${SLURM_JOB_ID:-0}}"
# MODEL_PREFIX feeds the trace-loader default; derive from MODEL_NAME if unset.
: "${MODEL_PREFIX:=${MODEL_NAME:-}}"

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
