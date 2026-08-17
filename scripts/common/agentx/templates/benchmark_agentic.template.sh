#!/bin/bash
# Agentic trace-replay benchmark hook TEMPLATE for <YOUR BACKEND>.
# CHANGE: replace <YOUR BACKEND> above with your backend/launcher name.
#
# Copy this file into your launcher's script dir as benchmark_agentic.sh, fill
# in the three `# CHANGE:` fields below, and wire it in (the launcher selects it
# via `export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh`, or an `AGENTIC=1`
# shorthand where supported). It replays real agentic traces via aiperf's
# inferencex-agentx-mvp scenario against an OpenAI-compatible endpoint on
# AGENTIC_PORT and writes aiperf artifacts + an aggregate JSON + plots.
#
# Testable standalone against a running server, and with DRY_RUN=1 without any
# server. Env knobs are documented in scripts/common/agentic_lib.sh and the core
# README (scripts/common/agentx/README.md).
#
# SINGLE-NODE / non-SLURM use: this hook does NOT start a server. Point
# AGENTIC_PORT at your already-running serve port and set RESULT_DIR to a real
# directory (the /run_logs/${SLURM_JOB_ID} default assumes the launcher). No
# sbatch is needed; test the wiring with DRY_RUN=1 first, e.g.:
#   DRY_RUN=1 AGENTIC_WORKLOAD=conformance_256k RESULT_DIR=/tmp/agentic bash benchmark_agentic.sh
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Locate the shared lib. In-container the launcher dir may be mounted somewhere
# whose sibling common/ does not exist, so search a few candidates and allow
# AGENTIC_LIB to override.
_agentic_lib=""
for _cand in \
    "${AGENTIC_LIB:-}" \
    "${_here}/../common/agentic_lib.sh" \
    "${YOUR_REPO_DIR:-}/../common/agentic_lib.sh" \
    "${_here}/agentic_lib.sh"; do
    # CHANGE (optional): replace the "${YOUR_REPO_DIR:-}/../common/agentic_lib.sh"
    # candidate above with your launcher's repo-dir env var (e.g. the host repo
    # path visible via a bind mount) if your launcher mounts the repo elsewhere;
    # leave it as-is (harmless — expands to /../common/... and is skipped) if not.
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then _agentic_lib="$_cand"; break; fi
done
[ -n "$_agentic_lib" ] || { echo "[agentic][ERROR] agentic_lib.sh not found (set AGENTIC_LIB)" >&2; exit 1; }
# shellcheck source=/dev/null
source "$_agentic_lib"

: "${AGENTIC_PORT:=8000}"                                   # CHANGE: your router/proxy serve port
: "${AGENTIC_CONC:=16}"
: "${DURATION:=120}"
RESULT_DIR="${RESULT_DIR:-/run_logs/${SLURM_JOB_ID:-0}}"    # CHANGE for non-SLURM: set RESULT_DIR to a real dir
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
