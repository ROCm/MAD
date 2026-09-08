#!/bin/bash
# Thin shim: execs the shared scripts/common/benchmark_agentic.sh --backend vllm.
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _cand in \
    "${_here}/../common/benchmark_agentic.sh" \
    "${NIXL_COOKBOOK_PATH:-}/../common/benchmark_agentic.sh" \
    "${NIXL_REPO_DIR:-}/../common/benchmark_agentic.sh" "${AGENTIC_LIB:+$(dirname "$AGENTIC_LIB")/benchmark_agentic.sh}"; do
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then exec bash "$_cand" --backend vllm "$@"; fi
done
echo "[agentic][ERROR] shared benchmark_agentic.sh not found (set AGENTIC_LIB)" >&2
exit 1
