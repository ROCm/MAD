#!/bin/bash
# Session 3 lean production image for Kimi-K3 MI300X 2P/2D MoRIIO disagg.
# MoRI-EP only (WITH_NIXL=0) — no UCX/RIXL/rocSHMEM/DeepEP build (~30–45 min saved).
# vLLM connector fixes are baked into VLLM_REF=v3; runtime patchers not required.
#
# Usage (from this directory):
#   GH_TOKEN=$(gh auth token) ./build_lean.sh
#   TAG=myregistry/kimik3-disagg-lean:v1 ./build_lean.sh
#
# See SESSION3_LEAN.md for upstream PR checklist and patcher migration.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-kimik3-wideep-disagg-lean:latest}"
VLLM_REF="${VLLM_REF:-kimi-k3-wideep-disagg-fullsource-v3}"
WITH_NIXL="${WITH_NIXL:-0}"
GH_TOKEN="${GH_TOKEN:-}"

BUILD_ARGS=(
  --build-arg "WITH_NIXL=${WITH_NIXL}"
  --build-arg "VLLM_REF=${VLLM_REF}"
)
[[ -n "$GH_TOKEN" ]] && BUILD_ARGS+=(--build-arg "GH_TOKEN=${GH_TOKEN}")

echo "=== Session 3 lean build ==="
echo "  Dockerfile: Dockerfile.kimik3_disagg"
echo "  TAG:        ${TAG}"
echo "  WITH_NIXL:  ${WITH_NIXL}"
echo "  VLLM_REF:   ${VLLM_REF}"
echo ""

docker build -f "${DIR}/Dockerfile.kimik3_disagg" \
  "${BUILD_ARGS[@]}" \
  -t "${TAG}" \
  "${DIR}"

echo ""
echo "=== done: ${TAG} ==="
echo "Push and set DOCKER_IMAGE_NAME=${TAG} for vllm_dissag / run_2p2d."
