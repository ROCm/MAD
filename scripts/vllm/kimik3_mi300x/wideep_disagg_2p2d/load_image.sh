#!/bin/bash
# Self-restoring loader for the K3 disagg image. Ensures TAG is present on this node,
# trying sources in order:
#   1. already present locally -> done
#   2. docker pull from a registry you control (set HUB_IMAGE + DOCKER_USER/DOCKER_PAT)
#   3. docker load from a tar (offline fallback; set TAR)
#   4. build from Dockerfile.kimik3_disagg (last resort; needs base image + gh token)
# Usage: bash load_image.sh   (env: TAG, HUB_IMAGE, DOCKER_USER, DOCKER_PAT, TAR)
#
# The image is built from Dockerfile.kimik3_disagg (see README). Push it to your own
# registry and point HUB_IMAGE at it, or rely on the local build in step 4.
set -euo pipefail

TAG="${TAG:-kimik3-wideep-disagg:latest}"
HUB_IMAGE="${HUB_IMAGE:-}"                         # e.g. <your-registry>/kimik3-wideep-disagg:latest
DOCKER_USER="${DOCKER_USER:-}"
DOCKER_PAT="${DOCKER_PAT:-}"                       # set to enable authed pull
TAR="${TAR:-}"                                     # optional: path to a saved image tar

_have() { docker image inspect "$1" >/dev/null 2>&1; }

# 1. already present (either the local tag or the hub tag)
if _have "$TAG"; then echo "[load] $(hostname -s): $TAG already present"; exit 0; fi
if [ -n "$HUB_IMAGE" ] && _have "$HUB_IMAGE"; then
  docker tag "$HUB_IMAGE" "$TAG" 2>/dev/null || true
  echo "[load] $(hostname -s): $HUB_IMAGE present -> tagged $TAG"; exit 0
fi

# 2. pull from a registry you control (only if HUB_IMAGE is set)
if [ -n "$HUB_IMAGE" ]; then
  echo "[load] $(hostname -s): pulling $HUB_IMAGE ..."
  if [ -n "$DOCKER_PAT" ]; then
    echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin >/dev/null 2>&1 || true
  fi
  if docker pull "$HUB_IMAGE" 2>&1 | tail -1; then
    if _have "$HUB_IMAGE"; then
      docker tag "$HUB_IMAGE" "$TAG" 2>/dev/null || true
      echo "[load] $(hostname -s): pulled + tagged $TAG"; exit 0
    fi
  fi
fi

# 3. offline tar fallback
if [ -f "$TAR" ]; then
  echo "[load] $(hostname -s): pull failed; loading $TAR ..."
  docker load -i "$TAR" 2>&1 | tail -1
  _have "$TAG" && { echo "[load] $(hostname -s): loaded $TAG from tar"; exit 0; }
fi

# 4. build from source (last resort)
echo "[load] $(hostname -s): ERROR: could not obtain $TAG from hub or tar."
echo "       Rebuild: docker build -f Dockerfile.kimik3_disagg \\"
echo "         --build-arg MORI_REF=v1.2.2 --build-arg WITH_NIXL=0 \\"
echo "         --build-arg GH_TOKEN=\$(gh auth token) -t $TAG ."
exit 1
