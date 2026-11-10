#!/bin/bash
# Build MoRI from source INSIDE the mori_host container (run via: docker exec mori_host bash build_mori.sh).
# Only spdlog+msgpack-c submodules (recursive clone stalls on the spdk submodule / HTTP2).
#
# PINNED MoRI commit (the CI-green commit these validation results were produced on):
#   ROCm/mori  12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4  (2026-07-31, version 0.1.1.dev1+g12d1bc32d)
#     = "Fix(ep): AsyncLL slot assignment double-allocates when top-k does not divide warpSize (#505)"
#   submodule 3rdparty/msgpack-c  9b801f087ab7434f2ab1ab3c0f48a966c19d3b70
#   submodule 3rdparty/spdlog     4a9ccf7e38e257feecce0c579a782741254eaeef
# Override with MORI_COMMIT=<sha|main> to build a different revision.
set -euo pipefail
export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
export MORI_GPU_ARCHS=gfx942
MORI_COMMIT="${MORI_COMMIT:-12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4}"

cd /tmp
rm -rf mori-src
# Fetch just the pinned commit (falls back to full clone + checkout if the server rejects the fetch-by-sha).
mkdir mori-src && cd mori-src && git init -q
git remote add origin https://github.com/ROCm/mori.git
if ! git fetch -q --depth 1 origin "$MORI_COMMIT" 2>/dev/null; then
  cd /tmp && rm -rf mori-src && git clone -q https://github.com/ROCm/mori.git mori-src && cd mori-src
fi
git checkout -q "$MORI_COMMIT"
echo "MoRI pinned at: $(git log -1 --format='%H %ci %s')"
git submodule update --init --depth 1 3rdparty/spdlog 3rdparty/msgpack-c

pip install meson==0.64.0 "pybind11[global]" tqdm prettytable
pip uninstall -y amd_mori amd-mori mori 2>/dev/null || true
BUILD_UMBP=OFF pip install .

python3 -c "import mori; print('MoRI OK', getattr(mori,'__version__','n/a'))"
echo "BUILD DONE"
