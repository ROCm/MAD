# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# =============================================================================
# vllm_disagg_inference.musespark.ubuntu.amd.Dockerfile
#   Muse-Spark-Open-816B-A42B-Quark-MXFP4 disagg image for MI355X (gfx950).
#   PER-MODEL image, isolated from the base vllm_disagg_inference Dockerfile.
#
#   docker build -f docker/vllm_disagg_inference.musespark.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:musespark .
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:musespark
#
# The vLLM release image already carries everything the moriio+TP recipe needs:
# the ROCm Muse Spark model (vllm/models/muse_spark/amd, first released in v0.33.0),
# the MoRIIO connector with hybrid KV cache support, MoRI and AITER. This file only
# adds the MoRIIO-aware vllm-router. Do not use an older base: Muse Spark is not
# registered there.
# =============================================================================

ARG BASE_DOCKER=vllm/vllm-openai-rocm:v0.34.0
FROM $BASE_DOCKER

USER root
ENTRYPOINT []
WORKDIR /app

# Checked by file, not import: importing vllm or mori needs a GPU runtime.
RUN python3 - <<'PYEOF'
import importlib.util as u, os
from importlib.metadata import version
root = u.find_spec("vllm").submodule_search_locations[0]
for rel in ("models/muse_spark/amd/model.py",
            "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"):
    assert os.path.isfile(os.path.join(root, rel)), f"vllm/{rel} missing from the base image"
assert u.find_spec("mori"), "mori missing from the base image"
print("vLLM", version("vllm"), "has Muse Spark (ROCm) + MoRIIO + MoRI")
PYEOF
RUN mkdir -p /app && python3 -c "from importlib.metadata import version; print('VLLM=' + version('vllm'))" >> /app/versions.txt

# -----------------------------------------------------------------------------
# vllm-router with the MoRIIO connector, same pin as the GLM-5.1 image. Built in,
# so no external vllm-router binary is needed (leave ROUTER_BINARY unset).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=82dc9811af17412e6e24b5942a5486bc502df23a
ARG RUST_TOOLCHAIN=1.88.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    cargo build --release && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_REF=${ROUTER_REPO}@${ROUTER_REF}" >> /app/versions.txt && \
    rm -rf /tmp/vllm-router-src

# Mount target for the launcher's persistent host JIT cache. Runtime recipe and
# platform env come from scripts/vllm_dissag/models.yaml and connectors/moriio.env.
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

# Stale MoRI JIT locks from the build host deadlock MoriAll2AllManager at runtime.
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori

RUN cat /app/versions.txt
