# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
###############################################################################
#
# vLLM ROCm nightly + LMCache built from source.
#
# Base: vllm/vllm-openai-rocm:nightly
#   - Use --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:<tag> to pin a specific release.
#   - Do NOT use rocm/vllm-dev as base: that image requires compiling vLLM from source
#     and produces an older vLLM version incompatible with current mori/aiter kernels.
#
# Architecture: defaults to both MI300X/MI325X (gfx942) and MI350X/MI355X (gfx950).
#   Override with --build-arg PYTORCH_ROCM_ARCH=gfx942 to target a single arch.
#
# LMCache: defaults to v0.5.5 (latest stable). Override with LMCACHE_VERSION=v0.5.3 etc.
#   Installed in editable mode at /opt/LMCache so launch-time patches work without rebuild.
#
# Example build:
#   docker build \
#     -f docker/pyt_vllm_lmcache.ubuntu.amd.Dockerfile \
#     --pull \
#     -t ghcr.io/<org>/vllm-rocm-lmcache:nightly-lm0.5.5-gfx942-gfx950 \
#     .
#
ARG BASE_IMAGE=vllm/vllm-openai-rocm:nightly
FROM ${BASE_IMAGE}

ARG LMCACHE_VERSION=v0.5.5
# Space-separated list: gfx942=MI300X/MI325X, gfx950=MI350X/MI355X
ARG PYTORCH_ROCM_ARCH="gfx942 gfx950"

RUN git clone https://github.com/LMCache/LMCache.git /opt/LMCache && \
    cd /opt/LMCache && \
    git checkout ${LMCACHE_VERSION} && \
    pip install -r requirements/build.txt && \
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" TORCH_DONT_CHECK_COMPILER_ABI=1 \
      CXX=hipcc BUILD_WITH_HIP=1 \
      pip install -e . --no-build-isolation && \
    python -c "import lmcache; print('lmcache', lmcache.__version__)"

# Pre-build aiter HIP extension modules (.so files compiled via hipcc, no GPU needed).
RUN VLLM_CACHE_ROOT=/opt/vllm_cache python3 -c \
    "import aiter; print('aiter pre-built OK')" 2>/dev/null || true

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
