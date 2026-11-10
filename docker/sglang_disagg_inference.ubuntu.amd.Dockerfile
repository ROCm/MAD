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
#################################################################################
# The stack follows the GPU architecture madengine (and the CI) pass as
# MAD_SYSTEM_GPU_ARCHITECTURE:
#   gfx942 (MI300X): sglang 0.5.12 + MoRI built at MORI_COMMIT below -- the recipes'
#                    validated stack, unchanged.
#   gfx950 (MI355X): sglang 0.5.20 with its own MoRI v1.2.3. MoRI before v1.2.2 registers
#                    GPU memory with ibv_reg_mr only, which fails ("errno 22") on nodes
#                    without a GPU peer-memory module, e.g. MI355X with AMD AINIC; v1.2.3
#                    falls back to dma-buf. The disagg launchers adapt to 0.5.20's options.
# BASE_DOCKER, when set, overrides the per-architecture base.
ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942
ARG BASE_DOCKER=
FROM lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x AS stack-gfx942
FROM rocm/sgl-dev:v0.5.20-rocm720-mi35x-20260924 AS stack-gfx950
FROM stack-${MAD_SYSTEM_GPU_ARCHITECTURE} AS stack-arch
FROM ${BASE_DOCKER:-stack-arch}
ARG MAD_SYSTEM_GPU_ARCHITECTURE

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

ENV PYTHONPATH=$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:

ARG GPU_ARCH=gfx942
WORKDIR /sgl-workspace

RUN pip install --upgrade sglang-router

# Runtime deps of the disagg launcher and proxy. Baked in so the launcher scripts
# do not have to install them on every node of every run.
RUN pip install py-spy flask pyyaml

WORKDIR /sgl-workspace/mori

ARG MORI_COMMIT="158c7e8335a0b19b3f1f422ff134d7869252135e"
# INSTALL_MORI=1 builds/installs MoRI at MORI_COMMIT, 0 keeps the base image's MoRI.
# auto (default): 1 on gfx942, whose base needs the pinned MoRI; 0 elsewhere, where the
# base brings a MoRI with the dma-buf fallback.
ARG INSTALL_MORI=auto

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git ibverbs-utils libibverbs-dev \
    openmpi-bin libopenmpi-dev \
    libpci-dev libdw1 locales \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    cmake

RUN if [ "${INSTALL_MORI}" = "auto" ]; then \
      if [ "${MAD_SYSTEM_GPU_ARCHITECTURE:-gfx942}" = "gfx942" ]; then INSTALL_MORI=1; else INSTALL_MORI=0; fi; \
    fi; \
    if [ "${INSTALL_MORI}" = "1" ]; then \
      echo "INSTALL_MORI=1: installing MoRI at ${MORI_COMMIT}" \
      && git checkout main \
      && git fetch origin \
      && git pull origin main \
      && git checkout ${MORI_COMMIT} \
      && pip install -r requirements-build.txt \
      && pip install -e . ; \
    else \
      echo "INSTALL_MORI=${INSTALL_MORI}: skipping MoRI installation"; \
    fi


WORKDIR /sgl-workspace

# Display installed packages for verification
RUN pip list
