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
ARG BASE_DOCKER=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
FROM $BASE_DOCKER

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

ENV PYTHONPATH=$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:

ARG GPU_ARCH=gfx942
WORKDIR /sgl-workspace

RUN pip install --upgrade sglang-router

WORKDIR /sgl-workspace/mori

# MoRI >= #363 (guard dispatch kernels vs out-of-range expert id) and #505 (AsyncLL slot
# double-alloc when top-k does not divide warpSize) are REQUIRED for DeepSeek-V4-Flash decode
# CUDA-graph capture (topk6 -> 6 does not divide warpSize 64). The older 158c7e83 pin (2026-06-08)
# predates both and crashes at capture (mori low_latency_async.cpp:360 pe-out-of-range).
ARG MORI_COMMIT="7c51d18fda59457cc9238ed262bd93c8cad906c9"
# Set INSTALL_MORI=1 to build/install MoRI at MORI_COMMIT; any other value skips it.
ARG INSTALL_MORI=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git ibverbs-utils libibverbs-dev \
    openmpi-bin libopenmpi-dev \
    libpci-dev libdw1 locales \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    cmake

RUN if [ "${INSTALL_MORI}" = "1" ]; then \
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
