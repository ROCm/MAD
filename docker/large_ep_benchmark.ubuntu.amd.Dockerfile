# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
ARG BASE_DOCKER=lmsysorg/sglang:v0.5.20-rocm10-mi30x
FROM $BASE_DOCKER
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list
ENV PYTHONPATH=/sgl-workspace/mori:/sgl-workspace/aiter
WORKDIR /sgl-workspace

# ========== MORI ==========
WORKDIR /sgl-workspace/mori
# v1.2.3.post1
ARG MORI_COMMIT="67632e80e2e492184b589904b63225f82d45537c"
# Set INSTALL_MORI=1 to build/install MoRI at MORI_COMMIT; any other value skips it.
ARG INSTALL_MORI=1
RUN apt-get update && apt-get install -y --no-install-recommends git ibverbs-utils libibverbs-dev openmpi-bin libopenmpi-dev libpci-dev libdw1 locales libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc cmake libdrm-dev
RUN if [ "${INSTALL_MORI}" = "1" ]; then echo "INSTALL_MORI=1: installing MoRI at ${MORI_COMMIT}" && git checkout main && git fetch origin && git pull origin main && git checkout ${MORI_COMMIT} && pip install -r requirements-build.txt Cython && MORI_GPU_ARCHS="${GPU_ARCH_LIST}" pip install --no-build-isolation -e . ; else echo "INSTALL_MORI=${INSTALL_MORI}: skipping MoRI installation"; fi
ENV MORI_ROOT=/sgl-workspace/mori
# The MoRI install can leave a half-built JIT cache whose lock files hang every later run; kernels recompile at first use.
RUN rm -rf /root/.mori /tmp/mori_jit_*

# ========== BENCHMARK DEPENDENCIES ==========
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && pip install tqdm prettytable

# ========== ROCSHMEM + DEEPEP (deprecated) ==========
WORKDIR /app
ARG ROCM_SYSTEMS_REPO="https://github.com/ROCm/rocm-systems.git"
# develop as of 2026-10-01 (rocSHMEM has no standalone releases)
ARG ROCSHMEM_COMMIT="62324a8858fe285313fa7f6d6bc99caeed5d611e"
ARG DEEPEP_REPO="https://github.com/ROCm/DeepEP.git"
# main as of 2026-10-01 (DeepEP has no releases)
ARG DEEPEP_COMMIT="0f63d3e97f0e43b47926179d7f3f304b3bc9554e"
ARG NIC_COMPILATION_ARCH="cx7"
# DeepEP is deprecated in this benchmark. Set INSTALL_DEEPEP=1 to build rocSHMEM + DeepEP at the commits above; any other value skips them.
ARG INSTALL_DEEPEP=0
RUN if [ "${INSTALL_DEEPEP}" = "1" ]; then echo "INSTALL_DEEPEP=1: installing rocSHMEM at ${ROCSHMEM_COMMIT} and DeepEP at ${DEEPEP_COMMIT}" && \
        apt-get update && apt-get install -y \
            autoconf pkg-config \
            libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev \
            libibverbs-dev ibverbs-utils libtool libboost-all-dev \
            libgrpc++-dev protobuf-compiler-grpc protobuf-compiler libprotobuf-dev \
            libaio-dev liburing-dev pybind11-dev ninja-build libgflags-dev \
            rdma-core infiniband-diags perftest openssh-server \
            psmisc vim cmake-curses-gui && \
        pip install "pybind11[global]" meson==0.64.0 && \
        git clone --no-checkout --filter=blob:none ${ROCM_SYSTEMS_REPO} /app/rocm-systems && cd /app/rocm-systems && \
        git sparse-checkout set --cone projects/rocshmem && git checkout ${ROCSHMEM_COMMIT} && \
        mkdir -p /app/rocshmem-build && cd /app/rocshmem-build && \
        /app/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS=${GPU_ARCH_LIST} && \
        git clone ${DEEPEP_REPO} /app/DeepEP && cd /app/DeepEP && git checkout ${DEEPEP_COMMIT} && \
        PYTORCH_ROCM_ARCH=${GPU_ARCH_LIST} CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC --offload-arch=${GPU_ARCH_LIST}" HIP_CXX_FLAGS="-O3 -fPIC" \
            python3 setup.py --variant rocm --nic ${NIC_COMPILATION_ARCH} build develop ; \
    else echo "INSTALL_DEEPEP=${INSTALL_DEEPEP}: skipping rocSHMEM and DeepEP installation"; fi
ENV ROCSHMEM_TEST_UUID=1
ENV ROCSHMEM_HEAP_SIZE=6442450944

WORKDIR /sgl-workspace
RUN pip list
