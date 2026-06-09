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
ARG BASE_DOCKER=rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216
FROM $BASE_DOCKER

ARG GPU_ARCH=gfx942
WORKDIR /app

RUN sed -i 's/http/https/g' /etc/apt/sources.list

RUN apt-get update && \
   apt-get install -y \
    autoconf pkg-config \
    libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev \
    libibverbs-dev ibverbs-utils libtool libboost-all-dev \
    libgrpc++-dev protobuf-compiler-grpc protobuf-compiler libprotobuf-dev \
    libaio-dev liburing-dev pybind11-dev ninja-build libgflags-dev \
    rdma-core infiniband-diags perftest openssh-server \
    psmisc vim cmake-curses-gui

RUN pip3 install "pybind11[global]" meson==0.64.0

ARG GFX_COMPILATION_ARCH="gfx942"
ARG NIC_COMPILATION_ARCH="cx7"

WORKDIR /app
RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && cd rocm-systems && \
    git sparse-checkout set --cone projects/rocshmem && \
    git checkout develop

WORKDIR /app/rocm-systems/projects/rocshmem

RUN mkdir -p /app/rocshmem-build
WORKDIR /app/rocshmem-build
RUN /app/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS=$GFX_COMPILATION_ARCH

WORKDIR /app
RUN git clone https://github.com/ROCm/DeepEP.git
WORKDIR /app/DeepEP
RUN PYTORCH_ROCM_ARCH=$GFX_COMPILATION_ARCH  CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC --offload-arch=$GFX_COMPILATION_ARCH" HIP_CXX_FLAGS="-O3 -fPIC" \
    python3 setup.py --variant rocm --nic $NIC_COMPILATION_ARCH build develop

WORKDIR /app

# not installing mori since its already installed in vllm container.
RUN pip install tqdm prettytable
RUN git clone --recursive $(grep '^MORI_REPO:' versions.txt | cut -d' ' -f2) && \
    cd mori && \
    git checkout $(grep '^MORI_BRANCH:' /app/versions.txt | cut -d' ' -f2) 

RUN cd /app/mori && git log | head

ENV ROCSHMEM_TEST_UUID=1 
ENV ROCSHMEM_HEAP_SIZE=6442450944
