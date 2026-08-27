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
ARG BASE_DOCKER=lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718
FROM $BASE_DOCKER

ARG ENABLE_ROCTX=0
ARG MORI_ROCTX_COMMIT=f7e6ac6863c53821bc7afb91a578cc6ce38fcad0
ARG SGLANG_ROCTX_COMMIT=48ae829f6e47f9348d8bd936b102d4d7a76f2743

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

ENV PYTHONPATH=$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:

ARG GPU_ARCH=gfx942
WORKDIR /sgl-workspace

RUN pip install --upgrade sglang-router

WORKDIR /sgl-workspace/mori

ARG MORI_COMMIT="158c7e8335a0b19b3f1f422ff134d7869252135e"
# Set INSTALL_MORI=1 to build/install MoRI at MORI_COMMIT; any other value skips it.
ARG INSTALL_MORI=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git ibverbs-utils libibverbs-dev \
    openmpi-bin libopenmpi-dev \
    libpci-dev libdw1 locales \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    cmake

COPY scripts/sglang_disagg/moriio_profiling/patches/ /tmp/roctx-patches/

# Upgrade the compatible ROCm 7.2.0 base to 7.2.3 for every build.
# ENABLE_ROCTX only controls marker patches and profiling tools below.
RUN set -eux; \
    sed -i 's#repo.radeon.com/rocm/apt/7.2 #repo.radeon.com/rocm/apt/7.2.3 #' /etc/apt/sources.list.d/rocm.list; \
    apt-get update; \
    apt-get install -y --only-upgrade $(dpkg -l | awk '$1 == "ii" {print $2}' \
      | grep -viE '^(amdgpu|libdrm)' \
      | grep -iE '^(rocm|hip|hsa|rccl|miopen|comgr|roc|rpp|rdc|amd-smi|composablekernel|tensile)'); \
    grep -q '^7\.2\.3' /opt/rocm/.info/version; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ "${ENABLE_ROCTX}" = "1" ]; then \
      git clone --quiet https://github.com/ROCm/mori.git /tmp/roctx-mori; \
      git -C /tmp/roctx-mori checkout --quiet "${MORI_ROCTX_COMMIT}"; \
      git -C /tmp/roctx-mori apply --index /tmp/roctx-patches/mori/01-roctx-instrumentation.patch; \
      git -C /tmp/roctx-mori diff --cached --name-only "${MORI_ROCTX_COMMIT}" | while IFS= read -r file; do \
        mkdir -p "/sgl-workspace/mori/$(dirname "${file}")"; \
        cp "/tmp/roctx-mori/${file}" "/sgl-workspace/mori/${file}"; \
      done; \
      git clone --quiet https://github.com/sgl-project/sglang.git /tmp/roctx-sglang; \
      git -C /tmp/roctx-sglang checkout --quiet "${SGLANG_ROCTX_COMMIT}"; \
      git -C /tmp/roctx-sglang apply --index /tmp/roctx-patches/sglang/01-roctx-instrumentation.patch; \
      git -C /tmp/roctx-sglang diff --cached --name-only "${SGLANG_ROCTX_COMMIT}" | while IFS= read -r file; do \
        mkdir -p "/sgl-workspace/sglang/$(dirname "${file}")"; \
        cp "/tmp/roctx-sglang/${file}" "/sgl-workspace/sglang/${file}"; \
      done; \
      echo "Installing complete patched pinned SGLang benchmark package"; \
      rm -rf /sgl-workspace/sglang/python/sglang/benchmark; \
      cp -a /tmp/roctx-sglang/python/sglang/benchmark /sgl-workspace/sglang/python/sglang/; \
      cp -a /tmp/roctx-sglang/python/sglang/benchmark/. /sgl-workspace/sglang/benchmark/; \
      rm -rf /tmp/roctx-mori /tmp/roctx-sglang; \
    fi

RUN set -eux; \
    if [ "${ENABLE_ROCTX}" = "1" ]; then \
      rm -rf /sgl-workspace/mori/build/CMakeCache.txt /sgl-workspace/mori/build/CMakeFiles; \
      cmake -S /sgl-workspace/mori -B /sgl-workspace/mori/build -G Ninja \
        -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Release -DWARP_ACCUM_UNROLL=1 \
        -DBUILD_SHMEM_DEVICE_WRAPPER=ON -DENABLE_DEBUG_PRINTF=OFF \
        -DENABLE_STANDARD_MOE_ADAPT=OFF -DGPU_TARGETS="${GPU_ARCH}" \
        -DENABLE_PROFILER=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
        -DBUILD_TESTS=OFF -DBUILD_UMBP=ON -DUSE_SPDK=OFF -DWITH_MPI=OFF \
        -DBUILD_TORCH_BOOTSTRAP=OFF -DBUILD_XLA_FFI_OPS=OFF -DBUILD_OPS_DEVICE=OFF \
        -DMORI_MULTITHREAD_SUPPORT=OFF; \
      cmake --build /sgl-workspace/mori/build -j"$(nproc)"; \
      for so in application collective io ops pybind shmem; do \
        libname="libmori_${so}.so"; \
        [ "${so}" = "pybind" ] && libname="libmori_pybinds.so"; \
        src="/sgl-workspace/mori/build/src/${so}/${libname}"; \
        [ ! -f "${src}" ] || cp "${src}" "/sgl-workspace/mori/python/mori/${libname}"; \
      done; \
    elif [ "${INSTALL_MORI}" = "1" ]; then \
      echo "INSTALL_MORI=1: installing MoRI at ${MORI_COMMIT}" \
      && git checkout main \
      && git fetch origin \
      && git pull origin main \
      && git checkout ${MORI_COMMIT} \
      && pip install -r requirements-build.txt \
      && pip install -e . ; \
    else \
      echo "ENABLE_ROCTX=${ENABLE_ROCTX}, INSTALL_MORI=${INSTALL_MORI}: skipping MoRI installation"; \
    fi

ENV SGLANG_ROCTX=0
ENV MORI_ROCTX=0
ENV MORI_ROCTX_TRANSFER=0

WORKDIR /sgl-workspace

# Display installed packages for verification
RUN pip list
