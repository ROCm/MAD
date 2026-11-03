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
# =============================================================================
# di_vllm.ubuntu.amd.Dockerfile -- the vLLM distributed-inference image, for any
# pin set in docker/pins/ (vllm-mori121, vllm-glm51, vllm-kimi-k3, ...).
#
#   docker build -f docker/di_vllm.ubuntu.amd.Dockerfile \
#     $(python3 docker/pins/pins.py build-args vllm-kimi-k3 --arch gfx950) \
#     -t <registry>/di-vllm:vllm-kimi-k3-gfx950 .
#
# Build context = repo root. The pin set says what is built: the base image per
# architecture, MoRI, AITER (a release wheel or a commit), vLLM, vllm-router and,
# when enabled, the UCX / RIXL / rocSHMEM / DeepEP transport stack -- each at a
# full upstream commit, with the pin set's patches (docker/patches/) applied on top
# where a change is not upstream yet. BASE_IMAGE must be the pin set's base for the architecture (the
# build stops otherwise); build-args prints it. /app/versions.txt lists every
# component's commit and /app/pins.json is the pin set the image was built from.
#
# The image carries no runtime recipe, tuning or platform env: models.yaml and
# scripts/vllm_dissag/connectors/<connector>.env are applied at launch.
# =============================================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG BASE_IMAGE
ARG PINS
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ARG MAX_JOBS=32

# -----------------------------------------------------------------------------
# 0. The pin set, resolved for this architecture and base: /opt/mad/pins/build.env
#    is sourced by every step below. Stops on an invalid set, an architecture the
#    set is not for, or another base image.
# -----------------------------------------------------------------------------
COPY docker/pins/ /opt/mad/pins/
COPY docker/patches/ /opt/mad/patches/
RUN test -n "${PINS}" || { echo "PINS is not set (a name in docker/pins/)" >&2; exit 1; } && \
    python3 /opt/mad/pins/pins.py env "${PINS}" --arch "${MAD_SYSTEM_GPU_ARCHITECTURE}" \
        --base "${BASE_IMAGE}" > /opt/mad/pins/build.env && \
    _ARG_MAX_JOBS="${MAX_JOBS}" && . /opt/mad/pins/build.env && \
    { [ "${_ARG_MAX_JOBS}" = "${MAX_JOBS}" ] || \
      { echo "MAX_JOBS=${_ARG_MAX_JOBS}, but pin set ${PINS} says ${MAX_JOBS} (use pins.py build-args)" >&2; exit 1; }; } && \
    mkdir -p /app && cp "/opt/mad/pins/${PINS}.json" /app/pins.json && \
    printf 'PIN_SET=%s\nGPU_ARCH=%s\nBASE_IMAGE=%s\n' "${PIN_SET}" "${GPU_ARCH}" "${BASE_IMAGE}" > /app/versions.txt && \
    printf '%s\n' "${PIP_CONSTRAINTS}" > /etc/pip-constraints.txt && \
    cat /opt/mad/pins/build.env

# Pip constraints from the pin set (empty file = none), in effect for every pip below.
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# Runtime and build env the legacy images set from the architecture.
ENV MORI_GPU_ARCHS=${MAD_SYSTEM_GPU_ARCHITECTURE} \
    BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${MAD_SYSTEM_GPU_ARCHITECTURE} \
    MAX_JOBS=${MAX_JOBS}

# -----------------------------------------------------------------------------
# 1. MoRI at the pinned commit, replacing the base's bundled MoRI. BUILD_UMBP=OFF
#    (UMBP needs gRPC, unrelated to EP); default NIC backends (disabling them
#    deadlocked the cross-node EP all-to-all init).
# -----------------------------------------------------------------------------
RUN . /opt/mad/pins/build.env && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    if [ "${MORI_BUILD}" = "1" ]; then \
        pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true && \
        rm -rf /tmp/mori-src && \
        git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
        cd /tmp/mori-src && git checkout "${MORI_COMMIT}" && git submodule update --init --recursive && \
        BUILD_UMBP=OFF pip install . && \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
        echo "MORI_COMMIT=$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
        rm -rf /tmp/mori-src ; \
    else \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI (base) OK at', mori.__path__[0])" && \
        echo "MORI_COMMIT=base" >> /app/versions.txt ; \
    fi

# -----------------------------------------------------------------------------
# 2. AITER: the pinned release wheel, or the pinned commit built from source; then
#    flydsl at its pinned version, and the stale prewarmed AITER JIT cache cleared.
# -----------------------------------------------------------------------------
RUN . /opt/mad/pins/build.env && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    case "${AITER_KIND}" in \
      wheel) \
        _W="/tmp/$(basename "${AITER_WHEEL_URL}" | sed 's/%2B/+/g')" && \
        curl -fL --retry 3 --retry-delay 2 -o "${_W}" "${AITER_WHEEL_URL}" && \
        pip install --no-deps "${_W}" && rm -f "${_W}" && \
        echo "AITER_VERSION=${AITER_VERSION}" >> /app/versions.txt ;; \
      source) \
        rm -rf /tmp/aiter-src && mkdir -p /tmp/aiter-src && cd /tmp/aiter-src && \
        git init -q && git remote add origin "${AITER_REPO}" && \
        git fetch -q --tags origin "${AITER_COMMIT}" && git checkout -q FETCH_HEAD && \
        git submodule update --init --recursive && \
        for _p in ${AITER_PATCHES}; do git apply --binary "$_p" && \
            echo "AITER_PATCH=$(basename "$_p") sha256:$(sha256sum "$_p" | cut -d' ' -f1)" >> /app/versions.txt || exit 1; done && \
        if [ "${AITER_ARCH_ENV}" = "1" ]; then export GPU_ARCHS="${GPU_ARCH}"; fi && \
        pip install --no-build-isolation --no-deps -v . && \
        echo "AITER_COMMIT=$(git rev-parse HEAD)" >> /app/versions.txt && \
        cd / && rm -rf /tmp/aiter-src ;; \
      *) echo "pin set has no AITER" >&2; exit 1 ;; \
    esac && \
    if [ -n "${FLYDSL_VERSION}" ]; then pip install ${FLYDSL_PIP_FLAGS} "flydsl==${FLYDSL_VERSION}"; fi && \
    python3 /opt/mad/pins/check_vllm_stack.py aiter && \
    rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache"

# -----------------------------------------------------------------------------
# 3. vLLM compiled at the pinned commit (the base ships another commit, so a
#    .py-only overlay would be ABI-mismatched), then the pin set's after-engine pip
#    packages, and a check that MoRI and AITER survived the vLLM install.
# -----------------------------------------------------------------------------
RUN . /opt/mad/pins/build.env && \
    rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_COMMIT}" && \
    echo "VLLM_COMMIT=$(git rev-parse HEAD)" >> /app/versions.txt && \
    for _p in ${VLLM_PATCHES}; do git apply --binary "$_p" && \
            echo "VLLM_PATCH=$(basename "$_p") sha256:$(sha256sum "$_p" | cut -d' ' -f1)" >> /app/versions.txt || exit 1; done && \
    pip uninstall -y vllm 2>/dev/null || true && \
    MAX_JOBS="${MAX_JOBS}" ${NVCC_THREADS:+NVCC_THREADS="${NVCC_THREADS}"} \
      pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    cd / && rm -rf /tmp/vllm-src && \
    if [ -n "${PIP_AFTER_ENGINE}" ]; then pip install ${PIP_AFTER_ENGINE}; fi && \
    python3 /opt/mad/pins/check_vllm_stack.py aiter mori

# -----------------------------------------------------------------------------
# 4. vllm-router at the pinned commit, built in (no external ROUTER_BINARY needed).
#    Rust toolchain from the pin set (router deps need rustc >= 1.88).
# -----------------------------------------------------------------------------
RUN . /opt/mad/pins/build.env && \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_COMMIT}" && \
    for _p in ${ROUTER_PATCHES}; do git apply --binary "$_p" && \
            echo "ROUTER_PATCH=$(basename "$_p") sha256:$(sha256sum "$_p" | cut -d' ' -f1)" >> /app/versions.txt || exit 1; done && \
    cargo build --release && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_COMMIT=$(git -C /tmp/vllm-router-src rev-parse HEAD)" >> /app/versions.txt && \
    cd / && rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 5. The rixl transport stack, when the pin set enables it: UCX, RIXL (+python
#    bindings), rocSHMEM and DeepEP, each at its pinned commit. One RUN, so a set
#    without it adds no layer.
# -----------------------------------------------------------------------------
ENV _ROCM_DIR=/opt/rocm \
    _UCX_INSTALL_DIR=/usr/local/ucx/ \
    _RIXL_INSTALL_DIR=/usr/local/RIXL/install \
    _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL
RUN . /opt/mad/pins/build.env && \
    if [ "${TRANSPORTS_BUILD}" != "1" ]; then \
      echo "transports off in ${PIN_SET}: no UCX/RIXL/rocSHMEM/DeepEP"; \
    else set -e && \
      apt-get update && apt-get install -y \
        autoconf automake libtool autogen pkg-config m4 gcc make \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core strace libgflags-dev \
        libaio-dev liburing-dev libcpprest-dev libgrpc-dev libgrpc++-dev \
        libprotobuf-dev protobuf-compiler-grpc wget && \
      pip install meson==0.64.0 "pybind11[global]" pyyaml && \
      cd /tmp && git clone "${UCX_REPO}" ucx && cd ucx && git checkout "${UCX_COMMIT}" && \
        ./autogen.sh && mkdir -p build && cd build && \
        ../configure --prefix="${_UCX_INSTALL_DIR}" --with-rocm="${_ROCM_DIR}" \
          --disable-go --disable-java --disable-assertions --enable-mt && \
        make -j && make install && \
      cd /tmp && wget -q -O googletest.tar.gz "${GTEST_TARBALL}" && \
        mkdir -p googletest && tar -xzf googletest.tar.gz -C googletest --strip-components=1 && \
        cd googletest && mkdir -p build && cd build && \
        cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install && \
      cd /tmp && git clone "${RIXL_REPO}" RIXL && cd RIXL && git checkout "${RIXL_COMMIT}" && \
        meson setup build/ --prefix="${_RIXL_INSTALL_DIR}" -Ducx_path="${_UCX_INSTALL_DIR}" \
          -Ddisable_gds_backend=true -Dcudapath_inc="${_ROCM_DIR}/include" -Dcudapath_lib="${_ROCM_DIR}/lib" && \
        cd build && ninja && ninja install && cd /tmp/RIXL && \
        pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                    --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                    --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                    --config-settings=setup-args="-Ddisable_gds_backend=true" . && \
      cd /tmp && git clone --no-checkout --filter=blob:none "${ROCSHMEM_REPO}" rocm-systems && \
        cd rocm-systems && git sparse-checkout set --cone projects/rocshmem && git checkout "${ROCSHMEM_COMMIT}" && \
        mkdir -p /tmp/rocshmem-build && cd /tmp/rocshmem-build && \
        /tmp/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends \
          -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS="${GPU_ARCH}" && \
      cd /tmp && git clone "${DEEPEP_REPO}" DeepEP && cd DeepEP && git checkout "${DEEPEP_COMMIT}" && \
        PYTORCH_ROCM_ARCH="${GPU_ARCH}" CFLAGS="-O3 -fPIC" \
          CXXFLAGS="-O3 -fPIC --offload-arch=${GPU_ARCH}" HIP_CXX_FLAGS="-O3 -fPIC" \
          python3 setup.py --variant rocm --nic "${NIC_ARCH}" build develop && \
      printf 'UCX_COMMIT=%s\nRIXL_COMMIT=%s\nROCSHMEM_COMMIT=%s\nDEEPEP_COMMIT=%s\n' \
        "$(git -C /tmp/ucx rev-parse HEAD)" "$(git -C /tmp/RIXL rev-parse HEAD)" \
        "$(git -C /tmp/rocm-systems rev-parse HEAD)" "$(git -C /tmp/DeepEP rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/ucx /tmp/googletest /tmp/googletest.tar.gz /tmp/rocm-systems /tmp/rocshmem-build; \
    fi
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:/usr/local/lib:/usr/local/RIXL/install/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/ucx/bin:${PATH}"

# -----------------------------------------------------------------------------
# 6. Cache locations: the mount target for the launcher's persistent JIT cache.
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

# -----------------------------------------------------------------------------
# 7. Smoke test against the pin set, then scrub build-time MoRI JIT state: the
#    `import mori` checks above leave .hsaco.lock files under /root/.mori whose owner
#    is gone at runtime, and MoriAll2AllManager then deadlocks at ep:0 init.
# -----------------------------------------------------------------------------
RUN python3 /opt/mad/pins/check_vllm_stack.py all && \
    rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED=1" >> /app/versions.txt && \
    cat /app/versions.txt
