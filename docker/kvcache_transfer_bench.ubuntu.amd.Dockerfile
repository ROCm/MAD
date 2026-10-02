# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
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

# ========== MOONCAKE ==========
WORKDIR /sgl-workspace
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
# v0.3.13.post1
ARG MOONCAKE_COMMIT="719735896c86b56fabec6cf3e825fb2ea640597a"
# Set INSTALL_MOONCAKE=1 to build/install Mooncake at MOONCAKE_COMMIT; any other value skips it.
ARG INSTALL_MOONCAKE=1
ENV MOONCAKE_HOME=/sgl-workspace/Mooncake
# ENABLE_MULTI_PROTOCOL=ON is required with USE_HIP=ON; without it cross-node writes are routed to HIP IPC and fail.
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then echo "INSTALL_MOONCAKE=1: installing Mooncake at ${MOONCAKE_COMMIT}" && \
        rm -rf ${MOONCAKE_HOME} /usr/local/go && \
        git clone ${MOONCAKE_REPO} ${MOONCAKE_HOME} && cd ${MOONCAKE_HOME} && \
        git checkout ${MOONCAKE_COMMIT} && git submodule update --init --recursive && \
        bash dependencies.sh -y && \
        mkdir build && cd build && \
        cmake .. -DBUILD_SHARED_LIBS=ON -DUSE_HIP=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_HTTP=ON -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON -DBUILD_UNIT_TESTS=OFF && \
        make -j16 && make install && cd .. && \
        pip install uv && uv venv --seed /tmp/mooncake-wheel-venv && \
        PATH="/tmp/mooncake-wheel-venv/bin:${PATH}" LD_LIBRARY_PATH="${MOONCAKE_HOME}/build/mooncake-asio:${LD_LIBRARY_PATH}" ./scripts/build_wheel.sh && \
        rm -rf /tmp/mooncake-wheel-venv && \
        pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-non-cuda && \
        rm -rf "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/mooncake" && \
        pip install --no-cache-dir mooncake-wheel/dist/*.whl && \
        python3 -c "from mooncake.engine import TransferEngine" && \
        cp build/mooncake-transfer-engine/benchmark/tebench /usr/local/bin/ ; \
    else echo "INSTALL_MOONCAKE=${INSTALL_MOONCAKE}: skipping Mooncake installation"; fi

# ========== UCX + NIXL ==========
WORKDIR /sgl-workspace
ARG UCX_REPO="https://github.com/ROCm/ucx.git"
ARG UCX_COMMIT="da3fac2a"
ARG NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
# v1.4.1
ARG NIXL_COMMIT="778edd1d1a50936b12c264879e12ef465e629002"
# ai-dynamo/nixl#2191 (ROCm fix, not in v1.4.1)
ARG NIXL_CHERRY_PICK="fe3244c875ee4a22da6b0140c94c334546e9b715"
# Set INSTALL_NIXL=1 to build UCX + NIXL at the commits above; any other value skips them.
ARG INSTALL_NIXL=1
ENV UCX_HOME=/sgl-workspace/ucx
ENV NIXL_HOME=/sgl-workspace/nixl
ENV PATH=${UCX_HOME}/bin:${PATH}:/usr/local/bin/etcd
ENV LD_LIBRARY_PATH=${NIXL_HOME}/lib/x86_64-linux-gnu:${UCX_HOME}/lib:${LD_LIBRARY_PATH}
# Replaces any NIXL the base ships; NIXL's build rejects distro Abseil without absl_log (e.g. Ubuntu 24.04's).
RUN if [ "${INSTALL_NIXL}" = "1" ]; then echo "INSTALL_NIXL=1: installing NIXL at ${NIXL_COMMIT}" && \
        pip install meson meson-python patchelf tomlkit uv && \
        if dpkg -s libabsl-dev >/dev/null 2>&1; then apt-get remove -y libabsl-dev; fi && \
        pip uninstall -y nixl nixl-rocm nixl-cu12 nixl-cu13 && \
        SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && rm -rf "$SITE/nixl" "$SITE/nixl_rocm" && \
        rm -rf ${UCX_HOME} ${NIXL_HOME} && \
        git clone ${UCX_REPO} ${UCX_HOME} && cd ${UCX_HOME} && git checkout ${UCX_COMMIT} && \
        ./autogen.sh && mkdir build && cd build && \
        ../configure --prefix=${UCX_HOME} --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations --enable-devel-headers \
            --with-rocm=/opt/rocm --with-verbs --with-dm --enable-mt --without-go && \
        make -j && make install && \
        git clone ${NIXL_REPO} ${NIXL_HOME} && cd ${NIXL_HOME} && \
        git checkout ${NIXL_COMMIT} && git cherry-pick --no-commit ${NIXL_CHERRY_PICK} && \
        meson setup build --prefix=${NIXL_HOME} -Ducx_path=${UCX_HOME} -Dwheel_variant=rocm -Denable_plugins=UCX && \
        ninja -C build install && \
        ./contrib/tomlutil.py --wheel-name nixl-rocm pyproject.toml && \
        pip install --no-cache-dir --no-build-isolation . \
            -Csetup-args=-Ducx_path=${UCX_HOME} -Csetup-args=-Dwheel_variant=rocm -Csetup-args=-Denable_plugins=UCX && \
        pip install --no-cache-dir --no-deps build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl && \
        python3 -c "from nixl._api import nixl_agent, nixl_agent_config; assert 'UCX' in nixl_agent('probe', nixl_agent_config(backends=[])).get_plugin_list()" && \
        wget -q https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
        mkdir -p /usr/local/bin/etcd && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && rm /tmp/etcd.tar.gz ; \
    else echo "INSTALL_NIXL=${INSTALL_NIXL}: skipping NIXL installation"; fi

WORKDIR /sgl-workspace
RUN pip list
