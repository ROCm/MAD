#
# Single-stage build: Mooncake + MORI + RIXL + nixlbench
#   docker build -f Dockerfile -t kv-cache-unified:latest .
#
ARG BASE_IMAGE=rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216
FROM ${BASE_IMAGE} AS base

ENV WORKSPACE_DIR=/workspace

RUN sed -i 's/http:/https:/g' /etc/apt/sources.list && \
    sed -i 's/http:/https:/g' /etc/apt/sources.list.d/*.list || true

RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's/http:/https:/g' /etc/apt/sources.list.d/ubuntu.sources || true; \
    fi

# Install dependencies for KV cache calculator, report generation, Mori, Mooncake, RIXL
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    pyyaml \
    tabulate \
    plotly \
    kaleido \
    pandas \
    meson \
    auditwheel \
    patchelf \
    tomlkit \
    setuptools \
    wheel

RUN apt-get update && \
    apt-get install -y \
    git \
    cython3 \
    ibverbs-utils \
    ibverbs-providers \
    rdmacm-utils \
    libibverbs1 \
    libibverbs-dev \
    librdmacm1 \
    librdmacm-dev \
    openmpi-bin \
    libopenmpi-dev \
    libpci-dev \
    cmake \
    libdw1 \
    locales \
    autoconf \
    libtool \
    pkg-config \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    libcpprest-dev \
    libaio-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libnuma-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    liburing-dev \
    libjemalloc-dev \
    libmsgpack-dev \
    libzstd-dev \
    libasio-dev \
    libyaml-cpp-dev \
    pybind11-dev \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install prettytable cmake setuptools setuptools-scm pytest-assume

WORKDIR ${WORKSPACE_DIR}

# ========== MOONCAKE ==========
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"

RUN git clone ${MOONCAKE_REPO} mooncake && \
    cd mooncake && \
    echo "Mooncake: $(git rev-parse HEAD)" >> ${WORKSPACE_DIR}/shas.txt && \
    ./dependencies.sh -y && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DUSE_HIP=ON -DUSE_HTTP=ON -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON && \
    export PATH="/usr/local/go/bin:${PATH}" && \
    make -j16 && \
    make install

RUN cd ${WORKSPACE_DIR}/mooncake && \
    export PATH="/usr/local/go/bin:${PATH}" && \
    export LD_LIBRARY_PATH="${WORKSPACE_DIR}/mooncake/build/mooncake-asio:${LD_LIBRARY_PATH}" && \
    ./scripts/build_wheel.sh

RUN pip install uv && uv pip install --system ${WORKSPACE_DIR}/mooncake/mooncake-wheel/dist/*.whl

RUN TEBENCH="${WORKSPACE_DIR}/mooncake/build/mooncake-transfer-engine/benchmark/tebench" && \
    if [ -f "$TEBENCH" ]; then cp "$TEBENCH" /usr/local/bin/; fi

ENV MOONCAKE_VERSION="built-from-source-rocm"
ENV PATH=/usr/local/go/bin:${PATH}
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages

# ========== MORI ==========
RUN git clone --recursive https://github.com/ROCm/mori.git
WORKDIR ${WORKSPACE_DIR}/mori
RUN pip3 install . --no-build-isolation
ENV PYTHONPATH=/workspace/mori:/usr/local/lib/python3.12/dist-packages

# ========== UCX + RIXL ==========
WORKDIR ${WORKSPACE_DIR}
ARG RIXL_BRANCH="f33a5599"
ARG RIXL_REPO="https://github.com/ROCm/RIXL.git"
ARG UCX_BRANCH="da3fac2a"
ARG UCX_REPO="https://github.com/ROCm/ucx.git"
ENV ROCM_PATH=/opt/rocm
ENV UCX_HOME=${WORKSPACE_DIR}/ucx
ENV RIXL_HOME=${WORKSPACE_DIR}/rixl

RUN git clone ${UCX_REPO} && cd ucx && git checkout ${UCX_BRANCH} && \
    ./autogen.sh && mkdir build && cd build && \
    ../configure --prefix=${UCX_HOME} --enable-shared --disable-static \
        --disable-doxygen-doc --enable-optimizations --enable-devel-headers \
        --with-rocm=/opt/rocm --with-verbs --with-dm --enable-mt && \
    make -j && make install

ENV PATH=${UCX_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${UCX_HOME}/lib:${LD_LIBRARY_PATH}

RUN git clone ${RIXL_REPO} ${RIXL_HOME} && cd ${RIXL_HOME} && git checkout ${RIXL_BRANCH} && \
    meson setup build --prefix=${RIXL_HOME} -Ducx_path=${UCX_HOME} -Drocm_path=${ROCM_PATH} && \
    cd build && ninja && ninja install

RUN cd ${RIXL_HOME} && mkdir -p install && \
    ./contrib/build-wheel.sh \
        --output-dir ${RIXL_HOME}/install \
        --rocm-dir ${ROCM_PATH} \
        --ucx-plugins-dir ${UCX_HOME}/lib/ucx \
        --nixl-plugins-dir ${RIXL_HOME}/lib/x86_64-linux-gnu/plugins

RUN pip install ${RIXL_HOME}/install/*.whl
ENV _RIXL_INSTALL_DIR=${RIXL_HOME}/install
ENV _NIXLBENCH_INSTALL_DIR=${RIXL_HOME}/
ENV _ROCM_DIR=/opt/rocm

RUN python -c "import site; print(site.getsitepackages()[0])" > /tmp/sitepk.txt && \
    SP=$(cat /tmp/sitepk.txt) && \
    echo "import rixl, sys; sys.modules['nixl'] = rixl" > ${SP}/nixl_alias.pth && \
    python -c "import nixl; print('nixl alias OK')"

# ========== etcd + etcd-cpp ==========
RUN wget -q https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && rm /tmp/etcd.tar.gz
ENV PATH=${PATH}:/usr/local/bin/etcd/

RUN git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
    cd etcd-cpp-apiv3 && mkdir build && cd build && \
    cmake -DCMAKE_FIND_ROOT_PATH=/usr/grpc .. && make -j && make install

# ========== nixlbench ==========
RUN cd ${RIXL_HOME}/benchmark/nixlbench && \
    export CXXFLAGS="-Wno-error" && \
    meson setup build \
        -Dnixl_path=${RIXL_HOME} \
        -Drocm_path=${ROCM_PATH} \
        -Duse_rocm=true \
        -Dprefix=/usr/local \
        --buildtype=release && \
    cd build && ninja && ninja install

ENV PATH=/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${RIXL_HOME}/lib/x86_64-linux-gnu:${UCX_HOME}/lib:${LD_LIBRARY_PATH}

# ========== Final ==========
RUN echo "Image ready: Mooncake + MORI + RIXL + nixlbench"
