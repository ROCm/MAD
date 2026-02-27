ARG BASE_IMAGE=rocm/vllm:v0.14.0_amd_dev
FROM ${BASE_IMAGE}

WORKDIR /root

ENV _ROCM_DIR=/opt/rocm

ENV _UCX_SOURCE=https://github.com/ROCm/ucx.git
ENV _UCX_BRANCH=v1.19.x
ENV _UCX_INSTALL_DIR=/usr/local/ucx/

ENV _RIXL_SOURCE=github.com/ROCm/RIXL.git
ENV _RIXL_BRANCH=develop
ENV _RIXL_INSTALL_DIR=/usr/local/RIXL/install
ENV _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL

RUN pip3 install meson==0.64.0
RUN pip3 install "pybind11[global]"

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    autogen \
    pkg-config \
    m4 || apt --fix-broken install -y

RUN set -e && apt update

RUN set -e && apt -y install gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace
RUN apt install -y libgflags-dev

# Install UCX
RUN git clone ${_UCX_SOURCE} -b ${_UCX_BRANCH} && \
    cd ucx && \
    ./autogen.sh && \
    mkdir -p build && \
    cd build && \
    ../configure --prefix=${_UCX_INSTALL_DIR} --with-rocm=${_ROCM_DIR} --disable-go --disable-java --disable-assertions --enable-mt && \
    make -j && \
    make install && \
    echo "UCX installation completed."


ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ucx/lib/
ENV PATH=$PATH:/usr/local/ucx/bin/

RUN set -e && apt update && \
    apt install -y libaio-dev liburing-dev etcd etcd-server etcd-client libcpprest-dev libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc && \
    wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
    tar -xzf v1.14.0.tar.gz && \
    cd googletest-1.14.0 && \
    mkdir -p build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=on .. && \
    make -j && \
    make install && \
    cd ../..

# Expected etcd at /usr/local/bin/etcd//etcd
RUN wget https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=$PATH:/usr/local/bin/etcd/

RUN set -e && echo "Compiling etcd-cpp API" && \
    git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
    cd etcd-cpp-apiv3 && \
    mkdir build && cd build && \
    cmake -DCMAKE_FIND_ROOT_PATH=/usr/grpc .. && \
    make -j && \
    make install && \
    cd ../.. && \
    echo "etcd-cpp installation completed."

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/lib/cmake/etcd-cpp-api/
ENV PATH=/root/.local/bin:${_UCX_INSTALL_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${_RIXL_INSTALL_DIR}/lib/x86_64-linux-gnu
ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake/etcd-cpp-api/:/usr/grpc/lib/cmake/:/usr/local/lib/cmake

RUN set -e && git clone https://${_RIXL_SOURCE} -b ${_RIXL_BRANCH} && \
    cd RIXL && \
    git checkout ed772c8d0d8a47c7b4e1a622b13c4f6087a4972a && \
    meson setup build/ --prefix=${_RIXL_INSTALL_DIR} \
        -Ducx_path=${_UCX_INSTALL_DIR} \
        -Ddisable_gds_backend=true \
        -Dcudapath_inc=${_ROCM_DIR}/include \
        -Dcudapath_lib=${_ROCM_DIR}/lib && \
    cd build && \
    ninja && \
    ninja install

RUN set -e && cd RIXL && \
    pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                --config-settings=setup-args="-Ddisable_gds_backend=true" .

ENV LD_LIBRARY_PATH=${_RIXL_INSTALL_DIR}/lib:$LD_LIBRARY_PATH

RUN set -e && echo "Compiling NixlBench" && \
    cd RIXL/benchmark/nixlbench && \
    meson setup build \
        -Dnixl_path=${_RIXL_INSTALL_DIR} \
        -Dcudapath_inc=${_ROCM_DIR}/include \
        -Dcudapath_lib=${_ROCM_DIR}/lib \
        --prefix=${_NIXLBENCH_INSTALL_DIR} && \
    cd build && \
    ninja && \
    ninja install && \
    echo "NixlBench compilation complete"

# Only need tests/ for toy_proxy_server.py; base image already has vLLM installed
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-src && \
    cp -r /tmp/vllm-src/tests /app/vllm/tests && \
    rm -rf /tmp/vllm-src

# Install Rust compiler (required for building vllm-router)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install vllm-router
RUN pip install vllm-router
