ARG BASE_IMAGE=rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216
FROM ${BASE_IMAGE}

ENTRYPOINT []

WORKDIR /root

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

ENV _ROCM_DIR=/opt/rocm

ENV _UCX_SOURCE=https://github.com/ROCm/ucx.git
ENV _UCX_BRANCH=da3fac2a
ENV _UCX_INSTALL_DIR=/usr/local/ucx/

ENV _RIXL_SOURCE=https://github.com/ROCm/RIXL.git
ENV _RIXL_BRANCH=f33a5599
ENV _RIXL_INSTALL_DIR=/usr/local/RIXL/install
ENV _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL

ARG GFX_COMPILATION_ARCH="gfx942"
ARG NIC_COMPILATION_ARCH="cx7"
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT=7d6917bef552d6aff70142ab9fb8af648081d4db

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
RUN git clone ${_UCX_SOURCE} && \
    cd ucx && \
    git checkout ${_UCX_BRANCH} && \
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
    apt install -y libaio-dev liburing-dev libcpprest-dev libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc wget && \
    wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
    tar -xzf v1.14.0.tar.gz && \
    cd googletest-1.14.0 && \
    mkdir -p build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=on .. && \
    make -j && \
    make install && \
    cd ../..

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
ENV PATH=/root/.local/bin:${_UCX_INSTALL_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${_RIXL_INSTALL_DIR}/lib/x86_64-linux-gnu

RUN set -e && git clone ${_RIXL_SOURCE} && \
    cd RIXL && \
    git checkout ${_RIXL_BRANCH} && \
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


# Install Rust compiler (required for building vllm-router)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install vllm-router
RUN pip install vllm-router

WORKDIR /app

# versions.txt is provided by the base image and contains MORI_REPO / MORI_BRANCH entries.
# Override MORI_BRANCH at build time: docker build --build-arg MORI_BRANCH=158c7e83 ...
ARG MORI_BRANCH="158c7e83"
RUN pip install tqdm prettytable
RUN git clone --recursive $(grep '^MORI_REPO:' /app/versions.txt | cut -d' ' -f2) && \
    cd mori && \
    git checkout ${MORI_BRANCH:-$(grep '^MORI_BRANCH:' /app/versions.txt | cut -d' ' -f2)} && \
    git submodule update --init --recursive && \
    pip install .

RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && cd rocm-systems && \
    git sparse-checkout set --cone projects/rocshmem && \
    git checkout develop

WORKDIR /app/rocm-systems/projects/rocshmem
RUN echo "ROCSHMEM_REPO=\"https://github.com/ROCm/rocm-systems.git\"" >> /app/versions.txt
RUN echo "ROCSHMEM_BRANCH=\"$(git log | head -1 | awk '{print $2}' | cut -c1-8)\"" >> /app/versions.txt
RUN pip install pyyaml
RUN mkdir -p /app/rocshmem-build
WORKDIR /app/rocshmem-build
RUN /app/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS=$GFX_COMPILATION_ARCH

WORKDIR /app
RUN git clone https://github.com/ROCm/DeepEP.git
WORKDIR /app/DeepEP
RUN echo "DEEPEP_REPO=\"https://github.com/ROCm/DeepEP.git\"" >> /app/versions.txt
RUN echo "DEEPEP_BRANCH=\"$(git log | head -1 | awk '{print $2}' | cut -c1-8)\"" >> /app/versions.txt
RUN PYTORCH_ROCM_ARCH=$GFX_COMPILATION_ARCH  CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC --offload-arch=$GFX_COMPILATION_ARCH" HIP_CXX_FLAGS="-O3 -fPIC" \
    python3 setup.py --variant rocm --nic $NIC_COMPILATION_ARCH build develop

# Uninstall vLLM from the base image, then install the pinned commit from source (ROCm).
# TODO: Remove this installation details after upstream vllm is stable.
RUN pip uninstall -y vllm || true
RUN pip install setuptools-scm huggingface-hub[cli]
RUN pip install quart msgpack  --ignore-installed blinker
RUN rm -rf /tmp/vllm-src && \
    git clone --recursive "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && \
    git checkout "${VLLM_COMMIT}" && \
    git submodule update --init --recursive && \
    pip install -r requirements/rocm.txt && \
    pip install -r requirements/kv_connectors_rocm.txt && \
    (PYTORCH_ROCM_ARCH=${GFX_COMPILATION_ARCH} python setup.py install || \
        echo "WARNING: vLLM build from source failed; container may be broken") && \
    mkdir -p /app/vllm && \
    cp -r tests /app/vllm/tests && \
    cp -r examples /app/vllm/examples && \
    cp -r benchmarks /app/vllm/benchmarks && \
    rm -rf /tmp/vllm-src

WORKDIR /app

ENV ROCSHMEM_TEST_UUID=1
ENV ROCSHMEM_HEAP_SIZE=6442450944

RUN pip install --upgrade vllm-router && \
    pip install py-spy && \
    pip install --ignore-installed --force-reinstall flask

RUN echo "UCX_REPO=${_UCX_SOURCE}" >> /app/versions.txt && \
    echo "UCX_BRANCH=${_UCX_BRANCH}" >> /app/versions.txt && \
    echo "RIXL_REPO=${_RIXL_SOURCE}" >> /app/versions.txt && \
    echo "RIXL_BRANCH=${_RIXL_BRANCH}" >> /app/versions.txt

RUN cat /app/versions.txt
