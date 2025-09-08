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
ARG BASE_DOCKER=lmsysorg/sglang:v0.4.9.post1-rocm630
FROM $BASE_DOCKER

ARG GPU_ARCH=gfx942
WORKDIR /sgl-workspace

RUN apt update && apt install -y zip unzip wget
RUN apt install -y gcc make libtool autoconf  librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool  libibverbs-dev rdma-core
RUN apt install -y openssh-server openmpi-bin openmpi-common libopenmpi-dev

RUN export  ETCD_VERSION="v3.6.0-rc.5" && \
        wget https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
        mkdir -p /usr/local/bin/etcd && \
        tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
        rm /tmp/etcd.tar.gz

ENV PATH=$PATH:/usr/local/bin/etcd/

ENV PATH=$PATH:/usr/local/go/bin

# Fix Go build environment
RUN git clone --recursive https://github.com/kvcache-ai/Mooncake.git && \
        cd Mooncake && \
        git checkout e386b1bf8f && \
        git rev-parse HEAD && \
        bash dependencies.sh -y && \
        rm -rf /usr/local/go && \
        wget https://go.dev/dl/go1.22.2.linux-amd64.tar.gz && \
        tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
        rm go1.22.2.linux-amd64.tar.gz && \
        mkdir -p build && \
        cd build && \
        cmake .. -DUSE_ETCD=ON && \
        make -j 20 && make install


RUN curl -k -o /tmp/MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64.tgz https://content.mellanox.com/ofed/MLNX_OFED-5.9-0.5.6.0.127/MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64.tgz && \
        cd /tmp && \
        tar xzf MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64.tgz && \
        cd MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64 && \
        ./mlnxofedinstall --user-space-only --without-fw-update --without-neohost-backend --force && \
        rm /tmp/MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64.tgz && \
        rm -rf /tmp/MLNX_OFED_LINUX-5.9-0.5.6.0.127-ubuntu22.04-x86_64 && \
        apt-get clean

# Display installed packages for verification
RUN pip list
