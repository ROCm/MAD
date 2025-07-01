# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
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

ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Environment variables
ENV HIP_FORCE_DEV_KERNARG=1
ARG MAX_JOBS_ARG=192
ENV MAX_JOBS=${MAX_JOBS_ARG}

# Argument to check current GPU arch
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}

# Install necessary system dependencies (if any, e.g., git, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    numactl \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    pip install "huggingface_hub[cli]"

#For multigpu run, install following
RUN pip install "xfuser>=0.4.1"

# ROCm gpg key
RUN wget -q -O - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
RUN apt update && apt install -y \
    unzip \
    jq

# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# Install flash attention
ARG BUILD_FA="1"
ARG FA_BRANCH="v3.0.0.r1-cktile"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
RUN if [ "$BUILD_FA" = "1" ]; then \
    cd ${WORKSPACE_DIR} \
    && pip uninstall -y flash-attention \
    && rm -rf flash-attention \
    && git clone ${FA_REPO} \
    && cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=${HIP_ARCHITECTURES} python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl \
    && python -c "import flash_attn; print(f'Flash Attention version == {flash_attn.__version__}')"; \
    fi

#Download wan2.1 source code
RUN cd $WORKSPACE_DIR \
    && git clone https://github.com/Wan-Video/Wan2.1.git \
    && cd Wan2.1 \
    && sed -i '/\(torch\|torchvision\|flash_attn\)/d' requirements.txt \
    && pip install -r requirements.txt

# Display installed packages for verification
RUN pip list

