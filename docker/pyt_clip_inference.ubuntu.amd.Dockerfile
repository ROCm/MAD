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

# Argument to check current GPU arch
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}

RUN apt-get update
RUN apt-get install -y \
    unzip \
    jq \
    git \
    vim \
    wget

# Update pip to latest version
RUN pip install --upgrade pip

# Install clip
# WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/mlfoundations/open_clip.git open_clip &&\
    cd open_clip &&\
    pip install -e . &&\
    cd ..

# WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/LAION-AI/CLIP_benchmark.git CLIP_benchmark &&\
    cd CLIP_benchmark &&\
    pip install -e . &&\
    cd ..

# Replace the original zeroshot_retrieval.py with this version
COPY pyt_clip_inference/zeroshot_retrieval.py CLIP_benchmark/clip_benchmark/metrics/zeroshot_retrieval.py

# Replace the original zeroshot_classification.py with this version
COPY pyt_clip_inference/zeroshot_classification.py CLIP_benchmark/clip_benchmark/metrics/zeroshot_classification.py
