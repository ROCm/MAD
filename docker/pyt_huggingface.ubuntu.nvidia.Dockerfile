# CONTEXT {'gpu_vendor': 'NVIDIA', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
ARG BASE_DOCKER=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
FROM $BASE_DOCKER
USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    unzip \
    jq \
    python3-pip \
    git \
    vim \
    wget

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
# pinning MINICONDA_VER from "latest" to "py311_24.1.2-0" due to "pkgutil has no attribute ImpImporter" issue
RUN export MINICONDA_VER="py311_24.1.2-0" && \
wget https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh -b && \
    rm -rf Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh

RUN conda --version && \
    conda init
RUN pip3 install --upgrade pip
RUN pip3 install typing-extensions
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

# Install huggingface transformers
RUN cd /workspace && git clone https://github.com/huggingface/transformers transformers &&\
    cd transformers &&\
    git show --oneline -s && \
    pip install -e .

# Install dependencies
RUN cd /workspace/transformers/examples/pytorch && pip3 install -r _tests_requirements.txt

RUN pip3 install protobuf==3.20.*

# Install dependencies
RUN pip3 install --no-cache-dir GPUtil azureml azureml-core tokenizers ninja cerberus sympy sacremoses sacrebleu==1.5.1 sentencepiece scipy scikit-learn
# setting datasets==1.9.0 due to Roberta SQUAD change, https://github.com/huggingface/transformers/issues/12880
#RUN pip3 install --no-cache-dir datasets==1.9.0
# https://github.com/huggingface/datasets/issues/3099
RUN pip3 install -U huggingface_hub

RUN apt update && apt install -y \
    unzip \
    jq

# add sshpass, sshfs for downloading from mlse-nas
RUN apt-get install -y sshpass sshfs
RUN apt-get install -y netcat

RUN pip3 install setuptools==59.5.0
# record configuration for posterity
RUN pip3 list
