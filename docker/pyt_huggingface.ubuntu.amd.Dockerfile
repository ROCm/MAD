# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
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
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# numpy is reinstalled because of pandas compatibility issues, remove the lines below once base image moves to numpy>1.20.3
RUN pip3 install -U numpy
RUN pip3 install -U scipy
# Install huggingface transformers
RUN cd /workspace && git clone https://github.com/huggingface/transformers transformers &&\
    cd transformers &&\
    git show --oneline -s && \
    pip install -e .

# Install dependencies
RUN cd /workspace/transformers/examples/pytorch && pip3 install -r _tests_requirements.txt
RUN pip3 install --no-cache-dir GPUtil azureml azureml-core tokenizers ninja cerberus sympy sacremoses sacrebleu==1.5.1 sentencepiece scipy scikit-learn
# setting datasets==1.9.0 due to Roberta SQUAD change, https://github.com/huggingface/transformers/issues/12880
#RUN pip3 install --no-cache-dir datasets==1.9.0
# https://github.com/huggingface/datasets/issues/3099
RUN pip3 install -U huggingface_hub

# ROCm gpg key
RUN wget -q -O - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
RUN apt update && apt install -y \
    unzip \
    jq

# add sshpass, sshfs for downloading from mlse-nas
RUN apt-get install -y sshpass sshfs
RUN apt-get install -y netcat

# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# record configuration for posterity
RUN pip3 list


