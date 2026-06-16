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
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Install necessary system dependencies (if any, e.g., git, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    numactl \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip

# numpy is reinstalled because of pandas compatibility issues, remove the lines below once base image moves to numpy>1.20.3
RUN pip3 install -U numpy
RUN pip3 install -U scipy

# Install pip-tools to compile the requirements.in file into requirements.txt
RUN pip install pip-tools

# ROCm gpg key
RUN wget -q -O - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt update && apt install -y \
    unzip \
    jq


# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# Clone the chai_lab repository
# Modify requirements.in to exclude torch and compile dependencies
# Install chai_lab without re-installing torch
RUN cd $WORKSPACE_DIR && \
ARG CHAI_BRANCH ="main",
    git clone --branch ${CHAI_BRANCH} https://github.com/chaidiscovery/chai-lab chai-lab && \
    cd chai-lab && \

    # Removing old branch v0.4.4, due to rdkit version incompatability
    sed '/torch/d' requirements.in > requirements.temp && \
    mv requirements.temp requirements.in && \
    pip-compile requirements.in && \
    pip install -r requirements.txt && \
    pip install -e .

# Display installed packages for verification
RUN pip list
