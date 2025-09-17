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
ARG BASE_DOCKER=docker.io/rocm/pytorch:latest
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/app
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx1100;gfx1101;gfx1200;gfx1201
RUN git clone ${FA_REPO}
RUN cd flash-attention \
    && git submodule update --init \
    && GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g') python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN pip install "huggingface_hub[cli]"
RUN pip install transformers
RUN pip install sentencepiece
RUN pip install ray
RUN pip install safetensors
# record configuration for posterity
RUN pip3 list
