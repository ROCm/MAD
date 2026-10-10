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
# Kimi K3 day-0 support landed in sgl-project/sglang#32541 and is not in a
# tagged SGLang ROCm release yet; this model-specific image is the only ROCm
# build carrying the KDA / Stable LatentMoE / AITER A8W4 support the checkpoint
# needs. Kept separate from docker/pyt_sglang, which is still on the v0.4.5
# rocm630 base that the existing SGLang entry is validated against.
#
# The tag is the day-0 image named in the AMD tracking issue
# https://github.com/sgl-project/sglang/issues/32548, which is what its MI355X
# performance tables were measured against. A newer rocm720-mi35x-k3-20260728
# tag exists on Docker Hub but nothing published ties it to the recipe, so it is
# deliberately not adopted here.
#
# Fold this back into docker/pyt_sglang once K3 lands in a versioned ROCm image.
ARG BASE_DOCKER=lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727

FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# record configuration for posterity
RUN pip3 list
