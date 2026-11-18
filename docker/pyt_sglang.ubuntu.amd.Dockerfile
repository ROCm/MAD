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
# NOTE (arch dependency): the sglang base image is arch-specific. The default below is the MI35X
# (gfx950) build -- its sgl_kernel custom ops are AOT-compiled for gfx950 ONLY (GPU_ARCH_LIST=gfx950),
# so running it on gfx942 (MI300X) segfaults the moment SGLang invokes an sgl_kernel op (verified:
# torch is multi-arch and base matmul works, but sgl_kernel .so contains gfx950 code only). For a
# gfx942 target, override BASE_DOCKER with the matching MI30X base, e.g.:
#   --build-arg BASE_DOCKER=<lmsysorg/sglang mi30x image of the same SGLang version as the default>
# Builds should select the base by the target runner arch (MI300X vs MI355X).
ARG BASE_DOCKER=lmsysorg/sglang:v0.5.22-rocm10-mi35x

FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
# huggingface_hub 1.x CLI uses `suggest_commands` which requires typer>=0.17.0;
# the base image ships with typer 0.16.1 causing `hf` to crash on import.
RUN pip install "typer>=0.17.0"

# FID accuracy benchmark dependencies (pytorch-fid, pillow)
RUN pip install --no-cache-dir pytorch-fid pillow

# record configuration for posterity
RUN pip3 list
