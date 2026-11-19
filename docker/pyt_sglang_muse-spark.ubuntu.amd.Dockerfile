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
# Standalone image for amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 (tag pyt_sglang_muse-spark-open-816b-a42b_fp4).
# MuseSparkForConditionalGeneration is only in the muse-spark SGLang branch, so the default
# docker/pyt_sglang base cannot load it. Otherwise identical to pyt_sglang.ubuntu.amd.Dockerfile.
# Keep BASE_DOCKER in sync with docker_base in scripts/sglang/configs/perf.yaml; fold this tag back
# into docker/pyt_sglang once the default base supports the architecture. gfx950 only.
ARG BASE_DOCKER=lmsysorg/sglang-rocm:muse-spark-open

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
