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
# FLUX.1-dev / FLUX.2-dev image. When docker/pyt_sglang moved to
# lmsysorg/sglang:v0.5.18-rocm720-mi35x (#55), the FLUX entry in
# scripts/sglang/configs/perf.yaml was deliberately held on the previous base via
# docker_base. run_sglang.py does not read docker_base, so this model-specific
# image carries that base instead. Keep BASE_DOCKER in sync with the FLUX entry
# in perf.yaml.
#
# MI35X (gfx950) only: sgl_kernel in this base is AOT-compiled for gfx950, so the
# tags using this image set skip_gpu_arch gfx942 (see docker/pyt_sglang).
#
# Everything below the base matches docker/pyt_sglang (including the FID
# accuracy dependencies). Fold this back into docker/pyt_sglang once FLUX is
# validated on its base.
ARG BASE_DOCKER=lmsysorg/sglang-rocm:v0.5.15-rocm720-mi35x-20260712

FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
# huggingface_hub 1.x CLI uses `suggest_commands` which requires typer>=0.17.0;
# the base image ships with typer 0.16.1 causing `hf` to crash on import.
RUN pip install "typer>=0.17.0"

# --- Profiling-only ROCR runtime (STAGED here; ACTIVATED in scripts/sglang/run.sh under --profile) ---
# rocm-trace-lite in `--mode full` injects profiling signals into EVERY HSA queue submission,
# including the batch (count>1) AQL packets that a CUDA-graph *replay* submits during decode.
# That injection is only memory-safe if the ROCR runtime carries rocm-systems commit 559d48b1;
# without it the queue interceptor's packet copy reads out of bounds and SEGFAULTs
# (rocm-trace-lite/src/hsa_intercept.cpp:queue_intercept_cb). The lmsysorg base is ROCm 7.2.0
# (hsa-rocr 1.18.0.70200), which LACKS the fix -> SGLang's full-CUDA-graph decode crashes the
# instant tracing is on. The fix shipped in ROCm 7.2.3 (hsa-rocr 1.18.0.70203); vLLM profiles green
# on 7.2.3 with the IDENTICAL `rtl --mode full` + CUDA graphs. lmsysorg publishes no rocm723 mi35x
# base, so we swap ONLY the hsa-rocr runtime .so to the 7.2.3 build (patch within soname 1.18.0 ->
# ABI-compatible). Keeps CUDA graphs ON (no enforce_eager, no mode change).
#
# SCOPING (why staged, not activated here): the segfault ONLY occurs under profiling
# (`rtl trace --mode full`). Activating 7.2.3 in the image would change the runtime for
# benchmark/accuracy runs too. So we DOWNLOAD + EXTRACT it ONCE at build (cached layer) and STAGE the
# .so under /opt/rocr723 WITHOUT touching the active 7.2.0 runtime; scripts/sglang/run.sh copies it
# over the active lib dir ONLY when invoked with --profile, before building rtl (rtl links the active
# runtime at run time). Staging at build (vs a run-time download) avoids adding a per-profiling-run
# network fetch to the critical path.
# We do NOT `dpkg -i`: this base uses versioned ROCm dirs (/opt/rocm -> /opt/rocm-7.2.0 via
# /etc/alternatives) and the stock package installs into a SEPARATE /opt/rocm-7.2.3/lib while removing
# the active 7.2.0 libhsa-runtime64.so.1 -> breaks the runtime (verified on cv350). Staging the raw
# .so and copying it in-place (run.sh) avoids that.
# Validated cv350 (gfx950) A/B: baseline .70200 SEGFAULTs on the first cuda-graph decode; with .70203
# swapped in the server serves prefill+decode (cuda graph ON) with no crash and emits a valid payload.
ARG INSTALL_HSA_ROCR723=false
RUN set -eux; \
    if [ "$INSTALL_HSA_ROCR723" = "true" ]; then \
        cd /tmp; \
        if ! command -v curl >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*; fi; \
        curl -fsSL -O https://repo.radeon.com/rocm/apt/7.2.3/pool/main/h/hsa-rocr/hsa-rocr_1.18.0.70203-90~22.04_amd64.deb; \
        dpkg-deb -x hsa-rocr_1.18.0.70203-90~22.04_amd64.deb /tmp/rocr723; \
        mkdir -p /opt/rocr723; \
        cp -avf /tmp/rocr723/opt/rocm-*/lib/libhsa-runtime64.so* /opt/rocr723/; \
        rm -rf /tmp/rocr723 /tmp/hsa-rocr_1.18.0.70203-90~22.04_amd64.deb; \
        ls -l /opt/rocr723/libhsa-runtime64.so.1; \
    else \
        echo "Skipping ROCR 7.2.3 staging (INSTALL_HSA_ROCR723=$INSTALL_HSA_ROCR723)"; \
    fi

# FID accuracy benchmark dependencies (pytorch-fid, pillow)
RUN pip install --no-cache-dir pytorch-fid pillow

# record configuration for posterity
RUN pip3 list
