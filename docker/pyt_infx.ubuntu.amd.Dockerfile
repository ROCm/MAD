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
# Image for configs generated from InferenceX e2e scenarios by tools/infx2mad.py.
#
# Engine-agnostic on purpose: an InferenceX scenario may name an SGLang, vLLM or
# ATOM image, and this file only adds what a MAD/SLURM run needs on top of
# whichever one it is.
#
# There is deliberately no default BASE_DOCKER: each InferenceX scenario names
# the image its numbers were measured against, and infx2mad.py writes that image
# into the manifest's docker_build_arg. A default here would let a scenario run
# silently on an image it was never measured on, which is the one failure this
# whole path exists to avoid.
ARG BASE_DOCKER

FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# On a cluster whose MAD workspace is an NFS export with root_squash, madengine
# has to run the container as the submitting user rather than root. AITER then
# JIT-builds kernels and writes its tuning cache into paths the image ships
# root-owned, so widen exactly those. /tmp keeps the sticky bit.
# When the cluster forces the container to run as the submitting user, that uid
# comes from sssd and has no passwd entry in the image. madengine probes the
# container with `whoami`, which then fails and aborts the run before the model
# starts, so give the uid a name at build time. Default 0 keeps the plain
# run-as-root case untouched.
ARG RUN_AS_UID=0
ARG RUN_AS_GID=0
RUN if [ "$RUN_AS_UID" != "0" ]; then \
        getent group "$RUN_AS_GID" >/dev/null || groupadd -g "$RUN_AS_GID" madrun; \
        getent passwd "$RUN_AS_UID" >/dev/null || \
            useradd -u "$RUN_AS_UID" -g "$RUN_AS_GID" -M -d /tmp -s /bin/bash madrun; \
    fi

ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton
RUN mkdir -p /tmp/aiter_configs /tmp/torchinductor /tmp/triton \
 && chmod 1777 /tmp \
 && chmod -R a+rwX /tmp/aiter_configs /tmp/torchinductor /tmp/triton \
 # The JIT cache lives in a different place per engine image; widen whichever
 # one this base actually has.
 && for d in /sgl-workspace/aiter/aiter/jit /opt/aiter/aiter/jit /usr/local/lib/aiter/jit; do \
        if [ -d "$d" ]; then mkdir -p "$d/build"; chmod -R a+rwX "$d"; fi; \
    done \
 # torch._inductor probes for a CUDA compiler by exec'ing the bare name `nvcc`.
 # There is none in a ROCm image, and as root the lookup ends in
 # FileNotFoundError, which inductor handles. As a non-root user a root-only
 # directory on PATH turns the same lookup into PermissionError, which it does
 # not handle, and cuda-graph capture dies. Make PATH searchable for everyone.
 && chmod -R a+rX /root || true

# record configuration for posterity
RUN pip3 list
