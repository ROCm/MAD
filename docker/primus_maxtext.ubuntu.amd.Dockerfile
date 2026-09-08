# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

# Primus JAX/MaxText launcher image for MAD: bakes the Primus repo onto a JAX training
# base so scripts/jax-maxtext/run.sh can run `train pretrain --config ...`.
#
# The base image owns the MaxText stack: it is installed at /workspace/maxtext, at the same
# commit as Primus's third_party/maxtext pin, so no submodule is needed here. This mirrors
# how primus_maxdiffusion takes maxdiffusion from the base.
#
# Check Primus out first with tools/fetch_primus.sh. It is gitignored here and baked from
# the build context, which keeps git auth for a private repo out of the build.
#
# Build from the repo root, as madengine does for dockerfile paths containing "primus":
#   docker build -f docker/primus_maxtext.ubuntu.amd.Dockerfile .

# madengine passes the base via docker_build_arg, which is how the v26.6 sweep put both
# maxtext and maxdiffusion on one unified CI image so their numbers share a toolchain.
ARG BASE_DOCKER=rocm/jax-training:maxtext-v26.6
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
# The Primus repo root, not /workspace: run.sh resolves examples/ relative to it.
ENV PRIMUS_ROOT=/workspace/Primus
# Pin the base's tree; prepare.py would otherwise default to
# $PRIMUS_ROOT/third_party/maxtext, which this image does not ship.
ENV MAXTEXT_PATH=/workspace/maxtext
RUN mkdir -p "$WORKSPACE_DIR"
WORKDIR $WORKSPACE_DIR

LABEL mad.launcher=primus

# The base may ship /workspace/Primus as a git clone, and COPY cannot replace a
# .git directory with a submodule checkout's .git file.
RUN rm -rf /workspace/Primus
COPY scripts/Primus/ /workspace/Primus/

RUN test -f /workspace/Primus/examples/run_pretrain.sh
RUN test -f /workspace/Primus/requirements-jax.txt

# Prove the base's stack is really there, so a wrong base fails the build instead
# of step 0 of a training run.
RUN test -f /workspace/maxtext/pyproject.toml \
    || (echo "ERROR: no MaxText at /workspace/maxtext. Use a base that bakes it, or point MAXTEXT_PATH at a checkout." >&2 && exit 1)

# Installed here rather than on every run: run.sh sets PRIMUS_SKIP_PIP=1 so a
# launch stays off the network. On this base it adds loguru, wandb, pre-commit.
RUN pip3 install --no-cache-dir -r /workspace/Primus/requirements-jax.txt

RUN pip3 list 2>/dev/null || true
