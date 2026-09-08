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

# Primus JAX/MaxDiffusion launcher image for MAD (WAN 2.1 + FLUX.1-dev): bakes the Primus
# repo onto a JAX training base so scripts/jax-maxdiffusion/run.sh can run
# `train pretrain --config ...`.
#
# The base image owns the maxdiffusion stack: maxdiffusion is installed and patched at
# /workspace/maxdiffusion, at the same commit as Primus's third_party/maxdiffusion pin. So
# this image runs no setup_maxdiffusion_env.sh and installs no maxdiffusion deps. For a
# base without the stack, use the setup-script build from git history before this commit.
#
# Check Primus out first with tools/fetch_primus.sh. It is gitignored here and baked from
# the build context, which keeps git auth for a private repo out of the build. That script
# initializes no submodules: third_party/maxdiffusion at the same commit is unpatched, and
# run_pretrain.sh would select it over the base's tree if MAXDIFFUSION_PATH were ever unset.
#
# Build from the repo root, as madengine does for dockerfile paths containing "primus":
#   docker build -f docker/primus_maxdiffusion.ubuntu.amd.Dockerfile .

# madengine passes the base via docker_build_arg, which is how the v26.6 sweep put both
# maxtext and maxdiffusion on one unified CI image so their numbers share a toolchain.
ARG BASE_DOCKER=rocm/jax-training:maxtext-v26.6
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
# The Primus repo root, not /workspace: run.sh resolves examples/ relative to it.
ENV PRIMUS_ROOT=/workspace/Primus
# Pin the base's patched tree; run_pretrain.sh would otherwise default to
# $PRIMUS_ROOT/third_party/maxdiffusion and insert it at sys.path[0].
ENV MAXDIFFUSION_PATH=/workspace/maxdiffusion
# Transformer Engine must load only its JAX extension (torch is present too).
ENV NVTE_FRAMEWORK=jax
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

LABEL mad.launcher=primus

# The base may ship /workspace/Primus as a git clone, and COPY cannot replace a
# .git directory with a submodule checkout's .git file.
RUN rm -rf /workspace/Primus
COPY scripts/Primus/ /workspace/Primus/

RUN test -f /workspace/Primus/examples/run_pretrain.sh
RUN test -d /workspace/Primus/primus/backends/maxdiffusion \
    || (echo "ERROR: Primus checkout lacks primus/backends/maxdiffusion; use Primus main branch." >&2 && exit 1)

# Prove the base's stack is really there, so a wrong base fails the build instead
# of step 0 of a training run. The patch fixes a segfault on TE import order.
RUN python3 -c "import maxdiffusion, os; print('maxdiffusion ->', os.path.dirname(maxdiffusion.__file__))"
RUN grep -q "preload before Transformer Engine" /workspace/maxdiffusion/src/maxdiffusion/train_utils.py \
    || (echo "ERROR: /workspace/maxdiffusion is missing or lacks the TF-preload patch." >&2 && exit 1)

# Primus's own requirements, not maxdiffusion's, which the base already covers.
# Installed here rather than on every run: run.sh sets PRIMUS_SKIP_PIP=1 so a
# launch stays off the network. On this base it adds loguru.
RUN pip3 install --no-cache-dir -r /workspace/Primus/requirements-maxdiffusion.txt

RUN pip3 list 2>/dev/null || true
