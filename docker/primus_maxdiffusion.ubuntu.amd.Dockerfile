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

# Primus's FLUX config still names the third-party Flax mirrors of CLIP-L and
# T5-XXL, which hold no PyTorch weights and so cannot be read once the text
# encoders run under Torchax. Repoint it at the text_encoder and text_encoder_2
# subfolders of the official black-forest-labs/FLUX.1-dev repo, the same weights
# maxdiffusion's own base_flux_dev.yml names. Same model, first-party copy.
# Submitted to Primus separately; this comes out when the submodule pin carries it.
COPY docker/patches/primus-flux-torch-text-encoders.patch /tmp/
RUN cd /workspace/Primus && git apply /tmp/primus-flux-torch-text-encoders.patch
RUN test -d /workspace/Primus/primus/backends/maxdiffusion \
    || (echo "ERROR: Primus checkout lacks primus/backends/maxdiffusion; use Primus main branch." >&2 && exit 1)

# Prove the base's stack is really there, so a wrong base fails the build instead
# of step 0 of a training run. The patch fixes a segfault on TE import order.
RUN python3 -c "import maxdiffusion, os; print('maxdiffusion ->', os.path.dirname(maxdiffusion.__file__))"
RUN grep -q "preload before Transformer Engine" /workspace/maxdiffusion/src/maxdiffusion/train_utils.py \
    || (echo "ERROR: /workspace/maxdiffusion is missing or lacks the TF-preload patch." >&2 && exit 1)

# The base pins transformers 4.57.3, which carries CVE-2026-4372, CVE-2026-5241
# and CVE-2026-9856; 5.10.0 is the first release fixing all three. v5 dropped
# every Flax implementation, including the FlaxCLIPTextModel and
# FlaxT5EncoderModel that FLUX's text encoders imported, so the patches below
# run FLUX's PyTorch CLIP-L and T5-XXL encoders under JAX through Torchax
# instead. The transformers upgrade itself comes after Primus's requirements,
# further down, because those pin the vulnerable version.
#
# Both patches are rebased onto the maxdiffusion commit Primus pins at
# third_party/maxdiffusion, which is the commit the base's tree is checked out
# at. The rev-parse guard makes a pin bump fail the build loudly rather than
# apply these to a tree they were never rebased onto.
ARG MAXDIFFUSION_COMMIT=68e069659f0af80694559e29939a17f879fe7f6a
COPY docker/patches/maxdiffusion-pin-transformers5-compat.patch /tmp/
COPY docker/patches/maxdiffusion-flux-transformers5.patch /tmp/
RUN cd $MAXDIFFUSION_PATH && \
    if [ "$(git rev-parse HEAD)" != "${MAXDIFFUSION_COMMIT}" ]; then \
      echo "ERROR: base ships maxdiffusion $(git rev-parse HEAD), patches expect ${MAXDIFFUSION_COMMIT}." >&2; \
      exit 1; \
    fi && \
    git apply /tmp/maxdiffusion-pin-transformers5-compat.patch && \
    git apply /tmp/maxdiffusion-flux-transformers5.patch && \
    pip3 install -e . --no-deps

# Primus's own requirements, not maxdiffusion's, which the base already covers.
# Installed here rather than on every run: run.sh sets PRIMUS_SKIP_PIP=1 so a
# launch stays off the network. On this base it adds loguru.
RUN pip3 install --no-cache-dir -r /workspace/Primus/requirements-maxdiffusion.txt

# After Primus's requirements on purpose: requirements-maxdiffusion.txt pins
# transformers==4.57.3, so upgrading any earlier just gets downgraded straight
# back onto the CVEs. The upgrade also pulls huggingface_hub 1.x over the base's
# 0.36.2, which is why the compat patch renames use_auth_token -> token.
#
# torchax goes in with --no-deps, or it pulls its own torch and jax over the
# ROCm builds this base ships. It is version-pinned because it dispatches
# against private torch overloads: 0.0.13 imports cleanly on this base's torch
# 2.12.0, but breaks on torch >= 2.14, which dropped the Dimname overloads it
# references at import time.
RUN pip3 install --no-cache-dir "transformers>=5.10.0" && \
    pip3 install --no-cache-dir --no-deps "torchax==0.0.13"

# Ship a vulnerable image never again silently: fail the build if anything above
# reorders, or adds another requirements file that pins transformers back down.
RUN python3 -c "import transformers; from packaging.version import Version; \
v = Version(transformers.__version__); \
assert v >= Version('5.10.0'), f'transformers {v} still carries CVE-2026-4372, CVE-2026-5241 and CVE-2026-9856'; \
print('transformers', v, 'OK')"

RUN pip3 list 2>/dev/null || true
