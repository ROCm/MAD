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

# The vLLM and ATOM base images declare a serving ENTRYPOINT. madengine keeps
# its container alive by running `<image> cat`, which such an entrypoint takes
# as a serving argument instead of executing, so the benchmark's later
# `docker exec` calls never reach a running container.
ENTRYPOINT []

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# A recipe may declare `setup_script: pip-runtime-deps.sh`, which installs the
# packages its base image lacks before the engine starts -- DeepSeek-V4's vLLM
# base needs Pillow, fastapi and uvicorn, and the server dies on import without
# them. That script does nothing but `pip install $SETUP_PIP_PACKAGES`, so
# infx2mad.py folds it into the build and passes the list here. Doing it at
# build time rather than per run also keeps the install out of the measured
# window. The list is part of the image tag's fingerprint, so two scenarios
# needing different packages cannot share an image.
ARG INFX_PIP_PACKAGES=""
RUN if [ -n "$INFX_PIP_PACKAGES" ]; then \
        pip_opts=""; \
        python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages \
            && pip_opts="--break-system-packages"; \
        python3 -m pip install --quiet $pip_opts $INFX_PIP_PACKAGES; \
    fi

# On a cluster whose MAD workspace is an NFS export with root_squash, madengine
# has to run the container as the submitting user rather than root. AITER then
# JIT-builds kernels and writes its tuning cache into paths the image ships
# root-owned, so widen exactly those. /tmp keeps the sticky bit.
# When the cluster forces the container to run as the submitting user, that uid
# comes from sssd and has no passwd entry in the image. madengine probes the
# container with `whoami`, which then fails and aborts the run before the model
# starts, so give the uid a name at build time. Default 0 keeps the plain
# run-as-root case untouched.

# madengine starts the container with `--group-add render --group-add video`,
# and docker resolves those names against the IMAGE's group file. The vLLM
# nightly images carry neither, so the run dies with "Unable to find group
# render" before the model is reached.
#
# Creating the names is not enough: the supplementary group only grants access
# if its gid matches the compute node's, because /dev/dri/renderD* is bind-
# mounted with that node's numeric ownership and a container running as the
# submitting user has no other way in. So the gid is forced -- an image group
# at a different gid is moved, and -o allows the gid to be shared with a group
# that already holds it, which is harmless and beats silently using the wrong
# one. Default 0 means "leave this image's groups alone".
ARG RENDER_GID=0
ARG VIDEO_GID=0
RUN for spec in "render:$RENDER_GID" "video:$VIDEO_GID"; do \
        name="${spec%%:*}"; gid="${spec##*:}"; \
        [ "$gid" = "0" ] && continue; \
        if getent group "$name" >/dev/null; then \
            cur="$(getent group "$name" | cut -d: -f3)"; \
            [ "$cur" = "$gid" ] || groupmod -o -g "$gid" "$name"; \
        else \
            groupadd -o -g "$gid" "$name"; \
        fi; \
    done

ARG RUN_AS_UID=0
ARG RUN_AS_GID=0
RUN if [ "$RUN_AS_UID" != "0" ]; then \
        getent group "$RUN_AS_GID" >/dev/null || groupadd -g "$RUN_AS_GID" madrun; \
        getent passwd "$RUN_AS_UID" >/dev/null || \
            useradd -u "$RUN_AS_UID" -g "$RUN_AS_GID" -M -d /tmp -s /bin/bash madrun; \
    fi

# Replaces the image's rdma-core with one built from source. /dev/infiniband
# and /sys/class/infiniband being bind-mounted is not enough for RDMA to work
# in a container: the verbs provider has to speak the ABI the HOST kernel's
# driver reports, and Ruby's bnxt_re kernel module (a Broadcom Thor2 fork,
# `uname -r` carrying `_brcmrdma...`) reports an ABI this image's distro
# rdma-core does not support -- `ibv_devinfo` fails with "does not support the
# kernel ABI" and every RDMA transport (disagg's MoRI kv-transfer among them)
# reports no active device. The bare host's own rdma-core (built against this
# kernel) has no such problem, which is what pins the version below rather
# than leaving this to "whatever the image ships". Lifted from
# docker/primus_maxtext_rccl_overlay.ubuntu.amd.Dockerfile, which carries the
# same stage for the same Thor2 family of bnxt_re fixes. Empty = keep the
# base image's rdma-core, which is the right choice for anything not doing
# RDMA transport (P2P/IPC-only serving points measured in rccl_surface_atom_serving
# never needed this).
ARG RDMA_CORE_VERSION=
RUN if [ -n "$RDMA_CORE_VERSION" ]; then \
      set -e; \
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential pkg-config make \
        libnl-3-dev libnl-route-3-dev libudev-dev libssl-dev libsystemd-dev \
        python3-docutils pandoc; \
      rm -rf /var/lib/apt/lists/*; \
      git clone --depth 1 --branch "v${RDMA_CORE_VERSION}" https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core; \
      cd /tmp/rdma-core && git log -1 --format='%H %s'; \
      mkdir build && cd build; \
      cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
        -DCMAKE_INSTALL_SYSCONFDIR=/etc \
        -DCMAKE_INSTALL_RUNDIR=/run \
        -DCMAKE_BUILD_TYPE=Release \
        -DNO_PYVERBS=1 \
        ..; \
      ninja -j"$(nproc)"; \
      DEBIAN_FRONTEND=noninteractive apt-get purge -y python3-docutils pandoc; \
      DEBIAN_FRONTEND=noninteractive apt-get autoremove -y; \
      rm -rf /var/lib/apt/lists/*; \
      to_remove=""; \
      for p in ibverbs-providers libibverbs1 libibverbs-dev ibverbs-utils \
               librdmacm1 librdmacm-dev rdma-core libibumad3 libibmad5 \
               infiniband-diags; do \
        if dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then \
          to_remove="$to_remove $p"; \
        fi; \
      done; \
      if [ -n "$to_remove" ]; then \
        echo "removing distro rdma packages:$to_remove"; \
        dpkg -r --force-all $to_remove; \
      fi; \
      ninja install && ldconfig; \
      rm -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav*.so /usr/local/lib/libbnxt_re-rdmav*.so; \
      cd / && rm -rf /tmp/rdma-core; \
    else \
      echo "RDMA_CORE_VERSION empty -- keeping the base image's rdma-core"; \
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
