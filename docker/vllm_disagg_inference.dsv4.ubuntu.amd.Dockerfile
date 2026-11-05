# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
#################################################################################
# =============================================================================
# vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile
#   DeepSeek-V4 Flash/Pro MoRI-EP WideEP disagg image on the upstream ROCm 7.2
#   nightly at vLLM 83c078d (2026-10-23). DSV4 needs its own pins, so it does
#   not use the shared vllm_disagg_inference.ubuntu.amd.Dockerfile.
#
#   PINS (queried 2026-10-23):
#   - BASE  -> vllm/vllm-openai-rocm:nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370
#              upstream docker/Dockerfile.rocm_72 output, pushed 2026-10-23.
#              vLLM is installed from 83c078d (49 behind main that day; nearest
#              tag v0.32.1rc0). Nothing is rebuilt from source here:
#              ROCm 7.2.3, torch ROCm/pytorch 733fca1, triton 669b31a,
#              AITER v0.1.26, flydsl 0.4.0,
#              triton_kernels 669b31ac (bundled as vllm.third_party.triton_kernels),
#              MoRI v1.2.3.post1, AINIC + bnxt NIC libs  <- all from the base.
#   - vllm-router -> vllm-project/router 37b37a43c577cd7d83bad7b10b8b4650325b4fc7
#              (crate 0.1.16, rustc 1.95), baked to /usr/local/bin/vllm-router.
#   - DSV4 patches -> PATCHES_REPO @ PATCHES_REF (a vLLM branch cut from
#              83c078d), mad/scripts/vllm_disagg/patches/.
#              Four patches, baked by bake.sh:
#                0001 SWA MLA bounded_replay drop   (vllm-project/vllm#63411)
#                0002 MoRI dispatch trim            (vllm-project/vllm#63412)
#                apply_moriio_dsv4_attn_backend_fix.py   (vllm-project/vllm#62311)
#                apply_moriio_dsv4_mixed_block_size_fix.py
#              Drop each one as its upstream PR lands in the base nightly.
#
#   /app/versions.txt must have VLLM_ROUTER_REF=37b37a43,
#   DSV4_PATCHES_SRC=… <PATCHES_REF> … and
#   DSV4_PATCHES_BAKED=bounded_replay,trim,attn_backend,mixed_bs.
#
# Build from the repo root on a GPU node. --ulimit is required: cargo hits
# EMFILE (os error 24) under the default container nofile of 1024.
#   docker build --ulimit nofile=1048576:1048576 \
#     -f docker/vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:dsv4 .
#   docker push <your-registry>/vllm-disagg:dsv4
# =============================================================================

ARG BASE_IMAGE=vllm/vllm-openai-rocm:nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

# Jobs start with docker --entrypoint /bin/bash then vllm_disagg.sh.

# Some clusters block plain-HTTP Ubuntu mirrors; HTTPS works everywhere.
# libssl-dev + pkg-config are for the router's openssl-sys.
RUN sed -i -E 's#http://(archive|security)\.ubuntu\.com#https://\1.ubuntu.com#g' \
        /etc/apt/sources.list $(ls /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null) && \
    apt-get update -q -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        git curl ca-certificates libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /app && \
    echo "BASE_IMAGE=${BASE_IMAGE}" >> /app/versions.txt && \
    python3 -c 'import vllm, torch; print(f"VLLM={vllm.__version__} TORCH={torch.__version__} HIP={torch.version.hip}")' >> /app/versions.txt && \
    python3 -c 'import importlib.util as u; m = [n for n in ("mori", "aiter", "vllm.third_party.triton_kernels") if u.find_spec(n) is None]; assert not m, m' && \
    pip list 2>/dev/null | grep -iE '^(amd[-_])?aiter |^flydsl |^mori |^triton |^triton[-_]kernels |^torch |^vllm ' | sed 's/^/BASE_PKG /' >> /app/versions.txt

# -----------------------------------------------------------------------------
# 1. vllm-router baked in -> /usr/local/bin/vllm-router.
#    The router builds vLLM's rust tokenizer crate, which needs
#    str::floor_char_boundary (rustc >= 1.91).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/vllm-project/router.git
ARG ROUTER_REF=37b37a43c577cd7d83bad7b10b8b4650325b4fc7
ARG ROUTER_CRATE=0.1.16
ARG RUST_TOOLCHAIN=1.95
ARG CARGO_BUILD_JOBS=8
RUN if ! command -v rustup >/dev/null 2>&1 && [ ! -x /root/.cargo/bin/rustup ]; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    rustup toolchain install "${RUST_TOOLCHAIN}" && \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    test "$(git rev-parse HEAD)" = "${ROUTER_REF}" && \
    grep -q "version = \"${ROUTER_CRATE}\"" Cargo.toml && \
    { ulimit -n 1048576 || ulimit -n 65536 || true; } && \
    echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} nofile=$(ulimit -n)" && \
    cargo +"${RUST_TOOLCHAIN}" build --release --locked -j "${CARGO_BUILD_JOBS}" && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_REPO=${ROUTER_REPO}" >> /app/versions.txt && \
    echo "VLLM_ROUTER_REF=${ROUTER_REF}@$(git rev-parse HEAD)" >> /app/versions.txt && \
    echo "VLLM_ROUTER_CRATE=${ROUTER_CRATE}" >> /app/versions.txt && \
    echo "VLLM_ROUTER_RUSTC=$(rustc +${RUST_TOOLCHAIN} --version)" >> /app/versions.txt && \
    cd / && rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 2. Bake the four DSV4 patches into the installed vLLM tree.
#    The patches live in the vLLM fork on a branch cut from the base image's
#    own vLLM commit (83c078d), so they are tested against exactly this tree.
#    PATCHES_REF is the pinned commit on branch 83c078d-mad; only
#    PATCHES_PATH is fetched (partial clone, depth 1).
# -----------------------------------------------------------------------------
ARG PATCHES_REPO=https://github.com/basemam/vllm.git
ARG PATCHES_REF=64fe5a00b544cdf574f20b1591e0d91e9c54415f
ARG PATCHES_PATH=mad/scripts/vllm_disagg/patches
RUN rm -rf /tmp/vllm-mad /opt/dsv4_patches && \
    git init -q /tmp/vllm-mad && cd /tmp/vllm-mad && \
    git remote add origin "${PATCHES_REPO}" && \
    git fetch -q --depth 1 --filter=blob:none origin "${PATCHES_REF}" && \
    git checkout -q FETCH_HEAD -- "${PATCHES_PATH}" && \
    test -f "${PATCHES_PATH}/bake.sh" && \
    mkdir -p /opt/dsv4_patches && cp -a "${PATCHES_PATH}/." /opt/dsv4_patches/ && \
    echo "DSV4_PATCHES_SRC=${PATCHES_REPO} ${PATCHES_REF}@$(git rev-parse FETCH_HEAD) ${PATCHES_PATH}" >> /app/versions.txt && \
    rm -rf /tmp/vllm-mad
RUN bash /opt/dsv4_patches/bake.sh

# -----------------------------------------------------------------------------
# Runtime env. The base sets HSA_ENABLE_IPC_MODE_LEGACY=1 for UCX rocm_ipc;
# the MoRIIO PD path was validated with 0.
# -----------------------------------------------------------------------------
ENV SKIP_RUNTIME_PATCH=1 \
    HSA_ENABLE_IPC_MODE_LEGACY=0 \
    AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN echo "=== Build versions ===" && cat /app/versions.txt
