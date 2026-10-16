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
# =============================================================================
# =============================================================================
# vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile
#   DeepSeek-V4 Flash-FP8 / Pro-FP8 MoRI-EP WideEP disagg image (v0.29.0 vehicle).
#   PER-MODEL image, isolated from the base vllm_disagg_inference Dockerfile
#   and from glmv5.1, so DSV4 can pin its own vLLM/AITER/MoRI.
#   Builds the stack from source and bakes the DSV4 vLLM patches at build time.
#
#   PINS (kept in this header, not the filename; queried 2026-09-09):
#   - BASE  -> vllm/vllm-openai-rocm:v0.29.0
#              digest sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1
#              Hub amd64, pushed 2026-09-09. Tag 98dff2a81d747d1dba01a47f939f48c3526d4206.
#   - vLLM  -> 98dff2a81d747d1dba01a47f939f48c3526d4206 (tag v0.29.0)
#   - AITER -> ROCm/aiter main 10f8874dc2cd69c07ed84b5f125c27d12baccb10 (2026-09-09)
#              +229 over 1d872fa. Built FROM SOURCE. Not the UFB TheRock wheel.
#   - flydsl -> ==0.3.2  (AITER main setup.py FLYDSL_VERSION)
#   - MoRI  -> ROCm/mori main 07bdace2ff7306928871f85afd92f1d2aae13ad0 (2026-09-09)
#              +24 over 6fcf6b3. Nearest tag v1.2.3 = 879983bdbd8c (+a few on main).
#   - triton_kernels -> ROCm/triton @ 0f380657 (v0.29.0 Dockerfile.rocm, unchanged)
#   - vllm-router -> vllm-project/router f962dfcf26094530fbee37289784c37b2ca00ca6
#                    (2026-09-18T01:30:59Z, crate v0.1.15)
#                    feat(tokenizer): add bounded L0 exact-match encode cache (#270)
#                    Baked to /usr/local/bin/vllm-router. rustc 1.88.0 (6b00bc388 2025-06-23).
#   - DSV4 patches -> github.com/basemam/vllm e5d7c9a74e60e17bb06d03c03e3dc12e734c2249
#                     (branch v0.29.0-mad, mad/scripts/vllm_disagg/patches/)
#
#   Prebuilt image (internal, rocm/pytorch-private; no public equivalent yet):
#     rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk
#     The recipes in scripts/vllm_dissag/models.json pin sha256:49e87ada… (2026-09-21),
#     which every validation run used. It baked byte-identical patches from in-repo
#     copies. The tag was re-pushed from this file on 2026-10-01 as sha256:631be721…
#     (patches fetched from basemam/vllm e5d7c9a7). sha256:bb1104fa… (2026-09-19) is
#     router-baked and UNPATCHED — do not serve DSV4 on it. Always pin by digest.
#     /app/versions.txt must have VLLM_ROUTER_REF=f962dfcf,
#     DSV4_PATCHES_SRC=… e5d7c9a7… and
#     DSV4_PATCHES_BAKED=combine,trim,attn_backend,storage_span,mixed_bs,gate,attn_xfer,rdma_wait.
#
# Hub v0.29.0 is STILL ROCm 7.2.3 (docker/Dockerfile.rocm_base
# rocm/dev-ubuntu-22.04:7.2.3-complete). Hub AITER_BRANCH/MORI_BRANCH in
# /app/versions.txt are still v0.1.19 / v1.1.0 residue — trust AITER_REF /
# MORI_REF written below.
#
# "AITER 10.1" here means latest AITER *main* (the same tree UFB stamps as
# amd_aiter-0.1.22+rocm10.1.0a…). We do NOT pip-install that TheRock wheel onto
# this image: the wheel is built against TheRock ROCm 10.1, this Hub base is
# 7.2.3. Source-build AITER against the image torch instead.
#
# FlyDSL pin is STILL REQUIRED. AITER is installed --no-deps, so setup.py's
# flydsl==0.3.2 is not pulled. Hub leftover flydsl 0.1.8 is 217850
# (ImportError: expected >=0.2.4). AITER main _MIN_FLYDSL_VERSION is still
# 0.2.4; setup.py wants exactly 0.3.2. Do not drop the uninstall+assert.
#
# Build on a host with docker (not WSL, not a login node), from the repo root.
# --ulimit is required: cargo -j $nproc on a 100-core node hits EMFILE
# (os error 24 / Too many open files) compiling wasmtime. CARGO_BUILD_JOBS=8
# is the Dockerfile default; do not drop --ulimit (container hard nofile is 1024).
#   docker pull vllm/vllm-openai-rocm:v0.29.0
#   docker build --ulimit nofile=1048576:1048576 \
#     -f docker/vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:dsv4 .
#
# THE IMAGE IS THE CONTRACT, as for glmv5.1: MAD ships no runtime patchers. The DSV4
# patches run only at image build (step 6). Do not substitute a stock v0.29.0 image:
# it boots but fails the DSV4 MoRIIO KV transfer.
#
# Runtime switches read by the baked code (a DSV4 recipe must set these in its
# scripts/vllm_dissag/models.yaml env:):
#   DSV4_TRANSFER_ATTN   default 0 — a DSV4 recipe MUST set 1; 0 skips the full-attention
#                        KV transfer and long-context retrieval fails.
#   MORI_TRIM_DISPATCH   default 1 — sizes the expert GEMM to live tokens, not the full
#                        MoRI recv buffer (decode ITL ~24 ms vs ~290 ms at con=1, EP8).
# =============================================================================

ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.29.0
FROM ${BASE_IMAGE}

# Keep the Hub ENTRYPOINT. Jobs start with docker --entrypoint /bin/bash
# then vllm_disagg.sh (connector env; DSV4 patches baked below).
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32

# Hub release bakes LEGACY=1; MoRI RDMA needs dmabuf (same as moriio.env).
ENV HSA_ENABLE_IPC_MODE_LEGACY=0

# -----------------------------------------------------------------------------
# 1. MoRI main 07bdace2 (2026-09-09), +24 over v0280's 6fcf6b3.
#    Do NOT disable ionic/bnxt (ep:0 init deadlock on RoCE/mlx5).
#    grpc/protobuf keep UMBP ON (same as mori624002 / v0280).
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=07bdace2ff7306928871f85afd92f1d2aae13ad0
ARG MORI_NEAREST_TAG=v1.2.3
ENV MORI_GPU_ARCHS=gfx942
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    pip uninstall -y mori amd-mori amd_mori 2>/dev/null || true && \
    rm -rf /tmp/mori-src && \
    git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    pip install -r requirements-build.txt && \
    pip install . --no-build-isolation && \
    python3 -c "import mori, mori.ops as o; print('mori OK; kernels:', [n for n in dir(o.EpDispatchCombineKernelType) if not n.startswith('_')])" && \
    mkdir -p /app && echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
    echo "MORI_NEAREST_TAG=${MORI_NEAREST_TAG} (+$(git -C /tmp/mori-src rev-list --count ${MORI_NEAREST_TAG}..${MORI_REF} 2>/dev/null || echo '?') commits; tag is provenance only, MORI_REF is the pin)" >> /app/versions.txt && \
    echo "MORI_HEAD_DATE=$(git -C /tmp/mori-src log -1 --format=%cI ${MORI_REF})" >> /app/versions.txt && \
    echo "MORI_UMBP_DEPS=grpc+protobuf installed (protoc=$(command -v protoc || echo none), grpc_cpp_plugin=$(command -v grpc_cpp_plugin || echo none))" >> /app/versions.txt && \
    rm -rf /tmp/mori-src

# -----------------------------------------------------------------------------
# 2. AITER main 10f8874 from source + flydsl==0.3.2.
#    Do not use Hub 0.1.19. Keep Hub ROCm triton (AITER_USE_SYSTEM_TRITON=1).
#    Do not install UFB +rocm10.1.0a wheels (TheRock ABI; this base is 7.2.3).
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=10f8874dc2cd69c07ed84b5f125c27d12baccb10
ARG FLYDSL_PIN="flydsl==0.3.2"
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV AITER_USE_SYSTEM_TRITON=1
# Docker build has no GPU. get_gfx() defaults GPU_ARCHS=native and runs rocminfo.
# GPU_ARCHS is RUN-local so serve still uses the live GPU.
# Do not import aiter in the flydsl check (cwd /tmp/aiter-src + rocminfo).
RUN echo "Compiling AITER from source: ${AITER_REPO}@${AITER_REF}" && \
    rm -rf /tmp/aiter-src && \
    git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
    cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
    git submodule update --init --recursive && \
    export GPU_ARCHS=gfx942 CU_NUM=304 && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    pip install --no-build-isolation --no-deps -v . && \
    echo "AITER_REF=${AITER_REF}@$(git rev-parse HEAD) (built from source)" >> /app/versions.txt && \
    rm -rf /tmp/aiter-src && \
    pip show amd-aiter 2>/dev/null | head -5 && echo "AITER OK (from source)"
RUN pip uninstall -y flydsl 2>/dev/null || true && \
    pip install --no-deps -U "${FLYDSL_PIN}" && \
    python3 -c "\
import importlib.metadata as m, importlib.util, pathlib, flydsl; \
v=m.version('flydsl'); \
assert v=='0.3.2', 'flydsl %s != 0.3.2 (217850 class: Hub leftover; AITER main wants 0.3.2)' % v; \
spec=importlib.util.find_spec('aiter'); \
assert spec and spec.origin, spec; \
p=pathlib.Path(spec.origin).parent/'ops'/'flydsl'/'kernels'/'mqa_logits'/'fp8_mqa_logits.py'; \
assert p.is_file(), p; \
assert 'def flydsl_fp8_mqa_logits' in p.read_text(), p; \
print('flydsl', v, getattr(flydsl,'__version__',None), flydsl.__file__); \
print('aiter flydsl_fp8_mqa_logits source OK', p)" && \
    echo "FLYDSL_PIN=${FLYDSL_PIN}" >> /app/versions.txt && \
    python3 -c 'import importlib.metadata as m; print("FLYDSL_VER="+m.version("flydsl"))' >> /app/versions.txt
RUN rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache"

# -----------------------------------------------------------------------------
# 3. vLLM: same ref as BASE_IMAGE (tag v0.29.0). Full compile so the Hub
#    wheel is not the one that ends up imported.
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=98dff2a81d747d1dba01a47f939f48c3526d4206
ARG VLLM_TAG=v0.29.0
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS}
RUN rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_REF}" && \
    echo "VLLM_REF=${VLLM_REF}@$(git rev-parse HEAD) (tag ${VLLM_TAG})" >> /app/versions.txt && \
    test -f vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    rm -rf /tmp/vllm-src

# -----------------------------------------------------------------------------
# 4. triton_kernels wheel into site-packages (Flash MXFP4 import).
#    0f380657 is v0.29.0's own ROCM_TRITON_KERNELS_COMMIT.
# -----------------------------------------------------------------------------
ARG TRITON_KERNELS_REPO=https://github.com/ROCm/triton.git
ARG TRITON_KERNELS_REF=0f380657dbf3ee86eb57558ff71df24f03b5d4e7
RUN pip install -U build && \
    rm -rf /tmp/triton-src && \
    git clone --filter=blob:none "${TRITON_KERNELS_REPO}" /tmp/triton-src && \
    cd /tmp/triton-src && git checkout "${TRITON_KERNELS_REF}" && \
    test -f python/triton_kernels/triton_kernels/matmul_ogs.py && \
    cd python/triton_kernels && python3 -m build --wheel && \
    pip uninstall -y triton_kernels triton-kernels 2>/dev/null || true && \
    pip install --no-deps dist/*.whl && \
    python3 -c "from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig; \
pc = PrecisionConfig.__dataclass_fields__; \
assert 'weight_scale' in pc, list(pc); \
print('triton_kernels.matmul_ogs OK')" && \
    echo "TRITON_KERNELS_REF=${TRITON_KERNELS_REF}@$(git -C /tmp/triton-src rev-parse HEAD) (wheel)" >> /app/versions.txt && \
    rm -rf /tmp/triton-src

RUN echo "=== AITER check ===" && pip show amd-aiter 2>/dev/null | head -5 && \
    echo "=== flydsl check ===" && pip show flydsl && \
    python3 -c "import importlib.metadata as m, flydsl; v=m.version('flydsl'); assert v=='0.3.2', v; print('flydsl', v, 'OK', flydsl.__file__)" && \
    echo "=== MoRI check ===" && python3 -c "import mori; print('mori', mori.__version__)" && \
    python3 -c "import mori.io; print('mori.io OK')" && \
    python3 -c "import mori.ops; print('mori.ops OK')" && \
    echo "=== MoRI-IO connector present ===" && \
    python3 -c "import vllm, pathlib; \
p = pathlib.Path(vllm.__file__).parent/'distributed'/'kv_transfer'/'kv_connector'/'v1'/'moriio'/'moriio_connector.py'; \
assert p.is_file(), p; print('moriio_connector.py OK', p)" && \
    echo "=== triton_kernels check ===" && \
    python3 -c "import triton_kernels, triton_kernels.matmul_ogs as m; print(triton_kernels.__file__); print('matmul_ogs', m.__file__)" && \
    echo "Post-vLLM cross-check OK"

# -----------------------------------------------------------------------------
# 5. vllm-router baked in — /usr/local/bin/vllm-router, no NODE0 cargo.
#    Pin (not `main`): vllm-project/router
#      SHA  f962dfcf26094530fbee37289784c37b2ca00ca6
#      date 2026-09-18T01:30:59Z
#      msg  feat(tokenizer): add bounded L0 exact-match encode cache (#270)
#      crate v0.1.15
#    Same binary 436879/436880 git-booted. rustc 1.88.0 required (time/home).
#    PROXY_TYPE=vllm_router uses this; moriio_toy still the Python :10001 path.
#    436919 cargo died EMFILE (os error 24): nproc-wide rustc on a compute node.
#    Cap CARGO_BUILD_JOBS and raise nofile (see docker build --ulimit above).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/vllm-project/router.git
ARG ROUTER_REF=f962dfcf26094530fbee37289784c37b2ca00ca6
ARG ROUTER_HEAD_DATE=2026-09-18T01:30:59Z
ARG ROUTER_CRATE=0.1.15
ARG RUST_TOOLCHAIN=1.88.0
ARG CARGO_BUILD_JOBS=8
RUN if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    if command -v rustup >/dev/null 2>&1; then \
        rustup default "${RUST_TOOLCHAIN}" >/dev/null 2>&1 || rustup toolchain install "${RUST_TOOLCHAIN}"; \
    fi && \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    test "$(git rev-parse HEAD)" = "${ROUTER_REF}" && \
    grep -q "version = \"${ROUTER_CRATE}\"" Cargo.toml && \
    ulimit -n 1048576 || ulimit -n 65536 || true && \
    echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} nofile=$(ulimit -n)" && \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS}" cargo build --release -j "${CARGO_BUILD_JOBS}" && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    command -v vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_REPO=${ROUTER_REPO}" >> /app/versions.txt && \
    echo "VLLM_ROUTER_REF=${ROUTER_REF}@$(git rev-parse HEAD)" >> /app/versions.txt && \
    echo "VLLM_ROUTER_HEAD_DATE=$(git log -1 --format=%cI) (pin ${ROUTER_HEAD_DATE})" >> /app/versions.txt && \
    echo "VLLM_ROUTER_CRATE=${ROUTER_CRATE}" >> /app/versions.txt && \
    echo "VLLM_ROUTER_RUSTC=$(rustc --version)" >> /app/versions.txt && \
    echo "PROXY=vllm-router baked @ ${ROUTER_REF} (moriio_toy still available)" >> /app/versions.txt && \
    rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 6. Bake DSV4 source patches into the installed vLLM tree.
#    Combine, trim, attn backend, storage span, mixed-bs, transfer gate,
#    .attn WRITE, RDMA wait. Same order as the retired moriio.sh apply list.
#    Trim and .attn WRITE stay env-gated inside the patched sources.
#    The patch scripts live in the vLLM fork (branch cut from v0.29.0), not in
#    this repo. Only PATCHES_PATH is fetched (partial clone, depth 1).
#    PATCHES_REF is the pinned commit on branch v0.29.0-mad. A branch name also
#    works (--build-arg PATCHES_REF=v0.29.0-mad) but is not reproducible.
# -----------------------------------------------------------------------------
ARG PATCHES_REPO=https://github.com/basemam/vllm.git
ARG PATCHES_REF=e5d7c9a74e60e17bb06d03c03e3dc12e734c2249
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

ENV SKIP_RUNTIME_PATCH=1
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN echo "=== Build versions ===" && cat /app/versions.txt
