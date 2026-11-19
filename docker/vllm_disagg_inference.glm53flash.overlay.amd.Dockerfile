# syntax=docker/dockerfile:1
# =============================================================================
# GLM-5.3-Flash-FP8 disaggregated (1P/1D over MoRIIO) — self-contained image.
#
# This is the PROVEN, shippable recipe: it starts FROM the verified base image
# and BAKES IN the 11 GLM-5.3-Flash disagg fixes as source files (no runtime
# overlay mounts). It reproduces the recall-correct stack verified live on a gold
# ionic rail pair — exact NIAH recall to 871K tokens, TP4 and EP8, MoRI WRITE
# mode (see ../scripts/vllm_dissag/glm53_flash/RESULTS.md).
#
# BUILD (context MUST be the recipe dir so `COPY patches/...` resolves):
#   docker build \
#     -f docker/vllm_disagg_inference.glm53flash.overlay.amd.Dockerfile \
#     -t rocmshared/vllm-glm53-flash:glm53-flash-disagg-overlays-v1 \
#     scripts/vllm_dissag/glm53_flash/
#
# The base image carries aiter-tip + the clr fix. This Dockerfile makes the image
# SELF-CONTAINED for ionic disagg by (a) rebuilding mori from the fork branch that
# carries the ionic atomic-MR strip + HIP-device-restore fixes (so the runtime no
# longer needs the MORI_PATCHED/shared_nfs .so mount), and (b) baking the 11
# sha256-verified vLLM/aiter overlays in-source. The 11 COPY sources are the exact
# overlays committed under patches/.
#
# Still host-provided at launch (node-specific, cannot be baked): GLIBC_SWAP=1
# HOSTLIBS=<glibc-2.39 closure> for the node's ionic libibverbs provider, and a
# warm aiter JIT cache (or AITER_KSPLIT=1 on a cold cache). See README.md.
# =============================================================================

ARG BASE_IMAGE=rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix
FROM ${BASE_IMAGE}

# --- stack provenance ---------------------------------------------------------
ARG BASE_IMAGE_DIGEST=sha256:2a359be8503efc2fd22c87a264bf0829bc23f098c4c8a0beefa41d0fe4ae1a6a
ARG AITER_REF=624e43586b                                    # aiter-tip + the 3 gfx950 overlays baked below
ARG PYTORCH_ROCM_ARCH=gfx950

# --- mori: rebuild from the fork with the ionic fixes (kills MORI_PATCHED) -----
# raviguptaamd/mori branch ionic-atomic-mr-strip = the 2 ionic C++ fixes rebased
# onto current ROCm/mori upstream (v1.2.3.post1): (1) strip IBV_ACCESS_REMOTE_ATOMIC
# from MR access flags when MORI_IO_DISABLE_ATOMIC_MR=1 (ionic reports IBV_ATOMIC_NONE,
# else ibv_reg_mr => EINVAL and KV transfer aborts); (2) restore the caller's HIP
# device across the RDMA backend's RegisterMemory/CreateSession (else the model's
# HIP context is left mutated -> next DSA indexer load fails HIP-209). Same commit
# is proposed upstream as a ROCm/mori PR. Override MORI_REPO/MORI_REF to build another.
ARG MORI_REPO=https://github.com/raviguptaamd/mori.git
ARG MORI_REF=b271ba2963782614f32f6707381ed99058d915a0      # ionic-atomic-mr-strip @ v1.2.3.post1-38
ENV MORI_GPU_ARCHS=gfx950 BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
# Bake the atomic strip ON so the image needs no runtime MORI_*ATOMIC* flag.
ENV MORI_IO_DISABLE_ATOMIC_MR=1
RUN set -e; \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true; \
    rm -rf /tmp/mori-src && \
    git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    BUILD_UMBP=OFF pip install . && \
    python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
    cd / && rm -rf /tmp/mori-src

# --- vllm-router: build from source into the image (kills the ROUTER_BIN mount) -
# raviguptaamd/router: pd-disaggregation router with moriio KV-connector + service
# discovery. Pinned to a sha for reproducible rebuilds. Installed to /usr/local/bin
# so the launcher finds it on PATH (no host-binary mount).
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=82dc9811af17412e6e24b5942a5486bc502df23a     # raviguptaamd/router (moriio pd-disagg)
ARG RUST_TOOLCHAIN=1.88.0
RUN set -e; \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi; \
    export PATH="/root/.cargo/bin:${PATH}"; \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    cargo build --release && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    cd / && rm -rf /tmp/vllm-router-src

ARG SP=/usr/local/lib/python3.12/dist-packages

# --- bake the 11 verified GLM-5.3-Flash disagg fixes in-source (as files) -----
# vLLM MoRIIO connector: per-group hybrid-KV WRITE-path routing + cross-chunk KV
# accumulation (CHUNKFIX) + HMA hybrid-KV support.
COPY patches/moriio_connector_hma.py   ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
COPY patches/moriio_engine.py          ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py
COPY patches/moriio_common.py          ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py
COPY patches/moriio_layout.py          ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py
# vLLM hybrid KDA(Mamba)+MLA KV-pool alloc + DSA sparse-indexer + GLM-5.3-Flash attention.
COPY patches/attn_utils.py             ${SP}/vllm/v1/worker/gpu/attn_utils.py
COPY patches/indexer.py                ${SP}/vllm/v1/attention/backends/mla/indexer.py
COPY patches/glm5next_attention.py     ${SP}/vllm/models/glm5next/nvidia/attention.py
# site-packages root: gfx950 inductor/dynamo guards.
COPY patches/usercustomize.py          ${SP}/usercustomize.py
# aiter gfx950 kernels (a8w8 blockscale GEMM, fused-MoE, batched a16wfp4 GEMM).
COPY patches/gemm_op_a8w8_tip.py       ${SP}/aiter/ops/gemm_op_a8w8.py
COPY patches/fused_moe_tip.py          ${SP}/aiter/fused_moe.py
COPY patches/batched_gemm_a16wfp4.py   ${SP}/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py

# --- gfx950 guards the overlays honor (also overridable at run time) ----------
ENV DISABLE_INDUCTOR_PM=1 \
    DISABLE_DYNAMO=1

# --- build-time sanity: overlays landed + CHUNKFIX present --------------------
# Static checks only: `import aiter` needs rocminfo/GPU, unavailable in the build
# sandbox. The live import + recall are the ship gate (GPU nodes, see RESULTS.md).
RUN set -e; \
    for f in \
      vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py \
      vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py \
      vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py \
      vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py \
      vllm/v1/worker/gpu/attn_utils.py \
      vllm/v1/attention/backends/mla/indexer.py \
      vllm/models/glm5next/nvidia/attention.py \
      usercustomize.py \
      aiter/ops/gemm_op_a8w8.py \
      aiter/fused_moe.py \
      aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py ; do \
      test -f ${SP}/$f || { echo "MISSING overlay: $f"; exit 1; }; \
    done; \
    grep -q CHUNKFIX ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py; \
    python3 -m py_compile ${SP}/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py; \
    echo "overlay sanity OK"; \
    python3 -c "import mori, mori.io, mori.ops" && echo "mori import OK"; \
    _MORI_DIR="$(python3 -c 'import mori,os;print(os.path.dirname(mori.__file__))')"; \
    if grep -rqa MORI_IO_DISABLE_ATOMIC_MR "$_MORI_DIR" 2>/dev/null; then \
      echo "mori atomic-MR strip present in built .so"; \
    else \
      echo "WARN: atomic-MR strip string not found in $_MORI_DIR (compiled out?); \
verify via live RegisterRdmaMemoryRegion (no errno 14) at the recall gate"; \
    fi

LABEL recipe="glm5.3-flash-disagg-overlays+mori" \
      verified_recall_tokens="871315" \
      mori_ref="${MORI_REF}" \
      base_image_digest="${BASE_IMAGE_DIGEST}"
