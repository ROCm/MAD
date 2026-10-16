# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# GLM-5.3-Flash-FP8 disaggregated (1P/1D) serving image for AMD Instinct gfx942
# (MI300X 192 GB and MI325X 256 GB — same image, same recipe).
#
# Every component is compiled from a pinned commit. The GLM-5.3 gfx942 fixes are
# commits on these fork branches — there is no patch overlay in this image:
#   vLLM    raviguptaamd/vllm    glm53-flash-gfx942  upstream-based glm5next + MoRIIO, plus 4 ROCm DSA/MoRI-EP
#                                                    fixes, upstream #57192 (ROCm TileLang JIT warmup) and #56734
#                                                    (MRV2 spec-decode slot mapping of padded rows), #57317 (no
#                                                    generic slot mapping for the kpool tail ring), then the
#                                                    GLM-5.3 gfx942 fixes (DSA indexer slot mapping,
#                                                    PD read barriers, MoRIIO kbpb transfer offsets, cudagraph ops)
#   aiter   raviguptaamd/aiter   glm53-flash-gfx942  ROCm/aiter b50066a9 + fp8_mqa_logits gfx942 LDS fit + int64
#                                                    paged-MQA-logits offsets + ROCm/aiter#3913 (gfx942 flydsl FP8
#                                                    MQA logits: the DSA-indexer prefill kernel, ~10x faster TTFT)
#   MoRI    ROCm/mori            78b7a5f3            upstream tip, unmodified
#   router  raviguptaamd/router  glm53-flash-gfx942  vllm-project/router main + PR#313 (per-DP-rank prefill
#                                                    routing) + dpfix (cross-pod DP KV-notify)
#
# Self-contained (no COPY), so it builds from any directory:
#   docker build --ulimit nofile=1048576:1048576 \
#     -f vllm_disagg_inference.glmv53flash.gfx942.ubuntu.amd.Dockerfile -t <image> .
# Serving is launched by the MAD vllm_disagg harness (models.yaml entry GLM-5.3-Flash-FP8-gfx942).
# =============================================================================

ARG BASE_IMAGE=rocm/vllm-dev:ci_base-build-01a0d6a8-b743-4bb5-802c-af6a7d97bf88@sha256:bd7f0c9b672659e6a0cbd9c110858b1a9e509229f7546d7fc42a52bdc57ae7e3
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
# The base exports an empty MAX_JOBS (vLLM's setup.py int() on it crashes the build), and an
# inherited ENV shadows an ARG of the same name — so the job count comes in as BUILD_JOBS.
ARG BUILD_JOBS=32
ARG VLLM_DIST=/usr/local/lib/python3.12/dist-packages/vllm

# -----------------------------------------------------------------------------
# 1. MoRI (ROCm/mori upstream). Keep the NIC backends enabled: turning off
#    USE_IONIC/USE_BNXT deadlocks cross-node EP all-to-all init. UMBP needs gRPC,
#    which this base lacks, and is unrelated to EP/IO.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=78b7a5f311c6e2ef4e3b929b7741ba3fed0d2fcf
ENV MORI_GPU_ARCHS=gfx942
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true && \
    git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    BUILD_UMBP=OFF pip install . && \
    python3 -c "import mori, mori.io, mori.ops" && \
    echo "MORI=${MORI_REPO}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
    rm -rf /tmp/mori-src

# -----------------------------------------------------------------------------
# 2. AITER. flydsl must match aiter's setup.py pin (it asserts the version).
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/raviguptaamd/aiter.git
ARG AITER_REF=6f65fba05b5dc51e8c7d786dcfec171769d8caab
ARG FLYDSL_VERSION=0.2.2
ENV GPU_ARCHS=${GFX_COMPILATION_ARCH} PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
RUN git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
    cd /tmp/aiter-src && git checkout "${AITER_REF}" && git submodule update --init --recursive && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    pip install --no-deps -U "flydsl==${FLYDSL_VERSION}" && \
    pip install --no-build-isolation --no-deps -v . && \
    echo "AITER=${AITER_REPO}@$(git rev-parse HEAD) flydsl==${FLYDSL_VERSION}" >> /app/versions.txt && \
    rm -rf /tmp/aiter-src /opt/vllm_cache/aiter_jit /root/.aiter

# -----------------------------------------------------------------------------
# 3. vLLM (full source compile; the base ships no vLLM).
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
ARG VLLM_REF=6a0cb9395797a124c594484e342989ed9e4f2f23
ENV VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} MAX_JOBS=${BUILD_JOBS}
RUN test -n "${MAX_JOBS}" && git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git -c advice.detachedHead=false checkout "${VLLM_REF}" && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm, vllm.models.glm5next; print('vLLM', vllm.__version__)" && \
    echo "VLLM=${VLLM_REPO}@$(git rev-parse HEAD)" >> /app/versions.txt && \
    rm -rf /tmp/vllm-src

# -----------------------------------------------------------------------------
# 4. Verify: MoRI + AITER survived the vLLM install, and every GLM-5.3 fix is in
#    the installed tree (located without importing aiter, which probes the GPU).
#    Fails the build if anything is missing.
# -----------------------------------------------------------------------------
RUN set -eu; \
    python3 -c "import mori, mori.io, mori.ops"; \
    pip show amd-aiter >/dev/null 2>&1 || pip show aiter >/dev/null; \
    A=$(python3 -c "import importlib.util as u; print(u.find_spec('aiter').submodule_search_locations[0])"); \
    V=$(python3 -c "import importlib.util as u; print(u.find_spec('vllm').submodule_search_locations[0])"); \
    M=$V/distributed/kv_transfer/kv_connector/v1/moriio; \
    miss=0; \
    for c in \
      "$V/tilelang_utils/__init__.py|class _DeferredTileLangJitKernel" \
      "$V/v1/worker/gpu/block_table.py|is_real_req = req_state_idx >= 0" \
      "$V/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py|GLM53_GDN_HASATTR_FIX" \
      "$V/models/common/ops/fused_qk_rmsnorm.py|GLM53_FUSED_QK_RMSNORM_OP" \
      "$V/v1/attention/backends/mla/indexer.py|GLM53_KPOOL_SLOT_MAPPING_FIX" \
      "$V/models/glm5next/amd/ops/kpool_compress.py|tl.program_id(0).to(tl.int64)" \
      "$V/v1/attention/ops/rocm_aiter_mla_sparse.py|GLM53_FLYDSL_FP8_MQA_IMPORT_GUARD" \
      "$V/model_executor/layers/sparse_attn_indexer_kpool.py|GLM53_DISAGG_INDEXER_KV_BARRIER" \
      "$M/moriio_layout.py|GLM53_INDEXER_KBPB" \
      "$M/moriio_connector.py|GLM53_INDEXER_KBPB_CONNECTOR" \
      "$M/moriio_common.py|GLM53_PERGROUP_REMOTE_BLOCKS" \
      "$M/moriio_connector.py|GLM53_PERGROUP_REMOTE_BLOCKS" \
      "$M/moriio_connector.py|GLM53_MTP_ALLOC_RECONCILE" \
      "$M/moriio_connector.py|GLM53_MORIIO_READ_STEP_BARRIER" \
      "$V/v1/core/sched/scheduler.py|GLM53_SYNC_LOAD_SKIP_ZERO" \
      "$V/v1/worker/gpu/spec_decode/rejection_sampler.py|def gather_draft_sampled" \
      "$A/ops/triton/attention/fp8_mqa_logits.py|GLM53_FP8_MQA_GFX942_LDS" \
      "$A/ops/triton/gluon/pa_mqa_logits.py|GLM53_PA_MQA_LOGITS_INT64" \
      "$A/ops/flydsl/kernels/fp8_mqa_logits.py|def flydsl_fp8_mqa_logits" ; do \
      f=${c%%|*}; n=${c#*|}; \
      grep -qF -- "$n" "$f" 2>/dev/null || { echo "MISSING GLM-5.3 fix: $n in $f"; miss=1; }; \
    done; \
    python3 -c "import re,sys; s=open(sys.argv[1]).read(); m=re.search(r'class KpoolTailSpec\b.*?(?=\n@dataclass|\nclass )', s, re.S); sys.exit(0 if m and 'def uses_slot_mapping' in m.group(0) else 1)" "$V/v1/kv_cache_interface.py" || { echo "MISSING GLM-5.3 fix: KpoolTailSpec.uses_slot_mapping"; miss=1; }; \
    [ "$miss" = 0 ]; echo "all 20 GLM-5.3 fix markers present"

# -----------------------------------------------------------------------------
# 5. vllm-router (Rust). Router main needs rustc >= 1.91 (its vllm-tokenizer git
#    dependency uses str::floor_char_boundary).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=9b2146ecb5cc8e46467ebe18eebc799370bd90d0
ARG RUST_TOOLCHAIN=1.91.0
RUN set -eu; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    export PATH="/root/.cargo/bin:${PATH}"; \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src; \
    cd /tmp/vllm-router-src; \
    git -c advice.detachedHead=false checkout "${ROUTER_REF}"; \
    cargo build --release; \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router; \
    vllm-router --help | grep -q -- --moriio-dp-size; \
    echo "VLLM_ROUTER=${ROUTER_REPO}@$(git rev-parse HEAD) rustc ${RUST_TOOLCHAIN}" >> /app/versions.txt; \
    rm -rf /tmp/vllm-router-src /root/.cargo /root/.rustup

# -----------------------------------------------------------------------------
# 6. RDMA userspace ABI. The base ships a stray libibverbs.so.1.16.62.0 next to the
#    v39 core the mlx5 (MI300X) and bnxt_re (MI325X) providers are built against;
#    ldconfig picks v62, ibv_devinfo then sees 0 devices and the first MoRIIO
#    CreateSession fails. Remove the stray and pin libibverbs/librdmacm to v39.
#    Must run after every ldconfig.
# -----------------------------------------------------------------------------
RUN set -eu; \
    LIBDIR=/usr/lib/x86_64-linux-gnu; \
    ldconfig || true; \
    find "$LIBDIR" -maxdepth 1 -name 'libibverbs.so.1.16.*' -print -delete || true; \
    test -e "$LIBDIR/libibverbs.so.1.14.39.0"; \
    ln -sf libibverbs.so.1.14.39.0 "$LIBDIR/libibverbs.so.1"; \
    if [ -e "$LIBDIR/librdmacm.so.1.3.39.0" ]; then ln -sf librdmacm.so.1.3.39.0 "$LIBDIR/librdmacm.so.1"; fi; \
    echo "LIBIBVERBS=v39 ($(readlink -f $LIBDIR/libibverbs.so.1))" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 7. JIT caches live under /opt/vllm_cache (bind-mount a host dir there to keep
#    warm restarts fast).
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr \
    TORCHINDUCTOR_CACHE_DIR=/opt/vllm_cache/inductor \
    MORI_JIT_CACHE_DIR=/opt/vllm_cache/mori_jit
# A cold JIT cache (aiter rmsnorm, MoE, MLA modules) takes >10 min on first start; the 600 s vLLM
# default then kills the frontend mid-build and leaves a stale build lock behind.
ENV VLLM_ENGINE_READY_TIMEOUT_S=3000

# -----------------------------------------------------------------------------
# 8. The import checks above leave MoRI JIT lock files under /root/.mori; at runtime
#    MoriAll2AllManager waits on them forever (owner PID gone). Ship it empty.
# -----------------------------------------------------------------------------
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori

LABEL org.opencontainers.image.title="glm53-flash-disagg-gfx942" \
      org.opencontainers.image.description="GLM-5.3-Flash-FP8 disaggregated serving, gfx942 (MI300X/MI325X), vLLM + MoRIIO + vllm-router from pinned fork branches" \
      glm53.vllm="raviguptaamd/vllm@6a0cb9395797 (glm53-flash-gfx942)" \
      glm53.aiter="raviguptaamd/aiter@6f65fba05b5d (glm53-flash-gfx942)" \
      glm53.mori="ROCm/mori@78b7a5f311c6" \
      glm53.router="raviguptaamd/router@9b2146ecb5cc (glm53-flash-gfx942)"
RUN echo "BASE=${BASE_IMAGE}" >> /app/versions.txt && cat /app/versions.txt

WORKDIR /opt
