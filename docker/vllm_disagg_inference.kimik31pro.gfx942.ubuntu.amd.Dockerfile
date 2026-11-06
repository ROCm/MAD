# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# Kimi-K3.1-Pro disaggregated (1P/1D, MoRIIO) serving image for AMD Instinct gfx942 (MI325X / MI300X).
#   base    vllm/vllm-openai-rocm nightly (ROCm 10.0, torch 2.13, MoRI 1.2.3.post1, FlyDSL 0.4.0)
#   vLLM    upstream Kimi K3.1 split tip (vllm-project/vllm PR #66027 head) python tree over the nightly; the
#           split changes no csrc vs the nightly commit, so the nightly's compiled extensions are kept
#   AITER   the gfx942 K3.1 build's AITER (fnuz fp8_dsv32_mla sparse_mla_fwd; public AITER lacks it)
#   patch   rocm_aiter_mla_sparse: allow kv_cache_dtype=fp8_ds_mla on gfx942 when AITER supports it
#   router  vllm-router (MoRIIO PD service discovery), raviguptaamd/router glm53-flash-gfx942-hybrid
# Self-contained (no COPY):
#   docker build --ulimit nofile=1048576:1048576 -f vllm_disagg_inference.kimik31pro.gfx942.ubuntu.amd.Dockerfile -t <image> .
# Serving: models.yaml entry Kimi-K3.1-Pro-gfx942 (CONNECTOR=moriio WIDE_EP=0).
# =============================================================================
ARG BASE_IMAGE=vllm/vllm-openai-rocm:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121
ARG K31_AITER_IMAGE=amdsiloai/vllm:vllm-openai-rocm-kimi-k31-gfx942

FROM ${K31_AITER_IMAGE} AS k31aiter

FROM ubuntu:22.04 AS router
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=076800bf43304281818a5353e1037e788ef24249
ARG RUST_TOOLCHAIN=1.91.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git ca-certificates build-essential pkg-config libssl-dev protobuf-compiler python3 python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    export PATH="/root/.cargo/bin:${PATH}"; \
    git clone --filter=blob:none "${ROUTER_REPO}" /src && cd /src && \
    git -c advice.detachedHead=false checkout "${ROUTER_REF}" && cargo build --release && \
    install -m 755 target/release/vllm-router /vllm-router && /vllm-router --help | grep -q -- --moriio-dp-size && \
    echo "VLLM_ROUTER=${ROUTER_REPO}@$(git rev-parse HEAD) rustc ${RUST_TOOLCHAIN}" > /router_version.txt

FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG K31_AITER_IMAGE
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_PR=66027
ARG VLLM_REF=ed04ad1e79d2b000606e518d9d9b31750629dde5
ENV DP=/usr/local/lib/python3.12/dist-packages
ENTRYPOINT []
WORKDIR /app
COPY --from=router /vllm-router /usr/local/bin/vllm-router
COPY --from=router /router_version.txt /tmp/router_version.txt
COPY --from=k31aiter /opt/runtime-venv/lib/python3.12/site-packages/aiter /tmp/k31_aiter/aiter
COPY --from=k31aiter /opt/runtime-venv/lib/python3.12/site-packages/aiter_meta /tmp/k31_aiter/aiter_meta
COPY --from=k31aiter /opt/runtime-venv/lib/python3.12/site-packages/aiter_worker_limits.py /tmp/k31_aiter/aiter_worker_limits.py
RUN set -eu; mkdir -p /app; \
    rm -rf $DP/aiter $DP/aiter_meta $DP/aiter_worker_limits.py; \
    mv /tmp/k31_aiter/aiter /tmp/k31_aiter/aiter_meta /tmp/k31_aiter/aiter_worker_limits.py $DP/; rm -rf /tmp/k31_aiter; \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git rsync >/dev/null; \
    git clone -q --filter=blob:none --no-checkout "$VLLM_REPO" /tmp/vllm-src; cd /tmp/vllm-src; \
    git fetch -q origin "pull/${VLLM_PR}/head"; git -c advice.detachedHead=false checkout -q "$VLLM_REF"; \
    rsync -a --exclude '*.so' --exclude '__pycache__' vllm/ $DP/vllm/; cd / && rm -rf /tmp/vllm-src; \
    printf '%s\n' \
      'import pathlib, sys' \
      'p = pathlib.Path(sys.argv[1]) / "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"' \
      's = p.read_text()' \
      'old = """            from vllm.platforms.rocm import on_gfx950\n\n            if not on_gfx950():\n"""' \
      'new = """            # K31_GFX942_FP8DSMLA_GATE: gfx942 works when aiter reads fp8_dsv32_mla in fnuz.\n            from vllm.platforms.rocm import on_gfx942, on_gfx950\n\n            fnuz_ok = False\n            if on_gfx942():\n                from aiter.ops.triton.attention import sparse_mla as _sm\n\n                fnuz_ok = \"gfx942\" in getattr(_sm, \"FNUZ_DSV32_ARCHS\", ())\n            if not (on_gfx950() or fnuz_ok):\n"""' \
      'assert s.count(old) == 1, "gate block not found"' \
      'p.write_text(s.replace(old, new))' > /tmp/gate.py; \
    python3 /tmp/gate.py $DP/vllm && rm /tmp/gate.py; \
    grep -q "FNUZ_DSV32_ARCHS" $DP/aiter/ops/triton/attention/sparse_mla.py; \
    grep -q K31_GFX942_FP8DSMLA_GATE $DP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py; \
    grep -q KimiK31ForConditionalGeneration $DP/vllm/model_executor/models/registry.py; \
    grep -q "_mamba_region_mr_offsets" $DP/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py; \
    vllm-router --help | grep -q -- --kv-connector; \
    { echo "BASE=${BASE_IMAGE}"; echo "VLLM=${VLLM_REPO} PR#${VLLM_PR} @ ${VLLM_REF} (python overlay)"; \
      echo "AITER=from ${K31_AITER_IMAGE}"; cat /tmp/router_version.txt; } >> /app/versions.txt; \
    rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori /opt/vllm_cache; cat /app/versions.txt
# JIT caches under /opt/vllm_cache: bind-mount a host dir there for warm restarts.
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit VLLM_CACHE_ROOT=/opt/vllm_cache/vllm TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr TORCHINDUCTOR_CACHE_DIR=/opt/vllm_cache/inductor \
    MORI_JIT_CACHE_DIR=/opt/vllm_cache/mori_jit VLLM_ENGINE_READY_TIMEOUT_S=7200
