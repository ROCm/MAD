# Thin derived image: GLM-5.2 serving image + the Thor2 AsyncLL vLLM patch.
# The base is pure-Python vLLM, so the patch is a single fast layer (no wheel rebuild).
#
# Build:
#   docker build -f Dockerfile -t rocm/vllm-dev:...mori121-async_ll \
#     --build-arg BASE=rocm/vllm-dev:vllm-wideep_06_29_2026_..._mori121 .
ARG BASE=rocm/vllm-dev:vllm-wideep_06_29_2026_Shiksha_dp16_2p2d_mori_v1.2.1_aiter_v0.1.16.post3_nightlybase_mori121
FROM ${BASE}

COPY apply_async_ll_patch.py /opt/thor2/apply_async_ll_patch.py
# Apply the vLLM MoRI-EP AsyncLL kernel-selection patch at build time (idempotent).
RUN python3 /opt/thor2/apply_async_ll_patch.py && \
    echo "thor2 async_ll patch baked in"

# Default the Thor2 EP knobs so the pod env doesn't have to (still overridable):
#   MORI_EP_FORCE_ASYNC_LL=1  -> vLLM multi-node EP uses the AsyncLL (WRITE+poll) kernel
#   MORI_ENABLE_SDMA=1        -> AsyncLL uses SDMA engines (else falls back to CUs, slower)
ENV MORI_EP_FORCE_ASYNC_LL=1 \
    MORI_ENABLE_SDMA=1
