# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# The image of Numeric Doctor's qwen_vllm_serve_nan example: vLLM's public ROCm
# release vllm/vllm-openai-rocm:v0.33.0-rocm100, pinned by digest, nothing installed.
# It exists to validate mad-debug-numeric, not as a benchmark.
ARG BASE_DOCKER=docker.io/vllm/vllm-openai-rocm@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba
FROM $BASE_DOCKER

ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# The base's entrypoint starts the OpenAI server.
ENTRYPOINT []
