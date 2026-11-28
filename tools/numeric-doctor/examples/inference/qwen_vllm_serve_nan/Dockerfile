# vLLM's public ROCm release image and the serving script. Nothing is installed: torch, vLLM and ROCm are the
# image's own.
#   docker build -t qwen-vllm-serve:v0.33.0 .
ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba
FROM ${BASE}

# The container runs as the host uid, which has no passwd entry in the image; getpass.getuser() fails without
# USER/LOGNAME. The engine runs in the serving process (no engine-core subprocess).
ENV USER=repro LOGNAME=repro HOME=/tmp HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# The base's entrypoint starts the OpenAI server.
ENTRYPOINT []

WORKDIR /opt/qwen_vllm_serve
COPY qwen_vllm_serve.py .
