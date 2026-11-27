# vLLM's public ROCm 7.2 release image and the serving script. Nothing is installed: torch, vLLM and ROCm are the
# image's own.
#   docker build -t qwen-vllm-tp2:v0.33.0-rocm72 .
ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm72@sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182
FROM ${BASE}

# The container runs as the host uid, which has no passwd entry in the image; getpass.getuser() fails without
# USER/LOGNAME. The engine core runs in the serving process; the TP workers are vLLM's multiproc workers, and
# apply_model() ships a Python function to them (VLLM_ALLOW_INSECURE_SERIALIZATION).
ENV USER=repro LOGNAME=repro HOME=/tmp HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1
# The base's entrypoint starts the OpenAI server.
ENTRYPOINT []

WORKDIR /opt/qwen_vllm_tp2
COPY qwen_vllm_serve.py ./
