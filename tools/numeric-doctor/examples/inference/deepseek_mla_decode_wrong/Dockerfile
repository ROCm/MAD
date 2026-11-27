# vLLM's public ROCm release image (torch, ROCm, amd_aiter) plus the workload script. Nothing is installed.
#   docker build -t deepseek-mla-decode:v0.33.0 .
ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba
FROM ${BASE}

ENV HOME=/tmp
# The base's entrypoint starts the OpenAI server.
ENTRYPOINT []

WORKDIR /opt/deepseek_mla_decode
COPY mla_decode_serve.py ./
