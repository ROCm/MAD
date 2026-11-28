# vLLM's public ROCm release image (amd-aiter 0.1.26) with aiter's mla.py from the aiter build this deployment
# uses (commit 95abba475), vendored here as aiter_mla.py.
#   docker build -t sparse-mla-decode:v1 .
ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba
FROM ${BASE}

ENV HOME=/tmp
# The base's entrypoint starts the OpenAI server.
ENTRYPOINT []

# (aiter cannot be imported without a GPU, so the site-packages path is spelled out.)
COPY aiter_mla.py /usr/local/lib/python3.12/dist-packages/aiter/mla.py
RUN echo "40cef25d2fb32419c7780f1314a2b8b9ec9fefc50d992078f864da1baf19281f  /usr/local/lib/python3.12/dist-packages/aiter/mla.py" | \
    sha256sum -c -

WORKDIR /opt/sparse_mla_decode
COPY sparse_mla_decode_serve.py ./
