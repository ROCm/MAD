# Image for the FP8 DiT training example. Only public layers: rocm/primus:v26.3 (ROCm 7.2.1, PyTorch 2.10.0,
# torchao 0.15.0), pinned by digest, plus the trainer script. Nothing is patched or installed.
#   docker build -t dit-fp8-train-nan:primus-v26.3 .
ARG BASE=rocm/primus:v26.3@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482
FROM ${BASE}

# run.sh runs the container as the host uid, which has no passwd entry and no writable home in the image.
ENV USER=repro LOGNAME=repro HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/inductor TRITON_CACHE_DIR=/tmp/triton

WORKDIR /opt/dit_fp8_train
COPY dit_fp8_train.py ./
