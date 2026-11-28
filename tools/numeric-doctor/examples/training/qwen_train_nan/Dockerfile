# Public image for the Qwen2.5-0.5B training reproducer. Only public layers: AMD's rocm/pytorch release image
# (ROCm 7.2.0, PyTorch 2.10.0, Ubuntu 24.04, Python 3.12) plus the trainer script.
#   docker build -t qwen-train-nan:rocm7.2.0 .
ARG BASE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4
FROM ${BASE}

# The container runs as the host uid, which has no passwd entry in the image. The optimizer imports dynamo,
# which calls getpass.getuser(): without USER/LOGNAME every trial exits silently.
ENV USER=repro LOGNAME=repro HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/inductor

WORKDIR /opt/qwen_train
COPY qwen_train.py .
