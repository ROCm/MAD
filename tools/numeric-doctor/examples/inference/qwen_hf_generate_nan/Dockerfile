# Image for the Qwen2.5-0.5B inference workload: AMD's public rocm/pytorch release image (ROCm 7.2.0,
# PyTorch 2.10.0, Ubuntu 24.04, Python 3.12), Hugging Face transformers (pinned) and the inference script. torch,
# numpy and the ROCm libraries stay the image's own: pip only adds what transformers needs on top.
#   docker build -t qwen-hf-generate:rocm7.2.0 .
ARG BASE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4
FROM ${BASE}
ARG TRANSFORMERS=5.20.2

# PIP_BREAK_SYSTEM_PACKAGES: the base's Python may be the distribution's (PEP 668); ignored in a venv.
RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir "transformers==${TRANSFORMERS}" && \
    python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# The container runs as the host uid, which has no passwd entry in the image; getpass.getuser() fails without
# USER/LOGNAME.
ENV USER=repro LOGNAME=repro HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/inductor HF_HUB_OFFLINE=1

WORKDIR /opt/qwen_hf_generate
COPY qwen_generate.py .
