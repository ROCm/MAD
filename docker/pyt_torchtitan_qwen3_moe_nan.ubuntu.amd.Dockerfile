# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# The image from nan-doctor's qwen3_moe_gradient_nan_nohint example (public Python
# 3.12 + the PyTorch 2.14 ROCm 7.14 wheel), with torchtitan pinned at the commit
# that example trains. It exists to validate mad-debug-numeric, not as a benchmark.
FROM docker.io/library/python@sha256:4209fae09a3e20afd863e528bfddc235da5fd5977983e1996cd54cc9a72e6267

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "torch==2.14.0+rocm7.14" "rocm[device-all,libraries]==7.14.0" \
        --index-url https://download.pytorch.org/whl/rocm7.14 --extra-index-url https://pypi.org/simple && \
    pip install "torchdata>=0.8.0" "datasets>=3.6.0,<4.8.0" "tokenizers>=0.15.0" safetensors "tyro>=1.0.5" \
        tensorboard einops pillow numpy "spmd_types==0.2.1" && \
    pip freeze > /opt/pip-freeze.txt

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --quiet https://github.com/pytorch/torchtitan.git /opt/torchtitan && \
    git -C /opt/torchtitan checkout --quiet ecae62f1591c3a73ed78a25d7c68f249bd61caa6 && \
    rm -rf /opt/torchtitan/.git

ENV PYTHONPATH=/opt/torchtitan \
    USER=repro LOGNAME=repro HOME=/tmp PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
