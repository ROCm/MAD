# Docker image for the Qwen3 MoE training example: the same layers as qwen3_moe.def (public Python 3.12 base +
# the public PyTorch ROCm wheel + torchtitan's runtime dependencies). torchtitan itself is not baked in:
# run.sh mounts ./src/torchtitan. No build context is needed:
#
#   docker build --network=host -t qwen3-moe - < Dockerfile
FROM docker.io/library/python@sha256:4209fae09a3e20afd863e528bfddc235da5fd5977983e1996cd54cc9a72e6267

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# The second pip install is torchtitan's (commit ecae62f15) dependencies.
RUN pip install "torch==2.14.0+rocm7.14" "rocm[device-all,libraries]==7.14.0" \
        --index-url https://download.pytorch.org/whl/rocm7.14 --extra-index-url https://pypi.org/simple && \
    pip install "torchdata>=0.8.0" "datasets>=3.6.0,<4.8.0" "tokenizers>=0.15.0" safetensors "tyro>=1.0.5" \
        tensorboard einops pillow numpy "spmd_types==0.2.1" && \
    pip freeze > /opt/pip-freeze.txt

# The container runs as the host uid, which has no passwd entry in the image (dynamo calls getpass.getuser()).
ENV USER=repro LOGNAME=repro HOME=/tmp PYTHONUNBUFFERED=1
