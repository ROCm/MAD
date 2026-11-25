# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# The image of Numeric Doctor's softmoe_train_nan example: AMD's public
# rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0, pinned by digest,
# nothing installed. It exists to validate mad-debug-numeric, not as a benchmark.
ARG BASE_DOCKER=docker.io/rocm/pytorch@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4
FROM $BASE_DOCKER

ENV PYTHONUNBUFFERED=1
