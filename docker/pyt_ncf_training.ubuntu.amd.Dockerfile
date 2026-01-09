# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch-training:v25.5
FROM $BASE_DOCKER

# Pin numpy to < 2.0.0
RUN pip install --upgrade numpy==1.26.4 scipy numba pandas
RUN apt-get update && apt-get install -y zip unzip
