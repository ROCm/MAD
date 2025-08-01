ARG BASE_DOCKER=rocm/pytorch-training:v25.5
FROM $BASE_DOCKER

WORKDIR /workspace
USER root

RUN python -c 'import torch; print(torch.__version__)'

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

ARG GPU_ARCH=gfx90a
ARG FLASH_ATTENTION_VERSION2=2.8.0.post2
ARG FLASH_ATTENTION_VERSION3=v3.0.0.r1-cktile
RUN pip install setuptools packaging


# Install v2 if GPU_ARCH includes gfx90a else install v3.
RUN \
    case "$GPU_ARCH" in \
      "gfx90a"|"gfx90a;gfx942"|"gfx942;gfx90a") \
        GPU_ARCHS=${GPU_ARCH} pip install flash-attn==${FLASH_ATTENTION_VERSION2} --no-build-isolation; \
        ;; \
      *) \
        git clone https://github.com/ROCm/flash-attention/ -b ${FLASH_ATTENTION_VERSION3} /app/flash-attention \
        && cd /app/flash-attention \
        && GPU_ARCHS=${GPU_ARCH} python setup.py install \
        && rm -rf /app/flash-attention; \
        ;; \
    esac

RUN cd /workspace && git clone https://github.com/MotifTechnologies/torchtitan_public.git -b open/torchtitan-train