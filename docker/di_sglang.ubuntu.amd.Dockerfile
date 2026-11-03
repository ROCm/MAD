# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# =============================================================================
# di_sglang.ubuntu.amd.Dockerfile -- the SGLang distributed-inference image, for
# any SGLang pin set in docker/pins/ (sglang-disagg, ...).
#
#   docker build -f docker/di_sglang.ubuntu.amd.Dockerfile \
#     $(python3 docker/pins/pins.py build-args sglang-disagg --arch gfx942) \
#     -t <registry>/di-sglang:sglang-disagg-gfx942 .
#
# Build context = repo root. The pin set names the SGLang image to build on per
# architecture (by digest), the pip packages added, and MoRI where the base needs
# it (gfx942's release image; gfx950's ships its own). BASE_IMAGE must be the pin
# set's base for the architecture; build-args prints it.
# =============================================================================

# The architecture defaults to gfx942, as the image this replaces did; madengine reads
# this default to pick BASE_IMAGE from the pin set.
ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG PINS
ARG MAD_SYSTEM_GPU_ARCHITECTURE

COPY docker/pins/ /opt/mad/pins/
COPY docker/patches/ /opt/mad/patches/
RUN test -n "${PINS}" || { echo "PINS is not set (a name in docker/pins/)" >&2; exit 1; } && \
    python3 /opt/mad/pins/pins.py env "${PINS}" --arch "${MAD_SYSTEM_GPU_ARCHITECTURE}" \
        --base "${BASE_IMAGE}" > /opt/mad/pins/build.env && \
    . /opt/mad/pins/build.env && \
    mkdir -p /app && cp "/opt/mad/pins/${PINS}.json" /app/pins.json && \
    printf 'PIN_SET=%s\nGPU_ARCH=%s\nBASE_IMAGE=%s\n' "${PIN_SET}" "${GPU_ARCH}" "${BASE_IMAGE}" > /app/versions.txt && \
    cat /opt/mad/pins/build.env

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list
ENV PYTHONPATH=$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:

WORKDIR /sgl-workspace
RUN . /opt/mad/pins/build.env && \
    if [ -n "${PIP_EARLY_UPGRADE}" ]; then pip install --upgrade ${PIP_EARLY_UPGRADE}; fi && \
    if [ -n "${PIP_EARLY}" ]; then pip install ${PIP_EARLY}; fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git ibverbs-utils libibverbs-dev \
    openmpi-bin libopenmpi-dev \
    libpci-dev libdw1 locales \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    cmake

# MoRI at the pinned commit, editable in the image's own /sgl-workspace/mori, where
# the pin set asks for it on this architecture.
WORKDIR /sgl-workspace/mori
RUN . /opt/mad/pins/build.env && \
    if [ "${MORI_BUILD}" = "1" ]; then \
      git fetch origin "${MORI_COMMIT}" && git checkout "${MORI_COMMIT}" && \
      pip install -r requirements-build.txt && \
      pip install -e . && \
      echo "MORI_COMMIT=$(git rev-parse HEAD)" >> /app/versions.txt ; \
    else \
      echo "MORI_COMMIT=base" >> /app/versions.txt ; \
    fi

WORKDIR /sgl-workspace
RUN pip list && \
    pip list --format=freeze | grep -iE '^(sglang|sglang-router|amd-aiter|mori|torch)==' >> /app/versions.txt; \
    cat /app/versions.txt
