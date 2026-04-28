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
ARG BASE_DOCKER=lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260417
FROM $BASE_DOCKER

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

ENV PYTHONPATH=$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:

ARG GPU_ARCH=gfx942
WORKDIR /sgl-workspace

RUN pip install --upgrade sglang-router

# the default already installs mori with AINIC. Reinstall from source for cx7 and bnxt; ainic only clears USE_* overrides.
ARG NIC_BACKEND="cx7"

RUN set -eux; \
  install_mori() { \
    pip uninstall -y mori; \
    cd /sgl-workspace/mori; \
    pip install -r requirements-build.txt; \
    pip install . --no-build-isolation; \
    export PYTHONPATH="${PYTHONPATH}:/sgl-workspace/mori"; \
  }; \
  profile="/etc/profile.d/50-mori-nic-backend.sh"; \
  case "${NIC_BACKEND}" in \
    cx7) \
      { echo "export USE_IONIC=OFF"; echo "export USE_BNXT=OFF"; } > "$profile"; \
      . "$profile"; \
      install_mori; \
      ;; \
    bnxt) \
      echo "export USE_BNXT=ON" > "$profile"; \
      . "$profile"; \
      install_mori; \
      ;; \
    ainic) \
      rm -f "$profile" || true; \
      ;; \
    *) \
      echo "ERROR: Unsupported NIC_BACKEND='${NIC_BACKEND}'. Supported values are: cx7, bnxt, ainic." >&2; \
      exit 1; \
      ;; \
  esac; \
  echo "NIC_BACKEND=${NIC_BACKEND} USE_IONIC=${USE_IONIC-} USE_BNXT=${USE_BNXT-} PYTHONPATH=${PYTHONPATH-}"

# Display installed packages for verification
RUN pip list
