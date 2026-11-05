# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
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

ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
USER root

# Additional step for base images that ship setup_pytorch_env.
# Guard on the setup command itself rather than on the workspace directory, so this
# same Dockerfile still builds against a plain upstream base, which does not ship it.
# The --mad flag installs into the image's existing environment instead of creating a
# venv. There is no /opt/venv-py<X.Y> to link, so this block has no symlink step and --
# importantly -- no "ENV PATH=/opt/venv/bin:${PATH}" line after it. Adding one would not
# fail the build (ENV never touches the filesystem, and the shell silently skips PATH
# entries that do not exist), it would just be dead PATH that hides which interpreter is
# actually in use. Leave it out.
RUN if command -v setup_pytorch_env >/dev/null 2>&1; then \
        echo "setup_pytorch_env found. Running env setup..."; \
        setup_pytorch_env --mad || { \
            echo "Setup failed."; \
            exit 1; \
        }; \
        echo "Setup successful."; \
        for pair in python:python3 pip:pip3; do \
            name="${pair%%:*}"; target="${pair#*:}"; \
            if ! command -v "$name" >/dev/null 2>&1; then \
                ln -s "$(command -v "$target")" "/usr/local/bin/$name" || { \
                    echo "Failed to link $name -> $target."; \
                    exit 1; \
                }; \
                echo "Linked $name -> $(command -v "$target")"; \
            fi; \
        done; \
    else \
        echo "setup_pytorch_env is not on PATH."; \
        echo "Nothing to set up."; \
    fi

# The ROCm pip SDK install location varies across base images (venv site-packages
# vs system dist-packages, and by Python version), so it cannot be hardcoded.
# Resolve it once here and expose stable symlinks that the ENV block below can
# reference -- ENV performs ${VAR} substitution only and never runs $(...).
#
# Guard on the rocm-sdk command, the same way Block 1 guards on setup_pytorch_env,
# so this Dockerfile still builds against a plain upstream base, which ships the
# classic /opt/rocm tree and no pip SDK. On that base the same three symlinks are
# pointed at the legacy layout, which keeps every ENV below valid on both bases
# and means the ENV wall itself needs no guard.
RUN set -eu; \
    if command -v rocm-sdk >/dev/null 2>&1; then \
        echo "ROCm pip SDK detected."; \
        ROCM_SDK_ROOT="$(rocm-sdk path --root)"; \
        [ -d "${ROCM_SDK_ROOT}" ] || { echo "ROCm SDK root not found: ${ROCM_SDK_ROOT}"; exit 1; }; \
        ln -sfn "${ROCM_SDK_ROOT}" /opt/rocm_sdk; \
        ROCM_SDK_CORE="$(dirname "${ROCM_SDK_ROOT}")/_rocm_sdk_core"; \
        [ -d "${ROCM_SDK_CORE}" ] || ROCM_SDK_CORE="${ROCM_SDK_ROOT}"; \
        ln -sfn "${ROCM_SDK_CORE}" /opt/rocm_sdk_core; \
        [ -d "${ROCM_SDK_ROOT}/share/amd_smi" ] || { echo "amd_smi bindings not found under ${ROCM_SDK_ROOT}"; exit 1; }; \
        ln -sfn "${ROCM_SDK_ROOT}/share/amd_smi" /opt/amd_smi; \
        echo "ROCm SDK root: ${ROCM_SDK_ROOT}"; \
        echo "ROCm SDK core: ${ROCM_SDK_CORE}"; \
    else \
        echo "rocm-sdk is not on PATH, so this is not a ROCm pip SDK image."; \
        [ -d /opt/rocm ] || { echo "No ROCm pip SDK and no /opt/rocm; cannot resolve a ROCm root."; exit 1; }; \
        ln -sfn /opt/rocm /opt/rocm_sdk; \
        ln -sfn /opt/rocm /opt/rocm_sdk_core; \
        if [ -d /opt/rocm/share/amd_smi ]; then \
            ln -sfn /opt/rocm/share/amd_smi /opt/amd_smi; \
        else \
            mkdir -p /opt/amd_smi; \
        fi; \
        echo "Falling back to the legacy /opt/rocm layout."; \
        echo "ROCm SDK root: /opt/rocm"; \
        echo "ROCm SDK core: /opt/rocm"; \
    fi


ENV ROCM_HOME="/opt/rocm_sdk" \
    ROCM_PATH="/opt/rocm_sdk" \
    ROCM_SOURCE_DIR="/opt/rocm_sdk" \
    ROCM_BIN="/opt/rocm_sdk/bin" \
    ROCM_CMAKE="/opt/rocm_sdk/lib/cmake" \
    SITE_PACKAGES="/opt/rocm_sdk/lib/python3.10/site-packages" \
    ROCM_CMAKE_PREFIX="/opt/rocm_sdk/lib/cmake"

# Explicitly set the library paths for ROCm and the ROCm SDK core to guarantee that llama.cpp always can find the necessary libraries
ENV CMAKE_PREFIX_PATH="${ROCM_CMAKE}" \
    HIP_DEVICE_LIB_PATH="${ROCM_HOME}/amdgcn/bitcode" \
    ROCM_DEVICE_LIB_PATH="${ROCM_HOME}/amdgcn/bitcode" \
    ROCM_SYSDEPS_INCLUDE="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    CPLUS_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    C_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    PKG_CONFIG_PATH="${ROCM_HOME}/lib/rocm_sysdeps/lib/pkgconfig" \
    LD_LIBRARY_PATH="/opt/rocm_sdk/lib:/opt/rocm_sdk/lib/host-math/lib:/opt/rocm_sdk/lib/rocm_sysdeps/lib:/opt/rocm_sdk_core/lib:/opt/rocm_sdk_core/lib/rocm_sysdeps/lib" \
    LIBRARY_PATH="/opt/rocm_sdk/lib:/opt/rocm_sdk/lib/host-math/lib:/opt/rocm_sdk/lib/rocm_sysdeps/lib" \
    PATH="/opt/rocm_sdk/bin:${PATH}"

# AMD-SMI Python bindings: ROCm 10.1+ pip SDK dropped setup.py from
# share/amd_smi (rocm-systems #9340); expose the module via PYTHONPATH instead.
ENV PYTHONPATH="/opt/amd_smi"

ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Environment variables
ENV HIP_FORCE_DEV_KERNARG=1
ARG MAX_JOBS_ARG=192
ENV MAX_JOBS=${MAX_JOBS_ARG}

# Argument to check current GPU arch
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}

# Install necessary system dependencies (if any, e.g., git, build-essential).
# libgl1 and libglib2.0-0 are opencv-python runtime libraries; Wan2.1's requirements
# install opencv-python, and diffusers imports cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    numactl \
    libgl1 \
    libglib2.0-0 \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    pip install "huggingface_hub[cli]"

#For multigpu run, install following
# xfuser 0.5+ imports update_npu_out from yunchang.ring.utils, which no PyPI yunchang release provides.
RUN pip install "xfuser>=0.4.1,<0.5"

# ROCm apt key. apt-key needs gpg-agent, which slim base images do not
# ship; gpg --dearmor works without it and needs no sudo.
RUN mkdir -p /etc/apt/trusted.gpg.d && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --batch --yes --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
RUN apt update && apt install -y \
    unzip \
    jq

# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# Install flash attention
ARG BUILD_FA="1"
# Pinned 2026-10-15 to upstream main commit 1fc02211 (flash_attn 2.8.4, committed 2026-10-12).
# Why: PyTorch 2.14+ headers require C++20, and upstream builds the ROCm path with -std=c++20
# since edb5c76 (#2899). No release tag includes that fix yet (v2.8.3.post2 still uses
# -std=c++17), so pin a tested commit instead of tracking main. Move to a release tag once
# one ships with -std=c++20
ARG FA_BRANCH="1fc0221183e799363001394a12dd9e78e15f5dda"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
RUN if [ "$BUILD_FA" = "1" ]; then \
    cd ${WORKSPACE_DIR} \
    && pip uninstall -y flash-attention \
    && rm -rf flash-attention \
    && git clone ${FA_REPO} \
    && cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=${HIP_ARCHITECTURES} python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl \
    && python -c "import flash_attn; print(f'Flash Attention version == {flash_attn.__version__}')"; \
    fi

#Download wan2.1 source code
RUN cd $WORKSPACE_DIR \
    && git clone https://github.com/Wan-Video/Wan2.1.git \
    && cd Wan2.1 \
    && sed -i '/\(torch\|torchvision\|flash_attn\)/d' requirements.txt \
    && pip install -r requirements.txt

# Display installed packages for verification
RUN pip list

