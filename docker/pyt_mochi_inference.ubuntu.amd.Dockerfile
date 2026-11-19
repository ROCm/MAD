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
# Those images ship their own setup command, which populates the workspace and
# creates a version-qualified venv at /opt/venv-py<X.Y>. Guard on the command
# itself rather than on the workspace directory, so this same Dockerfile still
# builds against a plain upstream base, which does not ship that command.
# The venv path carries the python version, but the ENV PATH below cannot run
# $(...) to discover it -- so the unversioned /opt/venv symlink is created here.
RUN if command -v setup_pytorch_env >/dev/null 2>&1; then \
        echo "setup_pytorch_env found. Running env setup..."; \
        setup_pytorch_env --qa || { \
            echo "Setup failed."; \
            exit 1; \
        }; \
        echo "Setup successful."; \
        PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; \
        if [ -e /opt/venv ]; then \
            echo "/opt/venv already exists."; \
        elif [ -d "/opt/venv-py${PY_VERSION}" ]; then \
            ln -s "/opt/venv-py${PY_VERSION}" /opt/venv || { \
                echo "Failed to link /opt/venv -> /opt/venv-py${PY_VERSION}."; \
                exit 1; \
            }; \
            echo "Created /opt/venv -> /opt/venv-py${PY_VERSION}"; \
        else \
            echo "No supported virtual environment found at /opt/venv-py${PY_VERSION}."; \
            exit 1; \
        fi; \
    else \
        echo "setup_pytorch_env is not on PATH."; \
        echo "Nothing to set up."; \
    fi

# Update PATH to include the virtual environment's bin directory
ENV PATH="/opt/venv/bin:${PATH}"

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

ENV WORKSPACE_DIR=/app
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# git for the source clones below (not present in slim base images)
RUN apt update && \
    apt install -y git

# Pinned 2026-10-15 to upstream main commit 1fc02211 (flash_attn 2.8.4, committed 2026-10-12).
# Why: PyTorch 2.14+ headers require C++20, and upstream builds the ROCm path with -std=c++20
# since edb5c76 (#2899). No release tag includes that fix yet (v2.8.3.post2 still uses
# -std=c++17), so pin a tested commit instead of tracking main. Move to a release tag once
# one ships with -std=c++20
ARG FA_BRANCH="1fc0221183e799363001394a12dd9e78e15f5dda"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
# Build for the GPU arch madengine detects (passed as a build arg)
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}
RUN git clone ${FA_REPO}
RUN cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=${HIP_ARCHITECTURES} python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN pip install "huggingface_hub[cli]"
RUN pip install transformers
RUN pip install sentencepiece
RUN pip install ray
RUN pip install safetensors
# record configuration for posterity
RUN pip3 list
