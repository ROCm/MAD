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

ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Install necessary system dependencies (if any, e.g., git, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    numactl \
    gnupg2 \
    dirmngr \
    wget \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade "pip>=26.0,<26.2"  # This version is required for chai_lab installation

# numpy is reinstalled because of pandas compatibility issues, remove the lines below once base image moves to numpy>1.20.3
RUN pip3 install -U numpy
RUN pip3 install -U scipy
RUN pip3 install filelock requests #  predict_structure.py imports & runs

# Install pip-tools to compile the requirements.in file into requirements.txt
RUN pip install pip-tools

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

# Clone the chai_lab repository
# Modify requirements.in to exclude torch and compile dependencies
# Install chai_lab without re-installing torch

ARG CHAI_BRANCH=main
RUN cd $WORKSPACE_DIR && \

    git clone --branch ${CHAI_BRANCH} https://github.com/chaidiscovery/chai-lab chai-lab && \
    cd chai-lab && \

    # Removing old branch v0.4.4, due to rdkit version incompatability
    sed '/torch/d' requirements.in > requirements.temp && \
    mv requirements.temp requirements.in && \
    pip-compile requirements.in && \
    pip install -r requirements.txt && \
    pip install -e .

# Display installed packages for verification
RUN pip list
