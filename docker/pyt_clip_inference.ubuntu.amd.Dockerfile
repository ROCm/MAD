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

# Argument to check current GPU arch
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}

RUN apt-get update
RUN apt-get install -y \
    unzip \
    jq \
    git \
    vim \
    wget

# Update pip to latest version
RUN pip install --upgrade pip

# Install clip
# WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/mlfoundations/open_clip.git open_clip &&\
    cd open_clip &&\
    pip install -e . &&\
    cd ..

# WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/LAION-AI/CLIP_benchmark.git CLIP_benchmark &&\
    cd CLIP_benchmark &&\
    pip install -e . &&\
    cd ..