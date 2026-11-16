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

ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# git for the source clones below (not present in slim base images)
RUN apt update && \
    apt install -y git

# Install huggingface transformers
RUN cd /workspace && git clone https://github.com/ROCm/transformers transformers &&\
    cd transformers &&\
    git show --oneline -s && \
    pip install -e .

# Intentionally skip torchaudio, else it force upgrades torch as well
RUN sed -i 's$torchaudio$$g' /workspace/transformers/examples/pytorch/_tests_requirements.txt

# Install dependencies
RUN cd /workspace/transformers/examples/pytorch && pip3 install -r _tests_requirements.txt
RUN pip3 install --no-cache-dir GPUtil azureml azureml-core tokenizers ninja cerberus sympy sacremoses sacrebleu==1.5.1 sentencepiece scipy scikit-learn
# setting datasets==1.9.0 due to Roberta SQUAD change, https://github.com/huggingface/transformers/issues/12880
#RUN pip3 install --no-cache-dir datasets==1.9.0
# https://github.com/huggingface/datasets/issues/3099
# transformers requires huggingface_hub<2.0.
RUN pip3 install -U "huggingface_hub<2"

# ROCm apt key. apt-key needs gpg-agent, which slim base images do not
# ship; gpg --dearmor works without it and needs no sudo.
RUN mkdir -p /etc/apt/trusted.gpg.d && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --batch --yes --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
RUN apt update && apt install -y \
    unzip \
    jq

# add sshpass, sshfs for downloading from mlse-nas
RUN apt-get install -y sshpass sshfs
RUN apt-get install -y netcat-traditional

# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# record configuration for posterity
RUN pip3 list


