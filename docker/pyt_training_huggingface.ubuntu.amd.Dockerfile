# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=docker.io/rocm/pytorch-training:v25.5
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

RUN apt-get update
RUN apt-get install -y \
    unzip \
    jq \
    git \
    vim \
    wget

# Update pip to latest version
RUN pip install --upgrade pip

# Pin numpy<2.0.0 for now; remove once numpy==2.0.0 is better supported
RUN pip install --upgrade numpy==1.26.4 scipy numba pandas

# Install dependencies
RUN pip install GPUtil azureml azureml-core tokenizers ninja cerberus sympy sacremoses sacrebleu==1.5.1 sentencepiece scipy scikit-learn "urllib3<2"

# Install DeepSpeed
ARG DEEPSPEED_REPO=https://github.com/microsoft/DeepSpeed
ARG DEEPSPEED_BRANCH=v0.15.1
RUN git clone -b $DEEPSPEED_BRANCH $DEEPSPEED_REPO DeepSpeed

RUN cd DeepSpeed && \
    git show --oneline -s && \
    pip install .[dev,1bit,autotuning]
RUN cd ~ && python -c "import deepspeed; print(deepspeed.__version__)"

# Install huggingface transformers
ARG TRANSFORMERS_REPO=https://github.com/ROCm/transformers
ARG TRANSFORMERS_BRANCH=main
RUN cd /workspace && git clone -b $TRANSFORMERS_BRANCH $TRANSFORMERS_REPO transformers
RUN cd transformers && \
    git show --oneline -s && \
    pip install -e .

# Intentionally skip torchaudio, else it force upgrades torch as well
RUN sed -i 's$torchaudio$$g' /workspace/transformers/examples/pytorch/_tests_requirements.txt

# Skip source install for accelerate, use specified version instead
RUN sed -i 's$git+https://github.com/huggingface/accelerate@main#egg=accelerate$$g' /workspace/transformers/examples/pytorch/_tests_requirements.txt

# Install test dependencies
RUN cd /workspace/transformers/examples/pytorch && pip install -r _tests_requirements.txt

# Install huggingface libraries
RUN pip install ftfy==6.3.1
RUN pip install peft==0.15.0
RUN pip install accelerate datasets huggingface_hub
RUN pip install triton==3.1.0

# Install xFormers
# Move to run.sh

RUN pip install git+https://github.com/huggingface/diffusers.git@66e50d4e248a32ef8f8698cf3e6f0e1040f74cfc

RUN pip install numpy==1.26.4

# Disable WANDB Logging
ENV WANDB_DISABLED=true
