# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# PyTorch 2.7.0a0+git3a58512
ARG BASE_DOCKER=rocm/pytorch-training:v25.5
FROM $BASE_DOCKER

WORKDIR /workspace
USER root

RUN python -c 'import torch; print(torch.__version__)'

RUN pip uninstall -y numpy
RUN rm -rf /opt/conda/envs/py_3.10/lib/python3.10/site-packages/numpy*
RUN apt-get remove -y python3-blinker

RUN bash -c "set -x && \
    export PYTORCH_VERSION=\$(python -c \"import torch; print(torch.__version__)\") && \
    export PYTORCHVISION_VERSION=\$(pip show torchvision | grep '^Version:' | awk '{print \$2}') && \
    if [[ \"\$PYTORCH_VERSION\" == 2.3* ]]; then \
        LLM_FOUNDRY_BRANCH=\"v0.11.0\"; \
        COMPOSER_BRANCH=\"v0.23.4\"; \
    elif [[ \"\$PYTORCH_VERSION\" == 2.4* ]]; then \
        LLM_FOUNDRY_BRANCH=\"v0.12.0\"; \
        COMPOSER_BRANCH=\"v0.25.0\"; \
    else \
        LLM_FOUNDRY_BRANCH=\"v0.17.1\"; \
        COMPOSER_BRANCH=\"v0.28.0\"; \
    fi && \
    git clone https://github.com/mosaicml/composer.git /workspace/composer && \
    cd /workspace/composer && \
    git checkout \$COMPOSER_BRANCH && \
    sed -i \"s/'torch>=.*/'torch==\$PYTORCH_VERSION',/\" setup.py && \
    sed -i \"s/'torchvision>=.*/'torchvision==\$PYTORCHVISION_VERSION',/\" setup.py && \
    pip install -e .[libcloud,wandb,oci,gcs,mlflow] && \
    cd /workspace && \
    git clone https://github.com/mosaicml/llm-foundry.git && \
    cd /workspace/llm-foundry && \
    git checkout \$LLM_FOUNDRY_BRANCH && \
    echo \"pytorch version \$PYTORCH_VERSION foundry branch version \$LLM_FOUNDRY_BRANCH\" && \
    sed -i \"s/'torch.*/'torch==\$PYTORCH_VERSION',/\" setup.py && \
    pip install -e ."
ENV HF_HOME="/data/llm-foundry"
