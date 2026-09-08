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
ENV APP_DIR=/app
RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

# Environment variables
ENV HIP_FORCE_DEV_KERNARG=1
ARG MAX_JOBS_ARG=192
ENV MAX_JOBS=${MAX_JOBS_ARG}

# Argument to check current GPU arch
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}

# Install flash attention
#ARG BUILD_FA="1"
#ARG FA_BRANCH="v3.0.0.r1-cktile"
#ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
#RUN if [ "$BUILD_FA" = "1" ]; then \
#    cd ${APP_DIR} \
#    && pip uninstall -y flash-attention \
#    && rm -rf flash-attention \
#    && git clone ${FA_REPO} \
#    && cd flash-attention \
#    && git checkout ${FA_BRANCH} \
#    && git submodule update --init \
#    && GPU_ARCHS=${HIP_ARCHITECTURES} python3 setup.py bdist_wheel --dist-dir=dist \
#    && pip install dist/*.whl \
#    && python -c "import flash_attn; print(f'Flash Attention version == {flash_attn.__version__}')"; \
#    fi

# install flash attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\
    cd flash-attention &&\
    python setup.py install

# Install Janus-pro (patches for Hugging Face transformers 5.x):
# - @dataclass rejects mutable default dict on PretrainedConfig subclasses; use default_factory
# - PreTrainedModel.post_init() sets all_tied_weights_keys (required by from_pretrained / loading)
RUN cd ${APP_DIR} \
  && git clone https://github.com/deepseek-ai/Janus.git \
  && cd Janus \
  && sed -i 's/^torch==/# torch==/' requirements.txt \
  && for f in janus/models/modeling_vlm.py janus/janusflow/models/modeling_vlm.py; do \
    sed -i '/^from transformers.configuration_utils import PretrainedConfig/a from dataclasses import field' "$f" \
    && sed -i 's/params: AttrDict = {}/params: AttrDict = field(default_factory=dict)/g' "$f"; \
  done \
  && sed -i '/self\.language_model = LlamaForCausalLM(language_config)/a\        self.post_init()' janus/models/modeling_vlm.py \
  && sed -i '/self\.vision_gen_dec_aligner = nn.Linear(2048, 768, bias=True)/a\        self.post_init()' janus/janusflow/models/modeling_vlm.py \
  && pip install -e . \
  && pip install datasets \
  && cd ..

WORKDIR /myworkspace

# record configuration for posterity
RUN pip3 list
