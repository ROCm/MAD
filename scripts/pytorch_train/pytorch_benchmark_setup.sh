#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
export HF_HOME=/workspace/huggingface

#Dependency for torchtune
echo "Current directory: $(pwd)"
TORCHTUNE_DIR="/workspace/torchtune"
cp -r "torchtune/wikitext_finetune.sh" ${TORCHTUNE_DIR}
cp -r "torchtune/wikitext_lora_finetune.sh" ${TORCHTUNE_DIR}
cp -r "torchtune/llama_3_1_70b_full_finetune_recipe.yaml" ${TORCHTUNE_DIR}
cp -r "torchtune/llama_3_1_70b_lora_finetune_recipe.yaml" ${TORCHTUNE_DIR}
cp -r "torchtune/dataset.py" ${TORCHTUNE_DIR}
cd $TORCHTUNE_DIR
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
       --local-dir ./models/Llama-3.1-70B-Instruct \
       --exclude 'original/*.pth'
cd $TORCHTUNE_DIR
python dataset.py

# Dependency for Flux
cd /workspace/FluxBenchmark
pip3 install --no-cache-dir --upgrade pip packaging
pip3 install --no-cache-dir -r requirements.txt
export ROCBLAS_USE_HIPBLASLT=1
export DISABLE_ADDMM_CUDA_LT=0
export HIP_FORCE_DEV_KERNARG=1
export TORCH_NCCL_HIGH_PRIORITY=0
export GPU_MAX_HW_QUEUES=8
make download_assets 
if [ $? -ne 0 ]; then
    # Except block (handle the error)
    make download assets # additional command to resolve huggingface download issue
fi

# Dependency for Llama3.1 (torchtitan)
cd /workspace/torchtitan
pip install -r requirements.txt

pip install numpy==1.26.4
