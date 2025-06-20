#!/bin/bash
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

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "Model repo: $MODEL_NAME"
echo "Setup script starting in directory $(pwd)"

export HF_HOME=/workspace/huggingface
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

if [[ "$MODEL_NAME" == "Llama-3.1-8B" ]]; then
  echo "Building torchtitan dependencies for $MODEL_NAME"
  TORCHTITAN_DIR="/workspace/torchtitan"
  echo "Torchtitan directory path: $TORCHTITAN_DIR"
  cd $TORCHTITAN_DIR
  python scripts/download_tokenizer.py \
    --repo_id meta-llama/Meta-Llama-3.1-8B \
    --tokenizer_path "original" \
    --hf_token=$HF_TOKEN 
fi 

if [[ "$MODEL_NAME" == "Llama-3.1-70B" ]]; then
  echo "Building torchtitan dependencies for $MODEL_NAME"
  TORCHTITAN_DIR="/workspace/torchtitan"
  echo "Torchtitan directory path: $TORCHTITAN_DIR"
  cd $TORCHTITAN_DIR
  python scripts/download_tokenizer.py \
      --repo_id meta-llama/Meta-Llama-3.1-70B \
      --tokenizer_path "original" \
      --hf_token=$HF_TOKEN 

  echo "Building torchtune dependencies for $MODEL_NAME"
  TORCHTUNE_DIR="/workspace/torchtune"
  echo "Torchtune directory path: $TORCHTUNE_DIR"
  cd $TORCHTUNE_DIR
  huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
    --local-dir ./models/Llama-3.1-70B-Instruct \
    --exclude 'original/*.pth'
  python dataset.py
  
fi

if [[ "$MODEL_NAME" == "Llama-3.3-70B" ]]; then
  echo "Building torchtune dependencies for $MODEL_NAME"
  TORCHTUNE_DIR="/workspace/torchtune"
  echo "Torchtune directory path: $TORCHTUNE_DIR"
  cd $TORCHTUNE_DIR
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
          --local-dir ./models/Llama-3.3-70B-Instruct \
          --exclude 'original/*.pth'
  python dataset.py
fi

# Dependency for Flux
if [ "$MODEL_NAME" == "Flux" ]; then
  echo "Building Flux dependencies for $MODEL_NAME"
  cd /workspace/FluxBenchmark
  pip3 install --no-cache-dir --upgrade pip packaging
  pip3 install --no-cache-dir -r requirements.txt
  export ROCBLAS_USE_HIPBLASLT=1
  export DISABLE_ADDMM_CUDA_LT=0
  export HIP_FORCE_DEV_KERNARG=1
  export TORCH_NCCL_HIGH_PRIORITY=0
  export GPU_MAX_HW_QUEUES=8
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  make download_assets
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "Building dependencies for all models"
  TORCHTITAN_DIR="/workspace/torchtitan"
  cd $TORCHTITAN_DIR
  python scripts/download_tokenizer.py \
      --repo_id meta-llama/Meta-Llama-3.1-8B \
      --tokenizer_path "original" \
      --hf_token=$HF_TOKEN 

  python scripts/download_tokenizer.py \
      --repo_id meta-llama/Meta-Llama-3.1-70B \
      --tokenizer_path "original" \
      --hf_token=$HF_TOKEN 

  TORCHTUNE_DIR="/workspace/torchtune"
  cd $TORCHTUNE_DIR
  # Llama 3.1 70B 
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
          --local-dir ./models/Llama-3.1-70B-Instruct \
          --exclude 'original/*.pth'
  # Llama 3.3 70B 
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
          --local-dir ./models/Llama-3.3-70B-Instruct \
          --exclude 'original/*.pth'
  python dataset.py

  cd /workspace/FluxBenchmark
  pip3 install --no-cache-dir --upgrade pip packaging
  pip3 install --no-cache-dir -r requirements.txt
  export ROCBLAS_USE_HIPBLASLT=1
  export DISABLE_ADDMM_CUDA_LT=0
  export HIP_FORCE_DEV_KERNARG=1
  export TORCH_NCCL_HIGH_PRIORITY=0
  export GPU_MAX_HW_QUEUES=8
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  make download_assets
fi
