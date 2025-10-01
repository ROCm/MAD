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

# Usage function
usage() {
    echo "Usage: $0 -m MODEL_NAME"
    echo "  -m: Model name (Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Flux, or empty for all models)"
    echo "  Example: $0 -m Llama-3.1-8B"
    exit 1
}

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

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here"
    exit 1
fi

export HF_HOME=/workspace/huggingface
hf login --token $HF_TOKEN --add-to-git-credential

TORCHTITAN_DIR="/workspace/torchtitan"
TORCHTUNE_DIR="/workspace/torchtune"

if [[ "$MODEL_NAME" == "Llama-3.1-8B" ]]; then
  echo "Building torchtitan dependencies for $MODEL_NAME"
  echo "Torchtitan directory path: $TORCHTITAN_DIR"
  cd $TORCHTITAN_DIR
  
  # Download tokenizer files
  python scripts/download_tokenizer.py --repo_id meta-llama/Llama-3.1-8B --hf_token=$HF_TOKEN
fi 

if [[ "$MODEL_NAME" == "Llama-3.1-70B" ]]; then
  echo "Building torchtitan dependencies for $MODEL_NAME"
  echo "Torchtitan directory path: $TORCHTITAN_DIR"
  cd $TORCHTITAN_DIR
  
  # Download tokenizer files
  python scripts/download_tokenizer.py --repo_id meta-llama/Llama-3.1-8B --hf_token=$HF_TOKEN

  echo "Building torchtune dependencies for $MODEL_NAME"
  echo "Torchtune directory path: $TORCHTUNE_DIR"
  cd $TORCHTUNE_DIR
  hf download meta-llama/Llama-3.1-70B-Instruct \
    --local-dir ./models/Llama-3.1-70B-Instruct \
    --exclude 'original/*.pth'
  python dataset.py
fi

if [[ "$MODEL_NAME" == "Llama-3.3-70B" ]]; then
  echo "Building torchtune dependencies for $MODEL_NAME"
  echo "Torchtune directory path: $TORCHTUNE_DIR"
  cd $TORCHTUNE_DIR
  hf login --token $HF_TOKEN --add-to-git-credential
  hf download meta-llama/Llama-3.3-70B-Instruct \
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
  hf login --token $HF_TOKEN --add-to-git-credential
  make download_assets
  make download_assets
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "Building dependencies for all models"
  TORCHTITAN_DIR="/workspace/torchtitan"
  cd $TORCHTITAN_DIR

  # Download tokenizer files
  python scripts/download_tokenizer.py --repo_id meta-llama/Llama-3.1-8B --hf_token=$HF_TOKEN
  
  TORCHTUNE_DIR="/workspace/torchtune"
  cd $TORCHTUNE_DIR
  # Llama 3.1 70B 
  hf login --token $HF_TOKEN --add-to-git-credential
  hf download meta-llama/Llama-3.1-70B-Instruct \
          --local-dir ./models/Llama-3.1-70B-Instruct \
          --exclude 'original/*.pth'
  # Llama 3.3 70B 
  hf login --token $HF_TOKEN --add-to-git-credential
  hf download meta-llama/Llama-3.3-70B-Instruct \
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
  hf login --token $HF_TOKEN --add-to-git-credential
  make download_assets
  make download_assets
fi
