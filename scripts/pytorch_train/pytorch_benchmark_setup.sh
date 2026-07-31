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
export HF_HOME=/workspace/huggingface

usage() {
    echo "Usage: $0 -m MODEL_NAME"
    echo "  -m: Model name (Flux, Stable-Diffusion-XL, Mochi-1, Hunyuan-video, Wan2_1-i2v, DLRM, or empty for all)"
    echo "  Example: $0 -m Flux"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "Model: $MODEL_NAME"
echo "Setup script starting in directory $(pwd)"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here"
    exit 1
fi

hf auth login --token $HF_TOKEN

if [[ "$MODEL_NAME" == "Flux" || "$MODEL_NAME" == "Stable-Diffusion-XL" || "$MODEL_NAME" == "Mochi-1" || "$MODEL_NAME" == "Hunyuan-video" || "$MODEL_NAME" == "Wan2_1-i2v" ]]; then
  echo "Building AMDiffusionBenchmark dependencies for $MODEL_NAME"
  cd /workspace/AMDiffusionBenchmark
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  make download_assets
fi

if [[ "$MODEL_NAME" == "DLRM" ]]; then
  echo "Building dependencies for $MODEL_NAME"
  cd /workspace/DLRMBenchmark
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "Building dependencies for all supported models"
  cd /workspace/AMDiffusionBenchmark
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential
  make download_assets
fi
