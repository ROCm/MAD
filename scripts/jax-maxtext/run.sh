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

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    	--model_repo) MODEL_REPO="$2"; shift ;;
    	*) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
      --quantization) QUANTIZATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "Model repo: $MODEL_REPO"

if [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.1-8b" ]]; then
  model="Llama-3.1-8B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.1-70b" ]]; then
  model="Llama-3.1-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.3-70b" ]]; then
  model="Llama-3.3-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-2-7b" ]]; then
  model="Llama-2-7B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-2-70b" ]]; then
  model="Llama-2-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_deepseek-v2-lite-16b" ]]; then
  model="DeepSeek-V2-lite"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_mixtral-8x7b" ]]; then
  model="Mixtral-8x7B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_qwen3-14b" ]]; then
  model="Qwen3-14B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_qwen3-30b-a3b" ]]; then
  model="Qwen3-30B-A3B"
fi

./jax-maxtext_benchmark_setup.sh -m $model
./jax-maxtext_benchmark_report.sh -m $model -q $QUANTIZATION

echo "performance: 1 pass"
