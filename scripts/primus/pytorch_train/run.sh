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
done

echo "=hyper params start="
echo $MODEL_REPO
#echo $TRAINING_MODE
#echo $DATATYPE
#echo $SEQUENCE_LENGTH
echo "=hyper params end="

datatypes=("FP8" "BF16")
sequence_lengths=("2048")

if [[ "$MODEL_REPO" == "primus_pyt_train_llama-3.1-8b" ]]; then
  model="Llama-3.1-8B"
  tasks=("pretrain")
elif [[ "$MODEL_REPO" == "primus_pyt_train_llama-3.1-70b" ]]; then
  model="Llama-3.1-70B"
  tasks=("pretrain")
elif [[ "$MODEL_REPO" == "primus_pyt_train_deepseek-v2" ]]; then
  model="DeepSeek-V2"
  tasks=("pretrain")
fi

# Run pytorch setup script
bash ./primus_pytorch_benchmark_setup.sh -m $model

echo "Model: $model"
# Loop through all combinations
for task in "${tasks[@]}"; do
  for datatype in "${datatypes[@]}"; do
    for sequence_length in "${sequence_lengths[@]}"; do
      echo "Running: $task - $model - $datatype - $sequence_length"
      ./primus_pytorch_benchmark_report.sh -t $task -m $model -p $datatype -s $sequence_length
    done
  done
done

