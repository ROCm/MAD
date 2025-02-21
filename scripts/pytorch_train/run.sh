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

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_REPO="$2"; shift ;;
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

# Run pytorch setup script
bash ./pytorch_benchmark_setup.sh

# Set script parameters
datatypes=("BF16" "FP8")
sequence_lengths=("8192" "4096" "2048")

# Function to run the benchmark for each combination
run_benchmark() {
  local model=$1
  local datatypes=$2
  local sequence_lengths=$3
  local tasks=("pretrain")

  # Add tasks based on the model
  if [[ "$model" == "pyt_train_llama-3.1-70b" && "$datatype" == "BF16" ]]; then
    tasks+=("finetune_fw" "finetune_lora")
  fi

  # Flux does not require datatype or sequence length
  if [[ "$model" == "pyt_train_flux" ]]; then
    datatype=""
    sequence_length=""
    tasks=("pretrain")  # Flux only runs pretrain
  fi

  #echo "Tasks: ${tasks[@]}"
  # Loop through tasks
  for task in "${tasks[@]}"; do
    echo "Running: $task - $model - $datatype - $sequence_length"
    #./pytorch_benchmark_report.sh -t $task -m $model -p $datatype -s $sequence_length
  done
}

# Loop through all combinations
for datatype in "${datatypes[@]}"; do
  for sequence_length in "${sequence_lengths[@]}"; do
    run_benchmark $MODEL_NAME $datatype $sequence_length
  done
done

#./pytorch_benchmark_report.sh -t $TRAINING_MODE -m $MODEL_REPO -d $DATATYPE -s $SEQUENCE_LENGTH
echo "performance: 1 pass"
