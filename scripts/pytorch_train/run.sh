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

datatypes=("BF16" "FP8")
sequence_lengths=("8192")

# Convert from MAD repo names to standalone script names and set training tasks
if [[ "$MODEL_REPO" == "pyt_train_llama-2-7b" ]]; then
  model="Llama-2-7B"
  tasks=("finetune_fw" "finetune_lora" "finetune_qlora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-2-13b" ]]; then
  model="Llama-2-13B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-2-70b" ]]; then
  model="Llama-2-70B"
  tasks=("finetune_lora" "finetune_qlora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3-8b" ]]; then
  model="Llama-3-8B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3-70b" ]]; then
  model="Llama-3-70B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.1-8b" ]]; then
  model="Llama-3.1-8B"
  tasks=("pretrain" "HF_pretrain" "finetune_fw" "finetune_lora")
  #tasks=("pretrain")
  #tasks=("HF_pretrain")
  #tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.1-70b" ]]; then
  model="Llama-3.1-70B"
  #datatypes=("FP8")
  tasks=("pretrain" "finetune_fw" "finetune_lora")
  #tasks=("pretrain")
  #tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.1-405b" ]]; then
  model="Llama-3.1-405B"
  tasks=("finetune_qlora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.2-1b" ]]; then
  model="Llama-3.2-1B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.2-3b" ]]; then
  model="Llama-3.2-3B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.2-vision-11b" ]]; then
  model="Llama-3.2-vision-11B"
  tasks=("finetune_fw")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.2-vision-90b" ]]; then
  model="Llama-3.2-vision-90B"
  tasks=("finetune_fw")

elif [[ "$MODEL_REPO" == "pyt_train_llama-3.3-70b" ]]; then
  model="Llama-3.3-70B"
  tasks=("finetune_fw" "finetune_lora" "finetune_qlora")

elif [[ "$MODEL_REPO" == "pyt_train_llama-4-scout-17b-16e" ]]; then
  model="Llama-4-scout-17B-16E"
  tasks=("finetune_fw" "finetune_lora")
  #tasks=("finetune_fw")

elif [[ "$MODEL_REPO" == "pyt_train_flux" ]]; then
  model="Flux"
  datatypes=("BF16")
  tasks=("posttrain")  

elif [[ "$MODEL_REPO" == "pyt_train_stable-diffusion-xl" ]]; then
  model="Stable-Diffusion-XL"
  datatypes=("BF16")
  tasks=("posttrain")
  
elif [[ "$MODEL_REPO" == "pyt_train_gpt_oss_20b" ]]; then
  model="GPT-OSS-20B"
  datatypes=("BF16")
  tasks=("HF_finetune_lora")  
  
elif [[ "$MODEL_REPO" == "pyt_train_gpt_oss_120b" ]]; then
  model="GPT-OSS-120B"
  datatypes=("BF16")
  tasks=("HF_finetune_lora") 

elif [[ "$MODEL_REPO" == "pyt_train_qwen2-1.5b" ]]; then
  model="Qwen2-1.5B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_qwen2-7b" ]]; then
  model="Qwen2-7B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_qwen2.5-32b" ]]; then
  model="Qwen2.5-32B"
  tasks=("finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_qwen2.5-72b" ]]; then
  model="Qwen2.5-72B"
  tasks=("finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_qwen3-8b" ]]; then
  model="Qwen3-8B"
  tasks=("finetune_fw" "finetune_lora")

elif [[ "$MODEL_REPO" == "pyt_train_qwen3-32b" ]]; then
  model="Qwen3-32B"
  tasks=("finetune_lora")
fi

# Run pytorch setup script
bash ./pytorch_benchmark_setup.sh -m $model

echo "Model: $model"
# Loop through all combinations
for task in "${tasks[@]}"; do
  if [[ "$task" == "HF_pretrain" ]]; then
    curr_datatypes=("FP8")
  elif [[ "$task" == "finetune_lora" || "$task" == "finetune_qlora" ]]; then
    curr_datatypes=("BF16")
  else 
    curr_datatypes=("${datatypes[@]}")
  fi
  for datatype in "${curr_datatypes[@]}"; do
    for sequence_length in "${sequence_lengths[@]}"; do
      echo "Running: $task - $model - $datatype - $sequence_length"
      ./pytorch_benchmark_report.sh -t $task -m $model -p $datatype -s $sequence_length
    done
  done
done

