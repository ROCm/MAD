#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
## Usage:
#./jax-maxtext_benchmark_report.sh  -m $model_name -q $quantization


# Parse command-line arguments
while getopts "m:q:" opt; do
    case "$opt" in
        m) MODEL_REPO="$OPTARG" ;;
        q) QUANTIZATION="$OPTARG" ;;
        *) usage ;;
    esac
done

# Set default values for additional parameters
MODE="pretrain"
NNODES=1 # default to 1 node
GPUS_PER_NODE=8 # default to 8 GPUs per node
NUM_GPUS=$((NNODES*GPUS_PER_NODE))

echo "=hyper params start="
echo $MODEL_REPO
echo $QUANTIZATION
echo "=hyper params end="

if [ -z "$QUANTIZATION" ]; then
  PERF_LOG="$(pwd)/../perf_${MODEL_REPO}.csv"
else
  PERF_LOG="$(pwd)/../perf_${MODEL_REPO}_${QUANTIZATION}.csv"
fi
perf_script="$(pwd)/jax-maxtext_benchmark_report.py"

# Run rocminfo and grep for "AMD Instinct"
DEVICE=$(/opt/rocm/bin/rocminfo | grep "AMD Instinct" | head -n1 | awk '{print $5}')
if [ -z "$DEVICE" ]; then
  ARCH=$(/opt/rocm/bin/rocminfo | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  case "$ARCH" in
    "gfx942") DEVICE="MI300X" ;;
    "gfx950") DEVICE="MI355X" ;;
    *) DEVICE="" ;;
  esac
fi             
echo "GPU DEVICE name: $DEVICE"

MAXTEXT="/workspace/maxtext"
MAXTEXT_DIR="/workspace/maxtext/MaxText"
ENV_SCRIPT_DIR="$(pwd)/env_scripts"

cd $MAXTEXT


execute_training(){
  gpu_architecture=$(rocminfo | grep -o -m 1 'gfx.*' | xargs )
  env_file=$ENV_SCRIPT_DIR/$1
  if test -e $ENV_SCRIPT_DIR/$gpu_architecture"_"$1; then
    env_file=$ENV_SCRIPT_DIR/$gpu_architecture"_"$1
  fi
  config_file=$ENV_SCRIPT_DIR/$2
  if test -e $ENV_SCRIPT_DIR/$gpu_architecture"_"$2; then
    config_file=$ENV_SCRIPT_DIR/$gpu_architecture"_"$2
  fi

  # output for logging
  echo "Using env file:"
  echo $env_file
  cat $env_file

  echo "Using yaml config file:"
  echo $config_file
  cat $config_file

  yaml() {
      python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
  }

  per_device_batch_size=$(yaml $config_file "['per_device_batch_size']")
  max_target_length=$(yaml $config_file "['max_target_length']")
  echo $per_device_batch_size
  echo $max_target_length

  # execute
  source $env_file
  python -m MaxText.train $config_file \
  quantization=$3 2>&1 |& tee -a  $2.log
  if [ -z "$3" ]; then
    python3 $perf_script --model $MODEL_REPO --input $MAXTEXT/$2.log --output $PERF_LOG --mode $MODE --quantization bf16 --batch_size $per_device_batch_size --seq_len $max_target_length --device $DEVICE --num_gpus $NUM_GPUS
  else
    python3 $perf_script --model $MODEL_REPO --input $MAXTEXT/$2.log --output $PERF_LOG --mode $MODE --quantization $3 --batch_size $per_device_batch_size --seq_len $max_target_length --device $DEVICE --num_gpus $NUM_GPUS
  fi

}


if [[ "$MODEL_REPO" == "Llama-2-7B" ]]; then
  echo "[INFO] LLAMA 2 7B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama2_7b_env.sh llama2_7b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Llama-2-70B" ]]; then
  echo "[INFO] LLAMA 2 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama2_70b_env.sh llama2_70b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Llama-3.1-8B" ]]; then
  echo "[INFO] LLAMA 3.1 8B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3_8b_env.sh llama3_8b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Llama-3.1-70B" ]]; then
  echo "[INFO] LLAMA 3.1 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3_70b_env.sh llama3_70b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Llama-3.3-70B" ]]; then
  echo "[INFO] LLAMA 3.3 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3.3_70b_env.sh llama3.3_70b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "DeepSeek-V2-lite" ]]; then
  echo "[INFO] DEEPSEEK V2 LITE TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training deepseek2_env_16b.sh deepseek2_16b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Mixtral-8x7B" ]]; then
  echo "[INFO] MIXTRAL-8x7B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training mixtral_8x7b_env.sh mixtral_8x7b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Qwen3-14B" ]]; then
  echo "[INFO] QWEN3-14B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training qwen3_14b_env.sh qwen3_14b.yml $QUANTIZATION

elif [[ "$MODEL_REPO" == "Qwen3-30B-A3B" ]]; then
  echo "[INFO] QWEN3-30B-A3B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training qwen3_30b_a3b_env.sh qwen3_30b_a3b.yml $QUANTIZATION

else
    echo "Error: Unsupported training mode."
    exit 1
fi
