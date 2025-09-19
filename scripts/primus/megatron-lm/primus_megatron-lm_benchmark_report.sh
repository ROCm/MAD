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
## Usage: 
#./primus_megatron-lm_benchmark_report.sh -m $model_name 
## example:
## Pretrain Llama 3.1 70B
#./primus_megatron-lm_benchmark_report.sh -m Llama-3.1-70B -p BF16
## Pretrain Llama 2 7B
#./primus_megatron-lm_benchmark_report.sh -m Llama-2-7B -p FP8
## Pretrain DeepSeek V2 Lite
#./primus_megatron-lm_benchmark_report.sh -m DeepSeek-V2-lite
## Pretrain DeepSeek V3
#./primus_megatron-lm_benchmark_report.sh -m DeepSeek-V3
## Pretrain Mixtral 8x7B
#./primus_megatron-lm_benchmark_report.sh -m Mixtral-8x7B -l 4
MODEL_REPO=""
DATATYPE="BF16"

usage() {
    echo "Usage: $0 -m <model_repo> -p <datatype> -l <layers>"
    echo "\nOptions:"
    echo "  -m <model_repo>      Model repository (Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, DeepSeek-V2-lite, DeepSeek-V3-proxy, Mixtral-8x7B, Mixtral-8x22B-proxy)"
    echo "  -p <datatype>        Precision type (FP8 or BF16)"
    exit 1
}

# Parse command-line arguments
while getopts "m:p:" opt; do
    case "$opt" in
        m) MODEL_REPO="$OPTARG" ;;
        p) DATATYPE="$OPTARG" ;;
        *) usage ;;
    esac
done

echo "=hyper params start="
echo $MODEL_REPO
echo $DATATYPE
echo "=hyper params end="

# Validate inputs
if [[ -z "$MODEL_REPO" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$DATATYPE" != "FP8" && "$DATATYPE" != "BF16" ]]; then
    echo "Error: Datatype must be either FP8 or BF16."
    exit 1
fi

# Run benchmark (Placeholder for actual script execution)
echo "Running primus training benchmark with the following parameters:"
echo "  Model Repository: $MODEL_REPO"

# config environment
export MOCK_DATA=1

# set performance output paths
TRAIN_LOG="$(pwd)/primus-$MODEL_REPO-pretrain.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_primus-$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"

perf_script="$(pwd)/primus_megatron-lm_benchmark_report.py"

# Set common environment variables
export NNODES=1
export CPUS_PER_TASK=128
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

cd /workspace/Primus

# run models
if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml
  if [ "$DATATYPE" == "FP8" ]; then
    bash ./examples/run_pretrain.sh --train_iters 50 --fp8 hybrid 2>&1 | tee $TRAIN_LOG
  elif [ "$DATATYPE" == "BF16" ]; then
    bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/llama3.1_70B-pretrain.yaml
  bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.1-70B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/llama3.1_70B-pretrain.yaml
  bash ./examples/run_pretrain.sh --train_iters 50 --num_layers 40 --fp8 hybrid 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.3-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/llama3_70B-pretrain.yaml
  bash ./examples/run_pretrain.sh --micro_batch_size 2 --global_batch_size 16 --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-2-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export PRIMUS_MBS=4
  export PRIMUS_GBS=256
  export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml
  if [ "$DATATYPE" == "FP8" ]; then
    bash ./examples/run_pretrain.sh --train_iters 50 --fp8 hybrid 2>&1 | tee $TRAIN_LOG
  elif [ "$DATATYPE" == "BF16" ]; then
    bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-2-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/llama2_70B-pretrain.yaml
  bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "DeepSeek-V2-lite" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/deepseek_v2_lite-pretrain.yaml
  bash ./examples/run_pretrain.sh --global_batch_size 256 --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "DeepSeek-V3-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/deepseek_v3-pretrain.yaml
  bash ./examples/run_pretrain.sh --num_layers 3 --moe_layer_freq 1 --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Mixtral-8x7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/mixtral_8x7B_v0.1-pretrain.yaml
  bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Mixtral-8x22B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  LAYERS=4 # default proxy model uses 4 layers
  echo "[INFO] Proxy model uses $LAYERS layers"
  export EXP=examples/megatron/configs/mixtral_8x22B_v0.1-pretrain.yaml
  bash ./examples/run_pretrain.sh --num_layers 4 --pipeline_model_parallel_size 1 --micro_batch_size 1  --global_batch_size 16 --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Qwen2.5-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/qwen2.5_7B-pretrain.yaml
  if [[ "$DATATYPE" == "BF16" ]]; then
    bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  elif [[ "$DATATYPE" == "FP8" ]]; then
    bash ./examples/run_pretrain.sh --train_iters 50 --fp8 hybrid 2>&1 | tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Qwen2.5-72B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"  
  export EXP=examples/megatron/configs/qwen2.5_72B-pretrain.yaml
  bash ./examples/run_pretrain.sh --train_iters 50 2>&1 | tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

else
    echo "Error: Unsupported training mode."
    exit 1
fi
