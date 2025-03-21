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
#./megatron-lm_benchmark_report.sh -m $model_name 
## example:
## Pretrain Llama 3.1 70B
#./megatron-lm_benchmark_report.sh -m Llama-3.1-70B -p BF16
## Pretrain Llama 2 7B
#./megatron-lm_benchmark_report.sh -m Llama-2-7B -p FP8
## Pretrain DeepSeek V2 Lite
#./megatron-lm_benchmark_report.sh -m DeepSeek-V2-lite
MODEL_REPO=""
DATATYPE="BF16"

usage() {
    echo "Usage: $0 -m <model_repo> -p <datatype>"
    echo "\nOptions:"
    echo "  -m <model_repo>      Model repository (Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, DeepSeek-V2-lite)"
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
echo "Running training benchmark with the following parameters:"
echo "  Model Repository: $MODEL_REPO"

# config environment
export MOCK_DATA=1

# set performance output paths
TRAIN_LOG="$(pwd)/$MODEL_REPO-pretrain.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"

perf_script="$(pwd)/megatron-lm_benchmark_report.py"

# run models
if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  if [ "$DATATYPE" == "FP8" ]; then
    TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 FSDP=0 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  elif [ "$DATATYPE" == "BF16" ]; then
    TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=0 SEQ_LENGTH=8192 MODEL_SIZE=8 FSDP=0 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  #rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  TEE_OUTPUT=1 RECOMPUTE=1 MBS=3 BS=24 TP=1 TE_FP8=0 SEQ_LENGTH=8192 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  #rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-2-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  if [ "$DATATYPE" == "FP8" ]; then
    TEE_OUTPUT=1 MBS=4 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=4096 MODEL_SIZE=7 FSDP=0 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh |& tee $TRAIN_LOG
  elif [ "$DATATYPE" == "BF16" ]; then
    TEE_OUTPUT=1 MBS=4 BS=256 TP=1 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=7 FSDP=0 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh |& tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  #rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-2-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  TEE_OUTPUT=1 RECOMPUTE=1 MBS=7 BS=56 TP=1 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  #rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "DeepSeek-V2-lite" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  GEMM_TUNING=1 PR=bf16 MBS=4 AC=none SEQ_LEN=4096 PAD_LEN=4096 TRAIN_ITERS=50 bash examples/deepseek_v2/train_deepseekv2.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  #rm $TRAIN_LOG

else
    echo "Error: Unsupported training mode."
    exit 1
fi
