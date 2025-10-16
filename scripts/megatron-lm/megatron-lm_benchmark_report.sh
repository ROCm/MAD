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
## Pretrain DeepSeek V3
#./megatron-lm_benchmark_report.sh -m DeepSeek-V3
## Pretrain Mixtral 8x7B
#./megatron-lm_benchmark_report.sh -m Mixtral-8x7B -l 4
MODEL_REPO=""
DATATYPE="BF16"

usage() {
    echo "Usage: $0 -m <model_repo> -p <datatype> -l <layers>"
    echo "\nOptions:"
    echo "  -m <model_repo>      Model repository (Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, DeepSeek-V2-lite, DeepSeek-V3-proxy, Mixtral-8x7B, Mixtral-8x22B-proxy)"
    echo "  -p <datatype>        Precision type (FP8 or BF16)"
    echo "  -l <layers>          Number of proxy layers in model"
    exit 1
}

# Parse command-line arguments
while getopts "m:p:l:" opt; do
    case "$opt" in
        m) MODEL_REPO="$OPTARG" ;;
        p) DATATYPE="$OPTARG" ;;
        l) LAYERS="$OPTARG" ;;
        *) usage ;;
    esac
done

echo "=hyper params start="
echo $MODEL_REPO
echo $DATATYPE
echo $LAYERS
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

if [[ "$MODEL_REPO" == "DeepSeek-V3-proxy" ]]; then
    if (( LAYERS < 0 || LAYERS > 3 )); then 
      echo "Error: Maximum number of proxy layers for DeepSeek-V3 is 3."
      exit 1
    fi 
fi

if [[ "$MODEL_REPO" == "Mixtral-8x22B-proxy" ]]; then
    if (( LAYERS < 0 || LAYERS > 4 )); then 
      echo "Error: Maximum number of proxy layers for Mixtral-8x22B is 4."
      exit 1
    fi
fi

if [[ "$MODEL_REPO" == "DeepSeek-V2-lite" ]]; then
    echo "Error: Running DeepSeek-V2-lite is not supported in Megatron-LM v25.9."
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
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 MBS=3 BS=24 TP=1 TE_FP8=0 SEQ_LENGTH=8192 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.1-70B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  FP8_WEIGHT_TRANSPOSE_CACHE=0 CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 MBS=3 BS=24 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=10 NUM_LAYERS=40 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-3.3-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  TOKENIZER_MODEL="meta-llama/Llama-3.3-70B-Instruct" CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 SEQ_LENGTH=8192 MBS=2 BS=16 TE_FP8=0 TP=1 PP=1 FSDP=1 MODEL_SIZE=70 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

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
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Llama-2-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 MBS=7 BS=56 TP=1 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

# elif [ "$MODEL_REPO" == "DeepSeek-V2-lite" ]; then
#   echo "[INFO] $MODEL_REPO TRAINING"
#   export NVTE_FUSED_ATTN_CK=0
#   cd /workspace/Megatron-LM
#   GEMM_TUNING=1 PR=bf16 MBS=4 AC=none SEQ_LEN=4096 PAD_LEN=4096 TRAIN_ITERS=20 bash examples/deepseek_v2/train_deepseekv2.sh |& tee $TRAIN_LOG
#   echo "[INFO] Benchmarking"
#   python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
#   export NVTE_FUSED_ATTN_CK=1
#   rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "DeepSeek-V3-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export NVTE_FUSED_ATTN_CK=0
  cd /workspace/Megatron-LM
  if [[ -z "$LAYERS" ]]; then
    LAYERS=3 # default proxy model uses 3 layers
  fi
  echo "[INFO] Proxy model uses $LAYERS layers"
  FORCE_BANLANCE=true RUN_ENV=cluster MODEL_SIZE=671B TRAIN_ITERS=50 SEQ_LEN=4096 NUM_LAYERS=$LAYERS MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 PR=bf16 TP=1 PP=1 ETP=1 EP=8 GEMM_TUNING=1 NVTE_CK_USES_BWD_V3=1 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true MOCK_DATA=1 bash examples/deepseek_v3/train_deepseekv3.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  export NVTE_FUSED_ATTN_CK=1
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Mixtral-8x7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  TOKENIZER_MODEL=/tokenizer/tokenizer.model RECOMPUTE_NUM_LAYERS=0 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=none PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true MOCK_DATA=1 RUN_ENV=cluster MODEL_SIZE=8x7B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh |& tee $TRAIN_LOG
  echo "[INFO] Proxy model uses $LAYERS layers"
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Mixtral-8x22B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  if [[ -z "$LAYERS" ]]; then
    LAYERS=4 # default proxy model uses 4 layers
  fi
  echo "[INFO] Proxy model uses $LAYERS layers"
  TOKENIZER_MODEL=/tokenizer/tokenizer.model RECOMPUTE_NUM_LAYERS=$LAYERS TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=full  NUM_LAYERS=$LAYERS PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=8192 FORCE_BALANCE=true MOCK_DATA=1 RUN_ENV=cluster MODEL_SIZE=8x22B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Qwen2.5-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  if [[ "$DATATYPE" == "BF16" ]]; then
    bash examples/qwen/train_qwen2.sh TP=1 CP=1 PP=1 MBS=10 BS=640 TE_FP8=0 MODEL_SIZE=7 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-7B |& tee $TRAIN_LOG
  elif [[ "$DATATYPE" == "FP8" ]]; then
    bash examples/qwen/train_qwen2.sh TP=1 CP=1 PP=1 MBS=10 BS=640 TE_FP8=1 MODEL_SIZE=7 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-7B |& tee $TRAIN_LOG
  fi
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

elif [ "$MODEL_REPO" == "Qwen2.5-72B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  cd /workspace/Megatron-LM
  bash examples/qwen/train_qwen2.sh FSDP=1 CP=1 PP=1 MBS=3 BS=24 TE_FP8=0 MODEL_SIZE=72 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-72B RECOMPUTE_ACTIVATIONS=full CKPT_FORMAT=torch_dist |& tee $TRAIN_LOG
  echo "[INFO] Benchmarking"
  python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG
  rm $TRAIN_LOG

else
    echo "Error: Unsupported training mode."
    exit 1
fi
