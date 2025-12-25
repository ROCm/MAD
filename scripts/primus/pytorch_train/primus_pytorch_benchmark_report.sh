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
#./pytorch_benchmark_report.sh -t $training_mode -m $model_name -d $datatype -s $sequence_length
## example:
## Pretrain Llama 3.1 70B with BF16 precision
#./pytorch_benchmark_report.sh -t pretrain -m Llama-3.1-70B -p BF16 -s 8192
## Pretrain Llama 3.1 8B with FP8 precision
#./pytorch_benchmark_report.sh -t pretrain -m Llama-3.1-8B -p FP8 -s 8192
## Torchtune full weight finetuning with Llama 3.1 70B
#./pytorch_benchmark_report.sh -t finetune_fw -m Llama-3.1-70B -p BF16 -s 8192
## Torchtune HF LoRA finetuning with Llama 2 70B
#./pytorch_benchmark_report.sh -t HF_finetune_lora -m Llama-2-70B -p BF16 -s 8192
# Function to display help message

TRAINING_MODE="pretrain"
DATATYPE="BF16"
MODEL_REPO=""
SEQUENCE_LENGTH="8192"
NUM_GPUS="8"
BATCH_SIZE=""

usage() {
    echo "Usage: $0 -t <training_mode> -m <model_repo> -p <datatype> -s <sequence_length> -n <num_gpus> -f <fsdp>"
    echo "\nOptions:"
    echo "  -t <training_mode>   Training mode (pretrain, HF_pretrain, finetune_fw, finetune_lora, finetune_qlora, HF_finetune_lora)"
    echo "  -m <model_repo>      Model repository (Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Flux)"
    echo "  -p <datatype>        Precision type (FP8 or BF16)"
    echo "  -s <sequence_length> Sequence length (between 2048 and 8192)"
    echo "  -n <num_gpus>        Number of GPUs (1 or 8)"
    echo "  -f <fsdp>            Use FSDP (default: False)"
    echo "  -b <fsdp>            Batch size (between 1 and 32)"
    exit 1
}

# Parse command-line arguments
while getopts "t:m:p:s:n:f:b:" opt; do
    case "$opt" in
        t) TRAINING_MODE="$OPTARG" ;;
        m) MODEL_REPO="$OPTARG" ;;
        p) DATATYPE="$OPTARG" ;;
        s) SEQUENCE_LENGTH="$OPTARG" ;;
        n) NUM_GPUS="$OPTARG" ;;
        f) FSDP="$OPTARG" ;;
		b) BATCH_SIZE="$OPTARG" ;;
        *) usage ;;
    esac
done

# Validate inputs
if [[ -z "$TRAINING_MODE" || -z "$MODEL_REPO" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$DATATYPE" != "FP8" && "$DATATYPE" != "BF16" ]]; then
    echo "Error: Datatype must be either FP8 or BF16."
fi

if ! [[ "$SEQUENCE_LENGTH" =~ ^[0-9]+$ ]] || (( SEQUENCE_LENGTH < 2048 || SEQUENCE_LENGTH > 8192 )); then
    echo "Error: Sequence length must be between 2048 and 8192."
fi

# Training mode validation
if [[ "$TRAINING_MODE" == "HF_finetune_lora" ]]; then
    if [[ ! ("$MODEL_REPO" == "GPT-OSS-20B" || "$MODEL_REPO" == "GPT-OSS-120B") || "$DATATYPE" != "BF16" ]]; then
        echo "Error: finetuning options are only supported for GPT-OSS-20B and GPT-OSS-120B with BF16."
    fi
fi

if [[ "$TRAINING_MODE" == "HF_pretrain" ]]; then
    if [[ "$MODEL_REPO" != "Llama-3.1-8B" ]]; then
        echo "Error: HF pretraining option are only supported for Llama_3.1_8B."
    fi
fi

# Check for incompatible FP8 + finetune_lora combination
if [[ "$TRAINING_MODE" == "finetune_lora" && "$DATATYPE" == "FP8" ]]; then
    echo "Error: finetune_lora is not supported with FP8 precision."
fi

# Check for incompatible finetune_fw + large Qwen models combination
if [[ "$TRAINING_MODE" == "finetune_fw" && ("$MODEL_REPO" == "Qwen3-32B" || "$MODEL_REPO" == "Qwen2.5-72B" || "$MODEL_REPO" == "Qwen2.5-32B") ]]; then
    echo "Error: finetune_fw is not supported for Qwen3-32B, Qwen2.5-72B, and Qwen2.5-32B models."
fi

if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "8" ]]; then
    echo "Error: Number of GPUs must be either 1 or 8."
fi

# Run benchmark (Placeholder for actual script execution)
echo "Running training benchmark with the following parameters:"
echo "  Training Mode: $TRAINING_MODE"
echo "  Model Repository: $MODEL_REPO"
echo "  Datatype: $DATATYPE"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Number of GPUs: $NUM_GPUS"
echo "  FSDP: $FSDP"
echo "  Batch size: $BATCH_SIZE"

TRAIN_LOG="$(pwd)/primus-pytorch-$MODEL_REPO-$TRAINING_MODE.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_primus-pytorch-$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"

perf_script="$(pwd)/primus_pytorch_benchmark_report.py"

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
if [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
  export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
  export NVTE_CK_IS_V3_ATOMIC_FP32=1
fi
export HSA_NO_SCRATCH_RECLAIM=1

if [[ "$TRAINING_MODE" == "pretrain" ]]; then
    echo "[INFO] Executing pretraining benchmark..."
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] Benchmarking LLAMA 3.1 8B TRAINING"
      cd /workspace/Primus
      SEQUENCE_LENGTH=8192
      CONFIG_FILE=$(pwd)/examples/torchtitan/configs/$DEVICE/llama3.1_8B-$DATATYPE-pretrain.yaml
      # Extract batch size from CONFIG_FILE if not provided
      if [ -z "$BATCH_SIZE" ] && [ -f "$CONFIG_FILE" ]; then
        BATCH_SIZE=$(grep -E "^\s*local_batch_size:" $CONFIG_FILE | head -n1 | awk '{print $2}' | tr -d '\r')
        echo "[INFO] Extracted batch size from config: $BATCH_SIZE"
      fi
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "FP8" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "FP8" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE
    fi

    if [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
      echo "[INFO] Benchmarking LLAMA 3.1 70B TRAINING"
      cd /workspace/Primus
      SEQUENCE_LENGTH=8192
      CONFIG_FILE=$(pwd)/examples/torchtitan/configs/$DEVICE/llama3.1_70B-$DATATYPE-pretrain.yaml
      # Extract batch size from CONFIG_FILE if not provided
      if [ -z "$BATCH_SIZE" ] && [ -f "$CONFIG_FILE" ]; then
        BATCH_SIZE=$(grep -E "^\s*local_batch_size:" $CONFIG_FILE | head -n1 | awk '{print $2}' | tr -d '\r')
        echo "[INFO] Extracted batch size from config: $BATCH_SIZE"
      fi
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "FP8" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "FP8" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE
    fi

    if [ "$MODEL_REPO" == "DeepSeek-V3-16B" ]; then
      echo "[INFO] Benchmarking DeepSeek-V3-16B TRAINING"
      cd /workspace/Primus
      SEQUENCE_LENGTH=4096
      CONFIG_FILE=$(pwd)/examples/torchtitan/configs/$DEVICE/deepseek_v3_16b-pretrain.yaml
      # Extract batch size from CONFIG_FILE if not provided
      if [ -z "$BATCH_SIZE" ] && [ -f "$CONFIG_FILE" ]; then
        BATCH_SIZE=$(grep -E "^\s*local_batch_size:" $CONFIG_FILE | head -n1 | awk '{print $2}' | tr -d '\r')
        echo "[INFO] Extracted batch size from config: $BATCH_SIZE"
      fi
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "BF16" ]]; then
        EXP=$CONFIG_FILE bash ./examples/run_pretrain.sh |& tee $TRAIN_LOG
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE
    fi

else
    echo "Error: Unsupported training mode."
    exit 1
fi
