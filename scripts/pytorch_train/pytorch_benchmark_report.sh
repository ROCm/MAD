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
## Usage: 
#./pytorch_benchmark_report.sh -t $training_mode -m $model_name -d $datatype -s $sequence_length
## example:
## Pretrain Llama 3.1 70B with BF16 precision
#./pytorch_benchmark_report.sh -t pretrain -m Llama-3.1-70B -p BF16 -s 8192
## Pretrain Llama 3.1 8B with FP8 precision 
#./pytorch_benchmark_report.sh -t pretrain -m Llama-3.1-8B -p FP8 -s 8192
## Torchtune full weight finetuning with Llama 3.1 70B
#./pytorch_benchmark_report.sh -t finetune_fw -m Llama-3.1-70B -p BF16 -s 8192
# Function to display help message
TRAINING_MODE="pretrain"
DATATYPE="BF16"
MODEL_REPO=""
SEQUENCE_LENGTH="8192"

usage() {
    echo "Usage: $0 -t <training_mode> -m <model_repo> -p <datatype> -s <sequence_length>"
    echo "\nOptions:"
    echo "  -t <training_mode>   Training mode (pretrain, finetune_fw, finetune_lora)"
    echo "  -m <model_repo>      Model repository (Llama_3.1_8B, Llama_3.1_70B, Flux)"
    echo "  -p <datatype>        Precision type (FP8 or BF16)"
    echo "  -s <sequence_length> Sequence length (between 2048 and 8192)"
    exit 1
}

# Parse command-line arguments
while getopts "t:m:p:s:" opt; do
    case "$opt" in
        t) TRAINING_MODE="$OPTARG" ;;
        m) MODEL_REPO="$OPTARG" ;;
        p) DATATYPE="$OPTARG" ;;
        s) SEQUENCE_LENGTH="$OPTARG" ;;
        *) usage ;;
    esac
done

echo "=hyper params start="
echo $TRAINING_MODE
echo $MODEL_REPO
echo $DATATYPE
echo $SEQUENCE_LENGTH
echo "=hyper params end="

# Validate inputs
if [[ -z "$TRAINING_MODE" || -z "$MODEL_REPO" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$DATATYPE" != "FP8" && "$DATATYPE" != "BF16" ]]; then
    echo "Error: Datatype must be either FP8 or BF16."
    exit 1
fi

if ! [[ "$SEQUENCE_LENGTH" =~ ^[0-9]+$ ]] || (( SEQUENCE_LENGTH < 2048 || SEQUENCE_LENGTH > 8192 )); then
    echo "Error: Sequence length must be between 2048 and 8192."
    exit 1
fi

# Training mode validation
if [[ "$TRAINING_MODE" == "finetune_fw" || "$TRAINING_MODE" == "finetune_lora" ]]; then
    if [[ "$MODEL_REPO" != "Llama-3.1-70B" || "$DATATYPE" != "BF16" ]]; then
        echo "Error: finetune_fw and finetune_lora are only supported for Llama_3.1_70B with BF16."
        exit 1
    fi
fi

# Run benchmark (Placeholder for actual script execution)
echo "Running training benchmark with the following parameters:"
echo "  Training Mode: $TRAINING_MODE"
echo "  Model Repository: $MODEL_REPO"
echo "  Datatype: $DATATYPE"
echo "  Sequence Length: $SEQUENCE_LENGTH"

# config environment
export HF_HOME=/workspace/huggingface
export ROCBLAS_USE_HIPBLASLT=1
export DISABLE_ADDMM_CUDA_LT=0
export HIP_FORCE_DEV_KERNARG=1
export TORCH_NCCL_HIGH_PRIORITY=0
export GPU_MAX_HW_QUEUES=8

if [[ "$TRAINING_MODE" == "pretrain" ]]; then
    echo "[INFO] Executing pretraining benchmark..."
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] LLAMA 3.1 8B TRAINING"
      cd llama3_1_8B
      echo "[INFO] Benchmarking"
      accelerate launch --config_file fsdp_fp8.yaml ./train_llama.py --max_seq_len=$SEQUENCE_LENGTH --batch_size=3
    fi

    if [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
      echo "[INFO] LLAMA 3.1 70B TRAINING"
      cd /workspace/torchtitan
      echo "[INFO] Benchmarking"
      bash run_llama_train.sh
    fi

    if [ "$MODEL_REPO" == "Flux" ]; then
      echo "[INFO] FLUX TRAINING"
      cd /workspace/FluxBenchmark
      echo "[INFO] Benchmarking"
      python launcher.py 
    fi

elif [[ "$TRAINING_MODE" == "finetune_fw" ]]; then
    echo "[INFO]Executing full-weight finetuning benchmark..."
    TORCHTUNE_DIR="/workspace/torchtune"
    echo "[INFO] LLAMA 3.1 70B"
    echo "[INFO] Benchmarking"
    cd $TORCHTUNE_DIR
    MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True \
		CPU_OFFLOAD=False PACKED=False SEQ_LEN=null \
		ACTIVATION_CHECKPOINTING=True TUNE_ENV=True \
		MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True \
		MAX_STEPS=30 bash wikitext_finetune.sh
    
elif [[ "$TRAINING_MODE" == "finetune_lora" ]]; then
    echo "Executing LoRA finetuning benchmark..."
    TORCHTUNE_DIR="/workspace/torchtune"
    echo "[INFO] LLAMA 3.1 70B"
    echo "[INFO] Benchmarking"
    cd $TORCHTUNE_DIR
    MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True \
		CPU_OFFLOAD=False PACKED=False SEQ_LEN=null \
		ACTIVATION_CHECKPOINTING=True TUNE_ENV=True \
		MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True \
		MAX_STEPS=30 bash wikitext_finetune.sh
else
    echo "Error: Unsupported training mode."
    exit 1
fi
