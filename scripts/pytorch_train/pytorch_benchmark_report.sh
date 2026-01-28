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
## Posttrain Flux with BF16 precision
#./pytorch_benchmark_report.sh -t posttrain -m Flux -p BF16 -s 8192
## Train DLRM with TF32 precision
#./pytorch_benchmark_report.sh -t posttrain -m DLRM -p TF32
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
    echo "  -t <training_mode>   Training mode (pretrain, posttrain, HF_pretrain, finetune_fw, finetune_lora, finetune_qlora, HF_finetune_lora)"
    echo "  -m <model_repo>      Model repository (Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Flux, DLRM)"
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

NNODES=1 # default to 1 node
GPUS_PER_NODE=8 # default to 8 GPUs per node
WORLD_SIZE=$((NNODES*GPUS_PER_NODE)) # default to 8 GPUs per node

# Validate inputs
if [[ -z "$TRAINING_MODE" || -z "$MODEL_REPO" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$MODEL_REPO" != "DLRM" ]]; then
    if [[ "$DATATYPE" != "FP8" && "$DATATYPE" != "BF16" ]]; then
        echo "Error: Datatype must be either FP8 or BF16."
        exit 1
    fi

elif [[ "$DATATYPE" != "FP32" && "$DATATYPE" != "TF32" ]]; then
    echo "Error: For DLRM model, datatype must be either FP32 or TF32."
    exit 1
fi

if ! [[ "$SEQUENCE_LENGTH" =~ ^[0-9]+$ ]] || (( SEQUENCE_LENGTH < 2048 || SEQUENCE_LENGTH > 8192 )); then
    echo "Error: Sequence length must be between 2048 and 8192."
    exit 1
fi

# Training mode validation
if [[ "$TRAINING_MODE" == "HF_finetune_lora" ]]; then
    if [[ ! ("$MODEL_REPO" == "GPT-OSS-20B" || "$MODEL_REPO" == "GPT-OSS-120B") || "$DATATYPE" != "BF16" ]]; then
        echo "Error: finetuning options are only supported for GPT-OSS-20B and GPT-OSS-120B with BF16."
        exit 1
    fi
fi

if [[ "$TRAINING_MODE" == "HF_pretrain" ]]; then
    if [[ "$MODEL_REPO" != "Llama-3.1-8B" ]]; then
        echo "Error: HF pretraining option are only supported for Llama_3.1_8B."
        exit 1
    fi
fi

# Check for incompatible FP8 + finetune_lora combination
if [[ "$TRAINING_MODE" == "finetune_lora" && "$DATATYPE" == "FP8" ]]; then
    echo "Error: finetune_lora is not supported with FP8 precision."
    exit 1
fi

# Check for incompatible finetune_fw + large Qwen models combination
if [[ "$TRAINING_MODE" == "finetune_fw" && ("$MODEL_REPO" == "Qwen3-32B" || "$MODEL_REPO" == "Qwen2.5-72B" || "$MODEL_REPO" == "Qwen2.5-32B") ]]; then
    echo "Error: finetune_fw is not supported for Qwen3-32B, Qwen2.5-72B, and Qwen2.5-32B models."
    exit 1
fi

if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "8" ]]; then
    echo "Error: Number of GPUs must be either 1 or 8."
    exit 1
fi

# Check for llama 3.2 vision 90b - not allowed to run
if [[ "$MODEL_REPO" == "Llama-3.2-Vision-90B" ]]; then
    echo "Error: Running the script for Llama 3.2 Vision 90B is not supported in Pytorch v25.8."
    exit 1
fi

# Check for llama 4 with finetune lora - not allowed to run
if [[ "$MODEL_REPO" == "Llama-4" && "$TRAINING_MODE" == "finetune_lora" ]]; then
    echo "Error: Running Llama 4 with finetune lora is not supported in Pytorch v25.8."
    exit 1
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

# config environment
export HF_HOME=/workspace/huggingface
export ROCBLAS_USE_HIPBLASLT=1
export DISABLE_ADDMM_CUDA_LT=0
export HIP_FORCE_DEV_KERNARG=1
export TORCH_NCCL_HIGH_PRIORITY=0
export GPU_MAX_HW_QUEUES=8
export WANDB_DISABLED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_LOG="$SCRIPT_DIR/$MODEL_REPO-$TRAINING_MODE.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$SCRIPT_DIR/../perf_$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"

perf_script="$SCRIPT_DIR/pytorch_benchmark_report.py"

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

if [[ "$TRAINING_MODE" == "pretrain" ]]; then
    echo "[INFO] Executing pretraining benchmark..."
    TORCHTITAN_DIR="/workspace/torchtitan/torchtitan/models/llama3/train_configs/"
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] Benchmarking LLAMA 3.1 8B TRAINING"
      MAD_CONFIG_FILE="$(pwd)/torchtitan_scripts/llama3_8b-$DATATYPE.toml" 
      cp $MAD_CONFIG_FILE $TORCHTITAN_DIR
      CONFIG_FILE=$TORCHTITAN_DIR/llama3_8b-$DATATYPE.toml
      cd /workspace/torchtitan
      SEQUENCE_LENGTH=2048
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        BATCH_SIZE=6
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "FP8" ]]; then
        BATCH_SIZE=8
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "BF16" ]]; then
        BATCH_SIZE=3
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "FP8" ]]; then
        BATCH_SIZE=4
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

    if [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
      echo "[INFO] Benchmarking LLAMA 3.1 70B TRAINING"
      MAD_CONFIG_FILE="$(pwd)/torchtitan_scripts/llama3_70b-$DATATYPE.toml"
      cp $MAD_CONFIG_FILE $TORCHTITAN_DIR
      CONFIG_FILE=$TORCHTITAN_DIR/llama3_70b-$DATATYPE.toml
      cd /workspace/torchtitan
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        BATCH_SIZE=8
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "FP8" ]]; then
        BATCH_SIZE=6
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "BF16" ]]; then
        BATCH_SIZE=3
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI300X" || "$DEVICE" == "MI325X") && "$DATATYPE" == "FP8" ]]; then
        BATCH_SIZE=4
        CONFIG_FILE=$CONFIG_FILE bash run_train.sh --training.batch_size $BATCH_SIZE |& tee $TRAIN_LOG
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

    if [ "$MODEL_REPO" == "DLRM" ]; then
      echo "[INFO] Benchmarking DLRM TRAINING"
      if [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" || "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        cd /workspace/DLRMBenchmark
        echo "[INFO] Removing all previous runs to avoid caching"
        rm -rf training_logs/
        rm results.csv
        HSA_NO_SCRATCH_RECLAIM=1 ./launch_training_single_node.sh -p $DATATYPE
        TRAIN_LOG=results.csv
        BATCH_SIZE=32768
        python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
      else
        echo "Error: DLRM training is not supported on $DEVICE."
        exit 1
      fi
    fi

elif [[ "$TRAINING_MODE" == "HF_pretrain" ]]; then
    echo "[INFO] Executing HF pretraining benchmark..."
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] LLAMA 3.1 8B TRAINING with $DATATYPE precision"
      cd llama3_1_8B
      echo "[INFO] Benchmarking"
      if [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "BF16" ]]; then
        bash run_multigpu.sh |& tee $TRAIN_LOG	
      elif [[ ("$DEVICE" == "MI355X" || "$DEVICE" == "MI350X") && "$DATATYPE" == "FP8" ]]; then
        bash run_multigpu.sh |& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        bash run_multigpu.sh |& tee $TRAIN_LOG	
      fi
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

elif [[ "$TRAINING_MODE" == "posttrain" ]]; then
    echo "[INFO] Executing post-training benchmark..."
	export HSA_NO_SCRATCH_RECLAIM=1
    if [ "$MODEL_REPO" == "Flux" ]; then
      echo "[INFO] Benchmarking FLUX training"
      cd /workspace/AMDiffusionBenchmark
      echo "[INFO] Removing all previous runs to avoid caching"
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256" 
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=16
        python launcher.py train_args=flux-dev train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=10
        python launcher.py train_args=flux-dev train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      fi
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE 
    fi

    if [ "$MODEL_REPO" == "Stable-Diffusion-XL" ]; then
      echo "[INFO] Benchmarking STABLE-DIFFUSION-XL training"
      cd /workspace/AMDiffusionBenchmark
      echo "[INFO] Removing all previous runs to avoid caching"
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=30
        python launcher.py train_args=stable-diffusion-xl train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=20
        python launcher.py train_args=stable-diffusion-xl train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      fi
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE 
    fi

    if [ "$MODEL_REPO" == "Mochi-1" ]; then
      echo "[INFO] Benchmarking MOCHI-1 training"
      cd /workspace/AMDiffusionBenchmark
      echo "[INFO] Removing all previous runs to avoid caching"
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=4
        python launcher.py train_args=mochi-1 train_args.train_batch_size=$BATCH_SIZE|& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=1
        python launcher.py train_args=mochi-1 train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      fi
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE 
    fi

    if [ "$MODEL_REPO" == "Hunyuan-video" ]; then
      echo "[INFO] Benchmarking HUNYUAN-VIDEO training"
      cd /workspace/AMDiffusionBenchmark
      echo "[INFO] Removing all previous runs to avoid caching"
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=3
        python launcher.py train_args=hunyuan-video train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=1
        python launcher.py train_args=hunyuan-video train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      fi
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE 
    fi

    if [ "$MODEL_REPO" == "Wan2_1-i2v" ]; then
      echo "[INFO] Benchmarking WAN2_1-I2V training"
      cd /workspace/AMDiffusionBenchmark
      echo "[INFO] Removing all previous runs to avoid caching"
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=1
        python launcher.py train_args=wan2_1-i2v train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=1
        python launcher.py train_args=wan2_1-i2v train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG	
      fi
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
          --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE 
    fi

elif [[ "$TRAINING_MODE" == "finetune_fw" || "$TRAINING_MODE" == "finetune_lora" || "$TRAINING_MODE" == "finetune_qlora" ]]; then
    echo "[INFO] Executing torchtune finetuning benchmark..."
    torchtune_parser="$(pwd)/parse_torchtune_args.py"
    output=$(python3 $torchtune_parser --mode $TRAINING_MODE --model $MODEL_REPO)

    # Extract the values after each colon using 'cut' and assign to variables
    MODEL_FAMILY=$(echo "$output" | grep "model_family" | cut -d ':' -f2 | xargs)
    MODEL_SIZE=$(echo "$output" | grep "model_size" | cut -d ':' -f2 | xargs)
    METHOD=$(echo "$output" | grep "method" | cut -d ':' -f2 | xargs)

    hf auth login --token $HF_TOKEN --add-to-git-credential
    
    # Choose the appropriate tester script based on model family and training mode
    if [[ ("$MODEL_FAMILY" == "qwen2" || "$MODEL_FAMILY" == "qwen2_5" || "$MODEL_FAMILY" == "qwen3") && ("$TRAINING_MODE" == "finetune_fw" || "$TRAINING_MODE" == "finetune_lora") ]]; then
        cp Torchtune_Tester_Qwen.sh /workspace
        TESTER_SCRIPT="Torchtune_Tester_Qwen.sh"
    else
        cp Torchtune_Tester.sh /workspace
        TESTER_SCRIPT="Torchtune_Tester.sh"
    fi
    cd /workspace
    # Llama 3.2 Vision
    if [[ "$MODEL_FAMILY" == "llama3_2_vision" && "$MODEL_SIZE" == "90B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            PACKED=False; SEQ_LEN=8192; MBS=32; COMPILE=$([ "$TRAINING_MODE" == "finetune_fw" ] && echo "False" || echo "True")
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            PACKED=False; SEQ_LEN=8192; MBS=16; COMPILE=$([ "$TRAINING_MODE" == "finetune_fw" ] && echo "False" || echo "True")
        fi
    elif [[ "$MODEL_FAMILY" == "llama3_2_vision" && "$MODEL_SIZE" == "11B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            PACKED=False; SEQ_LEN=8192; MBS=64; COMPILE=$([ "$TRAINING_MODE" == "finetune_fw" ] && echo "False" || echo "True")
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            PACKED=False; SEQ_LEN=8192; MBS=32; COMPILE=$([ "$TRAINING_MODE" == "finetune_fw" ] && echo "False" || echo "True")
        fi
    # Llama 3.2
    elif [[ "$MODEL_FAMILY" == "llama3_2" && "$MODEL_SIZE" == "1B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    elif [[ "$MODEL_FAMILY" == "llama3_2" && "$MODEL_SIZE" == "3B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=64
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    # Llama 2
    elif [[ "$MODEL_FAMILY" == "llama2" && "$MODEL_SIZE" == "70B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=16
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=8
        fi
    elif [[ "$MODEL_FAMILY" == "llama2" && "$MODEL_SIZE" == "13B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=64
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=32
        fi
    elif [[ "$MODEL_FAMILY" == "llama2" && "$MODEL_SIZE" == "7B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=64
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=4096; MBS=32
        fi
    # Llama 3
    elif [[ "$MODEL_FAMILY" == "llama3" && "$MODEL_SIZE" == "70B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4
        fi
    elif [[ "$MODEL_FAMILY" == "llama3" && "$MODEL_SIZE" == "8B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    # Llama 3.1        
    elif [[ "$MODEL_FAMILY" == "llama3_1" && "$MODEL_SIZE" == "405B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4; CPU_OFFLOAD=False
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=2; CPU_OFFLOAD=True # observe VRAM %, may disable
        fi
    elif [[ "$MODEL_FAMILY" == "llama3_1" && "$MODEL_SIZE" == "70B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4
        fi
    elif [[ "$MODEL_FAMILY" == "llama3_1" && "$MODEL_SIZE" == "8B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    # Llama 3.3
    elif [[ "$MODEL_FAMILY" == "llama3_3" && "$MODEL_SIZE" == "70B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4
        fi
    # Llama 4
    elif [[ "$MODEL_FAMILY" == "llama4" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4
        fi
    # Qwen 2    
    elif [[ "$MODEL_FAMILY" == "qwen2" && "$MODEL_SIZE" == "1.5B" && "$TRAINING_MODE" == "finetune_fw" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        fi
    elif [[ "$MODEL_FAMILY" == "qwen2" && "$MODEL_SIZE" == "1.5B" && "$TRAINING_MODE" == "finetune_lora" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    # Qwen 2.5
    elif [[ "$MODEL_FAMILY" == "qwen2" && "$MODEL_SIZE" == "7B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=32
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        fi
    # Qwen 2.5
    elif [[ "$MODEL_FAMILY" == "qwen2_5" && "$MODEL_SIZE" == "32B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        fi
    elif [[ "$MODEL_FAMILY" == "qwen2_5" && "$MODEL_SIZE" == "72B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=4
        fi
    # Qwen 3        
    elif [[ "$MODEL_FAMILY" == "qwen3" && "$MODEL_SIZE" == "8B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=14
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=14
        fi
    elif [[ "$MODEL_FAMILY" == "qwen3" && "$MODEL_SIZE" == "32B" ]]; then
        if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=16
        elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
            COMPILE=True; PACKED=True; SEQ_LEN=8192; MBS=8
        fi
    fi

    if [[ "$TRAINING_MODE" == "finetune_fw" && "$DATATYPE" == "FP8" ]]; then
        FP8=True
        echo "[INFO] MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD PACKED=$PACKED COMPILE=$COMPILE SEQ_LEN=$SEQ_LEN MBS=$MBS FP8=$FP8"
        MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD COMPILE=$COMPILE PACKED=$PACKED SEQ_LEN=$SEQ_LEN CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=$MBS GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash $TESTER_SCRIPT --fp8 |& tee $TRAIN_LOG 

    else
        FP8=False
        echo "[INFO] MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD PACKED=$PACKED COMPILE=$COMPILE SEQ_LEN=$SEQ_LEN MBS=$MBS FP8=$FP8"
        MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD COMPILE=$COMPILE PACKED=$PACKED SEQ_LEN=$SEQ_LEN CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=$MBS GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash $TESTER_SCRIPT |& tee $TRAIN_LOG 
    fi 

    echo "[INFO] Benchmarking"
    python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
            --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
            --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $WORLD_SIZE

elif [[ "$TRAINING_MODE" == "HF_finetune_lora" ]]; then
    echo "Executing Huggingface LoRA finetuning library..."
    export HSA_NO_SCRATCH_RECLAIM=1
    HF_PEFT_DIR="$(pwd)/HF_PEFT_FSDP"
    cd $HF_PEFT_DIR
    if [ "$MODEL_REPO" == "GPT-OSS-20B" ]; then
        BATCH_SIZE=8
        echo "[INFO] GPT-OSS-20B Finetuning with Ultrachat dataset using Huggingface library"
        bash run_peft_fsdp_gpt_20b.sh |& tee $TRAIN_LOG
    elif [ "$MODEL_REPO" == "GPT-OSS-120B" ]; then
        BATCH_SIZE=8
        echo "[INFO] GPT-OSS-120B Finetuning with Ultrachat dataset using Huggingface library"
        bash run_peft_fsdp_gpt_120b.sh |& tee $TRAIN_LOG
    fi
    echo "[INFO] Benchmarking"
    python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
            --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
            --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
        
else
    echo "Error: Unsupported training mode."
    exit 1
fi
