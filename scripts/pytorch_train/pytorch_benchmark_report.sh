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
    exit 1
fi

if ! [[ "$SEQUENCE_LENGTH" =~ ^[0-9]+$ ]] || (( SEQUENCE_LENGTH < 2048 || SEQUENCE_LENGTH > 8192 )); then
    echo "Error: Sequence length must be between 2048 and 8192."
    exit 1
fi

# Training mode validation
if [[ "$TRAINING_MODE" == "HF_finetune_lora" ]]; then
    if [[ ! ("$MODEL_REPO" == "Llama-2-70B" || "$MODEL_REPO" == "Llama-3.1-70B") || "$DATATYPE" != "BF16" ]]; then
        echo "Error: finetuning options are only supported for Llama_2_70B and Llama_3.1_70B with BF16."
        exit 1
    fi
fi

if [[ "$TRAINING_MODE" == "HF_pretrain" ]]; then
    if [[ "$MODEL_REPO" != "Llama-3.1-8B" ]]; then
        echo "Error: HF pretraining option are only supported for Llama_3.1_8B."
        exit 1
    fi
fi

if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "8" ]]; then
    echo "Error: Number of GPUs must be either 1 or 8."
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
    
TRAIN_LOG="$(pwd)/$MODEL_REPO-$TRAINING_MODE.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"

perf_script="$(pwd)/pytorch_benchmark_report.py"

update_config_param() {
    local key=$1
    local value=$2
    local CONFIG_FILE=$3

    # If the key exists, replace it
    if grep -q "^$key" "$CONFIG_FILE"; then
        sed -i "s/^$key *= *.*/$key = $value/" "$CONFIG_FILE"
    else
        echo "$key = $value" >> "$CONFIG_FILE"
    fi
}

remove_config_param() {
    local key=$1
    local value=$2
    local CONFIG_FILE=$3

    # Escape possible special chars in value for sed (simple approach)
    local escaped_value=$(printf '%s\n' "$value" | sed 's/[][\/.^$*]/\\&/g')

    sed -i "/^$key *= *$escaped_value$/d" "$CONFIG_FILE"
}


if [[ "$TRAINING_MODE" == "pretrain" ]]; then
    echo "[INFO] Executing pretraining benchmark..."
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] LLAMA 3.1 8B TRAINING"
      cd /workspace/torchtitan
      ls $(pwd) 
      echo "[INFO] Benchmarking"

      # Update torchtitan config file
      CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" 
      update_config_param "warmup_steps" 10 $CONFIG_FILE
      update_config_param "steps" 50 $CONFIG_FILE
      update_config_param "log_freq" 1 $CONFIG_FILE
      update_config_param "enable_profiling" false $CONFIG_FILE
      update_config_param "enable_tensorboard" false $CONFIG_FILE
      update_config_param "compile" true $CONFIG_FILE

      # Set default parallel strategy to FSDP=True
      # [parallelism]
      update_config_param "data_parallel_replicate_degree" 1 $CONFIG_FILE
      update_config_param "data_parallel_shard_degree" 8 $CONFIG_FILE
      update_config_param "seq_len" 8192 $CONFIG_FILE
      # [activation_checkpoint]
      update_config_param "mode" '"full"' $CONFIG_FILE

      # Set default datatype to BF16
      # [model]
      remove_config_param "converters" '["float8"]' $CONFIG_FILE
      # [float8]
      update_config_param "enable_fsdp_float8_all_gather" false $CONFIG_FILE
      update_config_param "precompute_float8_dynamic_scale_for_fsdp" false $CONFIG_FILE
      update_config_param "force_recompute_fp8_weight_in_bwd" false $CONFIG_FILE

      BATCH_SIZE=18
      echo "Training model with batch size = $BATCH_SIZE"
      update_config_param "batch_size" $BATCH_SIZE $CONFIG_FILE

      CONFIG_FILE=$CONFIG_FILE bash run_train.sh |& tee $TRAIN_LOG	
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG
    fi

    if [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
      echo "[INFO] LLAMA 3.1 70B TRAINING"
      cd /workspace/torchtitan
      echo "[INFO] Benchmarking"

      # Update torchtitan config file
      CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_70b.toml"
      update_config_param "warmup_steps" 10 $CONFIG_FILE
      update_config_param "steps" 50 $CONFIG_FILE
      update_config_param "log_freq" 1 $CONFIG_FILE
      update_config_param "seq_len" 8192 $CONFIG_FILE
      update_config_param "enable_profiling" false $CONFIG_FILE
      update_config_param "enable_tensorboard" false $CONFIG_FILE
      update_config_param "compile" true $CONFIG_FILE

      # Set default parallel strategy to FSDP=True
      # [parallelism]
      update_config_param "data_parallel_replicate_degree" 1 $CONFIG_FILE
      update_config_param "data_parallel_shard_degree" 8 $CONFIG_FILE
      update_config_param "tensor_parallel_degree" 1 $CONFIG_FILE
      # [activation_checkpoint]
      update_config_param "mode" '"full"' $CONFIG_FILE

      # Set default datatype to BF16
      # [model]
      remove_config_param "converters" '["float8"]' $CONFIG_FILE
      # [float8]
      update_config_param "enable_fsdp_float8_all_gather" false $CONFIG_FILE
      update_config_param "precompute_float8_dynamic_scale_for_fsdp" false $CONFIG_FILE
      update_config_param "force_recompute_fp8_weight_in_bwd" false $CONFIG_FILE

      BATCH_SIZE=4
      echo "Training model with batch size = $BATCH_SIZE"
      update_config_param "batch_size" $BATCH_SIZE $CONFIG_FILE

      CONFIG_FILE=$CONFIG_FILE bash run_train.sh |& tee $TRAIN_LOG
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
          --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG 
    fi

    if [ "$MODEL_REPO" == "Flux" ]; then
      echo "[INFO] FLUX TRAINING"
      cd /workspace/FluxBenchmark
      echo "[INFO] Benchmarking"
      python launcher.py 
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --input $TRAIN_LOG --output $PERF_LOG 
    fi

elif [[ "$TRAINING_MODE" == "HF_pretrain" ]]; then
    echo "[INFO] Executing HF pretraining benchmark..."
    if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
      echo "[INFO] LLAMA 3.1 8B TRAINING"
      cd llama3_1_8B
      echo "[INFO] Benchmarking"
      bash run_multigpu.sh |& tee $TRAIN_LOG	
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
	      --input $TRAIN_LOG --output $PERF_LOG
    fi

elif [[ "$TRAINING_MODE" == "finetune_fw" || "$TRAINING_MODE" == "finetune_lora" || "$TRAINING_MODE" == "finetune_qlora" ]]; then
    echo "[INFO] Executing torchtune finetuning benchmark..."
    torchtune_parser="$(pwd)/parse_torchtune_args.py"
    output=$(python3 $torchtune_parser --mode $TRAINING_MODE --model $MODEL_REPO)

    # Extract the values after each colon using 'cut' and assign to variables
    MODEL_FAMILY=$(echo "$output" | grep "model_family" | cut -d ':' -f2 | xargs)
    MODEL_SIZE=$(echo "$output" | grep "model_size" | cut -d ':' -f2 | xargs)
    METHOD=$(echo "$output" | grep "method" | cut -d ':' -f2 | xargs)

    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
    cp Torchtune_Tester.sh /workspace
    cd /workspace

    if [[ "$MODEL_FAMILY" == "llama3_2_vision" ]]; then
        PACKED=False
        SEQ_LEN=8192

        if [[ "$MODEL_SIZE" == "90B" ]]; then
            MBS=16
            if [[ "$TRAINING_MODE" == "finetune_fw" ]]; then
                COMPILE=False
            else
                COMPILE=True
            fi

        elif [[ "$MODEL_SIZE" == "11B" ]]; then
            MBS=32
            if [[ "$TRAINING_MODE" == "finetune_fw" ]]; then
                COMPILE=False
            else
                COMPILE=True
            fi
        fi

    elif [[ "$MODEL_FAMILY" == "llama3_2" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        if [[ "$MODEL_SIZE" == "1B" ]]; then
            MBS=16
        elif [[ "$MODEL_SIZE" == "3B" ]]; then
            MBS=16
        fi
   
    elif [[ "$MODEL_FAMILY" == "llama2" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=4096
        if [[ "$MODEL_SIZE" == "70B" ]]; then
            MBS=8
        elif [[ "$MODEL_SIZE" == "13B" ]]; then
            MBS=32
        elif [[ "$MODEL_SIZE" == "7B" ]]; then
            MBS=32
        fi

    elif [[ "$MODEL_FAMILY" == "llama3" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        if [[ "$MODEL_SIZE" == "70B" ]]; then
            MBS=4
        elif [[ "$MODEL_SIZE" == "8B" ]]; then
            MBS=16
        fi

    elif [[ "$MODEL_FAMILY" == "llama3_1" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        if [[ "$MODEL_SIZE" == "405B" ]]; then
            MBS=2
            CPU_OFFLOAD=True # Claire: observe VRAM%, may disable
        elif [[ "$MODEL_SIZE" == "70B" ]]; then
            MBS=4
        elif [[ "$MODEL_SIZE" == "8B" ]]; then
            MBS=16
        fi

    elif [[ "$MODEL_FAMILY" == "llama3_3" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        if [[ "$MODEL_SIZE" == "70B" ]]; then
            MBS=4
        fi

    elif [[ "$MODEL_FAMILY" == "llama3_2" ]]; then
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        if [[ "$MODEL_SIZE" == "3B" || "$MODEL_SIZE" == "1B" ]]; then
            MBS=16
        fi

    elif [[ "$MODEL_FAMILY" == "llama4" ]]; then
        if [[ "$TRAINING_MODE" == "finetune_fw" ]]; then
            COMPILE=False
        else
            COMPILE=True
        fi
        PACKED=True
        SEQ_LEN=8192
        MBS=4

    else
        COMPILE=True
        PACKED=True
        SEQ_LEN=8192
        MBS=4
    fi

    
    echo "[INFO] MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD PACKED=$PACKED COMPILE=$COMPILE SEQ_LEN=$SEQ_LEN MBS=$MBS"
    MODEL_FAMILY=$MODEL_FAMILY MODEL_SIZE=$MODEL_SIZE METHOD=$METHOD COMPILE=$COMPILE PACKED=$PACKED SEQ_LEN=$SEQ_LEN CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=$MBS GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash Torchtune_Tester.sh |& tee $TRAIN_LOG 

    echo "[INFO] Benchmarking"
    python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
            --input $TRAIN_LOG --output $PERF_LOG

elif [[ "$TRAINING_MODE" == "HF_finetune_lora" ]]; then
    echo "Executing Huggingface LoRA finetuning library..."
    HF_PEFT_DIR="$(pwd)/HF_PEFT_FSDP"
    cd $HF_PEFT_DIR
    if [ "$MODEL_REPO" == "Llama-2-70B" ]; then
        echo "[INFO] Llama-2-70b Finetuning with Ultrachat dataset using Huggingface library"
        bash run_peft_fsdp.sh |& tee $TRAIN_LOG
    elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
        echo "[INFO] Llama-3.1-70b Finetuning with Ultrachat dataset using Huggingface library"
        MODEL_DIR=meta-llama/Llama-3.1-70B-Instruct bash run_peft_fsdp.sh |& tee $TRAIN_LOG
    fi
    echo "[INFO] Benchmarking"
    python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
            --input $TRAIN_LOG --output $PERF_LOG
        
else
    echo "Error: Unsupported training mode."
    exit 1
fi
