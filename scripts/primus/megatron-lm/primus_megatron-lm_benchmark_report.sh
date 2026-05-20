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
MODE="pretrain"
DATATYPE="BF16"
POSTTRAIN_TYPE="lora"

usage() {
  echo "Usage: $0 -m <model_repo> -p <datatype> -t <mode> -f <posttrain_type>"
  echo "\nOptions:"
  echo "  -m <model_repo>      Model repository (Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, DeepSeek-V2-lite, DeepSeek-V3-proxy, Mixtral-8x7B, Mixtral-8x22B-proxy, Zebra-Llama-1B, Zebra-Llama-3B, Zebra-Llama-8B, Qwen-3-32B, GPT-OSS-20B, GPT-OSS-120B, Qwen-3-30B, Qwen-3-235B, Mamba-370M)"
  echo "  -p <datatype>        Precision type (FP8 or BF16)"
  echo "  -t <mode>            Training mode (pretrain or posttrain, default: pretrain)"
  echo "  -f <posttrain_type>  Post-training type (sft or lora, default: lora). Only used when mode is posttrain."
  exit 1
}

# Parse command-line arguments
while getopts "m:p:t:f:" opt; do
  case "$opt" in
    m) MODEL_REPO="$OPTARG" ;;
    p) DATATYPE="$OPTARG" ;;
    t) MODE="$OPTARG" ;;
    f) POSTTRAIN_TYPE="$OPTARG" ;;
    *) usage ;;
  esac
done

echo "=hyper params start="
echo $MODEL_REPO
echo $DATATYPE
echo $MODE
echo $POSTTRAIN_TYPE
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

if [[ "$MODE" != "pretrain" && "$MODE" != "posttrain" ]]; then
  echo "Error: Mode must be either pretrain or posttrain."
  exit 1
fi

if [[ "$POSTTRAIN_TYPE" != "sft" && "$POSTTRAIN_TYPE" != "lora" ]]; then
  echo "Error: Post-training type must be either sft or lora."
  exit 1
fi

# Run benchmark (Placeholder for actual script execution)
echo "Running primus training benchmark with the following parameters:"
echo "  Model Repository: $MODEL_REPO"

# config environment
export MOCK_DATA=1

unset_nvte_attn_backend_env() {
  unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
}

# set performance output paths
TRAIN_LOG="$(pwd)/primus-megatron-$MODEL_REPO-$MODE.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_primus-megatron-$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"
ls $(pwd)
perf_script="$(pwd)/primus_megatron-lm_benchmark_report.py"

# Run rocminfo and grep for "AMD Instinct"
DEVICE=$(/opt/rocm/bin/rocminfo | grep "AMD Instinct" | head -n1 | awk '{print $5}')
echo "DEVICE found: $DEVICE"
if [[ -z "$DEVICE" || ("$DEVICE" != "MI300X" && "$DEVICE" != "MI355X") ]]; then
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

# Map device names for config file paths
if [ "$DEVICE" == "MI350X" ]; then
  CONFIG_DEVICE="MI355X"
else
  CONFIG_DEVICE="$DEVICE"
fi

# Set common environment variables
export NNODES=1
export CPUS_PER_TASK=128
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
GPUS_PER_NODE=8 # default to 8 GPUs per node
NUM_GPUS=8

cd /workspace/Primus

# run models
if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama3.1_8B-$DATATYPE-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama3.1_70B-$DATATYPE-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.1-70B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING" # FP8 training only
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama3.1_70B-$DATATYPE-pretrain.yaml

  SEQ_LEN=8192
  # Set MBS/GBS through command line for proxy models 
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    echo "Error: Llama-3.1-70B-proxy model is not supported on $DEVICE. To train use the full Llama-3.1-70B model."
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      MBS=3
      GBS=24
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP --num_layers 40 --fp8 hybrid --micro_batch_size $MBS --global_batch_size $GBS --no_fp8_weight_transpose_cache true 2>&1 | tee $TRAIN_LOG
    elif [ "$DATATYPE" == "BF16" ]; then
      echo "Error: Datatype BF16 is not supported for $MODEL_REPO on $DEVICE. Only FP8 is supported."
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.3-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama3.3_70B-$DATATYPE-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-2-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama2_7B-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-2-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/llama2_70B-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "DeepSeek-V2-lite" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/deepseek_v2_lite-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP \
        --multi_latent_attention False 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "DeepSeek-V3-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/deepseek_v3-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  # Set MBS/GBS through command line for proxy models 
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=8
      GBS=64
      bash runner/primus-cli direct \
        --log_file /tmp/primus_deepseek_v3_proxy.log \
        -- train pretrain \
        --config examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml \
        --num_layers 3 \
        --moe_layer_freq 1 \
        --train_iters 50 \
        --micro_batch_size 8 \
        --global_batch_size 64 \
        --moe_use_fused_router_with_aux_score True \
        --moe_permute_fusion True \
        --pipeline_model_parallel_size 1 \
        --pipeline_model_parallel_layout null \
        --recompute_granularity null \
        --recompute_layer_ids null \
        --overlap_grad_reduce false \
        --overlap_param_gather false \
        --gradient_accumulation_fusion false 2>&1 | tee $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=3
      GBS=192
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP --num_layers 3 --moe_layer_freq 1 --micro_batch_size $MBS --global_batch_size $GBS 2>&1 | tee $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Mixtral-8x7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/mixtral_8x7B_v0.1-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Mixtral-8x22B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  LAYERS=4 # default proxy model uses 4 layers
  echo "[INFO] Proxy model uses $LAYERS layers"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/mixtral_8x22B_v0.1-$DATATYPE-pretrain.yaml

  SEQ_LEN=8192
  # Set MBS/GBS through command line for proxy models 
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=2
      GBS=16
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP --num_layers 4 --pipeline_model_parallel_size 1 --micro_batch_size $MBS --global_batch_size $GBS 2>&1 | tee $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=1
      GBS=16
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP --num_layers 4 --pipeline_model_parallel_size 1 --micro_batch_size $MBS --global_batch_size $GBS 2>&1 | tee $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen2.5-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/qwen2.5_7B-$DATATYPE-pretrain.yaml

  SEQ_LEN=2048
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [[ "$DATATYPE" == "BF16" ]]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    elif [[ "$DATATYPE" == "FP8" ]]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen2.5-72B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/qwen2.5_72B-$DATATYPE-pretrain.yaml

  SEQ_LEN=2048
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen-3-32B" ]; then
  unset_nvte_attn_backend_env
  echo "[INFO] $MODEL_REPO TRAINING"
  echo "[WARNING] Removing nemo_experiments, outputs, and logs folders in Primus directory for $MODEL_REPO"
  if [ -d "/workspace/Primus/nemo_experiments" ]; then
    rm -rf /workspace/Primus/nemo_experiments
    echo "[INFO] Removed nemo_experiments folder"
  fi
  if [ -d "/workspace/Primus/outputs" ]; then
    rm -rf /workspace/Primus/outputs
    echo "[INFO] Removed outputs folder"
  fi
  if [ -d "/workspace/Primus/logs" ]; then
    rm -rf /workspace/Primus/logs
    echo "[INFO] Removed logs folder"
  fi
  SEQ_LEN=8192
  if [[ "$MODE" == "posttrain" ]]; then
    export EXP=examples/megatron_bridge/configs/$CONFIG_DEVICE/qwen3_32b_${POSTTRAIN_TYPE}_posttrain.yaml
  
    MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
    GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
    echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
    if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
      if [ "$DATATYPE" == "FP8" ]; then
        echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
      elif [ "$DATATYPE" == "BF16" ]; then
        bash runner/primus-cli direct \
          --log_file /tmp/primus_$MODEL_REPO.log \
          -- train posttrain \
          --config $EXP 2>&1 | tee $TRAIN_LOG
      fi
    elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
      if [ "$DATATYPE" == "FP8" ]; then
        echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
      elif [ "$DATATYPE" == "BF16" ]; then
        bash runner/primus-cli direct \
          --log_file /tmp/primus_$MODEL_REPO.log \
          -- train posttrain \
          --config $EXP 2>&1 | tee $TRAIN_LOG
      fi
    fi
  elif [[ "$MODE" == "pretrain" ]]; then
    echo "Error: $MODEL_REPO pretrain mode is not supported. Use posttrain mode instead."
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --posttrain_type $POSTTRAIN_TYPE --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,posttrain_type,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$POSTTRAIN_TYPE,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$POSTTRAIN_TYPE,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Zebra-Llama-1B" ]; then
  unset_nvte_attn_backend_env
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/zebra_llama_1B-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Zebra-Llama-3B" ]; then
  unset_nvte_attn_backend_env
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/zebra_llama_3B-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Zebra-Llama-8B" ]; then
  unset_nvte_attn_backend_env
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/zebra_llama_8B-pretrain.yaml

  SEQ_LEN=8192
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "GPT-OSS-20B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/gpt_oss_20B-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "GPT-OSS-120B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  SEQ_LEN=4096
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    export EXP=examples/megatron/configs/MI355X/gpt_oss_120B-$DATATYPE-pretrain.yaml
    MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
    GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
    echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    echo "Error: $MODEL_REPO is not supported on $DEVICE. Only MI355X is supported."
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen-3-30B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/qwen3_30B_A3B-$DATATYPE-pretrain.yaml

  SEQ_LEN=4096
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen-3-235B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$CONFIG_DEVICE/qwen3_235B_A22B-$DATATYPE-pretrain.yaml

  SEQ_LEN=2048
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    bash runner/primus-cli direct \
      --log_file /tmp/primus_$MODEL_REPO.log \
      -- train pretrain \
      --config $EXP 2>&1 | tee -a $TRAIN_LOG
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Mamba-370M" ]; then
  unset_nvte_attn_backend_env
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/MI300X/mamba_370M-pretrain.yaml
  SEQ_LEN=2048
  MBS=$(grep -E '^\s*micro_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  GBS=$(grep -E '^\s*global_batch_size:' $EXP | head -n1 | awk '{print $2}' | tr -d '\r')
  echo "[INFO] Extracted MBS=$MBS, GBS=$GBS from config: $EXP"
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      bash runner/primus-cli direct \
        --log_file /tmp/primus_$MODEL_REPO.log \
        -- train pretrain \
        --config $EXP 2>&1 | tee -a $TRAIN_LOG
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --global_batch_size $GBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,global_batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$GBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

else
  echo "Error: Unsupported training mode."
  exit 1
fi
