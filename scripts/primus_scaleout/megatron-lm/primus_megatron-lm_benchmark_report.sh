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

usage() {
  echo "Usage: $0 -m <model_repo> -p <datatype> -l <layers>"
  echo "\nOptions:"
  echo "  -m <model_repo>      Model repository (Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-70B, Llama-3.1-405B, DeepSeek-V2-lite, DeepSeek-V3-proxy, Mixtral-8x7B, Mixtral-8x22B-proxy)"
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
TRAIN_LOG="$(pwd)/primus-megatron-$MODEL_REPO-pretrain.csv"
echo "TRAIN LOG: $TRAIN_LOG"

PERF_LOG="$(pwd)/../perf_primus-megatron-$MODEL_REPO.csv"
echo "PERF LOG: $PERF_LOG"
ls $(pwd)
perf_script="$(pwd)/primus_megatron-lm_benchmark_report.py"

# Detect device. Newer ROCm stacks may format rocminfo differently,
# so keep rocminfo first and fall back to amd-smi and the injected arch.
detect_device() {
  local device=""
  local arch=""

  if [[ -x /opt/rocm/bin/rocminfo ]]; then
    device=$(/opt/rocm/bin/rocminfo 2>/dev/null | awk '/AMD Instinct/ {print $5; exit}')
    arch=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  fi

  if [[ -z "$device" && -x /opt/rocm/bin/amd-smi ]]; then
    device=$(/opt/rocm/bin/amd-smi 2>/dev/null | awk '/AMD Instinct/ {print $5; exit}')
  fi

  if [[ -z "$arch" && -n "${MAD_SYSTEM_GPU_ARCHITECTURE:-}" ]]; then
    arch=$(printf '%s' "${MAD_SYSTEM_GPU_ARCHITECTURE}" | tr -d '[:space:]')
  fi

  case "$device" in
    MI300X|MI325X|MI350X|MI355X)
      ;;
    *)
      case "$arch" in
        gfx942) device="MI300X" ;;
        gfx950) device="MI355X" ;;
        *) device="" ;;
      esac
      ;;
  esac

  echo "$device"
}

DEVICE=$(detect_device)
echo "DEVICE found: $DEVICE"
echo "GPU DEVICE name: $DEVICE"
if [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
  export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
  export NVTE_CK_IS_V3_ATOMIC_FP32=1
fi

# Set common environment variables
# Keep compatibility with SLURM/madengine distributed env and only fallback to single-node defaults.
NNODES="${NNODES:-1}"
export NNODES
export CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE:-8}}"
if ! [[ "$NNODES" =~ ^[0-9]+$ ]]; then NNODES=1; fi
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then GPUS_PER_NODE=8; fi
NUM_GPUS=$((NNODES * GPUS_PER_NODE))
echo "[INFO] Distributed params: NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE NUM_GPUS=$NUM_GPUS"

normalize_global_batch_size() {
  local mbs="$1"
  local gbs="$2"
  local dp="$3"
  local unit=$((mbs * dp))
  if (( unit <= 0 )); then
    echo "$gbs"
    return
  fi
  if (( gbs % unit == 0 )); then
    echo "$gbs"
    return
  fi
  local adjusted=$(( ((gbs + unit - 1) / unit) * unit ))
  echo "$adjusted"
}

# Launch a single Primus pretrain run. Every rank must execute an identical
# command so the model shape stays consistent across nodes; a mismatch makes
# multi-node ranks rendezvous with different shapes and deadlocks NCCL. Pass
# the config first, then any shape/proxy/batch overrides -- they reach every
# rank because all ranks run this exact command.
run_primus() {
  local config="$1"; shift
  bash runner/primus-cli direct \
    --log_file "/tmp/primus_$MODEL_REPO.log" \
    -- train pretrain \
    --config "$config" "$@" 2>&1 | tee "$TRAIN_LOG"
}

cd /workspace/Primus

# run models
if [ "$MODEL_REPO" == "Llama-3.1-8B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/llama3.1_8B-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    MBS=4
    GBS=512
    run_primus "$EXP"
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    MBS=2
    GBS=128
    run_primus "$EXP"
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.1-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING" # BF16 training only
  export EXP=examples/megatron/configs/$DEVICE/llama3.1_70B-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      MBS=3
      GBS=24
      original_gbs=$GBS
      GBS=$(normalize_global_batch_size "$MBS" "$GBS" "$NUM_GPUS")
      if [[ "$GBS" != "$original_gbs" ]]; then
        echo "[INFO] Adjusted global batch size for distributed run: ${original_gbs} -> ${GBS} (MBS=${MBS}, NUM_GPUS=${NUM_GPUS})"
      fi
      run_primus "$EXP" --micro_batch_size $MBS --global_batch_size $GBS
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=4
      GBS=32
      original_gbs=$GBS
      GBS=$(normalize_global_batch_size "$MBS" "$GBS" "$NUM_GPUS")
      if [[ "$GBS" != "$original_gbs" ]]; then
        echo "[INFO] Adjusted global batch size for distributed run: ${original_gbs} -> ${GBS} (MBS=${MBS}, NUM_GPUS=${NUM_GPUS})"
      fi
      run_primus "$EXP" --micro_batch_size $MBS --global_batch_size $GBS
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=3
      GBS=24
      original_gbs=$GBS
      GBS=$(normalize_global_batch_size "$MBS" "$GBS" "$NUM_GPUS")
      if [[ "$GBS" != "$original_gbs" ]]; then
        echo "[INFO] Adjusted global batch size for distributed run: ${original_gbs} -> ${GBS} (MBS=${MBS}, NUM_GPUS=${NUM_GPUS})"
      fi
      run_primus "$EXP" --micro_batch_size $MBS --global_batch_size $GBS
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.1-405B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/llama3.1_405B-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  MBS=1
  GBS=8
  if [[ "$DATATYPE" == "FP8" ]]; then
    echo "Error: Datatype FP8 is not supported for $MODEL_REPO. Only BF16 is supported."
  elif [[ ! -f "$EXP" ]]; then
    echo "Error: Config file not found: $EXP"
    echo "Hint: add llama3.1_405B-$DATATYPE-pretrain.yaml in Primus configs."
  else
    run_primus "$EXP"
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.1-70B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING" # FP8 training only
  export EXP=examples/megatron/configs/$DEVICE/llama3.1_70B-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    echo "Error: Llama-3.1-70B-proxy model is not supported on $DEVICE. To train use the full Llama-3.1-70B model."
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      MBS=3
      GBS=24
      run_primus "$EXP" --num_layers 40 --fp8 hybrid --micro_batch_size $MBS --global_batch_size $GBS --no_fp8_weight_transpose_cache true
    elif [ "$DATATYPE" == "BF16" ]; then
      echo "Error: Datatype BF16 is not supported for $MODEL_REPO on $DEVICE. Only FP8 is supported."
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-3.3-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/llama3.3_70B-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=6
      GBS=48
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=2
      GBS=16
      run_primus "$EXP"
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-2-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/llama2_7B-$DATATYPE-pretrain.yaml
  SEQ_LEN=4096
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      export MBS=13
      export GBS=416
      run_primus "$EXP"
    elif [ "$DATATYPE" == "BF16" ]; then
      export MBS=10
      export GBS=640
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    export MBS=4
    export GBS=256
    if [ "$DATATYPE" == "FP8" ]; then
      run_primus "$EXP"
    elif [ "$DATATYPE" == "BF16" ]; then
      run_primus "$EXP"
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Llama-2-70B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/llama2_70B-$DATATYPE-pretrain.yaml
  SEQ_LEN=4096
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=17
      GBS=272
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=7
      GBS=56
      run_primus "$EXP"
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "DeepSeek-V2-lite" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/deepseek_v2_lite-$DATATYPE-pretrain.yaml
  SEQ_LEN=4096
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    MBS=12
    GBS=768
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=4
      GBS=640
      run_primus "$EXP"
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "DeepSeek-V3-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/deepseek_v3-$DATATYPE-pretrain.yaml
  SEQ_LEN=4096
  # DeepEP (Primus-Turbo) is opt-in via PRIMUS_USE_DEEPEP=1 and applies to every
  # supported device/precision branch below. DeepEP is incompatible with
  # moe_shared_expert_overlap (Primus ROCm validate_args asserts it), so disable
  # that overlap when enabling DeepEP. Default (unset/0) keeps alltoall over RCCL.
  DEEPEP_ARGS=""
  if [[ "${PRIMUS_USE_DEEPEP:-0}" == "1" ]]; then
    DEEPEP_ARGS="--use_turbo_deepep true --moe_shared_expert_overlap false"
    echo "[INFO] DeepEP enabled via PRIMUS_USE_DEEPEP=1 -> ${DEEPEP_ARGS}"
  fi
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=8
      GBS=64
      run_primus "$EXP" --num_layers 3 --moe_layer_freq 1 --micro_batch_size $MBS --global_batch_size $GBS $DEEPEP_ARGS
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=3
      GBS=192
      run_primus "$EXP" --num_layers 3 --moe_layer_freq 1 --micro_batch_size $MBS --global_batch_size $GBS $DEEPEP_ARGS
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Mixtral-8x7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/mixtral_8x7B_v0.1-$DATATYPE-pretrain.yaml
  SEQ_LEN=4096
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=4
      GBS=256
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=2
      GBS=32
      run_primus "$EXP"
    fi
  fi

  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Mixtral-8x22B-proxy" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  LAYERS=4 # default proxy model uses 4 layers
  echo "[INFO] Proxy model uses $LAYERS layers"
  export EXP=examples/megatron/configs/$DEVICE/mixtral_8x22B_v0.1-$DATATYPE-pretrain.yaml
  SEQ_LEN=8192
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=2
      GBS=16
      run_primus "$EXP" --num_layers 4 --pipeline_model_parallel_size 1 --micro_batch_size $MBS --global_batch_size $GBS
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=1
      GBS=16
      run_primus "$EXP" --num_layers 4 --pipeline_model_parallel_size 1 --micro_batch_size $MBS --global_batch_size $GBS
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen2.5-7B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/qwen2.5_7B-$DATATYPE-pretrain.yaml
  SEQ_LEN=2048
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [[ "$DATATYPE" == "BF16" ]]; then
      MBS=16
      GBS=768
      run_primus "$EXP"
    elif [[ "$DATATYPE" == "FP8" ]]; then
      MBS=20
      GBS=800
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    MBS=10
    GBS=640
    if [ "$DATATYPE" == "FP8" ]; then
      run_primus "$EXP"
    elif [ "$DATATYPE" == "BF16" ]; then
      run_primus "$EXP"
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  fi

elif [ "$MODEL_REPO" == "Qwen2.5-72B" ]; then
  echo "[INFO] $MODEL_REPO TRAINING"
  export EXP=examples/megatron/configs/$DEVICE/qwen2.5_72B-$DATATYPE-pretrain.yaml
  SEQ_LEN=2048
  if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=16
      GBS=256
      run_primus "$EXP"
    fi
  elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
    if [ "$DATATYPE" == "FP8" ]; then
      echo "Error: Datatype FP8 is not supported for $MODEL_REPO on $DEVICE. Only BF16 is supported."
    elif [ "$DATATYPE" == "BF16" ]; then
      MBS=4
      GBS=32
      run_primus "$EXP"
    fi
  fi
  if [ -f "$TRAIN_LOG" ]; then
    echo "[INFO] Benchmarking"
    python3 $perf_script --model $MODEL_REPO --input $TRAIN_LOG --output $PERF_LOG --mode $MODE --precision $DATATYPE --batch_size $MBS --seq_len $SEQ_LEN --device $DEVICE --num_gpus $NUM_GPUS
    rm $TRAIN_LOG
  else
    echo "[INFO] Training log not found - configuration not supported."
    echo "model,performance,metric,mode,precision,batch_size,seq_len,device,num_gpus" > $PERF_LOG
    echo "$MODEL_REPO,,tok_per_s_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
    echo "$MODEL_REPO,,TFLOPS_per_gpu,$MODE,$DATATYPE,$MBS,$SEQ_LEN,$DEVICE,$NUM_GPUS" >> $PERF_LOG
  fi

else
  echo "Error: Unsupported training mode."
  exit 1
fi
