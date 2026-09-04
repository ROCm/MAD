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
#./pytorch_benchmark_report.sh -t $training_mode -m $model_name -p $datatype
## example:
## Posttrain Flux with BF16 precision
#./pytorch_benchmark_report.sh -t posttrain -m Flux -p BF16
## Train DLRM with TF32 precision
#./pytorch_benchmark_report.sh -t pretrain -m DLRM -p TF32

TRAINING_MODE="posttrain"
DATATYPE="BF16"
MODEL_REPO=""
SEQUENCE_LENGTH="256"
NUM_GPUS="8"
BATCH_SIZE=""

usage() {
    echo "Usage: $0 -t <training_mode> -m <model_repo> -p <datatype> -n <num_gpus>"
    echo ""
    echo "Options:"
    echo "  -t <training_mode>   Training mode (pretrain, posttrain)"
    echo "  -m <model_repo>      Model (Flux, Stable-Diffusion-XL, Mochi-1, Hunyuan-video, Wan2_1-i2v, DLRM)"
    echo "  -p <datatype>        Precision type (BF16 for diffusion models; FP32 or TF32 for DLRM)"
    echo "  -n <num_gpus>        Number of GPUs (1 or 8)"
    echo "  -b <batch_size>      Batch size"
    exit 1
}

while getopts "t:m:p:s:n:b:" opt; do
    case "$opt" in
        t) TRAINING_MODE="$OPTARG" ;;
        m) MODEL_REPO="$OPTARG" ;;
        p) DATATYPE="$OPTARG" ;;
        s) SEQUENCE_LENGTH="$OPTARG" ;;
        n) NUM_GPUS="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        *) usage ;;
    esac
done

NNODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES*GPUS_PER_NODE))

if [[ -z "$TRAINING_MODE" || -z "$MODEL_REPO" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Validate model
SUPPORTED_MODELS="Flux Stable-Diffusion-XL Mochi-1 Hunyuan-video Wan2_1-i2v DLRM"
if ! echo "$SUPPORTED_MODELS" | grep -qw "$MODEL_REPO"; then
    echo "Error: Unsupported model '$MODEL_REPO'."
    echo "Supported models: $SUPPORTED_MODELS"
    exit 1
fi

# Validate precision
if [[ "$MODEL_REPO" == "DLRM" ]]; then
    if [[ "$DATATYPE" != "FP32" && "$DATATYPE" != "TF32" ]]; then
        echo "Error: For DLRM model, datatype must be either FP32 or TF32."
        exit 1
    fi
else
    if [[ "$DATATYPE" != "BF16" ]]; then
        echo "Error: For diffusion models, datatype must be BF16."
        exit 1
    fi
fi

if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "8" ]]; then
    echo "Error: Number of GPUs must be either 1 or 8."
    exit 1
fi

echo "Running training benchmark with the following parameters:"
echo "  Training Mode: $TRAINING_MODE"
echo "  Model: $MODEL_REPO"
echo "  Datatype: $DATATYPE"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Number of GPUs: $NUM_GPUS"
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

# Resolve rocminfo
ROCMINFO=""
if [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocminfo" ]]; then
  ROCMINFO="${ROCM_PATH}/bin/rocminfo"
elif [[ -x /opt/rocm/bin/rocminfo ]]; then
  ROCMINFO=/opt/rocm/bin/rocminfo
else
  shopt -s nullglob
  for _d in /opt/venv/lib/python*/site-packages/_rocm_sdk_devel/bin; do
    if [[ -x "${_d}/rocminfo" ]]; then
      ROCMINFO="${_d}/rocminfo"
      break
    fi
  done
  shopt -u nullglob
fi
if [[ -z "$ROCMINFO" ]]; then
  _rocm_which="$(command -v rocminfo 2>/dev/null || true)"
  if [[ -n "${_rocm_which}" && -x "${_rocm_which}" ]]; then
    ROCMINFO="${_rocm_which}"
  fi
fi

DEVICE=""
if [[ -n "$ROCMINFO" ]]; then
  DEVICE=$("$ROCMINFO" 2>/dev/null | grep "AMD Instinct" | head -n1 | awk '{print $5}')
  if [[ -z "$DEVICE" ]]; then
    ARCH=$("$ROCMINFO" 2>/dev/null | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
    case "$ARCH" in
      gfx942) DEVICE="MI300X" ;;
      gfx950) DEVICE="MI355X" ;;
    esac
  fi
fi

if [[ -z "$DEVICE" && -n "${MAD_SYSTEM_GPU_ARCHITECTURE:-}" ]]; then
  case "$MAD_SYSTEM_GPU_ARCHITECTURE" in
    *gfx950*) DEVICE="MI355X" ;;
    *gfx942*|*gfx941*) DEVICE="MI300X" ;;
    *)
      echo "[WARN] Could not map MAD_SYSTEM_GPU_ARCHITECTURE=$MAD_SYSTEM_GPU_ARCHITECTURE to a product name; using MI300X batch defaults." >&2
      DEVICE="MI300X"
      ;;
  esac
fi

if [[ -z "$DEVICE" ]]; then
  echo "[WARN] GPU name not detected; using MI300X batch defaults." >&2
  DEVICE="MI300X"
fi

echo "GPU DEVICE name: $DEVICE"

if [[ "$TRAINING_MODE" == "pretrain" && "$MODEL_REPO" == "DLRM" ]]; then
    echo "[INFO] Benchmarking DLRM TRAINING"
    if [[ ! -d /workspace/DLRMBenchmark ]]; then
      echo "Error: /workspace/DLRMBenchmark not found in container." >&2
      exit 1
    fi
    cd /workspace/DLRMBenchmark
    echo "[INFO] Removing all previous runs to avoid caching"
    rm -rf training_logs/
    rm -f results.csv
    HSA_NO_SCRATCH_RECLAIM=1 ./launch_training_single_node.sh -p $DATATYPE
    TRAIN_LOG=results.csv
    BATCH_SIZE=32768
    python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
      --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
      --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE

elif [[ "$TRAINING_MODE" == "posttrain" ]]; then
    echo "[INFO] Executing post-training benchmark..."
    export HSA_NO_SCRATCH_RECLAIM=1

    if [ "$MODEL_REPO" == "Flux" ]; then
      echo "[INFO] Benchmarking FLUX training"
      if [[ ! -d /workspace/AMDiffusionBenchmark ]]; then
        echo "Error: /workspace/AMDiffusionBenchmark not found in container." >&2
        exit 1
      fi
      cd /workspace/AMDiffusionBenchmark
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=16
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=10
      fi
      python launcher.py train_args=flux-dev train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
        --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
        --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

    if [ "$MODEL_REPO" == "Stable-Diffusion-XL" ]; then
      echo "[INFO] Benchmarking STABLE-DIFFUSION-XL training"
      if [[ ! -d /workspace/AMDiffusionBenchmark ]]; then
        echo "Error: /workspace/AMDiffusionBenchmark not found in container." >&2
        exit 1
      fi
      cd /workspace/AMDiffusionBenchmark
      rm -rf outputs/runs/*
      mkdir -p outputs/runs /root/.config/miopen
      export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=12
        python launcher.py train_args=stable-diffusion-xl \
          train_args.substitute_sdpa_with_flash_attn=false \
          accelerate_config.fsdp_config.fsdp_backward_prefetch=NO_PREFETCH \
          accelerate_config.fsdp_config.fsdp_sharding_strategy=SHARD_GRAD_OP \
          train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG
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
      if [[ ! -d /workspace/AMDiffusionBenchmark ]]; then
        echo "Error: /workspace/AMDiffusionBenchmark not found in container." >&2
        exit 1
      fi
      cd /workspace/AMDiffusionBenchmark
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=4
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=1
      fi
      python launcher.py train_args=mochi-1 train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
        --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
        --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

    if [ "$MODEL_REPO" == "Hunyuan-video" ]; then
      echo "[INFO] Benchmarking HUNYUAN-VIDEO training"
      if [[ ! -d /workspace/AMDiffusionBenchmark ]]; then
        echo "Error: /workspace/AMDiffusionBenchmark not found in container." >&2
        exit 1
      fi
      cd /workspace/AMDiffusionBenchmark
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
        BATCH_SIZE=3
      elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
        BATCH_SIZE=1
      fi
      python launcher.py train_args=hunyuan-video train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
        --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
        --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

    if [ "$MODEL_REPO" == "Wan2_1-i2v" ]; then
      echo "[INFO] Benchmarking WAN2_1-I2V training"
      if [[ ! -d /workspace/AMDiffusionBenchmark ]]; then
        echo "Error: /workspace/AMDiffusionBenchmark not found in container." >&2
        exit 1
      fi
      cd /workspace/AMDiffusionBenchmark
      rm -rf outputs/runs/*
      SEQUENCE_LENGTH="256"
      BATCH_SIZE=1
      python launcher.py train_args=wan2_1-i2v train_args.train_batch_size=$BATCH_SIZE |& tee $TRAIN_LOG
      TRAIN_LOG=$(find ./outputs/runs/ -type f -name "runs_summary.csv")
      python3 $perf_script --mode $TRAINING_MODE --model $MODEL_REPO \
        --precision $DATATYPE --input $TRAIN_LOG --output $PERF_LOG \
        --batch_size $BATCH_SIZE --seq_len $SEQUENCE_LENGTH --device $DEVICE --num_gpus $WORLD_SIZE
    fi

else
    echo "Error: Unsupported training mode '$TRAINING_MODE' for model '$MODEL_REPO'."
    echo "Supported: pretrain (DLRM), posttrain (Flux, Stable-Diffusion-XL, Mochi-1, Hunyuan-video, Wan2_1-i2v)"
    exit 1
fi
