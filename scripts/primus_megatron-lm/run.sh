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

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL_REPO="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-3.1-8b" ]]; then
  model="Llama-3.1-8B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-3.1-70b" ]]; then
  model="Llama-3.1-70B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-3.1-405b" ]]; then
  model="Llama-3.1-405B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-3.1-70b-proxy" ]]; then
  model="Llama-3.1-70B-proxy"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-3.3-70b" ]]; then
  model="Llama-3.3-70B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-2-7b" ]]; then
  model="Llama-2-7B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_llama-2-70b" ]]; then
  model="Llama-2-70B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_deepseek-v2-lite-16b" ]]; then
  model="DeepSeek-V2-lite"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_deepseek-v2" ]]; then
  model="DeepSeek-V2"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_deepseek-v3-proxy" ]]; then
  model="DeepSeek-V3-proxy"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_mixtral-8x7b" ]]; then
  model="Mixtral-8x7B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_mixtral-8x22b-proxy" ]]; then
  model="Mixtral-8x22B-proxy"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_qwen2.5-7b" ]]; then
  model="Qwen2.5-7B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_qwen2.5-72b" ]]; then
  model="Qwen2.5-72B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_zebra-llama-1b" ]]; then
  model="Zebra-Llama-1B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_zebra-llama-3b" ]]; then
  model="Zebra-Llama-3B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_zebra-llama-8b" ]]; then
  model="Zebra-Llama-8B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_qwen-3-32b" ]]; then
  model="Qwen-3-32B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_mamba-370m" ]]; then
  model="Mamba-370M"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_gpt-oss-20b" ]]; then
  model="GPT-OSS-20B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_gpt-oss-120b" ]]; then
  model="GPT-OSS-120B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_qwen-3-30b" ]]; then
  model="Qwen-3-30B"
elif [[ "$MODEL_REPO" == "primus_pyt_megatron_lm_train_qwen-3-235b" ]]; then
  model="Qwen-3-235B"
fi

# Run primus pytorch setup script
echo "Running setup script to download tokenizers"
bash ./primus_megatron-lm_benchmark_setup.sh -m $model

# Resolve ROCm CLI directory (UTD: ${ROCM_PATH}/bin or venv _rocm_sdk_devel; classic /opt/rocm/bin).
# Inlined so device detection works when only this script is copied into the container (no scripts/common/).
ROCM_BIN="/opt/rocm/bin"
if [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocminfo" ]]; then
  ROCM_BIN="${ROCM_PATH}/bin"
elif [[ -x /opt/rocm/bin/rocminfo ]]; then
  ROCM_BIN="/opt/rocm/bin"
else
  shopt -s nullglob
  for _d in /opt/venv/lib/python*/site-packages/_rocm_sdk_devel/bin; do
    if [[ -x "${_d}/rocminfo" ]]; then
      ROCM_BIN="${_d}"
      break
    fi
  done
  shopt -u nullglob
fi
if [[ ! -x "${ROCM_BIN}/rocminfo" ]]; then
  _rocm_which="$(command -v rocminfo 2>/dev/null || true)"
  if [[ -n "${_rocm_which}" && -x "${_rocm_which}" ]]; then
    ROCM_BIN="$(dirname "${_rocm_which}")"
  fi
fi
export ROCM_BIN
export PATH="${ROCM_BIN}:${PATH}"

# Detect device
DEVICE=$("${ROCM_BIN}/rocminfo" | grep "AMD Instinct" | head -n1 | awk '{print $5}')
# Fall back to amd-smi when the rocminfo output format changes or rocminfo is
# unavailable (some newer ROCm/scaleout stacks).
if [[ -z "$DEVICE" && -x "${ROCM_BIN}/amd-smi" ]]; then
  DEVICE=$("${ROCM_BIN}/amd-smi" 2>/dev/null | awk '/AMD Instinct/ {print $5; exit}')
fi
if [ -z "$DEVICE" ]; then
  ARCH=$("${ROCM_BIN}/rocminfo" | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  # Fall back to the arch injected by madengine when rocminfo cannot report it.
  if [[ -z "$ARCH" && -n "${MAD_SYSTEM_GPU_ARCHITECTURE:-}" ]]; then
    ARCH=$(printf '%s' "${MAD_SYSTEM_GPU_ARCHITECTURE}" | tr -d '[:space:]')
  fi
  case "$ARCH" in
    "gfx942") DEVICE="MI300X" ;;
    "gfx950") DEVICE="MI355X" ;;
    *) DEVICE="" ;;
  esac
fi
echo "GPU DEVICE name: $DEVICE"

# Define supported datatypes based on device and model
datatypes=()

if [[ "$DEVICE" == "MI355X" || "$DEVICE" == "MI350X" ]]; then
  # MI355X/MI350X support
  if [[ "$model" == "Llama-3.1-70B-proxy" ]]; then
    echo "Skipping $model - Not supported on $DEVICE"
  elif [[ "$model" == "Qwen-3-32B" || "$model" == "Mamba-370M" ]]; then
    datatypes=("BF16")  # Only BF16 supported on MI355X/MI350X
  elif [[ "$model" == "Zebra-Llama-1B" || "$model" == "Zebra-Llama-3B" || "$model" == "Zebra-Llama-8B" ]]; then
    datatypes=("BF16")  # Only BF16 supported on MI355X/MI350X
  elif [[ "$model" == "Llama-3.1-8B" ]]; then
    datatypes=("BF16" "FP8" "MXFP8" "MXFP4")  # MXFP8/MXFP4 only supported on MI355X/MI350X
  elif [[ "$model" == "Llama-3.1-70B" || "$model" == "Llama-2-7B" || "$model" == "Qwen2.5-7B" ]]; then
    datatypes=("BF16" "FP8")
  elif [[ "$model" == "GPT-OSS-20B" || "$model" == "GPT-OSS-120B" || "$model" == "Qwen-3-30B" || "$model" == "Qwen-3-235B" ]]; then
    datatypes=("BF16" "FP8")
  else
    # Most other models only support BF16 on MI355X/MI350X
    datatypes=("BF16")
  fi
elif [[ "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ]]; then
  # MI300X/MI325X support
  if [[ "$model" == "Llama-3.1-70B-proxy" ]]; then
    datatypes=("FP8")  # Only FP8 supported
  elif [[ "$model" == "GPT-OSS-120B" ]]; then
    datatypes=("BF16" "FP8")  # Reuses the MI355X config with gfx942 CLI overrides
  elif [[ "$model" == "Zebra-Llama-1B" || "$model" == "Zebra-Llama-3B" || "$model" == "Zebra-Llama-8B" || "$model" == "Mamba-370M" ]]; then
    datatypes=("BF16")  # Only BF16 supported on MI300X/MI325X
  elif [[ "$model" == "Llama-3.1-8B" || "$model" == "Llama-2-7B" || "$model" == "Qwen2.5-7B" ]]; then
    datatypes=("BF16" "FP8")  # Both supported
  elif [[ "$model" == "GPT-OSS-20B" || "$model" == "Qwen-3-30B" || "$model" == "Qwen-3-235B" ]]; then
    datatypes=("BF16" "FP8")  # Both supported
  else
    # Most large models only support BF16 on MI300X/MI325X
    datatypes=("BF16")
  fi
else
  # Unknown device, try both
  datatypes=("BF16" "FP8")
fi

# Determine mode (default to pretrain, but some models use posttrain)
TRAIN_MODE="pretrain"
posttrain_types=()
if [[ "$model" == "Qwen-3-32B" ]]; then
  TRAIN_MODE="posttrain"
  posttrain_types=("lora" "sft")
fi

# datatypes=("FP8")
# Loop through supported combinations
for datatype in "${datatypes[@]}"; do
  if [[ "$TRAIN_MODE" == "posttrain" && ${#posttrain_types[@]} -gt 0 ]]; then
    for pt_type in "${posttrain_types[@]}"; do
      echo "Running: $model - $datatype - $TRAIN_MODE - $pt_type"
      ./primus_megatron-lm_benchmark_report.sh -m $model -p $datatype -t $TRAIN_MODE -f $pt_type
    done
  else
    echo "Running: $model - $datatype - $TRAIN_MODE"
    ./primus_megatron-lm_benchmark_report.sh -m $model -p $datatype -t $TRAIN_MODE
  fi
done
