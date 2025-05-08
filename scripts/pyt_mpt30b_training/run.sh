#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
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
# set -ex

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tunableop) TUNABLEOP="$2"; shift ;;
        --model_repo) MODEL_NAME="$2"; shift ;;
        --test_option) TEST_OPTION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

## Running MPT 30B
LLM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Manual fixes to composer library as torch version checks in the following scripts : trainer.py, _scaler.py, _patch_pytorch.py; are not compatible to rocm torch versioning.
# Presently, the torch version being used to run MPT is this: https://github.com/ROCm/pytorch/commit/c32bf4d4a5ffab3623c54c3d672313f164c1425c
# PR tracking this issue: https://ontrack-internal.amd.com/browse/SWDEV-489989
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Python version: ${PYTHON_VERSION}"

# Manual fix for the following error duting MPT training: llmfoundry.utils.exceptions.UnknownExampleTypeError: Found keys {'source', 'prompt', 'response'} in dataset.
# This error is due to a bug in llm-foundry library which does an incorrect check for the number of keys in the mosaicml/instruct-v3 dataset. The fix below removes this check.
# # PR tracking this issue: https://ontrack-internal.amd.com/browse/SWDEV-489988
sed -i "/len(example.keys()) == 1 and any(/c any(" /workspace/llm-foundry/llmfoundry/data/finetuning/tasks.py
sed -i "/len(example.keys()) == 2 and/d" /workspace/llm-foundry/llmfoundry/data/finetuning/tasks.py

if [ "$LOAD_MODEL_FROM_COMPOSER" != "on" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/preload_models.py"
fi

export TORCH_BLAS_PREFER_HIPBLASLT=1

if [[ "$TUNABLEOP" == "on" ]]; then 
    echo "turning on pytorch turnableop"
    # Pytorch replace the tunable op on the model
    # TunableOp will search for the best GEMMs for your specific environment
    export PYTORCH_TUNABLEOP_ENABLED=1
    # Pytoch autotune operations. For rocm it usually means autotune rocblas/hipblaslt (Gemm)
    export PYTORCH_TUNEABLEOP_TUNING=1
    # Constraint the autotune time
    export PYTORCH_TUNEABLEOP_MAX_TUNING_DURATION_MS=30
    export PYTORCH_TUNEABLEOP_MAX_WARMUP_DURATION_MS=30
else
    echo "turning off pytorch turnableop"
    export PYTORCH_TUNABLEOP_ENABLED=0
    export PYTORCH_TUNEABLEOP_TUNING=0
    export TORCHINDUCTOR_MAX_AUTOTUNE=0
fi
if [ -z "$MAD_SYSTEM_GPU_ARCHITECTURE" ]; then
    export MAD_SYSTEM_GPU_ARCHITECTURE="gfx942"
fi
if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" != *"gfx94"* ]]; then 
    echo "Unsuported GPU arch detected, please use supported MI300X GPUs "
    echo $MAD_SYSTEM_GPU_ARCHITECTURE
fi
if [ -z "$MAD_SECRETS_HFTOKEN" ]; then
  export MAD_SECRETS_HFTOKEN=""
fi
export HF_TOKEN=$MAD_SECRETS_HFTOKEN

export DISABLE_ADDMM_CUDA_LT=0
echo "======== hyper params start ========"
printf "%-30s %s\n" "TUNABLEOP:"                  "${TUNABLEOP}"
printf "%-30s %s\n" "PYTORCH_TUNABLEOP_ENABLED:"  "${PYTORCH_TUNABLEOP_ENABLED}"
if [[ "${PYTORCH_TUNABLEOP_ENABLED}" == "1" ]]; then
    echo "PYTORCH_TUNABLEOP_TUNING: ${PYTORCH_TUNABLEOP_TUNING}"
    echo "PYTORCH_TUNABLEOP_VERBOSE: ${PYTORCH_TUNABLEOP_VERBOSE}"
    echo "PYTORCH_TUNABLEOP_FILENAME: ${PYTORCH_TUNABLEOP_FILENAME}"
fi
printf "%-30s %s\n" "DISABLE_ADDMM_CUDA_LT:"      "${DISABLE_ADDMM_CUDA_LT}"
echo "========= hyper params end ========="

HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2 USE_ROCMLINEAR=1 composer /workspace/llm-foundry/scripts/train/train.py $LLM_DIR/mpt-30b-instruct.yaml  2>&1 | tee output.txt

unset DISABLE_ADDMM_CUDA_LT

performance=$(grep "Train throughput/samples_per_sec:" output.txt | cut -d ':' -f 2 | tail -1)
performanceperdevice=$(grep "Train throughput/device/samples_per_sec:" output.txt | cut -d ':' -f 2 | tail -1)
LanguageCrossEntropy=$(grep "Train metrics/train/LanguageCrossEntropy:" output.txt | cut -d ':' -f 2 | tail -1)
TrainingLoss=$(grep "loss/train/total:" output.txt | cut -d ':' -f 2 | tail -1)
set +x

# Print the performance metric.
echo "performance: $performance samples_per_sec"

if [ -z "$MAD_MODEL_NAME" ]; then
  export MAD_MODEL_NAME="pyt_mpt30b_training"
fi
MAD_CSV="perf_$MAD_MODEL_NAME.csv"
echo "Performance, $performance, throughput/samples_per_sec" >> $MAD_CSV
echo "Performance per device, $performanceperdevice, throughput/samples_per_sec" >> $MAD_CSV
echo "Language Cross Entropy, $s_per_it_value, metrics/train/LanguageCrossEntropy" >> $MAD_CSV
echo "Training Loss, $s_per_it_value, loss/train/total" >> $MAD_CSV
