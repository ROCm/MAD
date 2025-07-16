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
set -ex

if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" != *"gfx94"* ]]; then 
    echo "Unsuported GPU arch detected, please use supported MI300X GPUs \n"
    exit 1
fi

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments
BENCHMARK_ARGS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL_NAME="$2"; shift ;;
        --test_option) TEST_OPTION="$2"; shift ;;
        --num_gpu) N_GPUS="$2"; shift ;;
        --datatype) DTYPE="$2"; shift ;;
        --tunableop) TUNABLEOP="$2"; shift ;;
        --vllm_v1) VLLM_V1="$2"; shift ;;
        --vllm_v1_split_attention) VLLM_V1_SPLIT_ATTENTION="$2"; shift ;;
        --aiter) AITER="$2"; shift ;;
        --aiter_pa) AITER_PA="$2"; shift ;;
        --aiter_mha) AITER_MHA="$2"; shift ;;
    esac
    shift
done

TEST_OPTION_SP=""
for i in $(echo $TEST_OPTION | tr "," "\n")
do
  TEST_OPTION_SP="$TEST_OPTION_SP $i"
done

DTYPE_SP=""
for i in $(echo $DTYPE | tr "," "\n")
do
  DTYPE_SP="$DTYPE_SP $i"
done

export HF_HUB_CACHE="/myworkspace"

if [[ $TUNABLEOP == "on" ]]; then 
    export PYTORCH_TUNABLEOP_ENABLED=1
elif [[ $TUNABLEOP == "off" ]]; then
    export PYTORCH_TUNABLEOP_ENABLED=0
fi

if [[ $VLLM_V1 == "on" ]]; then
    export VLLM_USE_V1=1
elif [[ $VLLM_V1 == "off" ]]; then
    export VLLM_USE_V1=0
fi

if [[ $VLLM_V1_SPLIT_ATTENTION == "on" ]]; then
    export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
elif [[ $VLLM_V1_SPLIT_ATTENTION == "off" ]]; then
    export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
fi

if [[ $AITER == "on" ]]; then
    export VLLM_ROCM_USE_AITER=1
elif [[ $AITER == "off" ]]; then
    export VLLM_ROCM_USE_AITER=0
fi

if [[ $AITER_PA == "on" ]]; then
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
elif [[ $AITER_PA == "off" ]]; then
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
fi

if [[ $AITER_MHA == "on" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=1
elif [[ $AITER_MHA == "off" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
fi

echo "=hyper params start="
echo "MODEL_NAME=$MODEL_NAME"
echo "TEST_OPTION_SP=$TEST_OPTION_SP"
echo "N_GPUS=$N_GPUS"
echo "DTYPE_SP=$DTYPE_SP"
echo "PYTORCH_TUNABLEOP_ENABLED=$PYTORCH_TUNABLEOP_ENABLED"
echo "VLLM_USE_V1=$VLLM_USE_V1"
echo "VLLM_V1_USE_PREFILL_DECODE_ATTENTION=$VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
echo "VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER"
echo "VLLM_ROCM_USE_AITER_PAGED_ATTN=$VLLM_ROCM_USE_AITER_PAGED_ATTN"
echo "VLLM_ROCM_USE_AITER_MHA=$VLLM_ROCM_USE_AITER_MHA"
echo "=hyper params end="

for scenario in $TEST_OPTION_SP; do
    for dtype in $DTYPE_SP; do
        ./vllm_benchmark_report.sh -s $scenario -m $MODEL_NAME -g $N_GPUS -d $dtype $BENCHMARK_ARGS
    done
done
