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

if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx90a"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-24}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx908"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-16}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx906"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-1}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"A100"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-8}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"V100"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-1}
else
	echo "Detected new GPU architecture: $MAD_SYSTEM_GPU_ARCHITECTURE"
	echo "If not using MAD_MODEL_BATCH_SIZE, setting batch size to 1"
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-1}
fi

HF_PATH='/workspace/transformers'

torchrun $HF_PATH/examples/pytorch/language-modeling/run_mlm.py \
     --model_name_or_path bert-large-uncased \
     --dataset_name wikitext \
     --dataset_config_name wikitext-2-raw-v1 \
     --do_train \
     --max_steps 150 \
     --logging_steps 1 \
     --output_dir /tmp/test-mlm-bbu \
     --overwrite_output_dir \
	 --per_device_train_batch_size="$MAD_MODEL_BATCH_SIZE" \
     --fp16 \
     --skip_memory_metrics=True \
     "$@" \
    2>&1 | tee log.txt

# output performance metric
performance=$(cat log.txt | grep -Eo "train_samples_per_second':[^,]+" | sed "s/train_samples_per_second': //g")

# unset printing trace to not confuse Jenkinsfile 
set +x
echo "performance: $performance samples_per_second"
