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
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-22}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx908"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-8}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx906"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-4}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"A100"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-8}
elif [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"V100"* ]]; then
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-4}
else
	echo "Detected new GPU architecture: $MAD_SYSTEM_GPU_ARCHITECTURE"
	echo "If not using MAD_MODEL_BATCH_SIZE, setting batch size to 1"
	MAD_MODEL_BATCH_SIZE=${MAD_MODEL_BATCH_SIZE:-1}
fi

# train model
HF_PATH='/workspace/transformers'

#set fp16 as a defult precision
precision_tag="--fp16"
#override default fp16
#pass -p=fp16 or --precision=fp16
for (( i=0; i<= $#; i=i+1 ));
do
        case ${@:$i:1}  in
                -p=*|--precision=*)
                        precision_tag=${@:$i:1}
                        precision_tag="--${precision_tag#*=}"
                        set -- ${@:1:$i-1} ${@:$i+1:$#}
                        ;;
        esac
done

# Add model-caching to resolve the hf multi processing error
hf download gpt2

torchrun --nproc_per_node="$MAD_RUNTIME_NGPUS" $HF_PATH/examples/pytorch/language-modeling/run_clm.py --output_dir output \
	--model_name_or_path gpt2 \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--do_train \
	--do_eval \
	--label_smoothing 0.1 \
	--logging_steps 1 \
	--logging_dir log $precision_tag \
	--dataloader_num_workers 1 \
	--skip_memory_metrics \
	--per_device_train_batch_size="$MAD_MODEL_BATCH_SIZE" \
	--overwrite_output_dir \
	--max_steps 150 "$@" \
	2>&1 | tee log.txt

# output performance metric
performance=$(cat log.txt | grep -Eo "train_samples_per_second':[^,]+" | sed "s/train_samples_per_second': //g" | head -n 1)

# unset printing trace to not confuse Jenkinsfile
set +x
echo "performance: $performance samples_per_second"
