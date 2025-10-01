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
set -ex

#set default value of N_GPUs
N_GPUS=1

#parse named arguments
while [[ "$#" -gt 0 ]]; do
   case $1 in
        --num_gpu) N_GPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
   esac
   shift
done

#Download the model
MODEL_DIR="/myworkspace/run_directory/Wan2.1-T2V-14B"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "Model dir already exists at $MODEL_DIR"
else
    echo "Model dir not found. Downloading"
    bash -c "hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$MODEL_DIR""
fi

#Copy generate.py inference script to Wan2.1 folder
cp generate.py ../../workspace/Wan2.1/

#Command for single GPU run
cmd_single="python ../../workspace/Wan2.1/generate.py --task t2v-14B --size 1280*720 \
--ckpt_dir $MODEL_DIR --prompt \"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.\" \
>output.log 2>&1"

#Command for multi GPU run
cmd_multi="torchrun --nproc_per_node=$N_GPUS ../../workspace/Wan2.1/generate.py --task t2v-14B --size 1280*720 --ckpt_dir $MODEL_DIR  --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt \"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.\">output.log 2>&1"

if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx"* ]];then
	echo $N_GPUS
	if [[ "$N_GPUS" == "1" ]]; then
		bash -c "$cmd_single"
	else
	        bash -c "$cmd_multi"	
	fi	
else
	echo "This platform doesn't support wan2.1"
fi

# unset printing trace to not confuse Jenkinsfile 
set +x

# Parse the e2e latency(seconds)
elapsed_time=$(grep "Elapsed time to generate" output.log | awk -F': | seconds' '{print $3}')
echo "Latency(e2e) : $elapsed_time seconds"
# Output the performance metrics
echo "performance: $elapsed_time seconds"
