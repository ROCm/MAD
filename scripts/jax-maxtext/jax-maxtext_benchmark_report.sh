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
## Usage: 
#./jax-maxtext_benchmark_report.sh  -m $model_name 


# Parse command-line arguments
while getopts "m:" opt; do
    case "$opt" in
        m) MODEL_REPO="$OPTARG" ;;
        *) usage ;;
    esac
done

echo "=hyper params start="
echo $MODEL_REPO
echo "=hyper params end="

PERF_LOG="$(pwd)/../perf_$MODEL_REPO.csv"
perf_script="$(pwd)/jax-maxtext_benchmark_report.py"

MAXTEXT="/workspace/maxtext"
MAXTEXT_DIR="/workspace/maxtext/MaxText"
ENV_SCRIPT_DIR="$(pwd)/env_scripts"

cd $MAXTEXT


execute_training(){
  # output for logging
  echo "Using env file:"
  echo $ENV_SCRIPT_DIR/$1
  cat $ENV_SCRIPT_DIR/$1

  echo "Using yaml config file:"
  echo $ENV_SCRIPT_DIR/$2
  cat $ENV_SCRIPT_DIR/$2

  # execute
  source $ENV_SCRIPT_DIR/$1
  python $MAXTEXT_DIR/train.py $ENV_SCRIPT_DIR/$2 2>&1 |& tee -a  $2.log
  python3 $perf_script --model $MODEL_REPO --input $MAXTEXT/$2.log --output $PERF_LOG 
}


if [[ "$MODEL_REPO" == "Llama-2-7B" ]]; then
  echo "[INFO] LLAMA 2 7B TRAINING"
  execute_training llama2_7b_env.sh llama2_7b.yml

elif [[ "$MODEL_REPO" == "Llama-2-70B" ]]; then
  echo "[INFO] LLAMA 2 70B TRAINING"
  execute_training llama2_70b_env.sh llama2_70b.yml

elif [[ "$MODEL_REPO" == "Llama-3.1-8B" ]]; then
  echo "[INFO] LLAMA 3.1 8B TRAINING"
  execute_training llama3_8b_env.sh llama3_8b.yml

elif [[ "$MODEL_REPO" == "Llama-3.1-70B" ]]; then
  echo "[INFO] LLAMA 3.1 70B TRAINING"
  execute_training llama3_70b_env.sh llama3_70b.yml

elif [[ "$MODEL_REPO" == "Llama-3.3-70B" ]]; then
  echo "[INFO] LLAMA 3.3 70B TRAINING"
  execute_training llama3.3_70b_env.sh llama3.3_70b.yml

elif [[ "$MODEL_REPO" == "DeepSeek-V3-lite" ]]; then
  echo "[INFO] DEEPSEEK V3 LITE TRAINING"
  execute_training deepseek2_env_16b.sh deepseek2_16b.yml

else
    echo "Error: Unsupported training mode."
    exit 1
fi
