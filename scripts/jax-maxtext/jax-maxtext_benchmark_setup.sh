#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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


cd $MAXTEXT
echo "Building dependencies for $MODEL_REPO"

set -x 
export HF_HOME=/hf_cache
mkdir /hf_cache
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# always download and use the real dataset
# huggingface-cli download legacy-datasets/c4 \
# 	  --include "en/partial-train/000*.parquet" \
# 	  --repo-type dataset \
# 	  --revision refs/convert/parquet 

# debug
# apt install tree -y > /dev/null
# tree /hf_cache/
# tree /hf_cache/hub/datasets--legacy-datasets--c4/snapshots/5abe0d085aa23dd9db2a6c1e86cfce4e4db6f0c3/
# ls /hf_cache/hub/datasets--legacy-datasets--c4/snapshots/5abe0d085aa23dd9db2a6c1e86cfce4e4db6f0c3/en/
# ls /hf_cache/hub/datasets--legacy-datasets--c4/snapshots/5abe0d085aa23dd9db2a6c1e86cfce4e4db6f0c3/en/partial-train/

download_tokenizer(){
  huggingface-cli download  $1 --include "**token**" 
}

if [[ "$MODEL_REPO" == "Llama-2-7B" ]]; then
  download_tokenizer "meta-llama/Llama-2-7b"
elif [[ "$MODEL_REPO" == "Llama-2-70B" ]]; then
  download_tokenizer "meta-llama/Llama-2-70b"
elif [[ "$MODEL_REPO" == "Llama-3.1-8B" ]]; then
  download_tokenizer "meta-llama/Meta-Llama-3-8B"
elif [[ "$MODEL_REPO" == "Llama-3.1-70B" ]]; then
  download_tokenizer "meta-llama/Meta-Llama-3-70B"
elif [[ "$MODEL_REPO" == "Llama-3.3-70B" ]]; then
  download_tokenizer "meta-llama/Llama-3.3-70B-Instruct"
elif [[ "$MODEL_REPO" == "DeepSeek-V2-lite" ]]; then
  echo "No tokenizer for download"
elif [[ "$MODEL_REPO" == "Mixtral-8x7B" ]]; then
  download_tokenizer "mistralai/Mixtral-8x7B-v0.1"
else
    echo "Error: Unsupported training mode."
    exit 1
fi
