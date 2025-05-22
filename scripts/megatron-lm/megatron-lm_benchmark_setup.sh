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
export HF_HOME=/workspace/huggingface

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "[INFO] Setup script starting in directory $(pwd)"

if [ "$MODEL_NAME" == "Mixtral-8x7B" ]; then
    mkdir -p /tokenizer
    cd /tokenizer
    # download tokenizer.model from https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/tokenizer.model
    wget --header="Authorization: Bearer $HF_TOKEN" -O ./tokenizer.model https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/resolve/main/tokenizer.model
    ls /tokenizer/tokenizer.model

elif [ "$MODEL_NAME" == "Mixtral-8x22B-proxy" ]; then
    mkdir -p /tokenizer
    cd /tokenizer
    # download tokenizer.model from https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/tokenizer.model
    wget --header="Authorization: Bearer $HF_TOKEN" -O ./tokenizer.model https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/resolve/main/tokenizer.model
    ls /tokenizer/tokenizer.model
fi
