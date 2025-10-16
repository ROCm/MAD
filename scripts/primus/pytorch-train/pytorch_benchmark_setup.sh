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

# Usage function
usage() {
    echo "Usage: $0 -m MODEL_NAME"
    echo "  -m: Model name (Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Flux, or empty for all models)"
    echo "  Example: $0 -m Llama-3.1-8B"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "Model repo: $MODEL_NAME"
echo "Setup script starting in directory $(pwd)"
export HSA_NO_SCRATCH_RECLAIM=1

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "No setup required for running Primus Pytorch v25.9"
