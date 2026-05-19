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

# Preliminary setup
if [[ -z "${HF_HUB_CACHE:-}" ]]; then
    export HF_HUB_CACHE="/myworkspace"
fi
MAD_MODEL_NAME=$(echo $MAD_MODEL_NAME | tr "/" "_")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL="$2"; shift ;;
        --config) CONFIG_ARG="$2"; shift ;;
        --benchmark) BENCHMARK_ARG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# By default run all benchmarks in configs/default.yaml; accept either CLI or env variable overrides
if [[ -z "$BENCHMARK" ]]; then
    BENCHMARK=${BENCHMARK_ARG:-"all"}
fi
if [[ -z "$CONFIG" ]]; then
    CONFIG=${CONFIG_ARG:-"configs/default.yaml"}
fi

# install lm-eval for accuracy testing
pip install -qqq lm-eval[api] hf-transfer

# Run benchmark; use -u to make python prints unbuffered
python3 -u run_vllm.py --config $CONFIG --model $MODEL --benchmark $BENCHMARK

# move the output csv to parent directory
MODEL_NAME=$(basename $MODEL)
OUTPUT_CSV="perf_${MODEL_NAME}.csv"
mv $OUTPUT_CSV ../
