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

PROFILE=false
PERF_ARGS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL="$2"; shift ;;
        --config) CONFIG_ARG="$2"; shift ;;
        --benchmark) BENCHMARK_ARG="$2"; shift ;;
        --output_csv) OUTPUT_CSV="$2"; shift ;;
        --perf) PERF_ARGS="$PERF_ARGS --perf $2"; shift ;;
        --perf-output) PERF_ARGS="$PERF_ARGS --perf-output $2"; shift ;;
        --no-perf-merge) PERF_ARGS="$PERF_ARGS --no-perf-merge"; shift ;;
	--profile) PROFILE=true ;;
        *)
            echo "Unknown parameter passed: $1" >&2
            echo "Usage: run.sh --model_repo <repo> [--config <yaml>] [--benchmark <name>] [--output_csv <csv>] [--perf <yaml>] [--perf-output <yaml>] [--no-perf-merge] [--profile]" >&2
            exit 1
            ;;
    esac
    shift
done

MODEL_NAME=$(basename $MODEL)

# By default run all benchmarks in configs/default.yaml; accept either CLI or env variable overrides
if [[ -z "$BENCHMARK" ]]; then
    # Infer benchmark from tag name if not explicitly set
    if [[ "$MAD_MODEL_NAME" == *"-accuracy"* ]]; then
        BENCHMARK=${BENCHMARK_ARG:-"accuracy"}
    elif [[ "$MAD_MODEL_NAME" == *"-perf"* ]]; then
        BENCHMARK=${BENCHMARK_ARG:-"serving"}
    else
        BENCHMARK=${BENCHMARK_ARG:-"all"}
    fi
fi
# CONFIG is always provided via --config flag from madengine or models.json
# If somehow not provided, use default.yaml as fallback
if [[ -z "$CONFIG" ]]; then
    CONFIG=${CONFIG_ARG:-"configs/default.yaml"}
fi

OUTPUT_CSV="perf_${MODEL_NAME}.csv"
if [[ ! -z "$MAD_OUTPUT_CSV" ]]; then
    OUTPUT_CSV="$MAD_OUTPUT_CSV"
fi

# install lm-eval for accuracy testing
pip install -qqq lm-eval[api] hf-transfer

# install profiling dependencies only when needed
if $PROFILE; then
    TRACELENS_BRANCH=${TRACELENS_BRANCH:-"main"}
    #pip install -qqq rocm-trace-lite-*.whl --find-links https://github.com/sunway513/rocm-trace-lite/releases/latest

    # TODO: Convert to support centos later
    apt install libsqlite3-dev g++
    git clone https://github.com/amathews-amd/rocm-trace-lite.git
    cd rocm-trace-lite
    git checkout amathews-amd/sig_handle
    sed -i 's/sys.exit(result.returncode if result is not None else 0)/sys.exit(getattr(result, "returncode", 0) if "result" in dir() else 0)/g' rocm_trace_lite/cmd_trace.py
    sed -i 's/    _generate_perfetto(output, json_file)/    pass  # perfetto JSON export disabled: 300+ GB penalty on Kimi K2, unused (kernels read from the .db)/' rocm_trace_lite/cmd_trace.py
    make -j
    make install
    pip install -e .
    cd ..
    #pip install -qqq "git+https://github.com/amathews-amd/TraceLens.git@${TRACELENS_BRANCH}"
fi

# Run benchmark; use -u to make python prints unbuffered
python3 -u run_vllm.py --config $CONFIG --model $MODEL --benchmark $BENCHMARK $($PROFILE && echo --profile) $PERF_ARGS

# move the output csv to parent directory
if [ -f "perf_${MODEL_NAME}.csv" ]; then                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    mv "perf_${MODEL_NAME}.csv" ../$OUTPUT_CSV                  
else
    echo "No perf_${MODEL_NAME}.csv to move (expected in --profile mode); skipping."
fi
