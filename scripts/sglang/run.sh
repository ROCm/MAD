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
set -ex

# Preliminary setup
export HF_HUB_CACHE="/myworkspace"
MAD_MODEL_NAME=$(echo $MAD_MODEL_NAME | tr "/" "_")

PERF_ARGS=""
PROFILE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL="$2"; shift ;;
        --config) CONFIG_ARG="$2"; shift ;;
        --benchmark) BENCHMARK_ARG="$2"; shift ;;
        --output_csv) OUTPUT_CSV="$2"; shift ;;
        --perf) PERF_ARGS="$PERF_ARGS --perf $2"; shift ;;
        --perf-output) PERF_ARGS="$PERF_ARGS --perf-output $2"; shift ;;
        --no-perf-merge) PERF_ARGS="$PERF_ARGS --no-perf-merge" ;;
        --profile) PROFILE=true ;;
        *)
            echo "Unknown parameter passed: $1" >&2
            echo "Usage: run.sh --model_repo <repo> [--config <yaml>] [--benchmark <name>] [--output_csv <csv>] [--perf <yaml>] [--perf-output <yaml>] [--no-perf-merge]" >&2
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

echo "=hyper params start="
echo "MODEL=$MODEL"
echo "CONFIG=$CONFIG"
echo "BENCHMARK=$BENCHMARK"
echo "=hyper params end="

# install lm-eval for accuracy testing
pip install -qqq lm-eval[api] hf-transfer

# In older SGLang images (e.g. v0.5.10.post1, before upstream commit a392ae8879)
# the metrics code has an unguarded comparison. When the server reports
# peak_memory_mb: null (None in Python), the o.peak_memory_mb > 0 check throws an error.
# This patch fixes the issue by adding a check for None.
BENCH_SERVING=$(find / -path "*/multimodal_gen/benchmarks/bench_serving.py" 2>/dev/null | head -1)
if [[ -n "$BENCH_SERVING" ]]; then
    sed -i 's/if o\.peak_memory_mb > 0/if o.peak_memory_mb is not None and o.peak_memory_mb > 0/g' "$BENCH_SERVING"
fi

# install profiling dependencies (rocm-trace-lite) only when profiling; mirrors scripts/vllm/run.sh
if $PROFILE; then
    # Activate the profiling-only ROCR 7.2.3 runtime that the Dockerfile STAGED under /opt/rocr723:
    # copy the fixed libhsa-runtime64.so over the active ROCm lib so `rtl trace --mode full` does not
    # SEGFAULT on cuda-graph replay (rocm-systems 559d48b1). Scoped to --profile so benchmark/accuracy
    # runs keep the untouched base 7.2.0 runtime. Must run BEFORE building rtl, which links the active
    # runtime. No network fetch here — the .so was downloaded once at image build.
    if [ -e /opt/rocr723/libhsa-runtime64.so.1 ]; then
        ROCR_DEST="$(readlink -f /opt/rocm)/lib"
        cp -avf /opt/rocr723/libhsa-runtime64.so* "$ROCR_DEST/"
        ldconfig
        ls -l "$ROCR_DEST"/libhsa-runtime64.so.1
    fi
    apt install libsqlite3-dev g++
    git clone https://github.com/amathews-amd/rocm-trace-lite.git
    cd rocm-trace-lite
    git checkout amathews-amd/sig_handle
    sed -i 's/sys.exit(result.returncode if result is not None else 0)/sys.exit(getattr(result, "returncode", 0) if "result" in dir() else 0)/g' rocm_trace_lite/cmd_trace.py
    sed -i 's/    _generate_perfetto(output, json_file)/    pass  # perfetto JSON export disabled: 300+ GB penalty, unused (kernels read from the .db)/' rocm_trace_lite/cmd_trace.py
    make -j
    make install
    pip install -e .
    cd ..
fi

# Run benchmark; use -u to make python prints unbuffered
python3 -u run_sglang.py --config $CONFIG --model $MODEL --benchmark $BENCHMARK $($PROFILE && echo --profile) $PERF_ARGS

# Move the output csv to parent directory (absent in --profile mode; guard the move)
if [ -f "perf_${MODEL_NAME}.csv" ]; then
    mv "perf_${MODEL_NAME}.csv" ../$OUTPUT_CSV
else
    echo "No perf_${MODEL_NAME}.csv to move (expected in --profile mode); skipping."
fi