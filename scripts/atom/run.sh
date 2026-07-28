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
    BENCHMARK=${BENCHMARK_ARG:-"all"}
fi
if [[ -z "$CONFIG" ]]; then
    CONFIG=${CONFIG_ARG:-"configs/default.yaml"}
fi

OUTPUT_CSV="perf_${MODEL_NAME}.csv"
if [[ ! -z "$MAD_OUTPUT_CSV" ]]; then
    OUTPUT_CSV="$MAD_OUTPUT_CSV"
fi

# install lm-eval for accuracy testing
pip install -qqq lm-eval[api] hf-transfer


# install profiling dependencies (rocm-trace-lite) only when profiling
if $PROFILE; then
    apt-get update || true
    apt-get install -y g++ || true
    apt-get install -y libsqlite3-dev || true
    if [ ! -f /usr/include/sqlite3.h ]; then
        SQV=$(apt-cache policy libsqlite3-0 | awk '/Candidate:/{print $2}')
        ( cd /tmp && (apt-get download libsqlite3-0=$SQV libsqlite3-dev=$SQV || apt-get download libsqlite3-0 libsqlite3-dev) \
          && for d in libsqlite3-0_*.deb libsqlite3-dev_*.deb; do [ -f "$d" ] && dpkg -x "$d" /; done && ldconfig )
    fi
    git clone https://github.com/amathews-amd/rocm-trace-lite.git
    cd rocm-trace-lite
    git checkout amathews-amd/sig_handle
    sed -i 's/    _generate_perfetto(output, json_file)/    pass  # perfetto JSON export disabled/' rocm_trace_lite/cmd_trace.py
    sed -i 's/sys.exit(result.returncode if result is not None else 0)/sys.exit(getattr(result, "returncode", 0) if "result" in dir() else 0)/g' rocm_trace_lite/cmd_trace.py
    make -j
    make install
    pip install -e .
    cd ..
fi

# Run benchmark; use -u to make python prints unbuffered
python3 -u run_atom.py --config $CONFIG --model $MODEL --benchmark $BENCHMARK $($PROFILE && echo --profile) $PERF_ARGS

# move the output csv to parent directory
mv "perf_${MODEL_NAME}.csv" ../$OUTPUT_CSV
