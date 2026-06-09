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

OPTIONS=$(getopt -o w:v --long workload:,verbose -- "$@")

if [ $? -ne 0 ]; then
  echo "Failed to parse options." >&2
  exit 1
fi

eval set -- "$OPTIONS"

# Parse options
while true; do
  case "$1" in
    -w|--workload)
      WORKLOAD="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT="/app/.ci/run.py"
BENCHMARK_CONFIGS="/app/.ci/benchmark_configs"
ARCH=$(amd-smi static | grep TARGET_GRAPHICS_VERSION | head -1 | cut -d ':' -f 2 | xargs)

CSV_OUTPUT_PATH="/outputs/results.csv"

if [ ! -e $SCRIPT ]; then
    echo "'$SCRIPT' not found" >&2
    exit 1
fi

if [ -z "$ARCH" ]; then
    echo "Failed to get TARGET_GRAPHICS_VERSION from amd-smi" >&2
    exit 1
fi

# set HF_TOKEN
export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# temporary fix to handle host ROCm version <6.4.2
export HSA_NO_SCRATCH_RECLAIM=1

# run workload
echo "Run configurations:"
python3 $SCRIPT --tag mad --tag ${ARCH} --dry-run ${BENCHMARK_CONFIGS}/${WORKLOAD}.yaml
python3 $SCRIPT --tag mad --tag ${ARCH} --csv-output-path ${CSV_OUTPUT_PATH} ${BENCHMARK_CONFIGS}/${WORKLOAD}.yaml

if [ $? -ne 0 ]; then
  echo "Failed to run workload" >&2
  exit 1
fi

# Strip architecture suffixes from model column (e.g., ".gfx942", ".gfx950") in the MAD CSV output file
awk -F, 'BEGIN{OFS=","} NR==1{print;next} {sub(/\.gfx(942|950)$/, "", $1); print}' \
  "${CSV_OUTPUT_PATH}" > "${CSV_OUTPUT_PATH}.tmp" && mv "${CSV_OUTPUT_PATH}.tmp" "${CSV_OUTPUT_PATH}"

cp ${CSV_OUTPUT_PATH} ../results.csv
