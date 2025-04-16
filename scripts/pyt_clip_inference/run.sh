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

#!/bin/bash
set -ex 

export GPU_MAX_HW_QUEUES=4

echo "Starting Inference."

############################################################################
# Select dataset for evaluation
############################################################################

TESTCASE=1  # Change this value (1-4) to select the dataset

if [ "$TESTCASE" -eq 1 ]; then
    DATASET="wds/mscoco_captions"
    DATASET_ROOT="https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions/tree/main"
elif [ "$TESTCASE" -eq 2 ]; then
    DATASET="wds/flickr30k"
    DATASET_ROOT="https://huggingface.co/datasets/clip-benchmark/wds_flickr30k/tree/main"
elif [ "$TESTCASE" -eq 3 ]; then
    DATASET="wds/vtab/pcam"
    DATASET_ROOT="https://huggingface.co/datasets/clip-benchmark/wds_vtab-pcam/tree/main"
elif [ "$TESTCASE" -eq 4 ]; then
    DATASET="wds/vtab/clevr_count_all"
    DATASET_ROOT="https://huggingface.co/datasets/clip-benchmark/wds_vtab-clevr_count_all/tree/main"
else
    echo "Invalid TESTCASE number. Please select a value between 1 and 4."
    exit 1
fi

############################################################################
# If "--tunableop on" flag is passed, set tunable op env vars and run twice.
############################################################################

if [[ "$*" == *"--tunableop on"* ]]; then
  # Map full dataset name to a simpler identifier.
  case "$DATASET" in
    "wds/mscoco_captions")
      DATA="coco"
      ;;
    "wds/flickr30k")
      DATA="flickr30k"
      ;;
    "wds/vtab/pcam")
      DATA="pcam"
      ;;
    "wds/vtab/clevr_count_all")
      DATA="clevr_count_all"
      ;;
    *)
      DATA="unknown"
      ;;
  esac

  # Export tunable op variables.
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_VERBOSE=0
  export PYTORCH_TUNABLEOP_FILENAME=./gemm_result_${DATA}.csv

  if [ -f "./gemm_result_${DATA}0.csv" ]; then
    echo "Found gemm_result_${DATA}0.csv, skipping warm-up run."
    export PYTORCH_TUNABLEOP_TUNING=0
    LOG_OUTPUT=$(
      clip_benchmark eval \
        --pretrained_model "ViT-B-32,laion2b_s34b_b79k" \
        --dataset "$DATASET" \
        --dataset_root "$DATASET_ROOT" \
        2>&1
    )
  else
    echo "Running first evaluation run (warm-up)..."
    export PYTORCH_TUNABLEOP_TUNING=1
    # First run: warm-up (output not captured)
    clip_benchmark eval \
      --pretrained_model "ViT-B-32,laion2b_s34b_b79k" \
      --dataset "$DATASET" \
      --dataset_root "$DATASET_ROOT" \
      2>&1 > /dev/null

    echo "Running second evaluation run (collecting tun result)..."
    export PYTORCH_TUNABLEOP_TUNING=0
    # Second run: capture output
    LOG_OUTPUT=$(
      clip_benchmark eval \
        --pretrained_model "ViT-B-32,laion2b_s34b_b79k" \
        --dataset "$DATASET" \
        --dataset_root "$DATASET_ROOT" \
        2>&1
    )
  fi
else
  # Normal run if tunable op is not enabled.
  export PYTORCH_TUNABLEOP_ENABLED=0
  export PYTORCH_TUNABLEOP_TUNING=0
  LOG_OUTPUT=$(
    clip_benchmark eval \
      --pretrained_model "ViT-B-32,laion2b_s34b_b79k" \
      --dataset "$DATASET" \
      --dataset_root "$DATASET_ROOT" \
      2>&1
  )
fi

# (Optional) Print the captured logs for debugging.
echo "$LOG_OUTPUT"

echo "Ending Inference."

############################################################################
# Parse iteration speed from lines like:
# "79it [00:11,  6.76it/s]"
############################################################################
SPEED_LINE=$(echo "$LOG_OUTPUT" | grep "it/s]" | tail -n1)
SPEED_VALUE=$(echo "$SPEED_LINE" | sed -E 's/.*,\s*([0-9.]+)it\/s\].*/\1/')

# If parsing fails, default to "NA"
if [ -z "$SPEED_VALUE" ]; then
    SPEED_VALUE="NA"
fi

set +x
echo "performance: $SPEED_VALUE iterations_per_second"
