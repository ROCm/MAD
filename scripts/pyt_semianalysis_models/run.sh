#!/usr/bin/env bash

set -ex

#Output csv file path
OUTPUT_FILE=$(pwd)/../perf_${MAD_MODEL_NAME}.csv

#Clone llm-train-bench repo
git clone https://${PUBLIC_GITHUB_ROCM_KEY_USERNAME}:${PUBLIC_GITHUB_ROCM_KEY_PASSWORD}@github.com/ROCm/llm-train-bench.git

cd llm-train-bench

# Run the bechmark script for all models
python3 ./benchmark_all.py 2>&1 | tee log.txt

cd ..
#Format the output csv to DLM csv format.
python3 process_csv.py 'llm-train-bench/outputs/benchmark_results.csv' $OUTPUT_FILE

echo "Results"
cat $OUTPUT_FILE

set +x
