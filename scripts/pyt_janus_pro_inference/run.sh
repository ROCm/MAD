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

# File containing multiple results
DLM_CSV="../perf_$MAD_MODEL_NAME.csv"

# Delete previous multiple results file if it exists
if [ -f "$DLM_CSV" ] ; then
    echo "Deleting previous multiple results file."
    rm "$DLM_CSV"
fi

# Create CSV header: model, performance, metric
echo "model,performance,metric" > "$DLM_CSV"

# Run model, collecting output
log_file_under="./${MAD_MODEL_NAME}_under.log"
echo "Running Janus-Pro Inference - Image understanding"
python3 run_inference.py under  2>&1 | tee $log_file_under

# Allow for GPU memory release
sleep 5

log_file_gen="./${MAD_MODEL_NAME}_gen.log"
echo "Running Janus-Pro Inference - Image generation"
python3 run_inference.py gen  2>&1 | tee $log_file_gen

# Parse output
understanding_line=$(cat "$log_file_under" | grep "Avg throughput:" | tail -n1)
img_generation_line=$(cat "$log_file_gen" | grep "Avg throughput:" | tail -n1)

understanding_speed=$(echo "$understanding_line" | sed -E 's/.*: ([0-9.]+) samples\/sec/\1/')
img_generation_speed=$(echo "$img_generation_line" | sed -E 's/.*: ([0-9.]+) prompts\/sec/\1/')

if [ -z "$understanding_speed" ]; then
    understanding_speed="NA"
fi
if [ -z "$img_generation_speed" ]; then
    img_generation_speed="NA"
fi

echo "${MAD_MODEL_NAME}-image_understanding,$understanding_speed,samples/sec" >> "$DLM_CSV"
echo "${MAD_MODEL_NAME}-text2img_generation,$img_generation_speed,prompts/sec" >> "$DLM_CSV"
