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

# File containing multiple results
MAD_CSV="../perf_$MAD_MODEL_NAME.csv"

# Delete previous multiple results file if it exists
if [ -f "$MAD_CSV" ] ; then
    echo "Deleting previous multiple results file."
    rm "$MAD_CSV"
fi

# Create CSV header file
echo "model, performance, metric" > $MAD_CSV

cmd="python ../../workspace/chai-lab/examples/predict_structure.py > output.log 2>&1"
if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" == *"gfx"* ]];then
	bash -c "$cmd"
else
	echo "This platform doesn't support chai1"
fi

# unset printing trace to not confuse Jenkinsfile 
set +x

# Parse the "s/it" value for Trunk recycles
trunk_line=$(grep "Trunk recycles:" output.log)
s_per_it=$(echo "$trunk_line" | sed -n 's/.*\[\(.*\)\]/\1/p' | awk -F',' '{print $2}' | xargs)
s_per_it_value=$(echo "$s_per_it" | sed 's/^\s*\([0-9.]*\)s\/it$/\1/')

# Parse the "it/s" value for Diffusion steps
diffusion_line=$(grep "Diffusion steps:" output.log)
it_per_s=$(echo "$diffusion_line" | sed -n 's/.*\[\(.*\)\]/\1/p' | awk -F',' '{print $2}' | xargs)
it_per_s_value=$(echo "$it_per_s" | sed 's/^\s*\([0-9.]*\)it\/s$/\1/')

# Output the performance metrics
echo "Trunk recycles performance: $s_per_it_value s/it"
echo "Diffusion steps performance: $it_per_s_value it/s"

echo "Trunk recycles, $s_per_it_value, s/it (latency)" >> $MAD_CSV
echo "Diffusion steps, $it_per_s_value, it/s (throughput)" >> $MAD_CSV
