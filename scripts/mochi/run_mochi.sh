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

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --tunableop) TUNABLEOP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if HF_TOKEN is present
if [[ "$MODEL_NAME" == *"genmo/mochi-1-preview"* ]]; then
	if [[ -z "$MAD_SECRETS_HFTOKEN" ]]; then
		echo "Mochi is a gated model and requires MAD_SECRETS_HFTOKEN=<your-huggingface-token> to be set as an environment variable."
		exit 1
	fi
	export HF_TOKEN=$MAD_SECRETS_HFTOKEN
fi

current_dir=$(pwd)

# Clone Mochi repo and install Python dependencies
cd /app
git clone https://github.com/genmoai/mochi.git
cd mochi && pip install -e . --no-build-isolation --no-deps
patch -p1 < "$current_dir/mochi_fix.patch"
pip install moviepy==1.0.3

# Download model
mkdir /mnt/mochi
python3 ./scripts/download_weights.py /mnt/mochi

cd "$current_dir"
echo $current_dir

model_org_name=(${MODEL_NAME//// })
model_name=${model_org_name[1]}
dump_file_name=perf_${model_name}.csv
echo "model,performance,metric" > $dump_file_name

if [[ "$TUNABLEOP" == "on" ]]; then 
    echo "turning on pytorch turnableop"
    export PYTORCH_TUNABLEOP_ENABLED=1
else
    echo "turning off pytorch turnableop"
    export PYTORCH_TUNABLEOP_ENABLED=0
fi

# Run inference
for batch_size in 1 
do
    for mode in eager graph 
	do 
        if [[ "$mode" == "graph" ]]; then
	        export COMPILE_DIT=1
	    fi
	    python3 benchmark.py --model_dir /mnt/mochi --warmup_steps 2 --benchmark_steps 5
	    # parse latency in benchmark csv result
	    latency=$(awk -v line="2" -v field="1" -F',' 'NR==line{print $field}' mochi_*.csv | tr -d '\r')
	    rm mochi_*.csv
        # store latency for each mode
        if [[ "$mode" == "eager" ]]; then
            eager_mode_value=$latency
        else
            graph_mode_value=$latency
        fi
        echo "mochi_${mode}_b${batch_size},${latency},msecs" >> $current_dir/$dump_file_name
        if [[ "$mode" == "graph" ]]; then
	        export COMPILE_DIT=0
	    fi
	done
done

mv $dump_file_name ../

# final summary output
echo "Eager mode performance: ${eager_mode_value} ms"
echo "Graph mode performance: ${graph_mode_value} ms"
