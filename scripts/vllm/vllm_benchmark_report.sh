#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

## Usage: 
#./vllm_benchmark_report.sh -s $mode -m $hf_model -g $n_gpu -d $datatype
## example:
## latency + throughput
#./vllm_benchmark_report.sh -s all -m NousResearch/Meta-Llama-3-8B -g 1 -d float16
## latency 
#./vllm_benchmark_report.sh -s latency -m NousResearch/Meta-Llama-3-8B -g 1 -d float16
## throughput
#./vllm_benchmark_report.sh -s throughput -m NousResearch/Meta-Llama-3-8B -g 1 -d float16

while getopts s:m:g:d: flag
do
    case "${flag}" in
        s) scenario=${OPTARG};;
        m) model=${OPTARG};;
        g) numgpu=${OPTARG};;
        d) datatype=${OPTARG};;
    esac
done
echo "MODEL: $model ";

# args
model_org_name=(${model//// })
model_name=${model_org_name[1]}
tp=$numgpu

# perf env setting
export VLLM_USE_TRITON_FLASH_ATTN=0

if [ $tp -eq 1 ]; then
    DIST_BE=" --enforce-eager "
else
    DIST_BE=" --distributed-executor-backend mp "
fi

if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "	
elif [[ $datatype == "float8" ]]; then
    DTYPE=" --dtype float16 --quantization fp8 --kv-cache-dtype fp8 " 
fi

OPTION_LATENCY_P=" --gpu-memory-utilization 0.9 --enforce-eager "
OPTION_LATENCY_D=" --gpu-memory-utilization 0.9 "
OPTION_THROUGHPUT=" --gpu-memory-utilization 0.9 --num-scheduler-steps 1 "

# latency conditions
Bat="1 2 4 8 16 32 64 128 256"
InLatency="128 2048"
OutLatency="128"

# throughput conditions
Req_In_Out=("30000:128:128" "3000:2048:128" "3000:128:2048" "1500:2048:2048")

report_dir="reports_${datatype}"
report_summary_dir="${report_dir}/summary"
tool_latency="/app/vllm/benchmarks/benchmark_latency.py"
tool_throughput="/app/vllm/benchmarks/benchmark_throughput.py"
tool_report="vllm_benchmark_report.py"
n_warm=3
n_itr=5
mkdir -p $report_dir
mkdir -p $report_summary_dir


if [ "$scenario" == "latency" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] LATENCY"
    mode="latency"
    out=1
    for inp in $InLatency;
    do
        for bat in $Bat;
        do
            outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${datatype}.json
            outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
            echo $model $mode $bat $tp $inp $out
            python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --output-json $outjson $DTYPE $DIST_BE $OPTION_LATENCY_P
            python3 $tool_report --mode $mode --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
        done
    done
    for out in $OutLatency;
    do
    inp=1
        for bat in $Bat;
        do
            outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${datatype}.json
            outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
            echo $model $mode $bat $tp $inp $out
            python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --output-json $outjson $DTYPE $DIST_BE $OPTION_LATENCY_D
            python3 $tool_report --mode $mode --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
        done
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    for req_in_out in ${Req_In_Out[@]}
    do
        req=$(echo $req_in_out | awk -F':' '{ print $1 }')
        inp=$(echo $req_in_out | awk -F':' '{ print $2 }')
        out=$(echo $req_in_out | awk -F':' '{ print $3 }')
        outjson=${report_dir}/${model_name}_${mode}_req${req}_in${inp}_out${out}_${datatype}.json
        outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
        echo $model $mode $req $tp $inp $out
        python3 $tool_throughput --model $model --num-prompts $req -tp $tp --input-len $inp --output-len $out --trust-remote-code --output-json $outjson $DTYPE $DIST_BE $OPTION_THROUGHPUT
        python3 $tool_report --mode $mode --model $model_name --num-prompts $req --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
    done
fi

echo "Generate report of multiple results"
tool_parser="parse_csv.py"
latency_summary_csv=${report_summary_dir}/${model_name}_latency_report.csv
throughput_summary_csv=${report_summary_dir}/${model_name}_throughput_report.csv
python3 $tool_parser --file_latency $latency_summary_csv --file_throughput $throughput_summary_csv

mv perf_${model_name}.csv ../