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

# only for ROCM
export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
export VLLM_TUNE_GEMM=0

# args
model_org_name=(${model//// })
model_name=${model_org_name[1]}
dtype=$datatype
tp=$numgpu

# latency conditions
Bat="1 2 4 8 16 32 64 128 256"
InLatency="128 2048"
OutLatency=128

# throughput conditions
Req="256 2000"
InThroughput="128 2048"
OutThroughput="128 2048"

report_dir="../reports_${dtype}"
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
    for inp in $InLatency;
    do
        out=1
        for bat in $Bat;
        do
            outjson=${report_dir}/${model_name}_${mode}_prefill_bs${bat}_in${inp}_out${out}_${dtype}.json
            outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
            echo $model $mode $bat $tp $inp $out
            if [ $tp -eq 1 ]; then
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype --enforce-eager --output-json $outjson
            else
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype --output-json $outjson --distributed-executor-backend "mp"
            fi
            python3 $tool_report --mode ${mode} --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --dtype $dtype --input-json $outjson --output-csv $outcsv
        done
    done
    for bat in $Bat;
    do
        inp=1
        for out in $OutLatency;
        do
            outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${dtype}.json
            outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
            echo $model $mode $bat $tp $inp $out
            if [ $tp -eq 1 ]; then
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype --enforce-eager --output-json $outjson
            else
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype --output-json $outjson --distributed-executor-backend "mp"
            fi
            python3 $tool_report --mode ${mode} --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --dtype $dtype --input-json $outjson --output-csv $outcsv
        done
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    for req in $Req;
    do
        for out in $OutThroughput;
        do
            for inp in $InThroughput;
            do
                outjson=${report_dir}/${model_name}_${mode}_req${req}_in${inp}_out${out}_${dtype}.json
                outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
                echo $model $mode $req $tp $inp $out
                if [ $tp -eq 1 ]; then
                    python3 $tool_throughput --model $model --num-prompts $req -tp $tp --input-len $inp --output-len $out --trust-remote-code --dtype $dtype --enforce-eager --output-json $outjson
                else
                    python3 $tool_throughput --model $model --num-prompts $req -tp $tp --input-len $inp --output-len $out --trust-remote-code --dtype $dtype --output-json $outjson --distributed-executor-backend "mp"
                fi
                python3 $tool_report --mode $mode --model $model_name --num-prompts $req --tp $tp --input-len $inp --output-len $out --dtype $dtype --input-json $outjson --output-csv $outcsv
            done
        done
    done
fi
