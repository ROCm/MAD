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

## Usage: 
#./sglang_benchmark_report.sh -s $mode -m $hf_model -g $n_gpu -d $datatype [-a $dataset]
## example:
## latency + throughput
#./sglang_benchmark_report.sh -s all -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -g 8 -d bfloat16 -a random
## latency 
#./sglang_benchmark_report.sh -s latency -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -g 8 -d bfloat16
## throughput
#./sglang_benchmark_report.sh -s throughput -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -g 8 -d bfloat16 -a random

while getopts s:m:g:d:a:b:l:r: flag
do
    case "${flag}" in
        s) scenario=${OPTARG};;
        m) model=${OPTARG};;
        g) numgpu=${OPTARG};;
        d) datatype=${OPTARG};;
        a) dataset=${OPTARG};;
        b) Bat=${OPTARG};;
        l) lat_in_out_len=${OPTARG};;
        r) req_in_out_len=${OPTARG};;
    esac
done
echo "MODEL: $model ";

# args
model_org_name=(${model//// })
model_name=${model_org_name[1]}
tp=$numgpu

# perf env setting
if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "	
elif [[ $datatype == "float8" ]]; then
    DTYPE=" --dtype float16 --quantization fp8 --kv-cache-dtype fp8_e5m2 " 
fi

if [[ -z "$dataset" ]]; then
    DATASET="random"
else
    DATASET=$dataset
fi

# latency conditions
if [[ -z "$Bat" ]]; then
    Bat="1,2,4,8,16,32,64,128"
fi

if [[ -z "$lat_in_out_len" ]]; then
    Lat_In_Out="128:1;2048:1;128:128;2048:128"
else
    Lat_In_Out=$lat_in_out_len
fi

# throughput conditions
if [[ -z "$req_in_out_len" ]]; then
    Req_In_Out="30000:128:128;3000:2048:128;3000:128:2048;1500:2048:2048"
else
    Req_In_Out=$req_in_out_len
fi

report_dir="reports_${datatype}"
report_summary_dir="${report_dir}/summary"

# latency sample command:
# python -m sglang.bench_one_batch --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype bfloat16 --batch-size 32 --input-len 128 --output-len 128 --tensor-parallel-size 8
tool_sglang_latency="-m sglang.bench_one_batch"
# throughput sample comand:
# python -m sglang.bench_offline_throughput --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset-name random --num-prompts 3000 --random-input-len 128 --random-output-len 128 --tensor-parallel-size 8 
tool_sglang_throughput="-m sglang.bench_offline_throughput"
tool_sglang_report="sglang_benchmark_report.py"

mkdir -p $report_dir
mkdir -p $report_summary_dir


if [ "$scenario" == "latency" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] LATENCY"
    mode="latency"
    echo "$Bat" | tr ',' '\n' | while read -r bat;
    do
        echo "$Lat_In_Out" | tr ';' '\n' | while read -r lat_in_out;
        do
            inp=$(echo $lat_in_out | awk -F':' '{ print $1 }')
            out=$(echo $lat_in_out | awk -F':' '{ print $2 }')
            outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${datatype}.json
            outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
            echo $model $mode $bat $tp $inp $out
            # benchmark SGLang latency: the results are automatically recorded into a file.
            python3 $tool_sglang_latency --model-path $model $DTYPE --batch-size $bat --input-len $inp --output-len $out --tensor-parallel-size $tp --result-filename $outjson
            python3 $tool_sglang_report --mode $mode --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
        done
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    echo "$Req_In_Out" | tr ';' '\n' | while read -r req_in_out;
    do
        req=$(echo $req_in_out | awk -F':' '{ print $1 }')
        inp=$(echo $req_in_out | awk -F':' '{ print $2 }')
        out=$(echo $req_in_out | awk -F':' '{ print $3 }')
        outjson=${report_dir}/${model_name}_${mode}_req${req}_in${inp}_out${out}_${datatype}.json
        outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
        echo $model $mode $req $tp $inp $out
	    python3 $tool_sglang_throughput --model-path $model --dataset-name $DATASET $DTYPE --num-prompts $req --random-input-len $inp --random-output-len $out --tensor-parallel-size $tp --result-filename $outjson
        python3 $tool_sglang_report --mode $mode --model $model_name --num-prompts $req --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
    done
fi

echo "Generate report of multiple results"
tool_parser="parse_csv.py"
latency_summary_csv=${report_summary_dir}/${model_name}_latency_report.csv
throughput_summary_csv=${report_summary_dir}/${model_name}_throughput_report.csv
python3 $tool_parser --file_latency $latency_summary_csv --file_throughput $throughput_summary_csv

mv perf_${model_name}.csv ../