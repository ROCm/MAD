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

# args
model_org_name=(${model//// })
model_name=${model_org_name[1]}
tp=$numgpu

# perf configuration
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
export VLLM_FP8_PADDING=1

if [ $tp -eq 1 ]; then
    DIST_BE=" "
else
    DIST_BE=" --distributed-executor-backend mp "
fi

if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "	
elif [[ $datatype == "float8" ]]; then
    DTYPE=" --dtype float16 --quantization fp8 --kv-cache-dtype fp8 " 
fi

OPTION_LATENCY=" --gpu-memory-utilization 0.9 "

# latency conditions
Bat="1 2 4 8 16 32 64 128 256"
InLatency="128 2048"
OutLatency="1 128"

# throughput conditions
In_Out=("128:128" "2048:128" "128:2048" "2048:2048")

tag="vllm_rocm6.3.1"

report_dir="reports_${datatype}_${tag}"
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
    for out in $OutLatency;
    do
        for inp in $InLatency;
        do
            for bat in $Bat;
            do
                if [ "$out" == "1" ]; then
                    NO_CUDA_GRAPH=" --enforce-eager "
                else
                    NO_CUDA_GRAPH=" "
                fi
                outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${datatype}.json
                outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
                echo $model $mode $bat $tp $inp $out
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --output-json $outjson $DTYPE $DIST_BE $OPTION_LATENCY $NO_CUDA_GRAPH
                python3 $tool_report --mode $mode --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
            done
        done
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    for in_out in ${In_Out[@]}
    do
        inp=$(echo $in_out | awk -F':' '{ print $1 }')
        out=$(echo $in_out | awk -F':' '{ print $2 }')

        # throughput config
        while IFS="," read -r model_cfg input_len output_len num_prompts max_num_seqs max_seq_len_to_capture max_num_batched_tokens	max_model_len gpu_memory_utilization num_scheduler_steps enable_chunked_prefill
        do
	    model_cfg_org_name=(${model_cfg//// })
	    model_cfg_name=${model_cfg_org_name[1]}
            if [ "$model_name" == "$model_cfg_name" ]; then
                if [ "$input_len" == "$inp" ] && [ "$output_len" == "$out" ];then
		    outjson=${report_dir}/${model_name}_${mode}_req${num_prompts}_in${inp}_out${out}_${datatype}.json
		    outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
		    if [ "$max_seq_len_to_capture" == "NA" ]; then
			OPTION_THROUGHPUT=" --num-prompts $num_prompts \
			    --max-num-seqs            $max_num_seqs            \
			    --gpu-memory-utilization  $gpu_memory_utilization  \
			    --num-scheduler-steps     $num_scheduler_steps     \
			    --enable-chunked-prefill $enable_chunked_prefill "
			else
			OPTION_THROUGHPUT=" --num-prompts $num_prompts \
			    --max-num-seqs            $max_num_seqs            \
			    --max-seq-len-to-capture  $max_seq_len_to_capture  \
			    --max-num-batched-tokens  $max_num_batched_tokens  \
			    --max-model-len           $max_model_len           \
			    --gpu-memory-utilization  $gpu_memory_utilization  \
			    --num-scheduler-steps     $num_scheduler_steps     \
			    --enable-chunked-prefill $enable_chunked_prefill "
		    fi
		    echo "[RUNNING] MODEL :" $model $mode $num_prompts $tp $inp $out
		    echo "[RUNNING] MODEL with OPTION: " $OPTION_THROUGHPUT
		    python3 $tool_throughput --model $model -tp $tp --input-len $inp --output-len $out --trust-remote-code --output-json $outjson $DTYPE $DIST_BE $OPTION_THROUGHPUT
		    python3 $tool_report --mode $mode --model $model_name --num-prompts $num_prompts --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
		fi
            fi
        done < <(tail -n +2 config.csv)
    done
fi

echo "Generate report of multiple results"
tool_parser="parse_csv.py"
latency_summary_csv=${report_summary_dir}/${model_name}_latency_report.csv
throughput_summary_csv=${report_summary_dir}/${model_name}_throughput_report.csv
python3 $tool_parser --file_latency $latency_summary_csv --file_throughput $throughput_summary_csv

mv perf_${model_name}.csv ../
