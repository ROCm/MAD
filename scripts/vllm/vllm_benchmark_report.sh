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
#./vllm_benchmark_report.sh -s|--scenario $scenario
#                           -m|--model $model
#                           -g|--numgpu $numgpu
#                           -d|--dtype $dtype
## example:
## latency + throughput + serving
#./vllm_benchmark_report.sh -s all -m meta-llama/Meta-Llama-3-8B -g 1 -d float16
## latency 
#./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3-8B -g 1 -d float16
## throughput
#./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3-8B -g 1 -d float16
## serving
#./vllm_benchmark_report.sh -s serving -m meta-llama/Meta-Llama-3-8B -g 1 -d float16

set -x
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--scenario) scenario="$2"; shift;;
        -m|--model) model="$2"; shift;;
        -g|--numgpu) numgpu="$2"; shift;;
        -d|--datatype) datatype="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
    shift
done

# args
model_org_name=(${model//// })
model_name=${model_org_name[1]}
tp=$numgpu

# perf configuration
if [[ $VLLM_USE_V1 == 0 ]]; then
    # Use CK Flash Attention
    export VLLM_USE_TRITON_FLASH_ATTN=0
    VLLM_ARGS="--num-scheduler-steps 10"
else
    # For V1 use full cuda graph compilation
    if [[ $scenario == "throughput" ]]; then
        # rms_norm compile runs into OOM currently
        VLLM_ARGS='--compilation-config {"full_cuda_graph":true,"custom_ops":["+silu_and_mul"],"pass_config":{"enable_noop":true,"enable_fusion":true}}'
    else
        VLLM_ARGS='--compilation-config {"full_cuda_graph":true,"custom_ops":["+rms_norm","+silu_and_mul"],"pass_config":{"enable_noop":true,"enable_fusion":true}}'
    fi
fi

if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "
elif [[ $datatype == "bfloat16" ]]; then
    DTYPE=" --dtype bfloat16 "
elif [[ $datatype == "float8" ]]; then
    DTYPE=" --dtype float16 "
    # Use FP8 kv cache for throughput
    if [[ $scenario == "throughput" ]]; then
        DTYPE=" ${DTYPE} --kv-cache-dtype fp8 "
    fi
fi

GPU_UTIL=" --gpu-memory-utilization 0.9 "

# latency conditions
Bat="1 8 32 128"
InLatency="128 2048"
OutLatency="128 2048"

# throughput conditions
InThroughput="128 2048"
OutThroughput="128 2048"

# serving conditions
NumPrompts="252"
MaxConcurrency="128"
InServing="128 2048"
OutServing="128 2048"
# override
if [ -n "$num_prompts" ]; then
    NumPrompts=$num_prompts
fi
if [ -n "$max_concurrency" ]; then
    MaxConcurrency=$max_concurrency
fi
if [ -n "$input_len" ]; then
    InServing=$input_len
fi
if [ -n "$output_len" ]; then
    OutServing=$output_len
fi

tag="vllm_rocm6.4.1"

report_dir="reports_${datatype}_${tag}"
report_summary_dir="${report_dir}/summary"
tool_latency="/app/vllm/benchmarks/benchmark_latency.py"
tool_throughput="/app/vllm/benchmarks/benchmark_throughput.py"
tool_serving="/app/vllm/benchmarks/benchmark_serving.py"
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
                outjson=${report_dir}/${model_name}_${mode}_decoding_bs${bat}_in${inp}_out${out}_${datatype}.json
                outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
                echo $model $mode $bat $tp $inp $out
                python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --output-json $outjson $DTYPE $GPU_UTIL $VLLM_ARGS
                python3 $tool_report --mode $mode --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
            done
        done
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    for inp in $InThroughput;
    do
        for out in $OutThroughput;
        do
            # throughput config
            while IFS="," read -r model_cfg input_len output_len num_prompts max_num_seqs max_seq_len_to_capture max_num_batched_tokens	max_model_len
            do
                model_cfg_org_name=(${model_cfg//// })
                model_cfg_name=${model_cfg_org_name[1]}
                if [ "$model_name" == "$model_cfg_name" ]; then
                    if [ "$input_len" == "$inp" ] && [ "$output_len" == "$out" ]; then
                        outjson=${report_dir}/${model_name}_${mode}_req${num_prompts}_in${inp}_out${out}_${datatype}.json
                        outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
                        if [ "$max_seq_len_to_capture" == "NA" ]; then
                            OPTION_THROUGHPUT=" --num-prompts $num_prompts         \
                                --max-num-seqs            $max_num_seqs            "
                        else
                            OPTION_THROUGHPUT=" --num-prompts $num_prompts         \
                                --max-num-seqs            $max_num_seqs            \
                                --max-seq-len-to-capture  $max_seq_len_to_capture  \
                                --max-num-batched-tokens  $max_num_batched_tokens  \
                                --max-model-len           $max_model_len           "
                        fi
                        echo "[RUNNING] MODEL :" $model $mode $num_prompts $tp $inp $out
                        echo "[RUNNING] MODEL with OPTION: " $OPTION_THROUGHPUT
                        python3 $tool_throughput --model $model -tp $tp --input-len $inp --output-len $out --trust-remote-code --output-json $outjson $DTYPE $GPU_UTIL $OPTION_THROUGHPUT $VLLM_ARGS
                        python3 $tool_report --mode $mode --model $model_name --num-prompts $num_prompts --tp $tp --input-len $inp --output-len $out --input-json $outjson --output-csv $outcsv --dtype $datatype
                    fi
                fi
            done < <(tail -n +2 config.csv)
        done
    done
fi

if [ "$scenario" == "serving" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] SERVING"
    mode="serving"
    # start server and send it to background using {command} &
    vllm serve $model --swap-space 16 --disable-log-requests --trust-remote-code  -tp $tp $DTYPE $GPU_UTIL $VLLM_ARGS 1>&1 2>&2 &
    # get the server pid
    server_pid=$!
    echo "vllm server pid: $server_pid"
    # wait for the server to start
    until curl http://localhost:8000/v1/models > /dev/null; do sleep 5; done
    # run serving benchmark
    for np in $NumPrompts;
    do
        for inp in $InServing;
        do
            for out in $OutServing;
            do
                for mc in $MaxConcurrency;
                do
                    outjson=${report_dir}/${model_name}_${mode}_req${np}_in${inp}_out${out}_mc${mc}_${datatype}.json
                    outcsv=${report_summary_dir}/${model_name}_${mode}_report.csv
                    python3 $tool_serving --model $model --percentile-metrics "ttft,tpot,itl,e2el" --dataset-name random --random-input-len $inp --random-output-len $out --num-prompts $np --max-concurrency $mc --ignore-eos --save-result --result-filename $outjson
                    python3 $tool_report --mode $mode --model $model_name --num-prompts $np --tp $tp --input-len $inp --output-len $out --max-concurrency $mc --input-json $outjson --output-csv $outcsv --dtype $datatype
                done
            done
        done
    done
    # stop server
    kill $server_pid && wait $server_pid
fi

echo "Generate report of multiple results"
tool_parser="parse_csv.py"
latency_summary_csv=${report_summary_dir}/${model_name}_latency_report.csv
throughput_summary_csv=${report_summary_dir}/${model_name}_throughput_report.csv
serving_summary_csv=${report_summary_dir}/${model_name}_serving_report.csv
python3 $tool_parser --file_latency $latency_summary_csv --file_throughput $throughput_summary_csv --file_serving $serving_summary_csv

mv perf_${model_name}.csv ../
