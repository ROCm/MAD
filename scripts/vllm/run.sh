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
set -ex

# Preliminary setup
export VLLM_DIR=/app/vllm
export HF_HUB_CACHE="/myworkspace"
MAD_MODEL_NAME=$(echo $MAD_MODEL_NAME | tr "/" "_")

# By default run all benchmarks in config
BENCHMARK="all"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --model_repo) MODEL="$2"; shift ;;
        # config overrides
        --benchmark) BENCHMARK="$2"; shift ;;
        --tp) TP="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --in) IN="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --num_prompts) NUM_PROMPTS="$2"; shift ;;
        --max_concurrency) MAX_CONCURRENCY="$2"; shift ;;
        --datatype) DATATYPE="$2"; shift ;;
        --max_num_seqs) MAX_NUM_SEQS="$2"; shift ;;
        --max_num_batched_tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$CONFIG" ]]; then
    echo "Please specify a config file from scripts/vllm/configs/*.csv e.g. configs/default.csv"
    exit 1
fi

# vLLM overrides
# gpt-oss on AITER gets best performance with --block-size 64 and FULL_AND_PIECEWISE graph capture
if [[ $MODEL == "openai/gpt-oss-20b" || $MODEL == "openai/gpt-oss-120b" ]]; then
    echo "Setting --block-size 64 and cudagraph_mode FULL_AND_PIECEWISE with AITER unified attention for gpt-oss"
    export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
    export VLLM_ROCM_USE_AITER_MHA=0
    VLLM_ARGS='--block-size 64'
fi

# Deepseek AITER MLA requires --block-size 1
if [[ $MODEL == *"DeepSeek-V3"* || $MODEL == *"DeepSeek-R1"* ]]; then
    if [[ $VLLM_ROCM_USE_AITER == 0 ]]; then
        echo "Deepseek Triton MLA does not support cudagraph capture; using cudagraph_mode PIECEWISE"
        VLLM_ARGS='--compilation-config {"cudagraph_mode":"PIECEWISE"}'
    else
        echo "Setting --block-size 1 for Deepseek AITER MLA"
        VLLM_ARGS='--block-size 1'
    fi
fi

# Llama 4 currently does not support full cudagraph or attn fusion
if [[ $MODEL == *"Llama-4"* ]]; then
    VLLM_ARGS='--compilation-config {"cudagraph_mode":"PIECEWISE","pass_config":{"enable_attn_fusion":false}}'
fi

# DeepSeek MoE models are not compatible with AITER (weight shuffling issue)
if [[ $MODEL == *"deepseek-moe"* ]]; then
    echo "Disabling AITER for DeepSeek MoE model (not compatible with current AITER implementation)"
    export VLLM_ROCM_USE_AITER=0
fi

# MXFP4 models are only supported on MI35x i.e. gfx950
if [[ $MODEL == *"MXFP4"* ]]; then
    if [[ $MAD_SYSTEM_GPU_ARCHITECTURE != *"gfx950"* ]]; then
        echo "MXFP4 models are only supported on MI35x i.e. gfx950 architecture"
        exit 1
    fi
fi

echo "=hyper params start="
echo "MODEL=$MODEL"
echo "PYTORCH_TUNABLEOP_ENABLED=$PYTORCH_TUNABLEOP_ENABLED"
echo "VLLM_USE_V1=$VLLM_USE_V1"
echo "VLLM_V1_USE_PREFILL_DECODE_ATTENTION=$VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
echo "VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER"
echo "VLLM_ROCM_USE_AITER_PAGED_ATTN=$VLLM_ROCM_USE_AITER_PAGED_ATTN"
echo "VLLM_ROCM_USE_AITER_MHA=$VLLM_ROCM_USE_AITER_MHA"
echo "=hyper params end="

# Read configs line-by-line and run benchmark
while IFS="," read -r model benchmark tp in out bs num_prompts max_concurrency datatype max_num_seqs max_num_batched_tokens max_model_len gpu_memory_utilization
do
    # If $MODEL matches model in config
    if [[ $model == $MODEL ]]; then

        # Use model name for logging
        model_name=$(basename $model)
        # Use dataprovider if present for model weights
        if [ -n "$MAD_DATAHOME" ]; then
            model=$MAD_DATAHOME
        else
            if [ -z "$MAD_SECRETS_HFTOKEN" ]; then
                echo "Warning: MAD_SECRETS_HFTOKEN is not set. If a gated model is used, please set MAD_SECRETS_HFTOKEN=<your-huggingface-token>"
            else
                export HF_TOKEN=$MAD_SECRETS_HFTOKEN
            fi
            # Explicitly download model before running benchmarks for easier debugging
            hf download $model --exclude "original/*" "*.tf" "*.onnx" "*.flax" "*.rust"
        fi

        # If $BENCHMARK is "all" or $BENCHMARK matches benchmark in config
        if [[ $BENCHMARK == "all" || $benchmark == $BENCHMARK ]]; then

            # Override config args if specified
            tp=${TP:-$tp}
            bs=${BS:-$bs}
            in=${IN:-$in}
            out=${OUT:-$out}
            num_prompts=${NUM_PROMPTS:-$num_prompts}
            max_concurrency=${MAX_CONCURRENCY:-$max_concurrency}
            datatype=${DATATYPE:-$datatype}
            max_num_seqs=${MAX_NUM_SEQS:-$max_num_seqs}
            max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$max_num_batched_tokens}
            max_model_len=${MAX_MODEL_LEN:-$max_model_len}
            gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-$gpu_memory_utilization}
            
            if [[ $MODEL == *"DeepSeek-V3"* || $MODEL == *"DeepSeek-R1"* ]]; then
                # Deepseek fp8 kv cache is currently broken on both Triton and AITER MLA backends
                DTYPE=" "
            elif [[ $datatype == "float16" ]]; then
                DTYPE=" --dtype float16 "
            elif [[ $datatype == "bfloat16" ]]; then
                DTYPE=" "
            elif [[ $datatype == "float8" ]]; then
                # Use --dtype float16 for float8 models for faster attention kernels
                # Use fp8 kv cache for throughput benchmarking for float4 and float8 models
                if [[ $benchmark == "throughput" ]]; then
                    DTYPE=" --dtype float16 --kv-cache-dtype fp8"
                else
                    DTYPE=" --dtype float16"
                fi
            elif [[ $datatype == "float4" ]]; then
                # Use fp8 kv cache for throughput benchmarking for float4 and float8 models
                if [[ $benchmark == "throughput" ]]; then
                    DTYPE=" --kv-cache-dtype fp8"
                fi
            else
                echo "Unknown datatype: $datatype"
                exit 1
            fi

            # list-type arguments
            TP_LIST=$(echo $tp | tr "," "\n")
            INP_LIST=$(echo $in | tr "," "\n")
            OUT_LIST=$(echo $out | tr "," "\n")
            BATCH_SIZES_LIST=$(echo $bs | tr "," "\n")
            NUM_PROMPTS_LIST=$(echo $num_prompts | tr "," "\n")
            MAX_CONCURRENCY_LIST=$(echo $max_concurrency | tr "," "\n")

            # optional arguments
            if [ -n "$max_num_seqs" ]; then
                MNS="--max-num-seqs $max_num_seqs"
            fi
            if [ -n "$max_num_batched_tokens" ]; then
                MNBT="--max-num-batched-tokens $max_num_batched_tokens"
            fi
            if [ -n "$max_model_len" ]; then
                MML="--max-model-len $max_model_len"
            fi
            if [ -n "$gpu_memory_utilization" ]; then
                GPU_UTIL="--gpu-memory-utilization $gpu_memory_utilization"
            fi

            OUTPUT_JSON="${MAD_MODEL_NAME}_${benchmark}.json"
            OUTPUT_CSV="${MAD_MODEL_NAME}_${benchmark}.csv"

            if [ $benchmark == "latency" ]; then
                for tp in $TP_LIST; do
                    for bs in $BATCH_SIZES_LIST; do
                        for in in $INP_LIST; do
                            for out in $OUT_LIST; do
                                echo " " 2>&1 | tee log.txt
                                echo "[INFO] LATENCY" 2>&1 | tee log.txt
                                echo $model $tp $bs $in $out $datatype 2>&1 | tee log.txt
                                vllm bench latency --model $model -tp $tp --batch-size $bs --input-len $in --output-len $out $DTYPE $MNS $MNBT $MML $GPU_UTIL --num-iters-warmup 3 --num-iters 5 --trust-remote-code --output-json $OUTPUT_JSON $VLLM_ARGS 2>&1 | tee log.txt
                                # convert vllm json to csv
                                python3 vllm_json_to_csv.py --benchmark $benchmark --model $model_name --vllm_json $OUTPUT_JSON --output_csv $OUTPUT_CSV --tensor-parallel-size $tp --batch-size $bs --input-len $in --output-len $out --dtype $datatype
                                # delete json
                                rm $OUTPUT_JSON
                            done
                        done
                    done
                done

            elif [ $benchmark == "throughput" ]; then
                for tp in $TP_LIST; do
                    for in in $INP_LIST; do
                        for out in $OUT_LIST; do
                            for num_prompts in $NUM_PROMPTS_LIST; do
                                echo " " 2>&1 | tee log.txt
                                echo "[INFO] THROUGHPUT" 2>&1 | tee log.txt
                                echo $model $tp $req $in $out $datatype 2>&1 | tee log.txt
                                vllm bench throughput --model $model -tp $tp --num-prompts $num_prompts --input-len $in --output-len $out $DTYPE $MNS $MNBT $MML $GPU_UTIL --trust-remote-code --output-json $OUTPUT_JSON $VLLM_ARGS 2>&1 | tee log.txt
                                # convert vllm json to perf csv
                                python3 vllm_json_to_csv.py --benchmark $benchmark --model $model_name --vllm_json $OUTPUT_JSON --output_csv $OUTPUT_CSV --tensor-parallel-size $tp --num-prompts $num_prompts --input-len $in --output-len $out --dtype $datatype
                                # delete json
                                rm $OUTPUT_JSON
                            done
                        done
                    done
                done

            elif [ $benchmark == "serving" ]; then
                for tp in $TP_LIST; do
                    # start server and send it to background using nohup {command} &
                    nohup vllm serve $model -tp $tp $DTYPE $MNS $MNBT $MML $GPU_UTIL --swap-space 16 --no-enable-prefix-caching --disable-log-requests --disable-uvicorn-access-log --trust-remote-code $VLLM_ARGS &
                    # get the server pid
                    server_pid=$!
                    echo "vllm server pid: $server_pid"
                    # wait for the server to start
                    until curl -s http://localhost:8000/v1/models; do sleep 30; done
                    for in in $INP_LIST; do
                        for out in $OUT_LIST; do
                            for max_concurrency in $MAX_CONCURRENCY_LIST; do
                                # if num_prompts is not specified use 10 * max_concurrency
                                if [ -z "$NUM_PROMPTS_LIST" ]; then
                                    NUM_PROMPTS_LIST_SP=$((10 * max_concurrency))
                                else
                                    NUM_PROMPTS_LIST_SP=$NUM_PROMPTS_LIST
                                fi
                                for num_prompts in $NUM_PROMPTS_LIST_SP; do
                                    # run serving benchmark
                                    echo " " 2>&1 | tee log.txt
                                    echo "[INFO] SERVING" 2>&1 | tee log.txt
                                    echo $model $tp $max_concurrency $num_prompts $in $out $datatype 2>&1 | tee log.txt
                                    vllm bench serve --model $model --percentile-metrics "tpot,itl,e2el" --dataset-name random --ignore-eos --max-concurrency $max_concurrency --num-prompts $num_prompts --random-input-len $in --random-output-len $out --trust-remote-code --save-result --result-filename $OUTPUT_JSON 2>&1 | tee log.txt
                                    # convert vllm json to perf csv
                                    python3 vllm_json_to_csv.py --benchmark $benchmark --model $model_name --vllm_json $OUTPUT_JSON --output_csv $OUTPUT_CSV --tensor-parallel-size $tp --max-concurrency $max_concurrency --num-prompts $num_prompts --input-len $in --output-len $out --dtype $datatype
                                    # delete json
                                    rm $OUTPUT_JSON
                                done
                            done
                        done
                    done
                    # kill the server
                    kill $server_pid && wait $server_pid
                done

            else
                echo "Unknown benchmark: $benchmark"
                exit 1
            fi
        fi
    fi
done < <(tail -n +2 $CONFIG)

# Generate multiple_results csv and move it to parent directory
latency_csv=${MAD_MODEL_NAME}_latency.csv
throughput_csv=${MAD_MODEL_NAME}_throughput.csv
serving_csv=${MAD_MODEL_NAME}_serving.csv
output_csv=perf_$(basename $MODEL).csv
python3 parse_csv.py --latency_csv $latency_csv --throughput_csv $throughput_csv --serving_csv $serving_csv --output_csv $output_csv
mv $output_csv ../
