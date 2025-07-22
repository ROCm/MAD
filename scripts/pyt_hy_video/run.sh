#!/bin/bash

export NCCL_MIN_NCHANNELS=112
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="hunyuan_video_usp_example.py"
#MODEL_ID="/cfs/dit/HunyuanVideo"
MODEL_ID="tencent/HunyuanVideo"
INFERENCE_STEP=30

mkdir -p ./results

# CogVideoX specific task args
height=720
TASK_ARGS="--height 720 --width 1280 --num_frames 129"

while [[ $# -gt 0 ]]; do
    argument=$1
    case $argument in
        "-h")
            if [[ -n "$2" && "$2" != -* ]]; then
                res_upper=$(echo "$2")
                if [[ "$res_upper" == "720" ]]; then
                    height=720
                    TASK_ARGS="--height 720 --width 1280 --num_frames 129"
                    shift
                    shift
                elif [[ "$res_upper" == "960" ]]; then
                    height=960
                    TASK_ARGS="--height 960 --width 960 --num_frames 129"
                    shift
                    shift
                elif [[ "$res_upper" == "1280" ]]; then
                    height=1280
                    TASK_ARGS="--height 1280 --width 720 --num_frames 129"
                    shift
                    shift
                else
                    echo "Error: suppport only height=720,960,1280"
                    exit 1
                fi
            else
                echo "Error: -r option requires a value"
                exit 1
            fi
            ;;
        *)
            echo "Invalid argument: $argument"
            exit 1
            ;;
    esac
done

echo $TASK_ARGS

# CogVideoX parallel configuration
N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 8 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
ENABLE_SLICING="--enable_slicing"
#ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"
# COMPILE_FLAG="--use_torch_compile"


export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=6

echo "__________________________________________________________________"
echo "!!!!! MIOPEN TUNNING !!!!!"
echo "__________________________________________________________________"

INFERENCE_STEP=1
torchrun --nproc_per_node=$N_GPUS /hunyuanvideo/xDiT/examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "A cat walks on the grass, realistic" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$ENABLE_MODEL_CPU_OFFLOAD \
$COMPILE_FLAG

export MIOPEN_FIND_MODE=5
unset MIOPEN_FIND_ENFORCE
unset MIOPEN_ENABLE_LOGGING
unset MIOPEN_ENABLE_LOGGING_CMD
unset MIOPEN_LOG_LEVEL

echo "__________________________________________________________________"
echo "!!!!! RUNNING !!!!!"
echo "__________________________________________________________________"

CUR_DIR=`pwd`
LOG_DIR=$CUR_DIR/logs
pushd $CUR_DIR

mkdir -p $LOG_DIR

INFERENCE_STEP=30
torchrun --nproc_per_node=$N_GPUS /hunyuanvideo/xDiT/examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "A cat walks on the grass, realistic" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$ENABLE_MODEL_CPU_OFFLOAD \
$COMPILE_FLAG 2>&1 | tee $LOG_DIR/hy_video_inference.log

python3 get_hy_video_metric.py $LOG_DIR/hy_video_inference.log perf_pyt_hy_video_${height}.csv
rm hy_video_inference.log
cp perf_pyt_hy_video_${height}.csv ..
