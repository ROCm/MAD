#!/bin/bash

batch_size=(1 2 4)
batch_size_enable_xformers=(1 2 4)

max_train_steps=100

export EXP_DIR=/myworkspace/scripts/pyt_huggingface_diffusers
export LOG_DIR=$EXP_DIR/logs
export CSV_DIR=$EXP_DIR/csvs
export PROFILING_DIR=$PWD/profiling
export RUN_ID=$(date '+%Y-%m-%d_%H-%M-%S')

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
#export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export ACCELERATE_CONFIG_FILE="/myworkspace/scripts/pyt_huggingface_diffusers/accelerate_ds_config.yaml"

export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2
#export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"

#export PYTORCH_TUNABLEOP_ENABLED=1
#export PYTORCH_TUNABLEOP_TUNING=0
#export PYTORCH_TUNABLEOP_VERBOSE=0
#export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=1
#export PYTORCH_TUNABLEOP_FILENAME=/myworkspace/scripts/pyt_huggingface_diffusers/tunableop/2K_tunableop_.csv

# Detect whether ROCm or CUDA is installed
if command -v rocm-smi &>/dev/null; then
    PLATFORM="ROCM"
    rocm-smi
elif command -v nvidia-smi &>/dev/null; then
    PLATFORM="CUDA"
    nvidia-smi
else
    echo "Neither ROCm nor CUDA could be detected. Exiting."
    exit 1
fi

mkdir -p $EXP_DIR
mkdir -p $LOG_DIR
mkdir -p $CSV_DIR

resolution=2048
while [[ $# -gt 0 ]]; do
    argument=$1
    case $argument in
        "-r")
            if [[ -n "$2" && "$2" != -* ]]; then
                res_upper=$(echo "$2" | tr '[:lower:]' '[:upper:]')
		if [[ "$res_upper" == "1K" ]]; then
                    resolution=1024
                    shift
                    shift
	        elif [[ "$res_upper" == "2K" ]]; then
                    resolution=2048
		    shift
		    shift
	        else
                    echo "Error: suppport only \"1k\" and \"2k\""
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

echo "Resolution :" $resolution
echo "Dataset :" $DATASET_NAME

if [[ "$resolution" -eq 1024 ]]; then
    batch_size=(24)
    #batch_size=(1 2 4 8 12 16 24)
    batch_size_enable_xformers=(4 8 12 16 24)
else
    batch_size=(3)
    #batch_size=(1 2 4)
    batch_size_enable_xformers=(1 2 3 4)
fi

export HF_HOME=/data/huggingface/

#rocm-smi --setperfdeterminism 1900

max_train_steps=20
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4
export MIOPEN_LOG_LEVEL=6

for BATCH_SIZE in "${batch_size[@]}"; do
    echo "_____________________tunning steps_____________________"
    echo "Running SDXL Fine Tune, BS=$BATCH_SIZE"
    echo "Enable xformers only for ${batch_size_enable_xformers[@]}."
    if [[ " ${batch_size_enable_xformers[@]} " =~ " ${BATCH_SIZE} " ]]; then
        echo "Enable xformers ${BATCH_SIZE} ..."
        ENABLED_XFORMERS="--enable_xformers_memory_efficient_attention"
    else
        echo "Disable xformers for ${BATCH_SIZE} ..."
        ENABLED_XFORMERS=""
    fi

    BATCH_SIZE_EX=${BATCH_SIZE} $RUN_CMD accelerate launch --config_file $ACCELERATE_CONFIG_FILE /myworkspace/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME --caption_column="text" \
    --resolution=${resolution}  \
    --train_batch_size=${BATCH_SIZE} \
    ${ENABLED_XFORMERS} \
    --num_train_epochs=2 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --max_train_steps=${max_train_steps} \
    --seed=1234 \
    --output_dir="sd-naruto-model-lora-sdxl-miopen" #2>&1 | tee "${LOG_DIR}/${RUN_ID}_${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.log"
done

max_train_steps=100

export MIOPEN_FIND_MODE=5
unset MIOPEN_FIND_ENFORCE
unset MIOPEN_LOG_LEVEL

for BATCH_SIZE in "${batch_size[@]}"; do
    echo "_____________________training steps_____________________"
    echo "Running SDXL Fine Tune, BS=$BATCH_SIZE"
    echo "Enable xformers only for ${batch_size_enable_xformers[@]}."
    if [[ " ${batch_size_enable_xformers[@]} " =~ " ${BATCH_SIZE} " ]]; then
        echo "Enable xformers ${BATCH_SIZE} ..."
        ENABLED_XFORMERS="--enable_xformers_memory_efficient_attention"
    else
        echo "Disable xformers for ${BATCH_SIZE} ..."
        ENABLED_XFORMERS=""
    fi

    BATCH_SIZE_EX=${BATCH_SIZE} $RUN_CMD accelerate launch --config_file $ACCELERATE_CONFIG_FILE /myworkspace/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME --caption_column="text" \
    --resolution=${resolution}  \
    --train_batch_size=${BATCH_SIZE} \
    ${ENABLED_XFORMERS} \
    --num_train_epochs=2 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --max_train_steps=${max_train_steps} \
    --seed=1234 \
    --output_dir="sd-naruto-model-lora-sdxl" 2>&1 | tee "${LOG_DIR}/${RUN_ID}_${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.log"
done

#rocm-smi --resetperfdeterminism

cp "${LOG_DIR}/${RUN_ID}_${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.log" sdxl_lora.log
python3 get_metric.py sdxl_lora.log "${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.csv"
rm sdxl_lora.log
cp "${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.csv" ..
mv "${PLATFORM}_SDXL_FINETUNE_bs${BATCH_SIZE}.csv" "$CSV_DIR" 
