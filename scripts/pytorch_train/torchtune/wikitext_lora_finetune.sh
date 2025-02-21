#!/bin/bash

# sample usage: COMPILE=False MAX_STEPS=10 EPOCHS=1 SAVE_WEIGHTS=False bash wikitext_finetune.sh
# torch.compile is currently working in our environment.

CONFIG="${CONFIG:-./llama_3_1_70b_lora_finetune_recipe.yaml}"
MODEL_DIR="${MODEL_DIR:-./models/Llama-3.1-70B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-./wikitext_train.json}"
TEST_FILE="${TEST_FILE:-./wikitext_test.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

PACKED="${PACKED:-False}"
MAX_STEPS="${MAX_STEPS:null}"
MBS="${MBS:-64}"
GAS="${GAS:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-True}"
CPU_OFFLOAD="${CPU_OFFLOAD:-True}"
COMPILE="${COMPILE:-True}"
EPOCHS="${EPOCHS:-3}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-True}"
SEQ_LEN="${SEQ_LEN:-null}"
TUNE_ENV="${TUNE_ENV:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_AUTOTUNE="${MAX_AUTOTUNE:-False}"
VALIDATE="${VALIDATE:-False}"
SEED="${SEED:-42}"

ROCBLAS_TUNED_GEMM="${ROCBLAS_TUNED_GEMM:-False}"
ROCBLAS_COLLECT_SHAPES="${ROCBLAS_COLLECT_SHAPES:-False}"

HIPBLAS_TUNED_GEMM="${HIPBLAS_TUNED_GEMM:-False}"
HIPBLAS_COLLECT_SHAPES="${HIPBLAS_COLLECT_SHAPES:-False}"

if [ "${HIPBLAS_TUNED_GEMM,,}" = "true" ] && [ "${HIPBLAS_COLLECT_SHAPES,,}" = "true" ]; then
    echo "HIPBLAS_TUNED_GEMM and HIPBLAS_COLLECT_SHAPES cannot both be set to true at the same time" >&2
    exit 1
fi

if [ "${ROCBLAS_TUNED_GEMM,,}" = "true" ] && [ "${ROCBLAS_COLLECT_SHAPES,,}" = "true" ]; then
    echo "ROCBLAS_TUNED_GEMM and ROCBLAS_COLLECT_SHAPES cannot both be set to true at the same time" >&2
    exit 1
fi

if ([ "${ROCBLAS_TUNED_GEMM,,}" = "true" ] || [ "${ROCBLAS_COLLECT_SHAPES,,}" = "true" ]) && ([ "${HIPBLAS_TUNED_GEMM,,}" = "true" ] || [ "${HIPBLAS_COLLECT_SHAPES,,}" = "true" ]); then
    echo "ROCBLAS and HIPBLAS cannot both be requested at the same time" >&2
    exit 1
fi


LOG_FILES="history.txt"

echo "Running with environment variables..." | tee -a $LOG_FILES
env | tee -a $LOG_FILES


if [ ! -f "$CONFIG" ]; then
    echo "$CONFIG not found, download it first" >&2
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ] ; then
    echo "$TRAIN_FILE not found, download it first" >&2
    exit 1
fi


if [ ! -d $MODEL_DIR ]; then
    echo "70B model not found in $MODEL_DIR" >&2
    echo "Download with: "
    echo "  huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir $MODEL_DIR --exclude 'original/*.pth'"
    exit 1
fi

VALIDATE_ARGS=""
if [ "${VALIDATE,,}" = "true" ]; then
    if [ ! -f "$TEST_FILE" ] ; then
        echo "$TEST_FILE not found, download it first" >&2
        exit 1
    fi
    VALIDATE_ARGS="val_dataset._component_=torchtune.datasets.text_completion_dataset val_dataset.source=json val_dataset.column=input val_dataset.data_files=$TEST_FILE"
fi

# if we're not saving weights, we'll need to edit the code, because there are no options to disable that.
if [ "${SAVE_WEIGHTS,,}" = "false" ]; then
    sed -i 's/.*self.save_checkpoint(epoch=curr_epoch)/            #self.save_checkpoint(epoch=curr_epoch)/' recipes/lora_finetune_distributed.py
else
    # theoretically this shouldn't be necessary, but incomplete runs might leave this behind.
    # i wonder if git checkout might be a better option here?
    sed -i 's/.*self.save_checkpoint(epoch=curr_epoch)/            self.save_checkpoint(epoch=curr_epoch)/' recipes/lora_finetune_distributed.py
    LOG_FILES="$TEE_FILES $CHECKPOINT_DIR/run_details.txt"
fi

if [ "${TUNE_ENV,,}" = "true" ]; then
    # copied wholesale from megatron script
    # i have no idea if some of these are megatron-specific or not
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export HIP_FORCE_DEV_KERNARG=1
fi


if [ "${ROCBLAS_TUNED_GEMM,,}" = "true" ]; then
    echo "Using optimized ROCBLAS GEMMs"
    export TORCH_BLAS_PREFER_HIPBLASLT=0 # disable hipblaslt to force rocblas
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=0 # disable tuning, just use the tuned gemm
    # CSV with best kernels per shapes
    # Hardcoded to MBS=64 currently
    export PYTORCH_TUNABLEOP_FILENAME=./tuned_shapes/mbs_64_compiled_1epoch/full_tuned%d.csv
fi

if [ "${ROCBLAS_COLLECT_SHAPES,,}" = "true" ]; then
    echo "Collecting shapes for ROCBLAS"
    export TORCH_BLAS_PREFER_HIPBLASLT=0 # disable hipblaslt to force rocblas
    export ROCBLAS_LAYER=4  # record shapes
    export PYTORCH_TUNABLEOP_ENABLED=0
fi

if [ "${MAX_AUTOTUNE,,}" = "true" ]; then
    # there may be other options to explore here.
    export TORCHINDUCTOR_MAX_AUTOTUNE=1
fi

if [ "${HIPBLAS_COLLECT_SHAPES,,}" = "true" ]; then
    echo "Collecting shapes for HIPBLAS"
    export TORCH_BLAS_PREFER_HIPBLASLT=1
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=1
    export PYTORCH_TUNABLEOP_VERBOSE=2
    export PYTORCH_TUNABLEOP_VERBOSE_FILENAME="out"
    export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=50
    export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=20
    export PYTORCH_TUNABLEOP_FILENAME=./tuned_shapes/mbs_64_compiled_fav3_tunableop_1epoch/hipblaslt_mbs64_tuning.csv
fi

if [ "${HIPBLAS_TUNED_GEMM,,}" = "true" ]; then
    echo "Using optimized ROCBLAS GEMMs"
    export TORCH_BLAS_PREFER_HIPBLASLT=1
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=0
    # CSV with best kernels per shapes
    # Hardcoded to MBS=64 currently
    export PYTORCH_TUNABLEOP_FILENAME=./tuned_shapes/mbs_64_compiled_test/hipblaslt_mbs64_tuning.csv
fi


tune run --nproc_per_node 8 \
    lora_finetune_distributed --config $CONFIG \
    log_peak_memory_stats=True \
    output_dir=./logs \
    checkpointer.output_dir=./checkpoints \
    dataset.data_files=$TRAIN_FILE \
    tokenizer.path=${MODEL_DIR}/original/tokenizer.model \
    tokenizer.max_seq_len=$SEQ_LEN \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    gradient_accumulation_steps=$GAS \
    max_steps_per_epoch=$MAX_STEPS \
    epochs=$EPOCHS \
    dataset.packed=$PACKED \
    fsdp_cpu_offload=$CPU_OFFLOAD \
    batch_size=$MBS \
    enable_activation_checkpointing=$ACTIVATION_CHECKPOINTING \
    compile=$COMPILE \
    seed=$SEED \
    $VALIDATE_ARGS \
    $EXTRA_ARGS \
        2>&1 | tee stdout.log
LOG_PATH=`cat stdout.log | grep 'Writing logs to ' | head -1 | awk '{print $4}'`

# revert code edits
if [ "${SAVE_WEIGHTS,,}" = "false" ]; then
    sed -i 's/.*self.save_checkpoint(epoch=curr_epoch)/            self.save_checkpoint(epoch=curr_epoch)/' recipes/lora_finetune_distributed.py
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")

echo ========================================================================== | tee -a history.txt
COMMIT=$(git rev-parse --short HEAD)
echo TORCH=$TORCH_VERSION COMMIT=$COMMIT | tee -a $LOG_FILES
echo COMPILE=$COMPILE CPU_OFFLOAD=$CPU_OFFLOAD PACKED=$PACKED SEQ_LEN=$SEQ_LEN ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING TUNE_ENV=$TUNE_ENV MAX_AUTOTUNE=$MAX_AUTOTUNE MBS=$MBS GAS=$GAS SEED=$SEED | tee -a $LOG_FILES
if [ ! -z "$EXTRA_ARGS" ] ; then
    echo "EXTRA_ARGS=$EXTRA_ARGS" | tee -a $LOG_FILES
fi

if [ -n "$LOG_PATH" ]; then
    grep -v val_num_tokens_total $LOG_PATH | awk -F'[: ]' 'NR>1 {sum+=$9; all_sum+=$11; if($13>max) max=$13} END {print "Max memory alloc:", max, "\nAverage tokens/s/gpu:", all_sum/(NR-1), "\nUnmasked tokens/s/gpu: ", sum/(NR-1)}' | tee -a $LOG_FILES
    grep time_per_validation_epoch $LOG_PATH | awk -F'[: ]'  '{ print "Step", $2, "validation loss:", $5}' | tee -a $LOG_FILES

    if [ "${SAVE_WEIGHTS,,}" = "true" ]; then
        cp $LOG_PATH $CHECKPOINT_DIR/steps.txt
        cp stdout.log $CHECKPOINT_DIR
    fi
else
    echo "No log path found in command output" >&2
    exit 1
fi
