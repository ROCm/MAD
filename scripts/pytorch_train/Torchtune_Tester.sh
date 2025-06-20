#!/bin/bash

# === Help Function ===
show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]
Example Usage: MODEL_FAMILY=llama4 MODEL_SIZE=17B_16E METHOD=full COMPILE=True PACKED=True SEQ_LEN=512 CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash YOUR_SCRIPT.sh

Required environment variables:
  MODEL_FAMILY   Model family (e.g., llama2, llama3, llama3_1, llama3_2, llama3_2_vision, llama3_3, llama4)
  MODEL_SIZE     Model size (e.g., 70B, 17B_16E)
  METHOD         Finetuning method (e.g., lora, full, qlora)

Optional environment variables:
  MODEL_DIR           Path to the model directory (default: ./models/\${MODEL_FAMILY^^}-\${MODEL_SIZE}-Instruct)
  CHECKPOINT_DIR      Path to the checkpoint directory (default: ./checkpoints)
  PACKED              Whether to use packed datasets (default: False)
  MAX_STEPS           Maximum steps per epoch (default: null)
  MBS                 Mini-batch size (default: 64)
  GAS                 Gradient accumulation steps (default: 1)
  ACTIVATION_CHECKPOINTING  Enable activation checkpointing (default: True)
  CPU_OFFLOAD         Enable CPU offloading (default: True)
  COMPILE             Enable model compilation (default: True)
  EPOCHS              Number of training epochs (default: 3)
  SAVE_WEIGHTS        Save model weights after training (default: True)
  SEQ_LEN             Maximum sequence length (default: null)
  EXTRA_ARGS          Additional arguments to pass to the training script
  SEED                Random seed (default: 42)

Available configurations:

llama2:
  - 13B_full
  - 13B_lora
  - 70B_lora
  - 70B_qlora
  - 7B_full
  - 7B_lora
  - 7B_qlora

llama3:
  - 70B_full
  - 70B_lora
  - 8B_full
  - 8B_lora

llama3_1:
  - 405B_qlora
  - 70B_full
  - 70B_lora
  - 8B_full
  - 8B_lora

llama3_2:
  - 1B_full
  - 1B_lora
  - 3B_full
  - 3B_lora

llama3_2_vision:
  - 11B_full
  - 11B_lora
  - 11B_qlora
  - 90B_full
  - 90B_lora
  - 90B_qlora

llama3_3:
  - 70B_full
  - 70B_lora
  - 70B_qlora

llama4:
  - scout_17B_16E_full
  - scout_17B_16E_lora

EOF
}

# === Parse Arguments ===
if [[ "$1" == "--help" ]]; then
  show_help
  exit 0
fi


# === USER-PROVIDED ARGS ===
MODEL_FAMILY="${MODEL_FAMILY:?Must set MODEL_FAMILY (e.g., llama3_3, llama4)}"
MODEL_SIZE="${MODEL_SIZE:?Must set MODEL_SIZE (e.g., 70B, 17B_16E)}"
METHOD="${METHOD:?Must set METHOD (e.g., lora, full, qlora)}"

# === Directories & Paths ===
CONFIG_DIR="/workspace/torchtune/recipes/configs/$MODEL_FAMILY"
if [[ "$MODEL_FAMILY" == "llama4" ]]; then
    CONFIG_FILE="${CONFIG_DIR}/scout_${MODEL_SIZE}_${METHOD}.yaml"
else
    CONFIG_FILE="${CONFIG_DIR}/${MODEL_SIZE}_${METHOD}.yaml"
fi

MODEL_DIR="${MODEL_DIR:-./models/${MODEL_FAMILY^^}-${MODEL_SIZE}-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# Normalize the model and config directory paths (case-insensitive)
MODEL_DIR="$(realpath "$MODEL_DIR" 2>/dev/null || echo "$MODEL_DIR")"
CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR" 2>/dev/null || echo "$CHECKPOINT_DIR")"

# === Optional Environment Overrides ===
PACKED="${PACKED:-False}"
MAX_STEPS="${MAX_STEPS:-null}"
MBS="${MBS:-64}"
GAS="${GAS:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-True}"
CPU_OFFLOAD="${CPU_OFFLOAD:-True}"
COMPILE="${COMPILE:-True}"
EPOCHS="${EPOCHS:-3}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-True}"
SEQ_LEN="${SEQ_LEN:-null}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SEED="${SEED:-42}"
LOG_FILES="history.txt"

# === Validate Config File ===
echo "Using config: $CONFIG_FILE" | tee -a $LOG_FILES
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# === Model Directory Check and Auto Download ===
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found in $MODEL_DIR. Attempting download from Hugging Face..."

    # Automatically extract HF model name from config
    HF_MODEL_NAME=$(grep -A1 'tune download' "$CONFIG_FILE" | grep -oE 'meta-llama/[A-Za-z0-9\.\-]+')
    if [ -z "$HF_MODEL_NAME" ]; then
        echo "Could not extract Hugging Face model name from $CONFIG_FILE" >&2
        exit 1
    fi

    echo "Downloading model $HF_MODEL_NAME to $MODEL_DIR..."
    huggingface-cli download "$HF_MODEL_NAME" --local-dir "$MODEL_DIR" --exclude "original/*.pth" || {
        echo "Model download failed"
        exit 1
    }
fi

# === Print Env Variables ===
echo "Running with environment variables..." | tee -a $LOG_FILES
env | tee -a $LOG_FILES

# === Special Handling for Tokenizer Path ===
if [[ "${MODEL_FAMILY,,}" == "llama2" || "${MODEL_FAMILY,,}" == "llama4" ]]; then
    TOKENIZER_PATH="${MODEL_DIR}/tokenizer.model"
fi

if [[ "$MODEL_SIZE" == "405B" ]]; then
    TOKENIZER_PATH="${MODEL_DIR}/original/mp8/tokenizer.model"
fi

if [[ -z "$TOKENIZER_PATH" ]]; then
    TOKENIZER_PATH="${MODEL_DIR}/original/tokenizer.model"
fi


# === qlora special handling ===
if [[ "$METHOD" == "lora" || "$METHOD" == "qlora" ]]; then
    TUNE_METHOD="lora"
else
    TUNE_METHOD="$METHOD"
fi

# === Launch Finetune ===
tune run --nproc_per_node 8 \
    ${TUNE_METHOD}_finetune_distributed --config "$CONFIG_FILE" \
    log_peak_memory_stats=True \
    output_dir=./logs \
    checkpointer.output_dir="$CHECKPOINT_DIR" \
    dataset.data_files="$TRAIN_FILE" \
    tokenizer.path="$TOKENIZER_PATH" \
    tokenizer.max_seq_len="$SEQ_LEN" \
    checkpointer.checkpoint_dir="$MODEL_DIR" \
    gradient_accumulation_steps="$GAS" \
    max_steps_per_epoch="$MAX_STEPS" \
    epochs="$EPOCHS" \
    dataset.packed="$PACKED" \
    fsdp_cpu_offload="$CPU_OFFLOAD" \
    batch_size="$MBS" \
    enable_activation_checkpointing="$ACTIVATION_CHECKPOINTING" \
    compile="$COMPILE" \
    seed="$SEED" \
    $EXTRA_ARGS \
        2>&1 | tee stdout.log

# === Parse Log Path ===
LOG_PATH=$(grep 'Writing logs to ' stdout.log | awk '{print $4}')
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")

echo ========================================================================== | tee -a $LOG_FILES
echo TORCH=$TORCH_VERSION | tee -a $LOG_FILES
echo COMPILE=$COMPILE CPU_OFFLOAD=$CPU_OFFLOAD PACKED=$PACKED SEQ_LEN=$SEQ_LEN ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING MBS=$MBS GAS=$GAS SEED=$SEED | tee -a $LOG_FILES
[ ! -z "$EXTRA_ARGS" ] && echo "EXTRA_ARGS=$EXTRA_ARGS" | tee -a $LOG_FILES

if [ -n "$LOG_PATH" ]; then
    # Memory alloc
    grep -Eo "peak_memory_alloc:[0-9]+\.[0-9]+" "$LOG_PATH" | grep -Eo "([^:]*)$" | \
      awk '{
          vals[NR] = $1
      }
      END {
          start = int(NR/2) + 1
          max = vals[start]
          for (i = start; i <= NR; i++) {
              if (vals[i] > max) max = vals[i]
          }
          print "Max memory alloc (last half):", max
      }' | tee -a $LOG_FILES

    # Tokens calculation
    grep -Eo "tokens_per_second_per_gpu:[0-9]+\.[0-9]+" "$LOG_PATH" | grep -Eo "([^:]*)$" | \
      awk '{
          vals[NR] = $1
      }
      END {
          start = int(NR/2) + 1
          sum = 0
          count = 0
          for (i = start; i <= NR; i++) {
              sum += vals[i]
              count++
          }
          avg = (count > 0) ? sum / count : 0
          print "Average tokens/s/gpu (last half):", avg
      }' | tee -a $LOG_FILES


    if [ "${SAVE_WEIGHTS,,}" = "true" ]; then
        cp "$LOG_PATH" "$CHECKPOINT_DIR/steps.txt"
        cp stdout.log "$CHECKPOINT_DIR"
    fi
else
    echo "No log path found in command output" >&2
    exit 1
fi
