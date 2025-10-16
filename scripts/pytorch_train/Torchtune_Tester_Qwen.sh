#!/bin/bash

# === Help Function ===
show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]
Example Usage: MODEL_FAMILY=qwen2_5 MODEL_SIZE=32B METHOD=lora COMPILE=True PACKED=True SEQ_LEN=8192 CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=4 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash YOUR_SCRIPT.sh

Required environment variables:
  MODEL_FAMILY   Model family (e.g., qwen2, qwen2_5, qwen3)
  MODEL_SIZE     Model size (e.g., 1.7B, 32B, 30B_A32B, 8B)
  METHOD         Finetuning method (e.g., lora, full)

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

EOF
}

# === Config Environment ===
export TORCHINDUCTOR_EXHAUSTIVE_FLEX_ATTENTION_EXPERIMENTAL=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1

# === Parse Arguments ===
if [[ "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# === Parse Optional Flags ===
FP8_ENABLED="false"
for arg in "$@"; do
  case $arg in
    --fp8)
      FP8_ENABLED="true"
      shift
      ;;
  esac
done

# === USER-PROVIDED ARGS ===
MODEL_FAMILY="${MODEL_FAMILY:?Must set MODEL_FAMILY (e.g., qwen2, qwen2_5, qwen3)}"
MODEL_SIZE="${MODEL_SIZE:?Must set MODEL_SIZE (e.g., 1.7B, 32B, 30B_A32B, 8B)}"
METHOD="${METHOD:?Must set METHOD (e.g., lora, full, qlora)}"

# === Directories & Paths ===
CONFIG_DIR="/workspace/torchtune/recipes/configs/$MODEL_FAMILY"
CONFIG_FILE="${CONFIG_DIR}/${MODEL_SIZE}_${METHOD}.yaml"


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
    HF_MODEL_NAME=$(grep -A1 'tune download' "$CONFIG_FILE" | grep -oE 'Qwen/[A-Za-z0-9\.\-]+')
    if [ -z "$HF_MODEL_NAME" ]; then
        echo "Could not extract Hugging Face model name from $CONFIG_FILE" >&2
        exit 1
    fi

    echo "Downloading model $HF_MODEL_NAME to $MODEL_DIR..."
    hf download "$HF_MODEL_NAME" --local-dir "$MODEL_DIR" || {
        echo "Model download failed"
        exit 1
    }
fi

if [[ "${MODEL_FAMILY,,}" == "qwen2" || "${MODEL_FAMILY,,}" == "qwen3" || "${MODEL_FAMILY,,}" == "qwen2_5" ]]; then
    TOKENIZER_PATH="${MODEL_DIR}/vocab.json"
fi

if [[ "${MODEL_FAMILY,,}" == "qwen2" || "${MODEL_FAMILY,,}" == "qwen3" || "${MODEL_FAMILY,,}" == "qwen2_5" ]]; then
    TOKENIZER_MERGE_FILE_PATH="${MODEL_DIR}/merges.txt"
fi


# === Print Env Variables ===
echo "Running with environment variables..." | tee -a $LOG_FILES
env | tee -a $LOG_FILES

# === qlora special handling ===
if [[ "$METHOD" == "lora" || "$METHOD" == "qlora" ]]; then
    TUNE_METHOD="lora"
else
    TUNE_METHOD="$METHOD"
fi

# === Handle FP8 Patch for YAML Config ===
if [[ "$FP8_ENABLED" == "true" ]]; then
  echo "Modifying YAML for FP8 and commenting out tensor parallel config: $CONFIG_FILE"

  TMP_FILE=$(mktemp)
  in_tensor_plan="false"
  found_enable_fp8="false"
  found_fp8_recipe="false"

  while IFS= read -r line; do
    trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')

    # Track FP8 flags
    [[ "$trimmed" == "enable_fp8_training:"* ]] && found_enable_fp8="true"
    [[ "$trimmed" == "fp8_recipe_name:"* ]] && found_fp8_recipe="true"

    # Handle tensor_parallel_plan block
    if [[ "$trimmed" == "tensor_parallel_plan:" ]]; then
      in_tensor_plan="true"
      echo "# $line" >> "$TMP_FILE"
      continue
    fi

    if [[ "$in_tensor_plan" == "true" && ! "$line" =~ ^[[:space:]]+ ]]; then
      in_tensor_plan="false"
    fi

    # Comment specific tensor config lines
    if echo "$trimmed" | grep -Eq '^(tensor_parallel_dim|tensor_parallel_plan|data_parallel_shard_dim|data_parallel_replicate_dim):'; then
      echo "# $line" >> "$TMP_FILE"
    elif [[ "$in_tensor_plan" == "true" && "$trimmed" == "_component_:"* ]]; then
      echo "# $line" >> "$TMP_FILE"
    else
      echo "$line" >> "$TMP_FILE"
    fi
  done < "$CONFIG_FILE"

  # Append FP8 flags if not already present
  if [[ "$found_enable_fp8" == "false" ]]; then
    echo "enable_fp8_training: true" >> "$TMP_FILE"
  fi
  if [[ "$found_fp8_recipe" == "false" ]]; then
    echo "fp8_recipe_name: tensorwise" >> "$TMP_FILE"
  fi

  mv "$TMP_FILE" "$CONFIG_FILE"

  if [[ "$found_enable_fp8" == "true" && "$found_fp8_recipe" == "true" ]]; then
    echo "FP8 config settings already present in $CONFIG_FILE"
  else
    echo "Inserted FP8 config settings into $CONFIG_FILE"
  fi

else
  echo "Removing FP8 settings from YAML: $CONFIG_FILE"

  TMP_FILE=$(mktemp)
  while IFS= read -r line; do
    trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
    if [[ "$trimmed" == "enable_fp8_training:"* || "$trimmed" == "fp8_recipe_name:"* ]]; then
      continue
    fi
    echo "$line" >> "$TMP_FILE"
  done < "$CONFIG_FILE"

  mv "$TMP_FILE" "$CONFIG_FILE"
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
    tokenizer.merges_file="$TOKENIZER_MERGE_FILE_PATH" \
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
