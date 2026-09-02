#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
#./jax-maxtext_benchmark_report.sh  -m $model_name -q $quantization


# Parse command-line arguments
while getopts "m:q:" opt; do
    case "$opt" in
        m) MODEL_REPO="$OPTARG" ;;
        q) QUANTIZATION="$OPTARG" ;;
        # `usage` is not defined in this file. Calling it printed "usage: command not found"
        # and carried on with MODEL_REPO empty, ending in "Unsupported training mode" - a
        # message that sends the reader looking in the wrong place entirely.
        *) echo "FATAL: usage: $0 -m <model> [-q <quantization>]" >&2; exit 1 ;;
    esac
done

# Set default values for additional parameters
MODE="pretrain"
# Multi-node aware: madengine (and any srun wrapper) exports NNODES; default to the
# previous single-node behaviour when it is absent. Same pattern as
# scripts/primus_megatron-lm/primus_megatron-lm_benchmark_report.sh.
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Reject anything that is not a plain integer, as the primus script does. Without this a
# non-numeric NNODES makes `[ "$NNODES" -gt 1 ]` exit 2, which reads as false: a real
# multi-node job would take every single-node branch below and emit no measurement at all.
# `[[ ]]` is worse still - it expands a name, so NNODES=GPUS_PER_NODE evaluates to 8 and
# writes num_gpus=64 for an 8-GPU run.
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [ "$NNODES" -lt 1 ]; then
  echo "WARNING: NNODES='$NNODES' is not a positive integer; treating this as a single node" >&2
  NNODES=1
fi
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || [ "$GPUS_PER_NODE" -lt 1 ]; then
  echo "WARNING: GPUS_PER_NODE='$GPUS_PER_NODE' is not a positive integer; assuming 8" >&2
  GPUS_PER_NODE=8
fi
if ! [[ "${_RANK_RAW:=${NODE_RANK:-0}}" =~ ^[0-9]+$ ]]; then
  echo "WARNING: NODE_RANK='${_RANK_RAW}' is not an integer; assuming rank 0" >&2
  _RANK_RAW=0
fi
NUM_GPUS=$((NNODES*GPUS_PER_NODE))
echo "Topology: NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE NUM_GPUS=$NUM_GPUS"

echo "=hyper params start="
echo $MODEL_REPO
echo $QUANTIZATION
echo "=hyper params end="

# Per-rank filename, with rank 0 owning the canonical name madengine ingests.
#
# On madengine's standard SLURM path each task copies the project into a node-local
# workspace (SLURM_TMPDIR or /tmp), so the ranks do NOT share this path and there is no race
# there. The per-rank naming is for launchers that give every rank the same shared
# workspace - which is how the multi-node runs behind the results page were made - where a
# single path means N writers opening the same file with mode='w' at once.
_RANK="${_RANK_RAW}"
if [ -z "$QUANTIZATION" ]; then
  _BASE="perf_${MODEL_REPO}"
else
  _BASE="perf_${MODEL_REPO}_${QUANTIZATION}"
fi
# Canonical name when MAD_COLLECT_METRICS says the workspaces are per-node (a _rankN file
# would never be collected there); rank suffix only for a possibly shared workspace.
# See .claude/skills/mad-slurm-multinode/references/gotchas.md, section jax_maxtext.
if [ -n "${MAD_COLLECT_METRICS:-}" ] || [ "$_RANK" = "0" ]; then
  PERF_LOG="$(pwd)/../${_BASE}.csv"
else
  PERF_LOG="$(pwd)/../${_BASE}_rank${_RANK}.csv"
fi
echo "Perf CSV for this rank: $PERF_LOG"
# Clear it HERE, before anything that can fail. The parser also removes it, but it only runs
# after training - so a missing config or a crashed run returns earlier and leaves a previous
# run's CSV sitting under the exact multiple_results name, ready to be collected as if it
# were this run's result.
if [ -e "$PERF_LOG" ]; then
  echo "removing stale $PERF_LOG from a previous run"
  rm -f "$PERF_LOG"
fi
perf_script="$(pwd)/jax-maxtext_benchmark_report.py"

# Run rocminfo and grep for "AMD Instinct"
DEVICE=$(/opt/rocm/bin/rocminfo | grep "AMD Instinct" | head -n1 | awk '{print $5}')
if [ -z "$DEVICE" ]; then
  ARCH=$(/opt/rocm/bin/rocminfo | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  case "$ARCH" in
    "gfx942") DEVICE="MI300X" ;;
    "gfx950") DEVICE="MI355X" ;;
    *) DEVICE="" ;;
  esac
fi             
echo "GPU DEVICE name: $DEVICE"

MAXTEXT="/workspace/maxtext"
MAXTEXT_DIR="/workspace/maxtext/src/maxtext"
ENV_SCRIPT_DIR="$(pwd)/env_scripts"

cd $MAXTEXT


execute_training(){
  gpu_architecture=$(rocminfo | grep -o -m 1 'gfx.*' | xargs )
  env_file=$ENV_SCRIPT_DIR/$1
  if test -e $ENV_SCRIPT_DIR/$gpu_architecture"_"$1; then
    env_file=$ENV_SCRIPT_DIR/$gpu_architecture"_"$1
  fi
  config_file=$ENV_SCRIPT_DIR/$2
  if test -e $ENV_SCRIPT_DIR/$gpu_architecture"_"$2; then
    config_file=$ENV_SCRIPT_DIR/$gpu_architecture"_"$2
  fi

  # Fail loudly on a missing env/config file. `source` on a nonexistent path only warns
  # and carries on, which would launch training with none of the tuning applied and no
  # obvious sign in the results - a slow run that looks merely disappointing.
  if [ ! -f "$env_file" ]; then
    echo "ERROR: env file not found: $env_file (gpu_architecture=$gpu_architecture)" >&2
    exit 1
  fi
  if [ ! -f "$config_file" ]; then
    echo "ERROR: config file not found: $config_file (gpu_architecture=$gpu_architecture)" >&2
    exit 1
  fi

  # output for logging
  echo "Using env file:"
  echo $env_file
  cat $env_file

  echo "Using yaml config file:"
  echo $config_file
  cat $config_file

  yaml() {
      python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
  }

  per_device_batch_size=$(yaml $config_file "['per_device_batch_size']")
  max_target_length=$(yaml $config_file "['max_target_length']")
  echo $per_device_batch_size
  echo $max_target_length

  # execute
  # Capture anything the caller set that the env file would otherwise clobber, then apply
  # the optional overrides centrally. Doing it here keeps all 17 model env scripts
  # untouched - the alternative was the same four lines duplicated into each of them.
  _caller_nccl_debug="${NCCL_DEBUG:-}"
  source $env_file
  [ -n "$_caller_nccl_debug" ] && export NCCL_DEBUG="$_caller_nccl_debug"
  if [ -n "${XLA_AUTOTUNE_LEVEL:-}" ]; then
    if ! [[ "$XLA_AUTOTUNE_LEVEL" =~ ^[0-9]+$ ]]; then
      echo "FATAL: XLA_AUTOTUNE_LEVEL='${XLA_AUTOTUNE_LEVEL}' is not a number" >&2
      return 1
    fi
    # Substitute if the flag is already there, append if it is not. Substitution alone was a
    # silent no-op on any env script that does not carry the flag - it printed the line below
    # and ran at the default level, which at 32 ranks is the difference between compiling and
    # not (see gotchas.md).
    if [[ "$XLA_FLAGS" == *--xla_gpu_autotune_level=* ]]; then
      # Safe to interpolate into sed: the value is digits only, checked just above.
      export XLA_FLAGS="$(echo "$XLA_FLAGS" | sed "s/--xla_gpu_autotune_level=[0-9]*/--xla_gpu_autotune_level=${XLA_AUTOTUNE_LEVEL}/")"
    else
      export XLA_FLAGS="$XLA_FLAGS --xla_gpu_autotune_level=${XLA_AUTOTUNE_LEVEL}"
    fi
    echo "XLA autotune level set to ${XLA_AUTOTUNE_LEVEL}; XLA_FLAGS=${XLA_FLAGS}"
  fi
  if [ -n "${XLA_EXTRA_FLAGS:-}" ]; then
    export XLA_FLAGS="$XLA_FLAGS ${XLA_EXTRA_FLAGS}"
    echo "XLA_FLAGS extended with: ${XLA_EXTRA_FLAGS}"
  fi
  # MAXTEXT_EXTRA_ARGS lets a manifest override config keys (e.g.
  # "per_device_batch_size=2 steps=30") without forking the YAML. The gfx950 YAMLs are
  # tuned for a single node; the same per-device batch can OOM once the run spans nodes.
  # Whatever is set here applies identically to every arm of an A/B, so it cannot bias a
  # comparison - but it does change absolute numbers, so it is echoed into the log.
  # read -a: must word-split, must not glob. Built once and used for both the metadata pass
  # and the argv, so the CSV cannot describe different arguments from the ones MaxText got.
  _extra_args=()
  [ -n "${MAXTEXT_EXTRA_ARGS:-}" ] && read -r -a _extra_args <<< "${MAXTEXT_EXTRA_ARGS}"
  # Rendezvous timeout is a MaxText config key (base.yml default 300), not an env var.
  # Appended only when the caller has not set it.
  if [ "${NNODES:-1}" -gt 1 ]; then
    case " ${MAXTEXT_EXTRA_ARGS:-} " in
      *jax_distributed_initialization_timeout=*) ;;
      *) _extra_args+=("jax_distributed_initialization_timeout=${JAX_RENDEZVOUS_TIMEOUT_S:-1800}") ;;
    esac
  fi
  # Keys that decide what the CSV row CLAIMS the run was are not overridable here. The row's
  # model and precision come from the card, not from the config, so an override of these
  # changes what was trained without changing what is reported: an 8B card passing
  # model_name=llama3.1-405b yields a perfectly well-formed row labelled 8B. A wrong number
  # in a results table is worse than a missing one, because nothing about it looks wrong.
  #
  # per_device_batch_size and max_target_length are deliberately NOT here - they are read
  # back from the overrides just below, so the row follows the run.
  for _kv in "${_extra_args[@]}"; do
    case "$_kv" in
      quantization=*|model_name=*|dtype=*|weight_dtype=*)
        echo "FATAL: MAXTEXT_EXTRA_ARGS must not set '${_kv%%=*}' ('$_kv'): the CSV takes model and" >&2
        echo "       precision from the model card, so this would be trained but not reported." >&2
        echo "       Use the card's --quantization, or add a card for the other model." >&2
        return 1 ;;
    esac
  done
  if [ -n "${MAXTEXT_EXTRA_ARGS:-}" ]; then
    echo "MaxText config overrides: ${MAXTEXT_EXTRA_ARGS}"
    # The CSV columns are read from the YAML above, but these overrides are what the run
    # actually used. Without re-deriving them the row reports the config's batch size for a
    # run made at a different one - a wrong number in a results table, which is worse than
    # a missing one because nothing about it looks wrong.
    for _kv in "${_extra_args[@]}"; do
      case "$_kv" in
        per_device_batch_size=*) per_device_batch_size="${_kv#*=}" ;;
        max_target_length=*)     max_target_length="${_kv#*=}" ;;
      esac
    done
    echo "reporting batch_size=${per_device_batch_size} seq_len=${max_target_length} (after overrides)"
  fi
  # PIPESTATUS[0], not $?: the training output is piped to tee, so $? is tee's status and
  # a crashed training run would look like a success all the way up to run.sh.
  # Only override quantization when the card actually declares one. Passing an empty
  # "quantization=" is not a no-op: MaxText parses it as None and it REPLACES the value in
  # the YAML. The 405B config sets quantization: "fp8" and its card passed no
  # --quantization, so the run trained in bf16 while every label said FP8.
  _quant_arg=()
  [ -n "$3" ] && _quant_arg=("quantization=$3")
  # Rank-specific log on the same condition as the CSV above: a shared workspace would
  # interleave step records from every rank into one file.
  _train_log="$2.log"
  if [ -z "${MAD_COLLECT_METRICS:-}" ] && [ "${NNODES:-1}" -gt 1 ]; then
    _train_log="$2_rank${_RANK}.log"
  fi
  echo "training log: $MAXTEXT/$_train_log"
  # Delete it before the run, not just rely on `tee` truncating it. If tee cannot OPEN the
  # file the old one survives untouched and stays perfectly readable, so the parser measures
  # the previous run and reports it as this one - the same stale-artifact trap the CSV is
  # cleared for. With the file gone, that failure has nothing to read.
  [ -e "$MAXTEXT/$_train_log" ] && { echo "removing stale $MAXTEXT/$_train_log"; rm -f "$MAXTEXT/$_train_log"; }
  # `tee`, not `tee -a`: a run emitting no step records leaves no boundary for the parser to
  # find, and the previous run's records would be read as this one's result.
# See .claude/skills/mad-slurm-multinode/references/gotchas.md, section jax_maxtext.
  python -m maxtext.trainers.pre_train.train "$config_file" \
  "${_quant_arg[@]}" "${_extra_args[@]}" |& tee "$_train_log"
  # The WHOLE array, captured immediately - the next command overwrites it. $? alone is
  # tee's status, so a crashed trainer would look like a success; but tee's own status
  # matters too, and in the opposite direction: a full disk or an unwritable path makes tee
  # fail while the trainer exits 0, leaving a truncated log that still parses.
  _pipe_rc=("${PIPESTATUS[@]}")
  train_rc="${_pipe_rc[0]}"
  tee_rc="${_pipe_rc[1]}"
  if [ "$train_rc" -ne 0 ]; then
    echo "ERROR: MaxText training exited $train_rc; not producing a measurement" >&2
    return "$train_rc"
  fi
  if [ "${tee_rc:-0}" -ne 0 ]; then
    echo "ERROR: tee exited ${tee_rc} writing $MAXTEXT/$_train_log; the log is incomplete or" >&2
    echo "       was never written, so any measurement taken from it would be wrong" >&2
    return "$tee_rc"
  fi
  if [ ! -s "$MAXTEXT/$_train_log" ]; then
    echo "ERROR: $MAXTEXT/$_train_log is missing or empty after a training run that exited 0" >&2
    return 1
  fi

  # Multi-node only: the median IS the measurement there, so the parser emits the scraped
  # line and fails without it. On a single node the card keeps its historical
  # "performance: 1 pass" contract - madengine scrapes the first match, so emitting the
  # median unconditionally would silently redefine what 19 existing cards report.
  # Rank 0 only: emitting from every rank yields one "measurement" per rank for one run.
  # The other side of the same split: a rank that is NOT the reporter must not fail the run
  # for lacking step metrics. MaxText logs them from process 0 only, and madengine already
  # exempts worker nodes from producing performance, so the parser's "no markers" refusal -
  # correct for a single-node card, where that rank is the only one - would otherwise turn
  # every worker into a failure.
  _emit_perf=""
  if [ "${NNODES}" -gt 1 ]; then
    if [ "${_RANK}" = "0" ]; then
      _emit_perf="--emit-performance-line"
    else
      _emit_perf="--worker-rank"
    fi
  fi
  if [ -z "$3" ]; then
    # No CLI quantization means the config's value trained; read it back rather than
    # assuming bf16 (gfx950_llama3.1_405b.yml sets fp8).
    _cfg_quant="$(yaml "$config_file" "['quantization']" 2>/dev/null || echo '')"
    [ -z "$_cfg_quant" ] || [ "$_cfg_quant" = "None" ] && _cfg_quant="bf16"
    echo "reporting precision=${_cfg_quant} (from $config_file; no --quantization given)"
    python3 $perf_script --model $MODEL_REPO --input "$MAXTEXT/$_train_log" --output $PERF_LOG --mode $MODE --quantization "$_cfg_quant" --batch_size $per_device_batch_size --seq_len $max_target_length --device $DEVICE --num_gpus $NUM_GPUS $_emit_perf
  else
    python3 $perf_script --model $MODEL_REPO --input "$MAXTEXT/$_train_log" --output $PERF_LOG --mode $MODE --quantization $3 --batch_size $per_device_batch_size --seq_len $max_target_length --device $DEVICE --num_gpus $NUM_GPUS $_emit_perf
  fi
  parse_rc=$?
  if [ "$parse_rc" -ne 0 ]; then
    echo "ERROR: perf parser exited $parse_rc" >&2
    return "$parse_rc"
  fi
  return 0

}


if [[ "$MODEL_REPO" == "Llama-2-7B" ]]; then
  echo "[INFO] LLAMA 2 7B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama2_7b_env.sh llama2_7b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Llama-2-70B" ]]; then
  echo "[INFO] LLAMA 2 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama2_70b_env.sh llama2_70b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Llama-3.1-8B" ]]; then
  echo "[INFO] LLAMA 3.1 8B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3_8b_env.sh llama3_8b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Llama-3.1-70B" ]]; then
  echo "[INFO] LLAMA 3.1 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3_70b_env.sh llama3_70b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Llama-3.1-405B" ]]; then
  echo "[INFO] LLAMA 3.1 405B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3.1_405b_env.sh llama3.1_405b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Llama-3.3-70B" ]]; then
  echo "[INFO] LLAMA 3.3 70B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training llama3.3_70b_env.sh llama3.3_70b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "DeepSeek-V2-lite" ]]; then
  echo "[INFO] DEEPSEEK V2 LITE TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training deepseek2_env_16b.sh deepseek2_16b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Mixtral-8x7B" ]]; then
  echo "[INFO] MIXTRAL-8x7B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training mixtral_8x7b_env.sh mixtral_8x7b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Qwen3-14B" ]]; then
  echo "[INFO] QWEN3-14B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training qwen3_14b_env.sh qwen3_14b.yml $QUANTIZATION || exit $?

elif [[ "$MODEL_REPO" == "Qwen3-30B-A3B" ]]; then
  echo "[INFO] QWEN3-30B-A3B TRAINING with following parameters"
  echo "  QUANTIZATION: $QUANTIZATION"
  execute_training qwen3_30b_a3b_env.sh qwen3_30b_a3b.yml $QUANTIZATION || exit $?

else
    echo "Error: Unsupported training mode."
    exit 1
fi
