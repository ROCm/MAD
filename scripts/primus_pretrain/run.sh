#!/usr/bin/env bash
# Wrapper for Primus pretrain when run via madengine (local, SLURM, or K8s).
# Sets EXP from PRIMUS_CONFIG_PATH or --config_path, infers BACKEND from path,
# then runs Primus examples/run_pretrain.sh. For HF-backed configs set HF_TOKEN
# or MAD_SECRET_HFTOKEN (e.g. via additional_context.docker_env_vars in madengine v2).
# Primus root: set PRIMUS_ROOT to override; else auto-detect.
# After training, extracts tps/tflops/mfu from log and writes primus_perf_output.csv for madengine multiple_results.
set -e

# run_directory when invoked by madengine (cd run_directory && bash run.sh ...); used for output CSV
RUN_DIR="$(pwd)"

# Primus root resolution (local bind-mount, K8s ConfigMap extract, image ENV, legacy paths):
# 1) Repo submodule scripts/Primus (local Docker / SLURM with project layout)
# 2) /workspace/Primus — Dockerfile COPY and madengine K8s init (keys Primus/examples/…)
# 3) PRIMUS_ROOT from environment (image default)
# 4) Legacy /opt/primus images
script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$script_dir/../Primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="$(cd "$script_dir/../Primus" && pwd)"
elif [[ -f "/workspace/Primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/workspace/Primus"
elif [[ -n "${PRIMUS_ROOT:-}" ]]; then
  :
elif [[ -f "/opt/primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/opt/primus"
elif [[ -f "/workspace/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/workspace"
else
  echo "ERROR: Could not find Primus run_pretrain.sh. Set PRIMUS_ROOT or use a repo with scripts/Primus submodule." >&2
  exit 1
fi

# EXP (required by Primus run_pretrain.sh): prefer PRIMUS_CONFIG_PATH (SLURM/K8s), else --config_path in args
if [[ -n "${PRIMUS_CONFIG_PATH:-}" ]]; then
  export EXP="$PRIMUS_CONFIG_PATH"
else
  export EXP="examples/megatron/exp_pretrain.yaml"
  args=("$@")
  for i in "${!args[@]}"; do
    if [[ "${args[i]}" == "--config_path" && -n "${args[i+1]:-}" ]]; then
      export EXP="${args[i+1]}"
      break
    fi
  done
fi

# Infer BACKEND from EXP path so run_pretrain.sh uses correct runner (torchtitan, megatron, maxtext, etc.)
# Primus expects BACKEND=MaxText for Jax/MaxText; lowercase for others.
exp_lower="$(echo "$EXP" | tr '[:upper:]' '[:lower:]')"
if [[ "$exp_lower" == *"/maxtext/"* ]]; then
  export BACKEND="MaxText"
elif [[ "$exp_lower" == *"/torchtitan/"* ]]; then
  export BACKEND="torchtitan"
elif [[ "$exp_lower" == *"/megatron_bridge/"* ]]; then
  export BACKEND="megatron_bridge"
elif [[ "$exp_lower" == *"/moe_package/"* ]]; then
  export BACKEND="moe_package"
else
  export BACKEND="megatron"
fi

# HF_TOKEN for Primus prepare (HF-backed configs): use MAD_SECRET_HFTOKEN from madengine v2
# (set via additional_context.docker_env_vars) if HF_TOKEN not already set
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
elif [[ -n "${MAD_SECRET_HFTOKEN:-}" ]]; then
  export HF_TOKEN="$MAD_SECRET_HFTOKEN"
fi

# Redirect Primus output/outputs to run_directory (workspace root when run via madengine).
# No changes to Primus: we set env vars that run_pretrain.sh already honors (TRAIN_LOG, DUMP_HLO_DIR)
# and pass --job.dump_folder so Torchtitan writes checkpoints here. output/ = logs; outputs/ = checkpoints.
mkdir -p "$RUN_DIR/output" "$RUN_DIR/outputs"
export TRAIN_LOG="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"
export DUMP_HLO_DIR="${DUMP_HLO_DIR:-$RUN_DIR/output/xla_dump_hlo}"

# Run from PRIMUS_ROOT so EXP path (e.g. examples/torchtitan/configs/...) resolves correctly.
# Do not use exec so we can run the perf extractor after training for madengine multiple_results.
# Pass --job.dump_folder so Torchtitan writes checkpoints to RUN_DIR/outputs (not scripts/Primus/outputs).
cd "$PRIMUS_ROOT" && bash "$PRIMUS_ROOT/examples/run_pretrain.sh" "$@" --job.dump_folder "$RUN_DIR/outputs"
exitcode=$?
# Extract tps/tflops/mfu from training log into primus_perf_output.csv (one row: model, performance, metric, tflops, model_flops_utilization)
LOG_PATH="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"
if [[ -f "$LOG_PATH" ]]; then
  python3 "$RUN_DIR/extract_primus_perf.py" "$LOG_PATH" "$RUN_DIR/primus_perf_output.csv" || true
fi
exit "$exitcode"
