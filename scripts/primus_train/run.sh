#!/usr/bin/env bash
# Wrapper for Primus pretrain when run via madengine (local, SLURM, or K8s).
# Sets EXP from PRIMUS_CONFIG_PATH or --config_path, infers BACKEND from path,
# then runs Primus examples/run_pretrain.sh. For HF-backed configs set HF_TOKEN
# or MAD_SECRETS_HFTOKEN (e.g. via additional_context.docker_env_vars in madengine v2).
# Primus root: set PRIMUS_ROOT to override; else auto-detect.
# After training, extracts tps/tflops/mfu from log and writes primus_perf_output.csv for madengine multiple_results.
set -e

# run_directory when invoked by madengine (cd run_directory && bash run.sh ...); used for output CSV
RUN_DIR="$(pwd)"

# Primus root resolution (local bind-mount, K8s ConfigMap extract, image ENV, legacy paths):
# 1) Repo submodule scripts/Primus (sibling of scripts/primus_train)
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

# EXP (required by Primus run_pretrain.sh): prefer PRIMUS_CONFIG_PATH (SLURM/K8s), else --config_path in args.
# --config_path is a wrapper-only flag that Primus' own CLI (primus/cli/main.py) does not
# recognize, so strip it (and its value) out of forward_args, which is what actually gets
# passed to run_pretrain.sh below instead of the raw "$@".
args=("$@")
forward_args=()
config_path_arg=""
i=0
while [[ $i -lt ${#args[@]} ]]; do
  if [[ "${args[i]}" == "--config_path" && -n "${args[i+1]:-}" ]]; then
    config_path_arg="${args[i+1]}"
    i=$((i + 2))
    continue
  fi
  forward_args+=("${args[i]}")
  i=$((i + 1))
done

if [[ -n "${PRIMUS_CONFIG_PATH:-}" ]]; then
  export EXP="$PRIMUS_CONFIG_PATH"
elif [[ -n "$config_path_arg" ]]; then
  export EXP="$config_path_arg"
else
  export EXP="examples/megatron/exp_pretrain.yaml"
fi

# BACKEND selects the runner (megatron, torchtitan, maxtext, ...) in run_pretrain.sh.
# Primus validates it against modules.pre_trainer.framework in the EXP and aborts on
# mismatch (examples/scripts/prepare_experiment.py), so read the framework from the config
# rather than guessing from the directory name — that also picks up launchers added in
# later Primus releases (maxdiffusion, nemo_automodel in v26.6) with no change here.
# Note the directory name is not always the framework: examples/moe_package/ configs
# declare framework: megatron.
# pre_trainer is the usual key; SFT/post-train configs declare framework under post_trainer.
framework="$(cd "$PRIMUS_ROOT" && python3 -c '
import sys, yaml
mods = (yaml.safe_load(open(sys.argv[1])) or {}).get("modules") or {}
for key in ("pre_trainer", "post_trainer"):
    fw = (mods.get(key) or {}).get("framework")
    if fw:
        print(fw)
        break
' "$EXP" 2>/dev/null)"

# Fallback for configs we cannot parse (missing PyYAML, non-standard layout): infer from the
# launcher directory, i.e. the component after examples/ in examples/<launcher>/configs/...
# Match on that component only: a plain substring test would mislabel
# examples/megatron/configs/<arch>/diffusion/*.yaml, which are framework: megatron.
if [[ -z "$framework" ]]; then
  exp_lower="$(echo "$EXP" | tr '[:upper:]' '[:lower:]')"
  launcher="${exp_lower##*examples/}"
  launcher="${launcher%%/*}"
  case "$launcher" in
    maxtext|maxdiffusion|torchtitan|megatron_bridge|nemo_automodel|diffusion|hummingbirdxt)
      framework="$launcher" ;;
    *)
      # megatron, moe_package (framework: megatron), and anything unrecognized
      framework="megatron" ;;
  esac
fi

# prepare_experiment.py compares case-insensitively, but run_pretrain.sh string-matches the
# literals "MaxText" and "MaxDiffusion" to skip LD_LIBRARY_PATH injection and
# HSA_NO_SCRATCH_RECLAIM for the JAX backends, so those two need exact casing.
case "$framework" in
  maxtext)      export BACKEND="MaxText" ;;
  maxdiffusion) export BACKEND="MaxDiffusion" ;;
  *)            export BACKEND="$framework" ;;
esac

# HF_TOKEN for Primus prepare (HF-backed configs): use MAD_SECRETS_HFTOKEN from madengine v2
# (set via additional_context.docker_env_vars) if HF_TOKEN not already set
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
elif [[ -n "${MAD_SECRETS_HFTOKEN:-}" ]]; then
  export HF_TOKEN="$MAD_SECRETS_HFTOKEN"
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
# Temporarily disable -e: with it on, a non-zero exit from training would abort this script
# immediately (via the && chain below) and skip perf extraction entirely.
set +e
cd "$PRIMUS_ROOT" && bash "$PRIMUS_ROOT/examples/run_pretrain.sh" "${forward_args[@]}" --job.dump_folder "$RUN_DIR/outputs"
exitcode=$?
set -e
# Extract tps/tflops/mfu from training log into primus_perf_output.csv (one row: model, performance, metric, tflops, model_flops_utilization)
LOG_PATH="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"
if [[ -f "$LOG_PATH" ]]; then
  extract_script="${script_dir}/extract_primus_perf.py"
  [[ -f "$RUN_DIR/extract_primus_perf.py" ]] && extract_script="$RUN_DIR/extract_primus_perf.py"
  python3 "$extract_script" "$LOG_PATH" "$RUN_DIR/primus_perf_output.csv" || true
fi
exit "$exitcode"
