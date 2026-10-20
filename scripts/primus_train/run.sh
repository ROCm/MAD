#!/usr/bin/env bash
# Wrapper for Primus training when run via madengine (local, SLURM, or K8s).
# Sets EXP from PRIMUS_CONFIG_PATH or --config_path, reads the suite
# (pretrain/posttrain) from the config, maps MAD GPU context to PRIMUS_GPU_MODEL
# so primus-cli loads runner/helpers/envs/<GPU_MODEL>.sh, then runs:
#   primus-cli direct -- train <suite> --config "$EXP"
# Performance env and per-config YAML env: are owned by Primus (not copied here).
# For HF-backed configs set HF_TOKEN or MAD_SECRETS_HFTOKEN
# (e.g. via additional_context.docker_env_vars in madengine v2).
# Primus root: set PRIMUS_ROOT to override; else auto-detect.
# After training, extracts tps/tflops/mfu from log and writes primus_perf_output.csv
# for madengine multiple_results.
set -e

# run_directory when invoked by madengine (cd run_directory && bash run.sh ...); used for output CSV
RUN_DIR="$(pwd)"

# Primus root resolution (local bind-mount, K8s ConfigMap extract, image ENV, legacy paths).
# v26.6 ships runner/primus-cli; v26.7+ (Primus #999) ships primus-cli at the repo root.
script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$script_dir/../Primus/runner/primus-cli" || -f "$script_dir/../Primus/primus-cli" ]]; then
  export PRIMUS_ROOT="$(cd "$script_dir/../Primus" && pwd)"
elif [[ -f "/workspace/Primus/runner/primus-cli" || -f "/workspace/Primus/primus-cli" ]]; then
  export PRIMUS_ROOT="/workspace/Primus"
elif [[ -n "${PRIMUS_ROOT:-}" ]]; then
  :
elif [[ -f "/opt/primus/runner/primus-cli" || -f "/opt/primus/primus-cli" ]]; then
  export PRIMUS_ROOT="/opt/primus"
elif [[ -f "/workspace/runner/primus-cli" || -f "/workspace/primus-cli" ]]; then
  export PRIMUS_ROOT="/workspace"
else
  echo "ERROR: Could not find Primus primus-cli. Set PRIMUS_ROOT or use a repo with scripts/Primus submodule." >&2
  exit 1
fi

# EXP: prefer PRIMUS_CONFIG_PATH (SLURM/K8s), else --config_path in args.
# --config_path and --variant (model_overrides.py) are wrapper-only flags that Primus' own
# CLI (primus/cli/main.py) does not recognize, so strip them (and their values) out of
# forward_args.
args=("$@")
forward_args=()
config_path_arg=""
variant=""
i=0
while [[ $i -lt ${#args[@]} ]]; do
  if [[ "${args[i]}" == "--config_path" && -n "${args[i+1]:-}" ]]; then
    config_path_arg="${args[i+1]}"
    i=$((i + 2))
    continue
  fi
  if [[ "${args[i]}" == "--variant" && -n "${args[i+1]:-}" ]]; then
    variant="${args[i+1]}"
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

# Suite selects `train pretrain` vs `train posttrain`. Read framework from the
# config rather than guessing from the directory name — examples/moe_package/
# configs declare framework: megatron. pre_trainer is the usual key; SFT/post-train
# configs declare framework under post_trainer.
framework_suite="$(cd "$PRIMUS_ROOT" && python3 -c '
import sys, yaml
mods = (yaml.safe_load(open(sys.argv[1])) or {}).get("modules") or {}
for key, suite in (("pre_trainer", "pretrain"), ("post_trainer", "posttrain")):
    fw = (mods.get(key) or {}).get("framework")
    if fw:
        print(fw, suite)
        break
' "$EXP" 2>/dev/null)"
framework="${framework_suite%% *}"
suite="${framework_suite##* }"

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

# Same fallback for the suite: post-train configs are named *_posttrain.yaml by convention
# (examples/megatron_bridge/configs/*/qwen3_32b_{sft,lora}_posttrain.yaml).
if [[ -z "$suite" ]]; then
  case "$(basename "$EXP" .yaml)" in
    *posttrain*) suite="posttrain" ;;
    *)           suite="pretrain" ;;
  esac
fi

# GPU model for primus-env.sh
# Map MAD GPU context to PRIMUS_GPU_MODEL so primus-cli sources
# runner/helpers/envs/<GPU_MODEL>.sh inside Docker (rocm-smi is often missing).
# An explicit PRIMUS_GPU_MODEL (docker_env_vars / shell) still wins.
if [[ -z "${PRIMUS_GPU_MODEL:-}" ]]; then
  gpu_name="${MAD_SYSTEM_GPU_PRODUCT_NAME:-}"
  arch="${MAD_SYSTEM_GPU_ARCHITECTURE:-}"
  case "$gpu_name" in
    *MI455*) PRIMUS_GPU_MODEL=MI455X ;;
    *MI325*) PRIMUS_GPU_MODEL=MI325X ;;
    *MI355*) PRIMUS_GPU_MODEL=MI355X ;;
    *MI350*) PRIMUS_GPU_MODEL=MI350X ;;
    *MI300*) PRIMUS_GPU_MODEL=MI300X ;;
    *)
      case "$arch" in
        gfx942*) PRIMUS_GPU_MODEL=MI300X ;;
        gfx950*) PRIMUS_GPU_MODEL=MI355X ;;
        gfx1250*) PRIMUS_GPU_MODEL=MI455X ;;
      esac
      ;;
  esac
  [[ -n "${PRIMUS_GPU_MODEL:-}" ]] && export PRIMUS_GPU_MODEL
fi
# end GPU model for primus-env.sh

echo "[primus_train] suite=$suite framework=$framework" \
     "gpu_model=${PRIMUS_GPU_MODEL:-<unset>}" \
     "arch=${MAD_SYSTEM_GPU_ARCHITECTURE:-unknown}" \
     "gpu=${MAD_SYSTEM_GPU_PRODUCT_NAME:-unknown}"

# HF_TOKEN for Primus prepare (HF-backed configs): use MAD_SECRETS_HFTOKEN from madengine v2
# (set via additional_context.docker_env_vars) if HF_TOKEN not already set
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
elif [[ -n "${MAD_SECRETS_HFTOKEN:-}" ]]; then
  export HF_TOKEN="$MAD_SECRETS_HFTOKEN"
fi

# Deployment knobs, set through context.docker_env_vars. Unset means whatever the config
# ships. They go after every other override because Primus builds the override map as a
# dict, so the last occurrence of a key wins.
knob_args=()
if [[ "$framework" == "megatron" ]]; then
  if [[ -n "${PRIMUS_TRAIN_ITERS:-}" ]]; then
    knob_args+=(--train_iters "$PRIMUS_TRAIN_ITERS")
  fi
  # A perf gate wants steady iterations, not a trace: the profiler window costs about 3x
  # a normal iteration and leaves the next ones elevated while the trace is written.
  if [[ "${PRIMUS_DISABLE_PROFILE:-0}" == "1" ]]; then
    knob_args+=(--profile false --use_pytorch_profiler false)
  fi
elif [[ -n "${PRIMUS_TRAIN_ITERS:-}" || "${PRIMUS_DISABLE_PROFILE:-0}" == "1" ]]; then
  echo "[primus_train] WARNING: PRIMUS_TRAIN_ITERS/PRIMUS_DISABLE_PROFILE apply to megatron" \
       "configs only; framework=$framework ignores them" >&2
fi
# Whitespace-separated Primus overrides for any framework, e.g. a parallel layout that has
# to change several keys together ("--pipeline_model_parallel_size 4 --use_torch_fsdp2 false").
if [[ -n "${PRIMUS_EXTRA_ARGS:-}" ]]; then
  read -r -a primus_extra_args <<< "$PRIMUS_EXTRA_ARGS"
  knob_args+=("${primus_extra_args[@]}")
fi
if [[ ${#knob_args[@]} -gt 0 ]]; then
  echo "[primus_train] Primus CLI knobs from environment: ${knob_args[*]}"
fi

# Give each rank its own RCCL log when the deployment points RCCL_LOG_DIR at a shared mount:
# the ranks stop overwriting each other in the shared stdout, and every rank is logged
# rather than the few that torchrun's --local-ranks-filter lets through.
# A directory RCCL cannot write to is worse than none: RCCL drops the output instead of
# falling back to stdout. The container writes as root squashed to nobody on NFS, so the
# directory has to be world-writable; when it is not, say so and keep the logs in stdout.
if [[ -n "${RCCL_LOG_DIR:-}" ]]; then
  if mkdir -p "$RCCL_LOG_DIR" 2>/dev/null && [[ -w "$RCCL_LOG_DIR" ]]; then
    export NCCL_DEBUG_FILE="$RCCL_LOG_DIR/$(basename "$EXP" .yaml).node_${NODE_RANK:-0}.%h.%p.log"
    echo "[primus_train] RCCL logs go to $NCCL_DEBUG_FILE (one file per rank)"
  else
    unset NCCL_DEBUG_FILE
    echo "[primus_train] WARNING: RCCL_LOG_DIR=$RCCL_LOG_DIR is not writable from the container" \
         "(chmod 0777 it on the host); RCCL keeps logging to stdout" >&2
  fi
fi

# Redirect Primus output/outputs to run_directory (workspace root when run via madengine).
# --log_file keeps the log where the perf extractor below expects it.
mkdir -p "$RUN_DIR/output" "$RUN_DIR/outputs"
export TRAIN_LOG="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"
export DUMP_HLO_DIR="${DUMP_HLO_DIR:-$RUN_DIR/output/xla_dump_hlo}"

# Run from PRIMUS_ROOT so EXP path (e.g. examples/torchtitan/configs/...) resolves.
# Do not use exec so we can run the perf extractor after training.
# Temporarily disable -e: a non-zero exit from training would skip perf extraction.
if [[ -f "$PRIMUS_ROOT/runner/primus-cli" ]]; then
  PRIMUS_CLI="$PRIMUS_ROOT/runner/primus-cli"
elif [[ -f "$PRIMUS_ROOT/primus-cli" ]]; then
  PRIMUS_CLI="$PRIMUS_ROOT/primus-cli"
else
  echo "ERROR: Could not find primus-cli under $PRIMUS_ROOT." >&2
  exit 1
fi
set +e
cd "$PRIMUS_ROOT"
cli_cmd=(
  bash "$PRIMUS_CLI" direct --log_file "$TRAIN_LOG" -- \
    train "$suite" --config "$EXP"
)
# --job.dump_folder is a Torchtitan flag; other backends reject it.
if [[ "$framework" == "torchtitan" ]]; then
  cli_cmd+=(--job.dump_folder "$RUN_DIR/outputs")
fi
world_size=$(( ${NNODES:-1} * ${GPUS_PER_NODE:-${NPROC_PER_NODE:-8}} ))
# Per-model overrides the config YAML does not carry (model_overrides.py). They come first
# so the caller's arguments and PRIMUS_EXTRA_ARGS can still replace any of them.
model_args=()
if [[ "$framework" == "megatron" ]]; then
  overrides_script="${script_dir}/model_overrides.py"
  [[ -f "$RUN_DIR/model_overrides.py" ]] && overrides_script="$RUN_DIR/model_overrides.py"
  if ! overrides_out="$(python3 "$overrides_script" "$EXP" --device "${PRIMUS_GPU_MODEL:-}" \
                          --world-size "$world_size" --variant "$variant")"; then
    echo "ERROR: $overrides_script failed for $EXP" >&2
    exit 1
  fi
  [[ -n "$overrides_out" ]] && mapfile -t model_args <<< "$overrides_out"
  if [[ -n "$variant" && ${#model_args[@]} -eq 0 ]]; then
    echo "ERROR: no '$variant' variant of $(basename "$EXP" .yaml) for ${PRIMUS_GPU_MODEL:-<unset>}" >&2
    exit 1
  fi
  if [[ ${#model_args[@]} -gt 0 ]]; then
    echo "[primus_train] model overrides${variant:+ ($variant)}: ${model_args[*]}"
  fi
elif [[ -n "$variant" ]]; then
  echo "ERROR: --variant applies to megatron configs only; framework=$framework" >&2
  exit 1
fi
cli_cmd+=("${model_args[@]}")
# Megatron rejects a global batch size that is not a multiple of micro batch size x
# data-parallel size, and the configs are tuned for one 8-GPU node. An explicit
# --global_batch_size from the caller or PRIMUS_EXTRA_ARGS wins; one from the model
# overrides is renormalized like the YAML value.
caller_args=("${forward_args[@]}" "${knob_args[@]}")
if [[ "$framework" == "megatron" ]] && ! printf '%s\n' "${caller_args[@]}" | grep -qx -- '--global_batch_size'; then
  scaleout_script="${script_dir}/scaleout_batch.py"
  [[ -f "$RUN_DIR/scaleout_batch.py" ]] && scaleout_script="$RUN_DIR/scaleout_batch.py"
  scaleout_gbs="$(python3 "$scaleout_script" "$EXP" --world-size "$world_size" \
                    -- "${model_args[@]}" "${caller_args[@]}" || true)"
  if [[ -n "$scaleout_gbs" ]]; then
    echo "[primus_train] global_batch_size -> $scaleout_gbs for world size $world_size"
    cli_cmd+=(--global_batch_size "$scaleout_gbs")
  fi
fi
cli_cmd+=("${caller_args[@]}")
"${cli_cmd[@]}"
exitcode=$?
set -e
# Extract tps/tflops/mfu and the run shape from the training log into primus_perf_output.csv
# (one row per metric; see extract_primus_perf.py for the columns)
LOG_PATH="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"
if [[ -f "$LOG_PATH" ]]; then
  extract_script="${script_dir}/extract_primus_perf.py"
  [[ -f "$RUN_DIR/extract_primus_perf.py" ]] && extract_script="$RUN_DIR/extract_primus_perf.py"
  extract_args=()
  # Megatron-Bridge omits printed TPS on pretrain and posttrain. Derive it
  # from elapsed ms + global batch size when seq_length/world_size are supplied.
  # seq_length is under pre_trainer (pretrain) or post_trainer (SFT/LoRA).
  # Values like ${PRIMUS_SEQ_LENGTH:2048} resolve to the env var or the default.
  seq_length="$(cd "$PRIMUS_ROOT" && python3 -c '
import os, re, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
mods = cfg.get("modules") or {}
sl = None
for key in ("pre_trainer", "post_trainer"):
    ov = (mods.get(key) or {}).get("overrides") or {}
    if "seq_length" in ov:
        sl = ov["seq_length"]
        break
if sl is None:
    raise SystemExit
if isinstance(sl, str):
    m = re.fullmatch(r"\$\{([^:}]+)(?::([^}]*))?\}", sl.strip())
    if m:
        sl = os.environ.get(m.group(1), m.group(2) or "")
print("" if sl is None or sl == "" else sl)
' "$EXP" 2>/dev/null || true)"
  vis="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -n "$vis" ]]; then
    num_gpus="$(awk -F',' '{print NF}' <<< "$vis")"
  else
    num_gpus=8
  fi
  [[ -n "$seq_length" ]] && extract_args+=(--seq-length "$seq_length")
  extract_args+=(--num-gpus "$num_gpus" --mode "$suite" --device "${PRIMUS_GPU_MODEL:-}")
  set +e
  python3 "$extract_script" "$LOG_PATH" "$RUN_DIR/primus_perf_output.csv" "${extract_args[@]}"
  extractcode=$?
  set -e
  # Megatron prints throughput on the last global rank only, so on a multi-node run
  # only the last node's log carries it; the other nodes have nothing to extract.
  if [[ "$extractcode" -ne 0 && "$exitcode" -eq 0 ]]; then
    if (( ${NODE_RANK:-0} == ${NNODES:-1} - 1 )); then
      echo "[primus_train] training log present but TPS extraction failed" >&2
      exitcode="$extractcode"
    else
      echo "[primus_train] no TPS on node ${NODE_RANK:-0} of ${NNODES:-1}; the last node reports it"
    fi
  fi
fi
exit "$exitcode"
