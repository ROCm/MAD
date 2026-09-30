#!/usr/bin/env bash
# Regression: MAD primus_train must not copy Primus performance env.
# Env lives in Primus runner/helpers/envs/ and YAML env:; run.sh only maps
# MAD GPU context to PRIMUS_GPU_MODEL and launches primus-cli.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/../run.sh"
PRIMUS_ROOT="$(cd "$SCRIPT_DIR/../../Primus" && pwd)"

assert_eq() {
  local expected="$1"
  local actual="$2"
  local variable="$3"

  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $variable=$expected, got $actual" >&2
    return 1
  fi
}

assert_file_contains() {
  local path="$1"
  local pattern="$2"
  if ! grep -qE -- "$pattern" "$path"; then
    echo "Expected $path to match /$pattern/" >&2
    return 1
  fi
}

assert_file_not_contains() {
  local path="$1"
  local pattern="$2"
  if grep -qE -- "$pattern" "$path"; then
    echo "Expected $path not to match /$pattern/" >&2
    return 1
  fi
}

gpu_model_block() {
  awk '
    /^# GPU model for primus-env.sh$/ { capture = 1 }
    /^# end GPU model for primus-env.sh$/ { print; exit }
    capture { print }
  ' "$RUN_SH"
}

eval_gpu_model() (
  local arch="$1"
  local gpu_name="$2"
  unset PRIMUS_GPU_MODEL
  export MAD_SYSTEM_GPU_ARCHITECTURE="$arch"
  export MAD_SYSTEM_GPU_PRODUCT_NAME="$gpu_name"
  eval "$(gpu_model_block)"
  printf '%s\n' "${PRIMUS_GPU_MODEL-<unset>}"
)

check_run_sh_does_not_copy_env() {
  assert_file_not_contains "$RUN_SH" 'export[[:space:]]+NVTE_CK_IS_V3_ATOMIC_FP32'
  assert_file_not_contains "$RUN_SH" 'export[[:space:]]+NCCL_PXN_DISABLE'
  assert_file_not_contains "$RUN_SH" 'export[[:space:]]+HSA_NO_SCRATCH_RECLAIM'
  assert_file_not_contains "$RUN_SH" 'export[[:space:]]+RCCL_WARP_SPEED_AUTO'
  assert_file_not_contains "$RUN_SH" 'export[[:space:]]+NVTE_USE_CAST_TRANSPOSE_TRITON'
  assert_file_not_contains "$RUN_SH" 'examples/run_pretrain.sh'
}

check_run_sh_always_primus_cli() {
  assert_file_contains "$RUN_SH" 'primus-cli'
  # A single launch path: train "$suite" covers pretrain and posttrain,
  # including megatron_bridge pretrain.
  assert_file_contains "$RUN_SH" 'train "\$suite" --config'
  if grep -q 'examples/run_pretrain.sh' "$RUN_SH"; then
    echo "run.sh must not invoke examples/run_pretrain.sh" >&2
    return 1
  fi
}

check_run_sh_extracts_bridge_pretrain() {
  # Megatron-Bridge pretrain (mamba) omits tokens/s/GPU. seq_length lives under
  # pre_trainer, not only post_trainer; --num-gpus is required to derive TPS.
  assert_file_contains "$RUN_SH" '\("pre_trainer", "post_trainer"\)'
  assert_file_contains "$RUN_SH" '--num-gpus'
}

check_dockerfile_requires_primus_cli() {
  local dockerfile="$SCRIPT_DIR/../../../docker/primus.ubuntu.amd.Dockerfile"
  if ! grep -qE -- '/workspace/Primus/(runner/)?primus-cli' "$dockerfile"; then
    echo "Expected $dockerfile to require primus-cli" >&2
    return 1
  fi
  assert_file_not_contains "$dockerfile" 'examples/run_pretrain.sh'
}

check_primus_gpu_env_files() {
  # gfx942 atomics stay at Primus base_env.sh defaults (Primus #1178 leaves
  # MI300X.sh / MI325X.sh unchanged), so only the gfx950 WarpSpeed override is asserted.
  assert_file_contains "$PRIMUS_ROOT/runner/helpers/envs/MI355X.sh" \
    'RCCL_WARP_SPEED_AUTO=\$\{RCCL_WARP_SPEED_AUTO:-0\}'
  assert_file_contains "$PRIMUS_ROOT/runner/helpers/envs/MI350X.sh" \
    'RCCL_WARP_SPEED_AUTO=\$\{RCCL_WARP_SPEED_AUTO:-0\}'
}

yaml_env_has_cast_transpose_zero() {
  local yaml="$1"
  python3 -c '
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
env = cfg.get("env") or {}
val = str(env.get("NVTE_USE_CAST_TRANSPOSE_TRITON", ""))
if val != "0":
    raise SystemExit(f"{sys.argv[1]}: expected env.NVTE_USE_CAST_TRANSPOSE_TRITON=0, got {val!r}")
' "$yaml"
}

check_mxfp4_yaml_env() {
  # Primus #1179 bakes NVTE_USE_CAST_TRANSPOSE_TRITON=0 into the MXFP4 YAML env: blocks.
  local llama_yaml="$PRIMUS_ROOT/examples/megatron/configs/MI355X/llama3.1_8B-MXFP4-pretrain.yaml"
  local flux_yaml="$PRIMUS_ROOT/examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_te_spec_mxfp4.yaml"
  python3 -c 'import yaml' 2>/dev/null || return 0
  yaml_env_has_cast_transpose_zero "$llama_yaml"
  yaml_env_has_cast_transpose_zero "$flux_yaml"
}

check_gpu_model_mapping() {
  assert_eq "MI300X" "$(eval_gpu_model gfx942 "AMD Instinct MI300X")" "PRIMUS_GPU_MODEL"
  assert_eq "MI325X" "$(eval_gpu_model gfx942 "AMD Instinct MI325X")" "PRIMUS_GPU_MODEL"
  assert_eq "MI355X" "$(eval_gpu_model gfx950 "AMD Instinct MI355X")" "PRIMUS_GPU_MODEL"
  assert_eq "MI350X" "$(eval_gpu_model gfx950 "AMD Instinct MI350X")" "PRIMUS_GPU_MODEL"
  assert_eq "MI300X" "$(eval_gpu_model gfx942 "")" "PRIMUS_GPU_MODEL"
  assert_eq "MI355X" "$(eval_gpu_model gfx950 "")" "PRIMUS_GPU_MODEL"
}

check_explicit_gpu_model_override() (
  export PRIMUS_GPU_MODEL="keep-me"
  export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
  export MAD_SYSTEM_GPU_PRODUCT_NAME="AMD Instinct MI355X"
  eval "$(gpu_model_block)"
  assert_eq "keep-me" "$PRIMUS_GPU_MODEL" "PRIMUS_GPU_MODEL"
)

check_fla_pretrain_hook() {
  # Primus #1181: FLA Triton autotune patch on the primus-cli pretrain path.
  local hook="$PRIMUS_ROOT/runner/helpers/hooks/train/pretrain/z_patch_fla_triton_autotune.sh"
  assert_file_contains "$hook" 'patch_fla_triton_autotune_hang.sh'
  assert_file_contains "$hook" 'hylo_\*|kda_\*|gdn_\*'
}

check_run_sh_does_not_copy_env
check_run_sh_always_primus_cli
check_run_sh_extracts_bridge_pretrain
check_dockerfile_requires_primus_cli
check_primus_gpu_env_files
check_mxfp4_yaml_env
check_gpu_model_mapping
check_explicit_gpu_model_override
check_fla_pretrain_hook
echo "primus_train uses primus-cli; Primus GPU env files and MXFP4 YAML env: are the source of truth."
