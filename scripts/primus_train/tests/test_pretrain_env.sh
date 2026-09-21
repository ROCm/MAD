#!/usr/bin/env bash
# Regression test: MAD primus_train pretrain env must match primus-cli
# (runner/helpers/envs/base_env.sh + MI355X.sh). ROCM-31034.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/../run.sh"

performance_block() {
  awk '
    /^# Architecture-aware performance environment$/ { capture = 1 }
    /^echo "\[primus_train\] suite=/ { exit }
    capture { print }
  ' "$RUN_SH"
}

assert_eq() {
  local expected="$1"
  local actual="$2"
  local variable="$3"

  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $variable=$expected, got $actual" >&2
    return 1
  fi
}

eval_env() (
  local arch="$1"
  local gpu_name="$2"
  local backend="${3:-megatron}"
  local exp="${4:-examples/megatron/configs/MI355X/gdn_1B_BF16-pretrain.yaml}"
  export MAD_SYSTEM_GPU_ARCHITECTURE="$arch"
  export MAD_SYSTEM_GPU_PRODUCT_NAME="$gpu_name"
  export BACKEND="$backend" EXP="$exp"
  unset NCCL_PXN_DISABLE RCCL_WARP_SPEED_AUTO HSA_NO_SCRATCH_RECLAIM
  unset NVTE_CK_IS_V3_ATOMIC_FP32 PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32
  unset NVTE_USE_CAST_TRANSPOSE_TRITON
  eval "$(performance_block)"
  printf '%s\n' \
    "NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE-<unset>}" \
    "RCCL_WARP_SPEED_AUTO=${RCCL_WARP_SPEED_AUTO-<unset>}" \
    "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM-<unset>}" \
    "NVTE_CK_IS_V3_ATOMIC_FP32=${NVTE_CK_IS_V3_ATOMIC_FP32-<unset>}" \
    "NVTE_USE_CAST_TRANSPOSE_TRITON=${NVTE_USE_CAST_TRANSPOSE_TRITON-<unset>}"
)

check_cli_parity_mi355x() {
  mapfile -t lines < <(eval_env gfx950 "AMD Instinct MI355X")
  declare -A got=()
  for line in "${lines[@]}"; do
    got["${line%%=*}"]="${line#*=}"
  done
  assert_eq "1" "${got[NCCL_PXN_DISABLE]}" "NCCL_PXN_DISABLE"
  assert_eq "0" "${got[RCCL_WARP_SPEED_AUTO]}" "RCCL_WARP_SPEED_AUTO"
  assert_eq "1" "${got[HSA_NO_SCRATCH_RECLAIM]}" "HSA_NO_SCRATCH_RECLAIM"
}

check_cli_parity_gfx950_mi350x() {
  mapfile -t lines < <(eval_env gfx950 "AMD Instinct MI350X")
  declare -A got=()
  for line in "${lines[@]}"; do
    got["${line%%=*}"]="${line#*=}"
  done
  assert_eq "1" "${got[NCCL_PXN_DISABLE]}" "NCCL_PXN_DISABLE"
  assert_eq "0" "${got[RCCL_WARP_SPEED_AUTO]}" "RCCL_WARP_SPEED_AUTO"
}

check_gfx942_atomics() {
  mapfile -t lines < <(eval_env gfx942 "AMD Instinct MI300X")
  declare -A got=()
  for line in "${lines[@]}"; do
    got["${line%%=*}"]="${line#*=}"
  done
  assert_eq "1" "${got[NVTE_CK_IS_V3_ATOMIC_FP32]}" "NVTE_CK_IS_V3_ATOMIC_FP32"
  assert_eq "<unset>" "${got[RCCL_WARP_SPEED_AUTO]}" "RCCL_WARP_SPEED_AUTO"
}

check_mxfp4_override() {
  mapfile -t lines < <(eval_env gfx950 "AMD Instinct MI355X" megatron \
    "examples/megatron/configs/MI355X/llama3.1_8B-MXFP4-pretrain.yaml")
  declare -A got=()
  for line in "${lines[@]}"; do
    got["${line%%=*}"]="${line#*=}"
  done
  assert_eq "0" "${got[NVTE_USE_CAST_TRANSPOSE_TRITON]}" \
    "NVTE_USE_CAST_TRANSPOSE_TRITON"
}

# Image / base_env.sh default NVTE_USE_CAST_TRANSPOSE_TRITON=1. The published
# MI355X MXFP4 command prefixes =0; ${VAR:-0} would keep the pre-set 1.
check_mxfp4_overrides_preset_one() (
  export NVTE_USE_CAST_TRANSPOSE_TRITON=1
  export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
  export MAD_SYSTEM_GPU_PRODUCT_NAME="AMD Instinct MI355X"
  export BACKEND=megatron
  export EXP="examples/megatron/configs/MI355X/llama3.1_8B-MXFP4-pretrain.yaml"
  eval "$(performance_block)"
  assert_eq "0" "$NVTE_USE_CAST_TRANSPOSE_TRITON" \
    "NVTE_USE_CAST_TRANSPOSE_TRITON"
)

check_non_mxfp4_keeps_preset_one() (
  export NVTE_USE_CAST_TRANSPOSE_TRITON=1
  export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
  export MAD_SYSTEM_GPU_PRODUCT_NAME="AMD Instinct MI355X"
  export BACKEND=megatron
  export EXP="examples/megatron/configs/MI355X/gdn_1B_BF16-pretrain.yaml"
  eval "$(performance_block)"
  assert_eq "1" "$NVTE_USE_CAST_TRANSPOSE_TRITON" \
    "NVTE_USE_CAST_TRANSPOSE_TRITON"
)

check_explicit_override() (
  export NCCL_PXN_DISABLE="keep-me"
  export RCCL_WARP_SPEED_AUTO="keep-rccl"
  export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
  export MAD_SYSTEM_GPU_PRODUCT_NAME="AMD Instinct MI355X"
  export BACKEND=megatron EXP="examples/megatron/configs/MI355X/gdn_1B_BF16-pretrain.yaml"
  eval "$(performance_block)"
  assert_eq "keep-me" "$NCCL_PXN_DISABLE" "NCCL_PXN_DISABLE"
  assert_eq "keep-rccl" "$RCCL_WARP_SPEED_AUTO" "RCCL_WARP_SPEED_AUTO"
)

check_cli_parity_mi355x
check_cli_parity_gfx950_mi350x
check_gfx942_atomics
check_mxfp4_override
check_mxfp4_overrides_preset_one
check_non_mxfp4_keeps_preset_one
check_explicit_override
echo "primus_train pretrain env matches primus-cli defaults and honors overrides."
