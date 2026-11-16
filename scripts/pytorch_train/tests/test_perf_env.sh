#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_SCRIPT="$SCRIPT_DIR/../pytorch_benchmark_report.sh"

performance_exports() {
  awk '
    /^# config environment$/ { capture = 1; next }
    /^SCRIPT_DIR=/ { capture = 0 }
    capture && /^export / { print }
  ' "$REPORT_SCRIPT"
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

check_defaults() (
  unset HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM GPU_MAX_HW_QUEUES
  unset CUDA_DEVICE_MAX_CONNECTIONS TORCH_NCCL_HIGH_PRIORITY
  eval "$(performance_exports)"

  assert_eq "1" "${HSA_ENABLE_SDMA-<unset>}" "HSA_ENABLE_SDMA"
  assert_eq "1" "${HSA_NO_SCRATCH_RECLAIM-<unset>}" "HSA_NO_SCRATCH_RECLAIM"
  assert_eq "2" "${GPU_MAX_HW_QUEUES-<unset>}" "GPU_MAX_HW_QUEUES"
  assert_eq "1" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}" \
    "CUDA_DEVICE_MAX_CONNECTIONS"
  assert_eq "1" "${TORCH_NCCL_HIGH_PRIORITY-<unset>}" \
    "TORCH_NCCL_HIGH_PRIORITY"
)

check_overrides() (
  export HSA_ENABLE_SDMA="override-sdma"
  export HSA_NO_SCRATCH_RECLAIM="override-scratch"
  export GPU_MAX_HW_QUEUES="override-queues"
  export CUDA_DEVICE_MAX_CONNECTIONS="override-connections"
  export TORCH_NCCL_HIGH_PRIORITY="override-priority"
  eval "$(performance_exports)"

  assert_eq "override-sdma" "$HSA_ENABLE_SDMA" "HSA_ENABLE_SDMA"
  assert_eq "override-scratch" "$HSA_NO_SCRATCH_RECLAIM" "HSA_NO_SCRATCH_RECLAIM"
  assert_eq "override-queues" "$GPU_MAX_HW_QUEUES" "GPU_MAX_HW_QUEUES"
  assert_eq "override-connections" "$CUDA_DEVICE_MAX_CONNECTIONS" \
    "CUDA_DEVICE_MAX_CONNECTIONS"
  assert_eq "override-priority" "$TORCH_NCCL_HIGH_PRIORITY" \
    "TORCH_NCCL_HIGH_PRIORITY"
)

check_defaults
check_overrides
echo "Performance environment defaults and overrides are correct."
