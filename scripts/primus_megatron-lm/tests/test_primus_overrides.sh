#!/usr/bin/env bash
# Regression test: the two override paths in primus_megatron-lm_benchmark_report.sh.
#
#   run_primus()  appends --train_iters / --profile only when the deployment
#                 asks for them, and leaves the command untouched otherwise.
#   MEM_OVERRIDE  lowers micro_batch_size for GPT-OSS-120B on 192GB parts and
#                 at the 16-GPU topology minimum, but not at 32 GPUs.
#
# Both blocks are evaluated in isolation, so this needs neither a GPU nor Primus.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_SH="$SCRIPT_DIR/../primus_megatron-lm_benchmark_report.sh"

run_primus_block() {
  awk '/^run_primus\(\) \{/, /^\}/' "$REPORT_SH"
}

mem_override_block() {
  awk '/^    MEM_OVERRIDE=""$/, /^    fi$/' "$REPORT_SH"
}

assert_eq() {
  local expected="$1"
  local actual="$2"
  local what="$3"

  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL $what" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
}

# Assembled primus-cli argv, with the launcher and the tee stubbed out. A shell
# function shadows the `bash` command, so the real runner is never reached.
primus_argv() (
  eval "$(run_primus_block)"
  bash() { printf 'ARGV %s\n' "$*"; }
  MODEL_REPO=GPT-OSS-120B
  TRAIN_LOG=/dev/null
  run_primus "$@" | sed -n 's/^ARGV //p'
)

BASE='runner/primus-cli direct --log_file /tmp/primus_GPT-OSS-120B.log -- train pretrain --config cfg.yaml --micro_batch_size 2'

check_no_knobs_leaves_command_untouched() (
  unset PRIMUS_TRAIN_ITERS PRIMUS_DISABLE_PROFILE
  assert_eq "$BASE" "$(primus_argv cfg.yaml --micro_batch_size 2)" \
    "unset knobs change the command"
)

# PRIMUS_DISABLE_PROFILE defaults to 0, so an explicit 0 must behave as unset.
check_disable_profile_zero_is_a_noop() (
  export PRIMUS_DISABLE_PROFILE=0
  unset PRIMUS_TRAIN_ITERS
  assert_eq "$BASE" "$(primus_argv cfg.yaml --micro_batch_size 2)" \
    "PRIMUS_DISABLE_PROFILE=0 is not a no-op"
)

check_train_iters() (
  export PRIMUS_TRAIN_ITERS=15
  unset PRIMUS_DISABLE_PROFILE
  assert_eq "$BASE --train_iters 15" \
    "$(primus_argv cfg.yaml --micro_batch_size 2)" "PRIMUS_TRAIN_ITERS"
)

check_disable_profile() (
  export PRIMUS_DISABLE_PROFILE=1
  unset PRIMUS_TRAIN_ITERS
  assert_eq "$BASE --profile false --use_pytorch_profiler false" \
    "$(primus_argv cfg.yaml --micro_batch_size 2)" "PRIMUS_DISABLE_PROFILE"
)

check_both_knobs() (
  export PRIMUS_TRAIN_ITERS=15 PRIMUS_DISABLE_PROFILE=1
  assert_eq "$BASE --train_iters 15 --profile false --use_pytorch_profiler false" \
    "$(primus_argv cfg.yaml --micro_batch_size 2)" "both knobs"
)

# The knobs land after the caller's own arguments: Primus builds the override
# map as a dict, so the last occurrence of a key is the one that takes effect.
check_knobs_come_last() (
  export PRIMUS_TRAIN_ITERS=15
  unset PRIMUS_DISABLE_PROFILE
  local argv
  argv="$(primus_argv cfg.yaml --recompute_num_layers 9)"
  assert_eq "--recompute_num_layers 9 --train_iters 15" \
    "${argv#*pretrain --config cfg.yaml }" "knob ordering"
)

mem_override() (
  DEVICE="$1"
  NUM_GPUS="$2"
  DATATYPE="$3"
  MBS=8
  eval "$(mem_override_block)" > /dev/null
  printf '%s|%s' "$MBS" "$MEM_OVERRIDE"
)

# MI355X has 288GB, so only the 16-GPU case pulls the overrides in: TP1 x PP2
# fixes model parallelism at 2, DP halves, and the distributed optimizer shard
# grows past what the shipped micro_batch_size 8 leaves room for.
check_mi355x_32gpu_is_untouched() {
  assert_eq "8|" "$(mem_override MI355X 32 BF16)" "MI355X at 32 GPUs"
  assert_eq "8|" "$(mem_override MI355X 32 FP8)" "MI355X at 32 GPUs, FP8"
}

check_mi355x_16gpu_gets_overrides() {
  assert_eq "2|--micro_batch_size 2 --recompute_num_layers 9" \
    "$(mem_override MI355X 16 BF16)" "MI355X at 16 GPUs"
  assert_eq "1|--micro_batch_size 1 --recompute_num_layers 9" \
    "$(mem_override MI355X 16 FP8)" "MI355X at 16 GPUs, FP8"
}

# 192GB parts need them at every scale, which is the pre-existing behaviour.
check_mi300x_keeps_overrides_at_every_scale() {
  assert_eq "2|--micro_batch_size 2 --recompute_num_layers 9" \
    "$(mem_override MI300X 32 BF16)" "MI300X at 32 GPUs"
  assert_eq "1|--micro_batch_size 1 --recompute_num_layers 9" \
    "$(mem_override MI325X 16 FP8)" "MI325X at 16 GPUs, FP8"
}

check_no_knobs_leaves_command_untouched
check_disable_profile_zero_is_a_noop
check_train_iters
check_disable_profile
check_both_knobs
check_knobs_come_last
check_mi355x_32gpu_is_untouched
check_mi355x_16gpu_gets_overrides
check_mi300x_keeps_overrides_at_every_scale
echo "primus_megatron-lm overrides: CLI knobs are opt-in and GPT-OSS-120B memory overrides follow device and world size."
