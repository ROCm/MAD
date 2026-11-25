#!/bin/bash
#
# pyt_torchtitan_qwen3_moe_nan: torchtitan's qwen3_moe_debug on 8 GPUs
# (FSDP 4 x TP 2 x EP 4, bf16, deterministic, seed 42). This is the MAD form of
# nan-doctor's qwen3_moe_gradient_nan_nohint example. A gradient-norm spike or a
# non-finite loss is the bug; a clean run prints the performance line.
#
# When madengine's nan_doctor tool is on (NAN_CAPTURE_DIR set), training starts
# through nan_doctor_rt.torchtitan_capture, which hooks Trainer.train_step. The
# generic optimizer hook does not see this loop.

set -u
TITAN=/opt/torchtitan
NGPU=${MAD_RUNTIME_NGPUS:-8}
STEPS=${STEPS:-100}
SEED_ROOT=/myworkspace/nan_hunts/qwen3-moe-seed
SEED=$SEED_ROOT/step-0
OUT=${OUT_DIR:-/myworkspace/run_directory/qwen3_moe_out}

mkdir -p "$SEED_ROOT" "$OUT/checkpoint"
if [ ! -f "$SEED/.metadata" ]; then
    work=$SEED_ROOT/work
    rm -rf "$work"
    echo "creating the seed checkpoint (1 GPU)"
    (cd "$TITAN" && MODULE=qwen3 CONFIG=qwen3_moe_debug NGPU=1 LOG_RANK=0 \
        ./run_train.sh --dump_folder="$work" --checkpoint.create_seed_checkpoint \
        --checkpoint.enable --debug.deterministic --debug.seed=42) || {
        echo "seed checkpoint failed" >&2
        exit 1
    }
    [ -f "$work/checkpoint/step-0/.metadata" ] || {
        echo "seed checkpoint has no step-0" >&2
        exit 1
    }
    cp -a "$work/checkpoint/step-0" "$SEED"
    rm -rf "$work"
fi
ln -sfn "$SEED" "$OUT/checkpoint/step-0"

# Device-mesh groups (FSDP, expert parallel) ignore --comm.init_timeout_seconds and
# keep PyTorch's 600s NCCL default. The first all-gather on this host outlives that
# while the GPUs are still compiling, so the job is killed at exactly 10 minutes.
pg=$OUT/pgtimeout
mkdir -p "$pg"
cat > "$pg/sitecustomize.py" <<'PY'
from datetime import timedelta
import torch.distributed.constants as _c
import torch.distributed.distributed_c10d as _d
_timeout = timedelta(seconds=1800)
# c10d binds the name at import, so both have to be replaced.
_c.default_pg_nccl_timeout = _d.default_pg_nccl_timeout = _timeout
PY
export PYTHONPATH="$pg:$TITAN${PYTHONPATH:+:$PYTHONPATH}"
# This image is a pip ROCm wheel. madengine's probe misses it and sets
# ROCM_PATH=/opt/rocm, which is not there. Point at the wheel's core instead.
if [ ! -x "${ROCM_PATH:-/opt/rocm}/bin/amd-smi" ]; then
    core=$(python3 -c 'import os, _rocm_sdk_core; print(os.path.dirname(_rocm_sdk_core.__file__))' 2>/dev/null || true)
    if [ -n "$core" ] && [ -x "$core/bin/amd-smi" ]; then
        export ROCM_PATH=$core
    fi
fi
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-http://localhost:29510}
if [ -n "${NAN_CAPTURE_DIR:-}" ]; then
    entry=nan_doctor_rt.torchtitan_capture
else
    entry=torchtitan.train
fi
echo "training: entry=$entry gpus=$NGPU steps=$STEPS"

cd "$TITAN"
set +e
torchrun --nproc_per_node="$NGPU" --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m "$entry" --module qwen3 --config qwen3_moe_debug \
    --dump_folder="$OUT" \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 4 \
    --debug.deterministic --debug.seed=42 \
    --metrics.enable_tensorboard --metrics.log_freq=1 \
    --training.steps="$STEPS" \
    --checkpoint.enable --checkpoint.export_dtype=bfloat16 --checkpoint.load_only \
    --comm.init_timeout_seconds 1800
rc=$?
set -e

log=$OUT/logs.txt
if [ -f "$log" ]; then
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -o 'step: *[0-9]* *loss: *[^ ]* *grad_norm: *[^ ]*' |
        awk '{ printf "  step %3d  loss %-10s grad_norm %s\n", $2, $4, $6 }'
fi
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
echo "performance: $STEPS samples_per_second"
