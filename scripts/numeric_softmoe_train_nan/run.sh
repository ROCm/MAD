#!/bin/bash
#
# pyt_numeric_softmoe_train_nan: a 20-layer soft mixture-of-experts transformer trained
# from scratch on one GPU, bf16 autocast, layers 3-18 each wrapped in torch.compile.
# This is the MAD form of Numeric Doctor's softmoe_train_nan example. A non-finite loss
# stops training with `RuntimeError: Loss is not finite`; a clean run prints the
# performance line.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEPS=${STEPS:-100}
COMPILE_LAYERS=${COMPILE_LAYERS:-3-18}
LOG=$HERE/train.log

echo "training: steps=$STEPS compile_layers=$COMPILE_LAYERS"
python3 "$HERE/train.py" --steps "$STEPS" --compile-layers "$COMPILE_LAYERS" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi

# tokens per second over the steps after the first five (compilation and warm-up)
tps=$(awk '/^step:/ && $2 > 5 { n++; t += $NF } END { if (n) printf "%.1f", n * 8 * 1024 / t }' "$LOG")
echo "performance: ${tps:-0} tokens_per_second"
