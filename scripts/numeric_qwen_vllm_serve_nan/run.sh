#!/bin/bash
#
# pyt_numeric_qwen_vllm_serve_nan: Qwen2.5-0.5B (random bf16 weights) served by vLLM's
# offline LLM.generate() on one GPU with vLLM's defaults; greedy requests whose
# generated-token logprobs must be finite. This is the MAD form of Numeric Doctor's
# qwen_vllm_serve_nan example. A request with non-finite logprobs prints
# `Logprobs are not finite at request <i>` and fails the run; a clean run prints the
# performance line.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REQUESTS=${REQUESTS:-200}
PROMPT=${PROMPT:-256}
NEW=${NEW:-8}
SEED=${SEED:-0}
LOG=$HERE/serve.log
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export HF_HUB_OFFLINE=1

echo "serving: requests=$REQUESTS prompt=$PROMPT new=$NEW seed=$SEED"
start=$SECONDS
python3 "$HERE/qwen_vllm_serve.py" --requests "$REQUESTS" --prompt-len "$PROMPT" \
    --new-tokens "$NEW" --seed "$SEED" --log-every 50 "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
elapsed=$((SECONDS - start))
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
res=$(grep -h '^RESULT' "$LOG")
if [ -z "$res" ]; then
    echo "no RESULT line" >&2
    exit 1
fi
if ! grep -q 'nan_request=-' <<<"$res"; then
    exit 1
fi
done_=$(grep -o 'requests_done=[0-9]*' <<<"$res" | cut -d= -f2)
echo "performance: $(awk -v n="$done_" -v t="$elapsed" 'BEGIN { printf "%.2f", t ? n / t : 0 }') requests_per_second"
