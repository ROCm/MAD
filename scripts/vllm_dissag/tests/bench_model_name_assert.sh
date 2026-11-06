#!/bin/bash
# Offline check: every `vllm bench serve` asks for the model under the name the server
# serves it as. A server started with --served-model-name answers 404 to any other
# name; the colocated launcher always serves under MODEL_NAME, and some disagg recipes
# set --served-model-name, so requesting MODEL_PATH failed every request there.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pass=0; fail=0
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (got %s, want %s)\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
for f in benchmark_xPyD.sh benchmark_long_context.sh; do
    calls=$(grep -c 'vllm bench serve' "$DIR/$f")
    named=$(grep -c -- '--served-model-name "${SERVED_MODEL_NAME:-$MODEL_PATH}"' "$DIR/$f")
    _is "$named" "$calls" "$f: all $calls vllm bench serve calls name the served model"
done
for f in benchmark_xPyD.sh benchmark_long_context.sh benchmark_agentic.sh; do
    _is "$(grep -v '^[[:space:]]*#' "$DIR/$f" | grep -c 'NIXL_COOKBOOK_PATH')" "0" "$f: finds its siblings from its own directory, not NIXL_COOKBOOK_PATH"
done
_is "$(grep -c '^export SERVED_MODEL_NAME="${MODEL_NAME:-model}"' "$DIR/../vllm_multinode/serve_colocated.sh")" "1" \
    "serve_colocated.sh exports the name it serves under"
_is "$(grep -c 'export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"' "$DIR/vllm_disagg.sh")" "1" \
    "vllm_disagg.sh still defaults it to MODEL_PATH (vLLM's own default)"
echo "======================================================"
echo "  bench_model_name_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
