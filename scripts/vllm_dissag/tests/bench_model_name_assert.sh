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
# Prompts per cell in the long-context harness, from its own lines (assignments and the
# per-cell formula), under the env a run would set.
_lc_prompts() { # con, then VAR=value...
    local con=$1; shift
    env -i PATH="$PATH" "$@" bash -c "
        $(grep -E '^(NUM_PROMPTS_FACTOR|MIN_PROMPTS)=' "$DIR/benchmark_long_context.sh")
        con=$con
        $(grep -E '^\s*(n_prompts=|\[ \"\$n_prompts\")' "$DIR/benchmark_long_context.sh" | sed 's/^ *//')
        echo \$n_prompts"
}
_is "$(_lc_prompts 8)" "32" "long_context default: con x 4"
_is "$(_lc_prompts 1)" "16" "long_context default: at least 16"
_is "$(_lc_prompts 8 BENCHMARK_PROMPTS_PER_CON=10 BENCHMARK_MIN_PROMPTS=10)" "80" "long_context follows BENCHMARK_PROMPTS_PER_CON"
_is "$(_lc_prompts 1 BENCHMARK_PROMPTS_PER_CON=10 BENCHMARK_MIN_PROMPTS=10)" "10" "long_context follows BENCHMARK_MIN_PROMPTS"
_is "$(_lc_prompts 8 BENCHMARK_PROMPTS_PER_CON=10 NUM_PROMPTS_FACTOR=3)" "24" "NUM_PROMPTS_FACTOR still wins in long_context"

echo "======================================================"
echo "  bench_model_name_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
