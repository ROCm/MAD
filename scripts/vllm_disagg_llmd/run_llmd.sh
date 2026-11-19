#!/bin/bash
#
# Benchmark an llm-d gateway. Prefill, decode, and the routing proxy are
# already running; this script only sends load to MAD_LLM_D_ENDPOINT.
#
set -euo pipefail

: "${MAD_LLM_D_ENDPOINT:?MAD_LLM_D_ENDPOINT is not set. Run this model with an llm_d additional context.}"
: "${MAD_LLM_D_MODEL:?MAD_LLM_D_MODEL is not set (llm_d.model.name).}"

# Same flag as scripts/vllm/run.sh. The model card is the Hugging Face id;
# madengine copies it into MAD_LLM_D_MODEL before the servers start.
MODEL_REPO=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL_REPO="$2"; shift ;;
        *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
    esac
    shift
done
if [[ -n "$MODEL_REPO" && "$MODEL_REPO" != "$MAD_LLM_D_MODEL" ]]; then
    echo "model card --model_repo (${MODEL_REPO}) does not match MAD_LLM_D_MODEL (${MAD_LLM_D_MODEL})" >&2
    exit 1
fi

NUM_PROMPTS="${LLM_D_NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${LLM_D_MAX_CONCURRENCY:-4}"
INPUT_LEN="${LLM_D_INPUT_LEN:-256}"
OUTPUT_LEN="${LLM_D_OUTPUT_LEN:-128}"
RESULT_JSON="${LLM_D_RESULT_JSON:-/tmp/llm-d-bench.json}"
PERF_CSV="${MAD_OUTPUT_CSV:-perf_Qwen3-8B_llmd.csv}"

python3 - "$MAD_LLM_D_ENDPOINT" "$MAD_LLM_D_MODEL" <<'PY'
import sys
import time
import urllib.request

endpoint = sys.argv[1].rstrip("/")
model = sys.argv[2]
print(f"llm-d gateway: {endpoint}", flush=True)
print(f"model: {model}", flush=True)
deadline = time.time() + 1800
while True:
    try:
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=10) as resp:
            body = resp.read().decode()
        if model in body:
            print("gateway is serving the model", flush=True)
            break
        print("gateway is up but the model is not listed yet", flush=True)
    except Exception as exc:
        print(f"waiting for gateway: {exc}", flush=True)
    if time.time() > deadline:
        raise SystemExit("gateway did not serve the model within 1800s")
    time.sleep(20)
PY

BENCH_CMD=(
    vllm bench serve
    --backend openai
    --base-url "$MAD_LLM_D_ENDPOINT"
    --model "$MAD_LLM_D_MODEL"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-rate inf
    --percentile-metrics ttft,tpot,itl,e2el
    --seed 1
    --save-result
    --result-filename "$RESULT_JSON"
)
"${BENCH_CMD[@]}"

# Same six serving rows as scripts/vllm/vllm_json_to_csv.py.
python3 "$(dirname "$0")/llmd_bench_csv.py" \
    --result "$RESULT_JSON" \
    --output "$PERF_CSV" \
    --model "$MAD_LLM_D_MODEL" \
    --tensor-parallel-size "${MAD_LLM_D_TP:-1}" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --dtype auto \
    --vllm-cmd "${BENCH_CMD[*]}"
