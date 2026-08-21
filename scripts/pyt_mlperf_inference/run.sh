#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

set -euo pipefail
set -x

MODEL_REPO=""

usage() {
  echo "Usage: $0 --model_repo <model>"
  echo "Supported models:"
  echo "  pyt_mlperf_inference_llama-3.1-8b"
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model_repo)
      MODEL_REPO="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      usage
      ;;
  esac
done

if [[ -z "${MODEL_REPO}" ]]; then
  usage
fi

# The harness lives in the image (see docker/pyt_mlperf_inference.ubuntu.amd.Dockerfile);
# model and dataset are mounted from the shared filesystem by the manifest.
MLPERF_HARNESS_DIR="${MLPERF_HARNESS_DIR:-/workspace/inference/language/llama3.1-8b}"
MLPERF_INF_MODEL_PATH="${MLPERF_INF_MODEL_PATH:-/model}"
MLPERF_INF_DATASET_PATH="${MLPERF_INF_DATASET_PATH:-/dataset/sample_cnn_eval_5000.json}"
MLPERF_INF_SCENARIO="${MLPERF_INF_SCENARIO:-Offline}"
# 5000 is the edge set, 13368 the datacenter one; keep it in step with
# MLPERF_INF_LG_MODEL_NAME, which selects the loadgen config and the ROUGE targets.
MLPERF_INF_TOTAL_SAMPLE_COUNT="${MLPERF_INF_TOTAL_SAMPLE_COUNT:-5000}"
MLPERF_INF_LG_MODEL_NAME="${MLPERF_INF_LG_MODEL_NAME:-llama3_1-8b-edge}"
MLPERF_INF_TENSOR_PARALLEL_SIZE="${MLPERF_INF_TENSOR_PARALLEL_SIZE:-1}"
MLPERF_INF_BATCH_SIZE="${MLPERF_INF_BATCH_SIZE:-16}"
MLPERF_INF_DTYPE="${MLPERF_INF_DTYPE:-bfloat16}"
MLPERF_INF_NUM_WORKERS="${MLPERF_INF_NUM_WORKERS:-1}"
MLPERF_INF_USER_CONF="${MLPERF_INF_USER_CONF:-user.conf}"
MLPERF_INF_RUN_PERFORMANCE="${MLPERF_INF_RUN_PERFORMANCE:-1}"
MLPERF_INF_RUN_ACCURACY="${MLPERF_INF_RUN_ACCURACY:-1}"

# madengine runs this from the model directory and collects the results CSV
# relative to that working directory.
MLPERF_RESULTS_DIR="${MLPERF_RESULTS_DIR:-${PWD}}"
RESULT_CSV="${MLPERF_RESULTS_DIR}/perf_${MODEL_REPO}.csv"
LOG_DIR="${MLPERF_RESULTS_DIR}/mlperf_inference_logs"
PERF_DIR="${LOG_DIR}/${MLPERF_INF_SCENARIO}_performance"
ACC_DIR="${LOG_DIR}/${MLPERF_INF_SCENARIO}_accuracy"
ACC_SCORE_LOG="${LOG_DIR}/accuracy_score.txt"

mkdir -p "${MLPERF_RESULTS_DIR}" "${LOG_DIR}"

# Offline validity needs the run to last at least user.conf's min_duration, and
# loadgen sizes its single coalesced query from target_qps alone. Left at the
# upstream default of 1 the query is far too small and loadgen reports
# "Min duration satisfied : NO", so the expected throughput has to be declared
# here (set it at or above the measured samples/s).
if [[ -n "${MLPERF_INF_OFFLINE_TARGET_QPS:-}" ]]; then
  USER_CONF_OVERRIDE="${LOG_DIR}/user.conf"
  cp "${MLPERF_HARNESS_DIR}/${MLPERF_INF_USER_CONF}" "${USER_CONF_OVERRIDE}"
  echo "*.Offline.target_qps = ${MLPERF_INF_OFFLINE_TARGET_QPS}" >> "${USER_CONF_OVERRIDE}"
  MLPERF_INF_USER_CONF="${USER_CONF_OVERRIDE}"
fi

echo "model,performance,metric" > "${RESULT_CSV}"
echo "${MODEL_REPO},0,run_completed" >> "${RESULT_CSV}"

if [[ ! -f "${MLPERF_INF_MODEL_PATH}/config.json" ]]; then
  echo "No checkpoint at ${MLPERF_INF_MODEL_PATH} (expected config.json)."
  echo "Stage meta-llama/Llama-3.1-8B-Instruct there and mount it into the container."
  exit 1
fi

if [[ ! -f "${MLPERF_INF_DATASET_PATH}" ]]; then
  echo "No dataset at ${MLPERF_INF_DATASET_PATH}."
  echo "Download it with the MLCommons R2 downloader, e.g."
  echo "  bash mlc-r2-downloader.sh -d <dir> \\"
  echo "    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri"
  exit 1
fi

echo "=== environment ==="
python3 -c "import sys, torch, vllm; \
print('vllm', vllm.__version__, '| torch', torch.__version__, '| hip', torch.version.hip); \
n = torch.cuda.device_count(); \
sys.exit('no GPU visible to the container: pass the devices through docker_gpus') if not n else None; \
print('gpus', n, torch.cuda.get_device_properties(0).gcnArchName)"

cd "${MLPERF_HARNESS_DIR}"

HARNESS_ARGS=(
  --scenario "${MLPERF_INF_SCENARIO}"
  --model-path "${MLPERF_INF_MODEL_PATH}"
  --dataset-path "${MLPERF_INF_DATASET_PATH}"
  --dtype "${MLPERF_INF_DTYPE}"
  --batch-size "${MLPERF_INF_BATCH_SIZE}"
  --user-conf "${MLPERF_INF_USER_CONF}"
  --total-sample-count "${MLPERF_INF_TOTAL_SAMPLE_COUNT}"
  --tensor-parallel-size "${MLPERF_INF_TENSOR_PARALLEL_SIZE}"
  --num-workers "${MLPERF_INF_NUM_WORKERS}"
  --lg-model-name "${MLPERF_INF_LG_MODEL_NAME}"
  --vllm
)

if [[ "${MLPERF_INF_RUN_PERFORMANCE}" == "1" ]]; then
  echo "=== ${MLPERF_INF_SCENARIO} performance run ==="
  python3 -u main.py "${HARNESS_ARGS[@]}" --output-log-dir "${PERF_DIR}"
fi

if [[ "${MLPERF_INF_RUN_ACCURACY}" == "1" ]]; then
  echo "=== ${MLPERF_INF_SCENARIO} accuracy run ==="
  python3 -u main.py "${HARNESS_ARGS[@]}" --output-log-dir "${ACC_DIR}" --accuracy

  echo "=== ROUGE scoring ==="
  python3 -u evaluation.py \
    --mlperf-accuracy-file "${ACC_DIR}/mlperf_log_accuracy.json" \
    --dataset-file "${MLPERF_INF_DATASET_PATH}" \
    --model-name "${MLPERF_INF_MODEL_PATH}" \
    --total-sample-count "${MLPERF_INF_TOTAL_SAMPLE_COUNT}" \
    --dtype int32 | tee "${ACC_SCORE_LOG}"
fi

echo "=== results ==="
MODEL_REPO="${MODEL_REPO}" \
RESULT_CSV="${RESULT_CSV}" \
PERF_DIR="${PERF_DIR}" \
ACC_SCORE_LOG="${ACC_SCORE_LOG}" \
LG_MODEL_NAME="${MLPERF_INF_LG_MODEL_NAME}" \
TOTAL_SAMPLE_COUNT="${MLPERF_INF_TOTAL_SAMPLE_COUNT}" \
TENSOR_PARALLEL_SIZE="${MLPERF_INF_TENSOR_PARALLEL_SIZE}" \
python3 - <<'PY'
import os
import re

# BF16 reference accuracy from the upstream README; the ROUGE gate is 99% of the
# target and the gen_len gate 90%, and gen_len is only comparable when the whole
# set was served.
TARGETS = {
    "llama3_1-8b": {
        "rouge1": 38.7792, "rouge2": 15.9075, "rougeL": 24.4957,
        "rougeLsum": 35.793, "gen_len": 8167644, "gen_num": 13368,
    },
    "llama3_1-8b-edge": {
        "rouge1": 39.06, "rouge2": 16.1147, "rougeL": 24.6375,
        "rougeLsum": 36.124, "gen_len": 3051113, "gen_num": 5000,
    },
}

rows = []


def summary_metrics(path):
    if not os.path.isfile(path):
        return
    text = open(path).read()
    patterns = {
        "samples_per_s": r"Samples per second:\s*([0-9.eE+-]+)",
        "tokens_per_s": r"Tokens per second:\s*([0-9.eE+-]+)",
        "completed_tokens_per_s": r"Completed tokens per second:\s*([0-9.eE+-]+)",
        "ttft_p99_ns": r"99.00 percentile first token latency \(ns\)\s*:\s*([0-9]+)",
    }
    for metric, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            rows.append((m.group(1), metric))
    m = re.search(r"Result is\s*:\s*(\w+)", text)
    if m:
        rows.append(("1" if m.group(1).upper() == "VALID" else "0", "result_valid"))


def accuracy_metrics(path, target_name):
    if not os.path.isfile(path):
        return
    # evaluation.py prints a plain dict, but numpy 2 renders gen_len as
    # `np.int64(...)`, so the keys are picked out one by one.
    text = open(path).read()
    scores = {}
    for key in ("rouge1", "rouge2", "rougeL", "rougeLsum", "gen_len", "gen_num"):
        found = re.search(rf"'{key}':\s*(?:np\.\w+\()?'?([0-9.eE+-]+)'?", text)
        if found:
            scores[key] = found.group(1)
    if not scores:
        return
    targets = TARGETS.get(target_name, {})
    gates = []
    for key in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        if key not in scores:
            continue
        value = float(scores[key])
        rows.append((f"{value:g}", key))
        if key in targets:
            gates.append(value >= 0.99 * targets[key])
    gen_num = int(scores.get("gen_num", 0))
    gen_len = int(scores.get("gen_len", 0))
    rows.append((str(gen_len), "gen_len"))
    rows.append((str(gen_num), "gen_num"))
    if targets.get("gen_num") == gen_num and "gen_len" in targets:
        gates.append(gen_len >= 0.90 * targets["gen_len"])
    if gates:
        rows.append(("1" if all(gates) else "0", "accuracy_pass"))


summary_metrics(os.path.join(os.environ["PERF_DIR"], "mlperf_log_summary.txt"))
accuracy_metrics(os.environ["ACC_SCORE_LOG"], os.environ["LG_MODEL_NAME"])
rows.append((os.environ["TOTAL_SAMPLE_COUNT"], "total_sample_count"))
rows.append((os.environ["TENSOR_PARALLEL_SIZE"], "tensor_parallel_size"))
rows.append(("1", "run_completed"))

model = os.environ["MODEL_REPO"]
with open(os.environ["RESULT_CSV"], "w") as handle:
    handle.write("model,performance,metric\n")
    for value, metric in rows:
        handle.write(f"{model},{value},{metric}\n")
print(open(os.environ["RESULT_CSV"]).read(), end="")
PY
