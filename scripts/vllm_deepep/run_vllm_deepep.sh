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
#
# vLLM expert-parallel serving over an all-to-all backend, single node, 8 GPUs.
#
# Selects the backend with ALL2ALL_BACKEND (deepep_v2 | mori_high_throughput)
# so the same script produces both arms of a backend comparison with nothing
# else changed -- which is the only way the comparison means anything.
#
# Emits perf_vllm_deepep.csv (models.json: multiple_results).
#
#################################################################################
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V2-Lite}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-deepep_v2}"
PORT="${PORT:-18000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
ISL="${ISL:-256}"
OSL="${OSL:-128}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"

# AITER tunes GEMM shapes at run time behind a global lock, so an unwarmed run
# measures the tuner, not the server: P99 TTFT lands around 20 s on the first
# pass against a few hundred ms once warm, and roughly 270 s on DeepSeek-R1.
# The discarded count is a reported field, not a hidden constant -- a
# comparison that does not state it is not interpretable.
WARMUP_RUNS="${WARMUP_RUNS:-2}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"

PYTHON="${PYTHON:-/opt/venv/bin/python}"
VLLM="${VLLM:-/opt/venv/bin/vllm}"
LOG_DIR="${LOG_DIR:-$(pwd)}"
SERVER_LOG="${LOG_DIR}/vllm_${ALL2ALL_BACKEND}_server.log"
# MAD ingests multiple_results in long form: it requires exactly the columns
# model / performance / metric and drops everything else (tools/utils.py:1231
# raises if one is missing; :1247-1250 reads only those three). So the
# diagnostic context that makes a number interpretable -- backend, load shape,
# discarded warm-ups -- cannot live in that file. It goes to a companion CSV
# that is not registered as multiple_results and is therefore not parsed.
RESULT_CSV="${LOG_DIR}/perf_vllm_deepep.csv"
DETAIL_CSV="${LOG_DIR}/perf_vllm_deepep_detail.csv"

echo "=== vllm_deepep: ${MODEL_REPO} over ${ALL2ALL_BACKEND} ==="

# --- backend-specific environment -------------------------------------------
# NCCL_CUMEM_ENABLE is required by BOTH backends: RCCL refuses symmetric memory
# (and ncclDevCommCreate) without VMM.
export NCCL_CUMEM_ENABLE=1
export VLLM_ROCM_USE_AITER=1
# vLLM refuses MoRI without AITER fused MoE, and running the two arms on
# different MoE implementations would make the comparison meaningless, so both
# arms use it.
export VLLM_ROCM_USE_AITER_MOE=1
export PYTHONPATH="${PYTHONPATH:-}:/opt/rocm/share/amd_smi"
export HSA_NO_SCRATCH_RECLAIM=1

case "${ALL2ALL_BACKEND}" in
  deepep_v2)
    # GIN stays on. It carries no payload inside a single XGMI domain, but it
    # is the path that matters for scale-out, and testing a mode nobody
    # deploys is not testing. Requires /dev/infiniband in the container.
    export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-2}"
    export EP_GIN_QUEUE_DEPTH="${EP_GIN_QUEUE_DEPTH:-0}"
    export EP_NIC_NAME="${EP_NIC_NAME:-bnxt_re0}"
    export VLLM_DEEPEP_TIMEOUT_SECS="${VLLM_DEEPEP_TIMEOUT_SECS:-1800}"
    ;;
  mori_high_throughput)
    # Works around a null dereference in MoRI's CollectAndSortCandidates. It
    # skips GPU/NIC affinity matching, which should not matter for a
    # single-node run where traffic stays on XGMI -- but that is an
    # expectation, so it is recorded in the CSV rather than left implicit.
    export MORI_DISABLE_TOPO="${MORI_DISABLE_TOPO:-1}"
    ;;
  *)
    echo "unknown ALL2ALL_BACKEND: ${ALL2ALL_BACKEND}" >&2
    exit 1
    ;;
esac

# --- model ------------------------------------------------------------------
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL="${MODEL_PATH}"
else
  MODEL="${MODEL_REPO}"
fi

# --- server -----------------------------------------------------------------
"${VLLM}" serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --all2all-backend "${ALL2ALL_BACKEND}" \
  --enforce-eager \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --trust-remote-code > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo "server pid ${SERVER_PID}, log ${SERVER_LOG}"

# Startup is dominated by AITER JIT (~5-10 min cold, more on a large model),
# so wait on the endpoint rather than on a fixed sleep.
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server died during startup; tail of ${SERVER_LOG}:" >&2
    tail -40 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 15
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

bench_once() {
  "${VLLM}" bench serve --backend vllm --model "${MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dataset-name random \
    --random-input-len "${ISL}" --random-output-len "${OSL}" \
    --max-concurrency "${CONCURRENCY}" --num-prompts "${NUM_PROMPTS}" \
    --ignore-eos 2>&1
}

echo "--- ${WARMUP_RUNS} warm-up run(s), discarded ---"
for i in $(seq 1 "${WARMUP_RUNS}"); do
  bench_once > /dev/null || { echo "warm-up run ${i} failed" >&2; exit 1; }
done

echo "model,performance,metric" > "${RESULT_CSV}"
echo "backend,model,concurrency,isl,osl,run,total_tok_s,median_tpot_ms,p99_ttft_ms,warmup_discarded,mori_disable_topo" \
  > "${DETAIL_CSV}"

for i in $(seq 1 "${MEASURED_RUNS}"); do
  out="$(bench_once)"
  echo "${out}" | tail -25
  tput_val=$(echo "${out}" | awk '/Total token throughput/ {print $NF}')
  tpot_val=$(echo "${out}" | awk '/Median TPOT/ {print $NF}')
  ttft_val=$(echo "${out}" | awk '/P99 TTFT/ {print $NF}')

  # `vllm bench serve` exits 0 while reporting failed requests, and still
  # prints the three metric lines -- as zeros. Ingested, those are
  # indistinguishable from a successful measurement, which is worse than no
  # data. The most likely cause is ISL + OSL exceeding MAX_MODEL_LEN, where
  # every request fails and the run looks merely slow.
  failed=$(echo "${out}" | awk '/Failed requests/ {print $NF}')
  if [[ -n "${failed}" && "${failed}" != "0" ]]; then
    echo "run ${i}: ${failed} failed request(s); refusing to record a result." >&2
    echo "check that ISL(${ISL}) + OSL(${OSL}) < MAX_MODEL_LEN(${MAX_MODEL_LEN})." >&2
    rm -f "${RESULT_CSV}" "${DETAIL_CSV}"
    exit 1
  fi
  # An empty field means the summary did not parse -- a changed bench output
  # format, or a run that died mid-way. Either way it is not a zero.
  if [[ -z "${tput_val}" || -z "${tpot_val}" || -z "${ttft_val}" ]]; then
    echo "run ${i}: could not parse the benchmark summary; refusing to record a result." >&2
    echo "${out}" | tail -40 >&2
    rm -f "${RESULT_CSV}" "${DETAIL_CSV}"
    exit 1
  fi

  # Each run is its own set of rows rather than a pre-averaged one, so a run
  # that has not converged stays visible instead of being hidden in a mean.
  # MAD prefixes these labels with the model-card name, which is what carries
  # the backend and the model.
  printf '%s\n' \
    "run${i}_throughput,${tput_val},total_token_throughput_tok_s" \
    "run${i}_tpot,${tpot_val},median_tpot_ms" \
    "run${i}_ttft,${ttft_val},p99_ttft_ms" \
    >> "${RESULT_CSV}"

  # MODEL, not MODEL_REPO: with MODEL_PATH set the server and the benchmark
  # both ran against the local path, and recording the repo id would label the
  # row with a model that was never loaded.
  echo "${ALL2ALL_BACKEND},${MODEL},${CONCURRENCY},${ISL},${OSL},${i},${tput_val},${tpot_val},${ttft_val},${WARMUP_RUNS},${MORI_DISABLE_TOPO:-}" \
    >> "${DETAIL_CSV}"
done

echo "=== results (${RESULT_CSV}) ==="
cat "${RESULT_CSV}"
echo "=== detail (${DETAIL_CSV}) ==="
cat "${DETAIL_CSV}"
