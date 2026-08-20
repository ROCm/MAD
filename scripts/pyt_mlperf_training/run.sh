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
  echo "  pyt_mlperf_training_llama-3.1-8b"
  echo "  pyt_mlperf_training_llama-3.1-405b"
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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(pwd)

MLPERF_TRAINING_REF="${MLPERF_TRAINING_REF:-d15c39403c20786633aac6876c014b074a22feb9}"
MLPERF_EXECUTION_MODE="${MLPERF_EXECUTION_MODE:-dryrun}"
MLPERF_RESULTS_DIR="${MLPERF_RESULTS_DIR:-${SCRIPT_DIR}/..}"
RESULT_CSV="${MLPERF_RESULTS_DIR}/perf_${MODEL_REPO}.csv"
RUN_LOG="${MLPERF_RESULTS_DIR}/${MODEL_REPO}.log"

mkdir -p "${MLPERF_RESULTS_DIR}"

write_results() {
  local model_name="$1"
  shift
  echo "model,performance,metric" > "${RESULT_CSV}"
  while [[ "$#" -gt 1 ]]; do
    echo "${model_name},$1,$2" >> "${RESULT_CSV}"
    shift 2
  done
}

append_envvar() {
  local key="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    if [[ "${value}" == *,* ]]; then
      export "${key}=${value}"
      return
    fi
    if [[ -n "${ENVVARS}" ]]; then
      ENVVARS="${ENVVARS},${key}=${value}"
    else
      ENVVARS="${key}=${value}"
    fi
  fi
}

_tokenizer_present() {
  [[ -f "${TOKENIZER_PATH}/tokenizer.json" || -f "${TOKENIZER_PATH}/tokenizer.model" ]]
}

_dataset_present() {
  compgen -G "${PREPROCESSED_PATH}/c4-train.en_*_text_document.bin" > /dev/null \
    && compgen -G "${PREPROCESSED_PATH}/c4-validation*text_document.bin" > /dev/null
}

_wait_for_sentinel() {
  # Non-rank-0 barrier: wait for rank 0 to publish the "ready" sentinel on the
  # (presumed shared) FS. Exits 0 if sentinel appears and the artifact itself
  # is still present, 1 on timeout.
  local sentinel="$1"
  local check_fn="$2"
  local label="$3"
  local timeout="${MLPERF_DATA_FETCH_TIMEOUT:-14400}"
  local waited=0
  echo "[data-fetch] rank=${DATA_FETCH_RANK} waiting for ${label} sentinel ${sentinel} (timeout=${timeout}s)"
  while true; do
    if [[ -f "${sentinel}" ]] && "${check_fn}"; then
      echo "[data-fetch] rank=${DATA_FETCH_RANK} ${label} ready after ${waited}s"
      return 0
    fi
    if (( waited >= timeout )); then
      echo "[data-fetch] ERROR: rank=${DATA_FETCH_RANK} timed out after ${timeout}s waiting for ${label} sentinel ${sentinel}"
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

ensure_mlperf_data() {
  # Prefetch the HuggingFace Llama-3.1-8B tokenizer (gated) and the MLCommons
  # preprocessed C4 dataset before training.
  #
  # Multi-node coordination: when the dataset directory is a shared filesystem
  # (manifest maps /shared_inference/...), a naive per-node download creates a
  # write race on the same files. Use a rank-0 barrier via a sentinel file:
  #   * node_rank == 0 downloads, then atomically publishes .ready
  #   * non-zero ranks poll for .ready (with the artifact still present)
  # If the artifacts are already materialized, the sentinel is (re)published
  # idempotently so that subsequent jobs on fresh allocations skip the wait.
  if [[ "${MLPERF_SKIP_DATA_FETCH:-0}" == "1" ]]; then
    echo "[data-fetch] MLPERF_SKIP_DATA_FETCH=1, skipping tokenizer/dataset prefetch"
    return 0
  fi

  DATA_FETCH_RANK="${NODE_RANK:-${RANK:-0}}"
  local tokenizer_sentinel="${TOKENIZER_PATH}/.mlperf_ready"
  local dataset_sentinel="${PREPROCESSED_PATH}/.mlperf_ready"
  mkdir -p "${TOKENIZER_PATH}" "${PREPROCESSED_PATH}"

  if [[ "${DATA_FETCH_RANK}" == "0" ]]; then
    if _tokenizer_present; then
      echo "[data-fetch] rank=0 tokenizer already present at ${TOKENIZER_PATH}"
    else
      # `set -x` traces expanded command lines, so every command that mentions
      # the token — the assignment, the emptiness test and the download itself —
      # would put its value in the run log. Trace off for all three; the
      # enclosing group keeps the `set +x` line itself out of the trace.
      { set +x; } 2>/dev/null
      local hf_token="${MAD_SECRETS_HFTOKEN:-${HF_TOKEN:-}}"
      if [[ -z "${hf_token}" ]]; then
        set -x
        echo "[data-fetch] ERROR: tokenizer missing at ${TOKENIZER_PATH} and no HF token provided (MAD_SECRETS_HFTOKEN/HF_TOKEN)."
        return 1
      fi
      echo "[data-fetch] rank=0 downloading meta-llama/Llama-3.1-8B tokenizer -> ${TOKENIZER_PATH}"
      rm -f "${tokenizer_sentinel}"
      # Only the tokenizer is needed; without --include this pulls the whole
      # repo, ~16 GB of weights, to reach a few MB of vocabulary.
      HF_TOKEN="${hf_token}" huggingface-cli download \
        meta-llama/Llama-3.1-8B \
        --include "tokenizer*" "special_tokens_map.json" "config.json" \
                  "generation_config.json" \
        --local-dir "${TOKENIZER_PATH}" \
        --local-dir-use-symlinks False
      set -x
      if ! _tokenizer_present; then
        echo "[data-fetch] ERROR: download finished but no tokenizer under ${TOKENIZER_PATH}"
        return 1
      fi
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${tokenizer_sentinel}.tmp"
    mv -f "${tokenizer_sentinel}.tmp" "${tokenizer_sentinel}"

    if _dataset_present; then
      echo "[data-fetch] rank=0 preprocessed C4 already present at ${PREPROCESSED_PATH}"
    else
      local dataset_uri="${MLPERF_DATASET_URI:-https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri}"
      local downloader_url="${MLPERF_R2_DOWNLOADER:-https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh}"
      echo "[data-fetch] rank=0 downloading MLCommons preprocessed C4 dataset -> ${PREPROCESSED_PATH}"
      echo "[data-fetch] URI=${dataset_uri}"
      rm -f "${dataset_sentinel}"
      # Fetch the downloader to a file first: `bash <(curl ...)` hides a failed
      # curl behind bash's exit 0 (an empty script), which `set -e` accepts and
      # which would publish the sentinel below over a missing dataset.
      local downloader
      downloader="$(mktemp)"
      if ! curl -fsSL "${downloader_url}" -o "${downloader}" || [[ ! -s "${downloader}" ]]; then
        echo "[data-fetch] ERROR: could not fetch the MLCommons downloader from ${downloader_url}"
        rm -f "${downloader}"
        return 1
      fi
      (
        cd "${PREPROCESSED_PATH}"
        bash "${downloader}" -d "${PREPROCESSED_PATH}" "${dataset_uri}"
      )
      rm -f "${downloader}"
      if ! _dataset_present; then
        echo "[data-fetch] ERROR: downloader finished but no C4 shards under ${PREPROCESSED_PATH}"
        return 1
      fi
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${dataset_sentinel}.tmp"
    mv -f "${dataset_sentinel}.tmp" "${dataset_sentinel}"
    echo "[data-fetch] rank=0 published sentinels: ${tokenizer_sentinel} ${dataset_sentinel}"
  else
    _wait_for_sentinel "${tokenizer_sentinel}" _tokenizer_present tokenizer || return 1
    _wait_for_sentinel "${dataset_sentinel}" _dataset_present "preprocessed C4" || return 1
  fi
}

validate_runtime_env() {
  if [[ "${MLPERF_INSTALL_RUNTIME_DEPS:-0}" == "1" ]]; then
    echo "Runtime dependency installation is no longer supported."
    echo "Rebuild the Docker image so all Python dependencies are present at build time."
    return 1
  fi

  (
  cd "${REPO_ROOT}"
  python3 - <<'PY'
import lightning
import nemo
import nemo_run
import transformers
import sentencepiece
import h5py
import ijson
import wget
import mlperf_logging
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

print(f"torch.__version__={torch.__version__}")
print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"IS_HIP_EXTENSION={IS_HIP_EXTENSION}")
print(f"lightning.__version__={lightning.__version__}")
print(f"nemo.__version__={getattr(nemo, '__version__', 'unknown')}")
print(f"nemo_run.__version__={getattr(nemo_run, '__version__', 'unknown')}")
print(f"transformers.__version__={transformers.__version__}")
if not IS_HIP_EXTENSION:
    raise RuntimeError("ROCm PyTorch extension support is unavailable in the container runtime")
PY
  )
}

git checkout "${MLPERF_TRAINING_REF}"
git show --oneline -s

case "${MODEL_REPO}" in
  pyt_mlperf_training_llama-3.1-8b)
    MODEL_NAME="Llama-3.1-8B"
    UPSTREAM_DIR="small_llm_pretraining/nemo"
    SIZE="8b"
    TARGET_LOG_PPL="${MLPERF_TARGET_LOG_PPL:-3.3}"
    GBS="${MLPERF_GBS:-32}"
    MBS="${MLPERF_MBS:-4}"
    MAX_STEPS="${MLPERF_MAX_STEPS:-16}"
    WARMUP_STEPS="${MLPERF_WARMUP_STEPS:-1}"
    EVAL_EVERY="${MLPERF_EVAL_EVERY:-12288}"
    START_EVAL_AT="${MLPERF_START_EVAL_AT:-0}"
    STEP_TIME_ATOL="${MLPERF_STEP_TIME_ATOL:-18000}"
    ;;
  pyt_mlperf_training_llama-3.1-405b)
    MODEL_NAME="Llama-3.1-405B"
    write_results "${MODEL_NAME}" "1" "foundation_only" "${MLPERF_TRAINING_REF}" "training_ref"
    set +x
    exit 0
    ;;
  *)
    echo "Unsupported model repo: ${MODEL_REPO}"
    usage
    ;;
esac

if [[ ! -d "${UPSTREAM_DIR}" ]]; then
  echo "Expected upstream path not found: ${UPSTREAM_DIR}"
  exit 1
fi

JOB_ROOT="${MLPERF_JOB_ROOT:-${REPO_ROOT}/mlperf_jobs}"
NODE_SUFFIX="node_${RANK:-${NODE_RANK:-0}}"
JOB_DIR="${JOB_ROOT}/${MODEL_REPO}/${NODE_SUFFIX}"
mkdir -p "${JOB_DIR}"

# Upstream callbacks open the MLPerf log file at import time.
# Prepare the absolute paths they expect before invoking Python.
mkdir -p /mlperf-outputs /output

DATA_ROOT="${MLPERF_DATA_ROOT:-${MAD_DATAHOME:-${REPO_ROOT}/mlperf_data}}"
# Resolution order for dataset-related paths:
#   1. MLPERF_* prefixed env (explicit, manifest-level override).
#   2. Bare PREPROCESSED_PATH / TOKENIZER_PATH / CONTINUAL_CKPT env vars, which
#      let mad.env + docker_env_vars route data to a shared filesystem so the
#      expensive MLCommons download happens once across all nodes.
#   3. DATA_ROOT-derived defaults (single-node / local dev fallback).
PREPROCESSED_PATH="${MLPERF_PREPROCESSED_PATH:-${PREPROCESSED_PATH:-${DATA_ROOT}/llama3_1_8b/preprocessed}}"
TOKENIZER_PATH="${MLPERF_TOKENIZER_PATH:-${TOKENIZER_PATH:-${DATA_ROOT}/llama3_1_8b/tokenizer}}"
CONTINUAL_CKPT="${MLPERF_CONTINUAL_CKPT:-${CONTINUAL_CKPT:-${DATA_ROOT}/llama3_1_8b/checkpoints}}"
TMP_NPY_INDEX="${MLPERF_TMP_NPY_INDEX:-${TMP_NPY_INDEX:-${DATA_ROOT}/llama3_1_8b/npy_index}}"
echo "[data-paths] PREPROCESSED_PATH=${PREPROCESSED_PATH}"
echo "[data-paths] TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "[data-paths] CONTINUAL_CKPT=${CONTINUAL_CKPT}"
echo "[data-paths] TMP_NPY_INDEX=${TMP_NPY_INDEX}"

mkdir -p "${PREPROCESSED_PATH}" "${TOKENIZER_PATH}" "${CONTINUAL_CKPT}" "${TMP_NPY_INDEX}"

USER_NAME="${USER:-mlperf}"
HOST_NAME="${MLPERF_HOST:-localhost}"
ACCOUNT_NAME="${MLPERF_ACCOUNT:-${SLURM_JOB_ACCOUNT:-local}}"
PARTITION_NAME="${MLPERF_PARTITION:-${SLURM_JOB_PARTITION:-local}}"
TIME_LIMIT="${MLPERF_TIME_LIMIT:-${SLURM_TIMELIMIT:-04:00:00}}"
NNODES="${NNODES:-1}"

# GPU-per-node resolution:
#   1. Explicit manifest override wins (MLPERF_GPUS_PER_NODE).
#   2. Autodetect the GPUs actually usable by this process, which is
#      authoritative regardless of what NPROC_PER_NODE contains.
#   3. Legacy env fallbacks (kept for single-node/local runs).
detect_local_gpu_count() {
  # torch.cuda.device_count() respects HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
  # and returns the number of GPUs actually available to the process, unlike
  # `/dev/dri/renderD*` which exposes every host render node (including ones
  # not passed via docker --device=).
  local count
  count=$(python3 - <<'PY' 2>/dev/null
import os
try:
    import torch
    n = torch.cuda.device_count()
except Exception:
    n = 0
if n <= 0:
    vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES") or ""
    if vis:
        n = len([v for v in vis.split(",") if v.strip()])
print(int(n))
PY
)
  count=$(echo "${count}" | tr -d '[:space:]')
  if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
    echo "${count}"
    return 0
  fi
  return 1
}

if [[ -n "${MLPERF_GPUS_PER_NODE:-}" ]]; then
  GPUS_PER_NODE="${MLPERF_GPUS_PER_NODE}"
  echo "[gpu-detect] using MLPERF_GPUS_PER_NODE override = ${GPUS_PER_NODE}"
elif DETECTED_GPUS=$(detect_local_gpu_count); then
  GPUS_PER_NODE="${DETECTED_GPUS}"
  echo "[gpu-detect] detected ${GPUS_PER_NODE} local GPU(s) via torch.cuda.device_count()"
  if [[ -n "${NPROC_PER_NODE:-}" && "${NPROC_PER_NODE}" != "${GPUS_PER_NODE}" ]]; then
    echo "[gpu-detect] WARNING: env NPROC_PER_NODE=${NPROC_PER_NODE} differs from detected ${GPUS_PER_NODE}; using detected value."
  fi
else
  GPUS_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-${MAD_RUNTIME_NGPUS:-8}}}"
  echo "[gpu-detect] autodetect unavailable, falling back to GPUS_PER_NODE=${GPUS_PER_NODE}"
fi
IMAGE_NAME="${MLPERF_IMAGE_NAME:-${MAD_CONTAINER_IMAGE:-${MODEL_REPO}:local}}"
REMOTE_SUBMIT="${MLPERF_REMOTE_SUBMIT:-0}"

ENVVARS=""
append_envvar "MASTER_ADDR" "${MASTER_ADDR:-}"
append_envvar "MASTER_PORT" "${MASTER_PORT:-}"
append_envvar "WORLD_SIZE" "${WORLD_SIZE:-}"
append_envvar "RANK" "${RANK:-}"
append_envvar "NODE_RANK" "${NODE_RANK:-}"
append_envvar "NNODES" "${NNODES}"
append_envvar "NPROC_PER_NODE" "${GPUS_PER_NODE}"
append_envvar "NCCL_SOCKET_IFNAME" "${NCCL_SOCKET_IFNAME:-}"
append_envvar "GLOO_SOCKET_IFNAME" "${GLOO_SOCKET_IFNAME:-}"
append_envvar "NCCL_IB_DISABLE" "${NCCL_IB_DISABLE:-}"
append_envvar "NCCL_IB_HCA" "${NCCL_IB_HCA:-}"
append_envvar "NCCL_IB_GID_INDEX" "${NCCL_IB_GID_INDEX:-}"
append_envvar "MIOPEN_USER_DB_PATH" "${MIOPEN_USER_DB_PATH:-}"

MOUNTS="${JOB_DIR}:/output,${JOB_DIR}:/mlperf-outputs,${PREPROCESSED_PATH}:/preproc_data,${TOKENIZER_PATH}:/tokenizer,${CONTINUAL_CKPT}:/continual,${TMP_NPY_INDEX}:/npy_index"

COMMON_ARGS=(
  --user "${USER_NAME}"
  --host "${HOST_NAME}"
  --job_dir "${JOB_DIR}"
  --account "${ACCOUNT_NAME}"
  --partition "${PARTITION_NAME}"
  --nodes "${NNODES}"
  --gpus_per_node "${GPUS_PER_NODE}"
  --time "${TIME_LIMIT}"
  --mounts "${MOUNTS}"
  --image "${IMAGE_NAME}"
  --size "${SIZE}"
  --gbs "${GBS}"
  --mbs "${MBS}"
  --max_steps "${MAX_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --eval_every "${EVAL_EVERY}"
  --start_eval_at "${START_EVAL_AT}"
  --continual_ckpt_path "${CONTINUAL_CKPT}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --target_log_ppl "${TARGET_LOG_PPL}"
  --step_time_atol "${STEP_TIME_ATOL}"
)

if [[ -n "${ENVVARS}" ]]; then
  COMMON_ARGS+=(--envvars "${ENVVARS}")
fi

ensure_mlperf_data
validate_runtime_env
cd "${UPSTREAM_DIR}"

# Upstream pretrain_llama31.py reads GBS and PREPROCESSED_PATH via os.getenv
# (see get_pretrain / get_data). Export them so both dryrun/local and
# upstream_slurm modes have them available at import/execution time.
export GBS="${GBS}"
export MBS="${MBS}"
export SIZE="${SIZE}"
export MAX_STEPS="${MAX_STEPS}"
export WARMUP_STEPS="${WARMUP_STEPS}"
export EVAL_EVERY="${EVAL_EVERY}"
export START_EVAL_AT="${START_EVAL_AT}"
export PREPROCESSED_PATH="${PREPROCESSED_PATH}"
export TOKENIZER_PATH="${TOKENIZER_PATH}"
export CONTINUAL_CKPT="${CONTINUAL_CKPT}"

if [[ "${MLPERF_EXECUTION_MODE}" == "dryrun" ]]; then
  python3 pretrain_llama31.py "${COMMON_ARGS[@]}" --dryrun | tee "${RUN_LOG}"
  write_results "${MODEL_NAME}" "1" "dryrun_success" "${NNODES}" "requested_nodes" "${GPUS_PER_NODE}" "gpus_per_node"
elif [[ "${MLPERF_EXECUTION_MODE}" == "local" ]]; then
  if [[ "${NNODES}" != "1" ]]; then
    echo "Execution mode 'local' only supports NNODES=1 for this integration."
    exit 1
  fi
  python3 pretrain_llama31.py "${COMMON_ARGS[@]}" | tee "${RUN_LOG}"
  write_results "${MODEL_NAME}" "1" "local_launch_success" "${NNODES}" "requested_nodes" "${GPUS_PER_NODE}" "gpus_per_node"
elif [[ "${MLPERF_EXECUTION_MODE}" == "torchrun_in_alloc" ]]; then
  # Real multi-node training inside an existing madengine Slurm allocation.
  # Upstream's NeMo-Run LocalExecutor is single-node and SlurmExecutor submits
  # a fresh sbatch, neither of which fits an already-allocated job. Instead,
  # we bypass NeMo-Run's Experiment and drive the pretrain recipe directly
  # from torchrun via our in-process entrypoint (sibling to pretrain_llama31.py).
  : "${MASTER_ADDR:?MASTER_ADDR required for torchrun_in_alloc}"
  : "${MASTER_PORT:=29500}"
  NODE_RANK_VAL="${NODE_RANK:-${RANK:-0}}"
  ENTRYPOINT_SRC="${SCRIPT_DIR}/mlperf_pretrain_entrypoint.py"
  # We already cd'd into UPSTREAM_DIR above, so the destination is the current
  # working directory; do not prepend UPSTREAM_DIR again.
  ENTRYPOINT_DST="./mlperf_pretrain_entrypoint.py"
  if [[ ! -f "${ENTRYPOINT_SRC}" ]]; then
    echo "Missing entrypoint: ${ENTRYPOINT_SRC}"
    exit 1
  fi
  cp -f "${ENTRYPOINT_SRC}" "${ENTRYPOINT_DST}"
  # data_index_sentinel MUST live on a shared filesystem so that rank 0 can
  # publish completion visible to ranks on other nodes. TMP_NPY_INDEX is
  # routed to ``/shared_inference/...`` via manifest docker_mounts, which is
  # the same path mounted into every container; per-container paths like
  # ``/mlperf-outputs`` (the old default) are NOT shared across nodes and
  # produced a 1-hour wait-then-timeout on node_1 (ranks 8..15).
  # Include SLURM_JOB_ID to avoid stale sentinels from previous jobs.
  DATA_INDEX_SENTINEL_DIR="${TMP_NPY_INDEX}"
  mkdir -p "${DATA_INDEX_SENTINEL_DIR}"
  DATA_INDEX_SENTINEL="${DATA_INDEX_SENTINEL_DIR}/.data_index_done_${SLURM_JOB_ID:-local}"
  echo "[data-index] sentinel=${DATA_INDEX_SENTINEL}"
  # Seed the results CSV with a launch-success marker so MAD still sees a CSV
  # even if the run aborts before extract_perf.py gets a chance to write real
  # metrics. Node 0's post-run extract_perf.py invocation overwrites this.
  write_results "${MODEL_NAME}" "1" "torchrun_launch_success" "${NNODES}" "requested_nodes" "${GPUS_PER_NODE}" "gpus_per_node"
  TORCHRUN_EXIT=0
  python3 -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK_VAL}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    mlperf_pretrain_entrypoint.py \
      --size "${SIZE}" \
      --gbs "${GBS}" \
      --mbs "${MBS}" \
      --max_steps "${MAX_STEPS}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --eval_every "${EVAL_EVERY}" \
      --start_eval_at "${START_EVAL_AT}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --continual_ckpt_path "${CONTINUAL_CKPT}" \
      --target_log_ppl "${TARGET_LOG_PPL}" \
      --step_time_atol "${STEP_TIME_ATOL}" \
      --data_index_sentinel "${DATA_INDEX_SENTINEL}" \
    | tee "${RUN_LOG}" || TORCHRUN_EXIT=$?

  # MAD only scrapes the perf CSV from node_0 workdir, so the extraction/write
  # is a node-0 concern. The RUN_LOG on node_0 aggregates stdout for local
  # ranks 0..GPUS_PER_NODE-1, which includes rank-0 and therefore all
  # ``:::MLLOG`` entries emitted by the mlperf logger.
  if [[ "${NODE_RANK_VAL}" == "0" ]]; then
    EXTRACT_PERF="${SCRIPT_DIR}/extract_perf.py"
    if [[ -f "${EXTRACT_PERF}" && -f "${RUN_LOG}" ]]; then
      echo "[perf] extracting metrics from ${RUN_LOG} -> ${RESULT_CSV}"
      TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
      # extract_perf.py defaults to the MI325X BF16 peak, so mfu_pct is only
      # meaningful when the accelerator's own peak is supplied.
      PEAK_ARGS=()
      if [[ -n "${MLPERF_PEAK_BF16_TFLOPS:-}" ]]; then
        PEAK_ARGS=(--peak-bf16-tflops "${MLPERF_PEAK_BF16_TFLOPS}")
      fi
      if ! python3 "${EXTRACT_PERF}" "${RUN_LOG}" \
          --n-gpus "${TOTAL_GPUS}" \
          --size "${SIZE}" \
          "${PEAK_ARGS[@]}" \
          --csv-out "${RESULT_CSV}" \
          --csv-model-name "${MODEL_NAME}" \
          --csv-nodes "${NNODES}" \
          --csv-gpus-per-node "${GPUS_PER_NODE}"; then
        echo "[perf] extract_perf.py failed; keeping torchrun_launch_success marker"
      fi
    else
      echo "[perf] extract_perf.py or run log not found; keeping default perf CSV"
    fi
  fi

  if [[ "${TORCHRUN_EXIT}" != "0" ]]; then
    echo "torchrun exited with ${TORCHRUN_EXIT}"
    exit "${TORCHRUN_EXIT}"
  fi
elif [[ "${MLPERF_EXECUTION_MODE}" == "upstream_slurm" ]]; then
  export USER="${USER_NAME}"
  export HOST="${HOST_NAME}"
  export ACCOUNT="${ACCOUNT_NAME}"
  export PARTITION="${PARTITION_NAME}"
  export TIME="${TIME_LIMIT}"
  export JOB_DIR="${JOB_DIR}"
  export IMAGE="${IMAGE_NAME}"
  export USE_CKPT="${MLPERF_USE_CKPT:-0}"
  export FROM_HF="${MLPERF_FROM_HF:-1}"
  export SAVE_CKPT="${MLPERF_SAVE_CKPT:-0}"
  export NNODES="${NNODES}"
  export GPUS_PER_NODE="${GPUS_PER_NODE}"
  export REMOTE="${REMOTE_SUBMIT}"
  bash run_llama31.sh | tee "${RUN_LOG}"
  write_results "${MODEL_NAME}" "1" "upstream_slurm_submit_success" "${NNODES}" "requested_nodes" "${GPUS_PER_NODE}" "gpus_per_node"
else
  echo "Unsupported MLPERF_EXECUTION_MODE=${MLPERF_EXECUTION_MODE}"
  exit 1
fi

set +x
