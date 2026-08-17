#!/bin/bash
# Colocated (single-instance) multi-node vLLM serve — per-node entry point.
# =============================================================================
# Runs INSIDE the container, one process per node, launched by run_multinode.slurm.
#
# "Colocated" = one vLLM instance whose parallelism spans several nodes, with NO
# prefill/decode disaggregation and no KV connector. It is the other half of the
# multi-node story from scripts/vllm_dissag/: use this for lowest single-request
# latency (one request uses every GPU), use vllm_dissag for concurrent throughput.
#
# Node roles (by NODE_RANK):
#   0        -> head: serves the OpenAI API on SERVE_PORT, then benchmarks
#   1..N-1   -> worker: --headless, no API server
#
# The per-model recipe (env:) is read from the SAME scripts/vllm_dissag/models.yaml
# the disagg harness uses, so a model's gfx/quant/KV knobs have one home. Serve
# flags specific to a colocated variant come from COLOCATED_EXTRA_ARGS.
#
# Required env: MODEL_PATH, NODE_RANK, NNODES, MASTER_ADDR, IPADDRS
# Optional:     MODEL_NAME, PP_SIZE, TP_SIZE, ENABLE_EP, ALL2ALL_BACKEND,
#               SERVE_PORT, GPU_MEMORY_UTILIZATION, COLOCATED_EXTRA_ARGS,
#               BENCHMARK_SCRIPT_FILE, DRY_RUN
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SHARED_DIR="${SHARED_DIR:-${SCRIPT_DIR}/../vllm_dissag}"

: "${MODEL_PATH:?MODEL_PATH must be set (path to the model dir)}"
MODEL_NAME="${MODEL_NAME:-}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
IPADDRS="${IPADDRS:-localhost}"
SERVE_PORT="${SERVE_PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Parallelism. Default is the natural colocated shape: TP within a node, PP across
# nodes (PP_SIZE = NNODES). Expert parallelism is opt-in; when on, the EP group is
# the TP group inside each node, so the expert all2all stays intra-node and the
# only cross-node traffic is the PP activation hand-off.
TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"
PP_SIZE="${PP_SIZE:-${NNODES}}"
ENABLE_EP="${ENABLE_EP:-0}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-}"

echo "[colocated] node=$(hostname -s) rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}"
echo "[colocated] TP=${TP_SIZE} PP=${PP_SIZE} EP=${ENABLE_EP} all2all=${ALL2ALL_BACKEND:-<none>}"

# -----------------------------------------------------------------------------
# Per-model env from the shared models.yaml (same precedence rule as the disagg
# driver: connector default < models.yaml env: < submit-time -e).
# -----------------------------------------------------------------------------
MODELS_YAML="${MODELS_YAML:-${SHARED_DIR}/models.yaml}"
if [[ -n "$MODEL_NAME" && -f "$MODELS_YAML" ]]; then
    export MODELS_YAML MODEL_NAME
    _yaml_env="$(python3 - <<'PY'
import os, yaml, shlex
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
cfg = m.get(os.environ["MODEL_NAME"]) or {}
for k, v in (cfg.get("env") or {}).items():
    if k in os.environ:   # submit-time -e wins
        continue
    print(f'export {k}={shlex.quote(str(v))}')
PY
)"
    [[ -n "$_yaml_env" ]] && eval "$_yaml_env"
    echo "[colocated] applied models.yaml env for '${MODEL_NAME}'"
fi

# Recipe knobs that models.yaml may have supplied.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# -----------------------------------------------------------------------------
# Assemble argv
# -----------------------------------------------------------------------------
serve_args=(
    --served-model-name "${MODEL_NAME:-model}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --trust-remote-code
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
)
[[ -n "${KV_CACHE_DTYPE:-}" ]]        && serve_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[[ -n "${KV_BLOCK_SIZE:-}" ]]         && serve_args+=(--block-size "${KV_BLOCK_SIZE}")
[[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]] && serve_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
if [[ "${ENABLE_EP}" == "1" ]]; then
    serve_args+=(--enable-expert-parallel)
    [[ -n "${ALL2ALL_BACKEND}" ]] && serve_args+=(--all2all-backend "${ALL2ALL_BACKEND}")
fi
if [[ "${NODE_RANK}" -eq 0 ]]; then
    serve_args+=(--port "${SERVE_PORT}")
else
    serve_args+=(--headless)
fi
# Per-variant serve flags (reasoning parser, max-model-len, quantization-config...).
# Word-split like the disagg driver does, so JSON values must carry their own quotes.
if [[ -n "${COLOCATED_EXTRA_ARGS:-}" ]]; then
    eval "extra_args=(${COLOCATED_EXTRA_ARGS})"
    serve_args+=("${extra_args[@]}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "===DRYRUN colocated role=$([ "$NODE_RANK" -eq 0 ] && echo head || echo worker) NODE_RANK=${NODE_RANK}==="
    printf '%s\n' vllm serve "${MODEL_PATH}" "${serve_args[@]}"
    echo "===END==="
    exit 0
fi

# -----------------------------------------------------------------------------
# Container barrier — every node's container must exist before any rank dials the
# master, otherwise early ranks burn their connect retries against a dead port.
# Reuses the disagg harness's barrier rather than re-implementing a sleep.
# -----------------------------------------------------------------------------
_BARRIER_PORT="${CONTAINER_BARRIER_PORT:-2223}"
host_ip=$(hostname -I | awk '{print $1}')
if [[ -f "${SHARED_DIR}/socket_barrier.py" ]]; then
    echo "[colocated] waiting at container barrier on $(hostname -s)"
    python3 "${SHARED_DIR}/socket_barrier.py" \
        --local-ip "${host_ip}" --local-port "${_BARRIER_PORT}" --enable-port \
        --node-ips "${IPADDRS}" --node-ports "${_BARRIER_PORT}"
fi

mkdir -p "/run_logs/${SLURM_JOB_ID:-0}"
LOG="/run_logs/${SLURM_JOB_ID:-0}/colocated_NODE${NODE_RANK}.log"

vllm serve "${MODEL_PATH}" "${serve_args[@]}" 2>&1 | tee "${LOG}" >/dev/null &
WORKER_PID=$!

if [[ "${NODE_RANK}" -ne 0 ]]; then
    # Workers have no API server: hold until the head tears the job down.
    echo "[colocated] worker ${NODE_RANK} serving headless; log ${LOG}"
    wait "${WORKER_PID}"
    exit 0
fi

# ---- head: wait for readiness, benchmark, then shut down -------------------
echo "[colocated] head waiting for 'Application startup complete.' in ${LOG}"
_TIMEOUT="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"; _elapsed=0
until grep -Fq "Application startup complete." "${LOG}" 2>/dev/null; do
    if [ "${_elapsed}" -ge "${_TIMEOUT}" ]; then
        echo "[colocated] TIMEOUT (${_TIMEOUT}s): server never became ready. Tail:" >&2
        tail -40 "${LOG}" >&2
        kill "${WORKER_PID}" 2>/dev/null
        exit 1
    fi
    sleep 10; _elapsed=$((_elapsed + 10))
done
echo "[colocated] server ready after ${_elapsed}s"

# Same benchmark scripts + CSV parser as the disagg harness. xP/yD are only used
# in their log filenames; a colocated run reports as 1 instance, 0 decode pools.
export BENCHMARK_PORT="${SERVE_PORT}"
export xP="${xP:-1}" yD="${yD:-0}"

# The shared harness assumes the disagg launcher's conventions. Override the two
# that do not hold here, so a colocated run is benchmarked and reported as itself:
#
#   NIAH_MODEL          we pass --served-model-name, so the served tag is MODEL_NAME,
#                       not MODEL_PATH. Without this every NIAH request 404s.
#   PERF_DEPLOYMENT_TYPE/PERF_TAGS
#                       xP/yD above are filename placeholders, not a topology; left
#                       alone the CSV would label this 2-node run `disagg_1P0D`.
export NIAH_MODEL="${MODEL_NAME:-model}"
_ep_tag="$([ "${ENABLE_EP}" = "1" ] && echo "ep_${ALL2ALL_BACKEND:-default}" || echo "noep")"
export PERF_DEPLOYMENT_TYPE="${PERF_DEPLOYMENT_TYPE:-colocated_pp${PP_SIZE}xtp${TP_SIZE}}"
export PERF_TAGS="${PERF_TAGS:-vllm_multinode,colocated,${_ep_tag}}"

bash "${SHARED_DIR}/${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}"

echo "[colocated] benchmark complete; stopping server"
pkill -P "${WORKER_PID}" 2>/dev/null; kill "${WORKER_PID}" 2>/dev/null || true
exit 0
