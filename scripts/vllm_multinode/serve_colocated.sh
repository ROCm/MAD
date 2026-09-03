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
    # Separate field from the one above: --distributed-timeout-seconds only
    # reaches the device/NCCL groups, while the startup barrier that killed
    # job 239755 runs on the gloo CPU group and takes its deadline from this
    # one. Unset, both fall back to PyTorch's stock 1800s.
    --cpu-distributed-timeout-seconds "${CPU_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
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
# Checkpoint page-cache pre-warm. OFF by default -- it made things worse here.
# The idea was to even out the cross-node gap that breaks vLLM's all-or-nothing
# startup barrier by faulting the checkpoint in before it. In practice job 240624
# never got past this block: both nodes were still reading at the 7200s pipeline
# timeout, so vllm serve never launched at all.
# The flaw is that this reads the WHOLE checkpoint on EVERY node, while PP2xTP8
# means each node only ever loads its own shard. On a 1453 GiB checkpoint that is
# ~4x the necessary I/O against one NFS export, with both nodes competing for it
# -- self-inflicted contention, which is the very thing it was meant to relieve.
# Left in, opt-in, because the per-node duration is still a useful storage probe:
# a large spread between nodes means the storage path is the problem and no
# timeout will fix it. Bounded by PREWARM_TIMEOUT_SECONDS so it can never eat the
# job's budget again -- on expiry it warns and proceeds rather than burning the
# whole allocation.
# Only sound while the checkpoint fits in RAM (1453 GiB of 1838 GiB here).
# -----------------------------------------------------------------------------
if [[ "${PREWARM_CHECKPOINT:-0}" == "1" ]]; then
    _warm_start=${SECONDS}
    _warm_budget="${PREWARM_TIMEOUT_SECONDS:-900}"
    echo "[prewarm] reading ${MODEL_PATH} into page cache on $(hostname -s) (budget ${_warm_budget}s)"
    if timeout "${_warm_budget}" bash -c '
        find -L "$1" -name "*.safetensors" -print0 2>/dev/null \
            | xargs -0 -r -P "$2" cat >/dev/null 2>&1
    ' _ "${MODEL_PATH}" "${PREWARM_JOBS:-4}"; then
        echo "[prewarm] done on $(hostname -s) in $(( SECONDS - _warm_start ))s"
    else
        echo "[prewarm] BUDGET EXPIRED on $(hostname -s) after $(( SECONDS - _warm_start ))s; continuing anyway." >&2
        echo "[prewarm] storage delivered less than the whole checkpoint in that window - the shared filesystem is the bottleneck." >&2
    fi
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

# Redirect through process substitution rather than a pipeline: after `a | b &`
# the shell reports b's PID, so WORKER_PID used to be tee's. Every `kill` below
# then hit tee and left vLLM running, and no liveness check was possible.
vllm serve "${MODEL_PATH}" "${serve_args[@]}" > >(tee "${LOG}" >/dev/null) 2>&1 &
WORKER_PID=$!

# `kill -0` alone is not enough: a child that has exited but not been reaped is a
# zombie and still answers signal 0, so a dead engine would read as alive forever.
_alive() {
    kill -0 "$1" 2>/dev/null || return 1
    case "$(ps -o stat= -p "$1" 2>/dev/null)" in
        *Z*) return 1 ;;
    esac
    return 0
}

# Shutdown sentinel on the shared log volume (/run_logs is $LOG_PATH on the host,
# visible from every node). The head writes it when it is done; workers watch for
# it. Without this the job DEADLOCKS: the head finishes the benchmark, kills only
# its own server and exits, while each worker sits in `wait` on a headless vLLM
# that nothing ever stops. srun then waits on those tasks until the wall clock --
# observed as 4.5 hours of two idle exclusive nodes after results were written.
SHUTDOWN_FLAG="/run_logs/${SLURM_JOB_ID:-0}/.shutdown"

if [[ "${NODE_RANK}" -ne 0 ]]; then
    # Workers have no API server: hold until the head tears the job down.
    echo "[colocated] worker ${NODE_RANK} serving headless; log ${LOG}"
    while _alive "${WORKER_PID}"; do
        if [ -f "${SHUTDOWN_FLAG}" ]; then
            echo "[colocated] worker ${NODE_RANK} got shutdown signal; stopping"
            pkill -P "${WORKER_PID}" 2>/dev/null
            kill "${WORKER_PID}" 2>/dev/null || true
            break
        fi
        sleep 5
    done
    wait "${WORKER_PID}" 2>/dev/null || true
    exit 0
fi

# Raise the sentinel however the head leaves -- including on error or timeout,
# so a failed head cannot strand the workers either.
trap 'touch "${SHUTDOWN_FLAG}" 2>/dev/null || true' EXIT

# ---- head: wait for readiness, benchmark, then shut down -------------------
echo "[colocated] head waiting for 'Application startup complete.' in ${LOG}"
_TIMEOUT="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"; _elapsed=0
until grep -Fq "Application startup complete." "${LOG}" 2>/dev/null; do
    # A dead engine used to be indistinguishable from a slow one: job 239755's
    # head sat here for the full 4000s polling a log whose writer had already
    # died 20 minutes earlier, and reported a timeout instead of the traceback.
    if ! _alive "${WORKER_PID}"; then
        echo "[colocated] vllm serve exited after ${_elapsed}s without becoming ready. Tail:" >&2
        tail -80 "${LOG}" >&2
        exit 1
    fi
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
#                       Only consulted on the LEGACY full-schema reporting path
#                       (BENCHMARK_SCRIPT=sweep). With BENCHMARK_SCRIPT=niah the CSV
#                       is narrow and madengine supplies these fields itself.
export NIAH_MODEL="${MODEL_NAME:-model}"
_ep_tag="$([ "${ENABLE_EP}" = "1" ] && echo "ep_${ALL2ALL_BACKEND:-default}" || echo "noep")"
export PERF_DEPLOYMENT_TYPE="${PERF_DEPLOYMENT_TYPE:-colocated_pp${PP_SIZE}xtp${TP_SIZE}}"
export PERF_TAGS="${PERF_TAGS:-vllm_multinode,colocated,${_ep_tag}}"

bash "${SHARED_DIR}/${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}"

echo "[colocated] benchmark complete; stopping server"
pkill -P "${WORKER_PID}" 2>/dev/null; kill "${WORKER_PID}" 2>/dev/null || true
exit 0
