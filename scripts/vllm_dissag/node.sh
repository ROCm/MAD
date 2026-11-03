#!/bin/bash
# One node of a vLLM disaggregated run: started once per node by run_xPyD_models.slurm
# (through the dispatcher, common/dispatch/dispatch.sh: srun, a job-array task or a pod),
# with SLURM_PROCID as the node rank and the launcher's
# exported settings as its environment. Starts this node's container and exits
# with the container's status.
. "$(dirname "${BASH_SOURCE[0]}")/../common/node/node_lib.sh"
echo "Rank $SLURM_PROCID on $(hostname)";
# Leftover containers of this launcher (any job) only, by name; any other container
# on the node is not ours to stop. A node whose GPUs are still held fails the start-up
# GPU check with the reason (vllm_disagg.sh _check_gpus_clean).
docker ps -q --filter "name=^container_" | xargs --no-run-if-empty docker stop;
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true;
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true;

# --- Create host-local compilation cache dirs (ext4, survives container restarts) ---
mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true;

# --- Persistent JIT cache (node_jit_cache) ---
# TP-within-EP (EP_TP_SIZE>1): prefill and decode compile different AITER kernel
# variants, so they get separate caches.
_JIT_ROLE=""
if (( ${EP_TP_SIZE:-1} > 1 )) && [ "${JIT_CACHE_SPLIT_ROLE:-1}" = "1" ]; then
    if [ "${SLURM_PROCID}" -lt "${xP}" ]; then _JIT_ROLE="prefill"; else _JIT_ROLE="decode"; fi
fi
node_jit_cache disagg "$_JIT_ROLE"

# --- Host RDMA library mounts (node_rdma_mounts) ---
node_rdma_mounts
echo "[host-rdma] mounts: $_RDMA_MOUNTS"
# AGENTIC_CONFIG may use $HOME (e.g. from models.json); expand at submit time.
[ -n "${AGENTIC_CONFIG:-}" ] && AGENTIC_CONFIG="${AGENTIC_CONFIG/#$HOME/$HOME}"

# GPU_MEMORY_UTILIZATION precedence:
#   1. Submit-time `-e GPU_MEMORY_UTILIZATION=X` (explicit override)
#   2. models.yaml env: block (static per-model value)
#   3. Launcher fallback below (topology-aware: 0.70 at EP>=32, else 0.8)
#
# Models needing topology-dependent values (e.g., GLM) should OMIT the key from
# yaml and use the fallback. Models needing a static value can specify it in yaml.
# The 0.70 at EP>=32 accounts for larger MoRI dispatch/combine buffers at high EP
# width; measured on GLM but applies to any MoRI model as a safe default.
_EP_WIDTH=$(( yD * ${GPUS_PER_NODE:-8} ))
if [ "$_EP_WIDTH" -ge 32 ]; then _GPU_MEM_UTIL_LAUNCHER_DEFAULT=0.70; else _GPU_MEM_UTIL_LAUNCHER_DEFAULT=0.8; fi
echo "GPU_MEMORY_UTILIZATION launcher fallback: ${_GPU_MEM_UTIL_LAUNCHER_DEFAULT} (decode EP width ${_EP_WIDTH})"

# Stops the GPU memory sampler (below) when the container ends or the task is signalled.
_stop_trace() { [ -n "$_trace_pid" ] && kill "$_trace_pid" 2>/dev/null; }
node_site_mounts
# This node's part of the run (topology.py), passed in as MAD_TOPO_*.
node_topology vllm_dissag
# Any further variables the run needs inside the container, by name: CONTAINER_ENV is a
# comma- or space-separated list of names, each passed as a bare `-e NAME`, so docker
# takes the value from this environment (values with spaces or quotes pass unchanged).
# For settings no launcher knob covers: the env vars a request names, or debug switches.
_EXTRA_ENV=""
for _v in ${CONTAINER_ENV//,/ }; do _EXTRA_ENV="$_EXTRA_ENV -e $_v"; done
[ -n "$_EXTRA_ENV" ] && echo "[env] passed into the container:${_EXTRA_ENV}"
# Which processes take GPU memory while the servers start, sampled from the host, where
# the kernel per-process GPU records resolve to command lines and containers (the
# container cannot see host pids). See gpu_mem_trace.sh. Written straight into LOG_PATH:
# the job directory under it is created by the container root user. Off unless
# GPU_MEM_TRACE=1 (a diagnostic for start-up memory failures); GPU_MEM_TRACE_SECS /
# GPU_MEM_TRACE_INTERVAL set its length and rate.
_trace_pid="";
if [ "${GPU_MEM_TRACE:-0}" = "1" ] && [ -f "$NIXL_REPO_DIR/gpu_mem_trace.sh" ]; then
    _trace_out="${LOG_PATH}/gpu_mem_trace_${SLURM_JOB_ID}_NODE${SLURM_PROCID}.log";
    bash "$NIXL_REPO_DIR/gpu_mem_trace.sh" "$_trace_out" </dev/null >/dev/null 2>&1 &
    _trace_pid=$!;
    echo "[gpu-mem-trace] $(hostname): GPU memory holders sampled into $_trace_out";
fi
_node_docker_run() {
docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    ${_SITE_MOUNTS} \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size ${DOCKER_SHM_SIZE:-256G} \
    --ulimit nofile=524288:524288 \
    --ulimit memlock=-1:-1 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    ${_JIT_CACHE_MOUNT} \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-unknown} \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID \
    ${_TOPO_ENV} \
    -e MASTER_ADDR=$MASTER_ADDR \
    ${ROUTER_ADDR:+-e ROUTER_ADDR=$ROUTER_ADDR} \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH \
    -e NIXL_REPO_DIR=$NIXL_REPO_DIR \
    ${AGENTIC_CONC:+-e AGENTIC_CONC=$AGENTIC_CONC} \
    ${DURATION:+-e DURATION=$DURATION} \
    ${AGENTIC_PORT:+-e AGENTIC_PORT=$AGENTIC_PORT} \
    ${MODEL_PREFIX:+-e MODEL_PREFIX=$MODEL_PREFIX} \
    ${WEKA_LOADER_OVERRIDE:+-e WEKA_LOADER_OVERRIDE=$WEKA_LOADER_OVERRIDE} \
    ${AIPERF_UNSAFE_OVERRIDE:+-e AIPERF_UNSAFE_OVERRIDE=$AIPERF_UNSAFE_OVERRIDE} \
    ${AGENTIC_CACHE_WARMUP_DURATION:+-e AGENTIC_CACHE_WARMUP_DURATION=$AGENTIC_CACHE_WARMUP_DURATION} \
    ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN=$MAX_MODEL_LEN} \
    ${AGENTIC_CONFIG:+-e AGENTIC_CONFIG=$AGENTIC_CONFIG} \
    ${AGENTIC_WORKLOAD:+-e AGENTIC_WORKLOAD=$AGENTIC_WORKLOAD} \
    ${SUITE_CORPUS_DIR:+-e SUITE_CORPUS_DIR=$SUITE_CORPUS_DIR} \
    ${SUITE_CORPUS_FORCE:+-e SUITE_CORPUS_FORCE=$SUITE_CORPUS_FORCE} \
    ${AGENTIC_STRICT_CONTEXT:+-e AGENTIC_STRICT_CONTEXT=$AGENTIC_STRICT_CONTEXT} \
    ${AGENTIC_HF_ISL_TAIL:+-e AGENTIC_HF_ISL_TAIL=$AGENTIC_HF_ISL_TAIL} \
    ${AIPERF_PIN:+-e AIPERF_PIN=$AIPERF_PIN} \
    ${AGENTIC_UTILS_PIN:+-e AGENTIC_UTILS_PIN=$AGENTIC_UTILS_PIN} \
    ${HF_HOME:+-e HF_HOME=$HF_HOME} \
    ${ENABLE_SERVER_METRICS:+-e ENABLE_SERVER_METRICS=$ENABLE_SERVER_METRICS} \
    ${ENABLE_PREFIX_CACHE:+-e ENABLE_PREFIX_CACHE=$ENABLE_PREFIX_CACHE} \
    ${DRY_RUN:+-e DRY_RUN=$DRY_RUN} \
    -e xP=$xP \
    -e yD=$yD \
    -e EP_TP_SIZE=$EP_TP_SIZE \
    ${PERF_GPU_ARCH:+-e PERF_GPU_ARCH=$PERF_GPU_ARCH} \
    ${GPUS_PER_NODE:+-e GPUS_PER_NODE=$GPUS_PER_NODE} \
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_ITR=$BENCHMARK_ITR \
    -e BENCHMARK_CON="${BENCHMARK_CON}" \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}" \
    ${BENCHMARK_PROMPTS_PER_CON:+-e BENCHMARK_PROMPTS_PER_CON=$BENCHMARK_PROMPTS_PER_CON} \
    ${BENCHMARK_MIN_PROMPTS:+-e BENCHMARK_MIN_PROMPTS=$BENCHMARK_MIN_PROMPTS} \
    ${PERF_LATENCY_METRICS:+-e PERF_LATENCY_METRICS=$PERF_LATENCY_METRICS} \
    ${STEP_TIMEOUT:+-e STEP_TIMEOUT=$STEP_TIMEOUT} \
    ${GPU_CLEAN_CHECK:+-e GPU_CLEAN_CHECK=$GPU_CLEAN_CHECK} \
    ${GPU_CLEAN_MAX_USED_GIB:+-e GPU_CLEAN_MAX_USED_GIB=$GPU_CLEAN_MAX_USED_GIB} \
    ${JIT_CACHE_HOST:+-e JIT_CACHE_HOST=$JIT_CACHE_HOST} \
    ${JIT_LOCK_STALE_MIN:+-e JIT_LOCK_STALE_MIN=$JIT_LOCK_STALE_MIN} \
    ${BENCHMARK_PORT:+-e BENCHMARK_PORT=$BENCHMARK_PORT} \
    ${PROXY_TYPE:+-e PROXY_TYPE=$PROXY_TYPE} \
    ${ROUTER_PORT:+-e ROUTER_PORT=$ROUTER_PORT} \
    -e IPADDRS=$IPADDRS \
    ${CONNECTOR:+-e CONNECTOR=$CONNECTOR} \
    ${WIDE_EP:+-e WIDE_EP=$WIDE_EP} \
    ${EP_BACKEND:+-e EP_BACKEND=$EP_BACKEND} \
    ${BARRIER_PORT:+-e BARRIER_PORT=$BARRIER_PORT} \
    ${BARRIER_PORT_MORI:+-e BARRIER_PORT_MORI=$BARRIER_PORT_MORI} \
    ${RUN_MORI:+-e RUN_MORI=$RUN_MORI} \
    ${RUN_DEEPEP:+-e RUN_DEEPEP=$RUN_DEEPEP} \
    ${VLLM_ALL2ALL_BACKEND:+-e VLLM_ALL2ALL_BACKEND=$VLLM_ALL2ALL_BACKEND} \
    ${PREFILL_MORI_BACKEND:+-e PREFILL_MORI_BACKEND=$PREFILL_MORI_BACKEND} \
    ${DECODE_MORI_BACKEND:+-e DECODE_MORI_BACKEND=$DECODE_MORI_BACKEND} \
    ${KV_BLOCK_SIZE:+-e KV_BLOCK_SIZE=$KV_BLOCK_SIZE} \
    ${KV_CACHE_MEMORY_BYTES:+-e KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES} \
    ${VLLM_ROCM_USE_AITER_MLA:+-e VLLM_ROCM_USE_AITER_MLA=$VLLM_ROCM_USE_AITER_MLA} \
    ${ROUTER_BINARY:+-e ROUTER_BINARY=$ROUTER_BINARY} \
    ${KV_CACHE_DTYPE:+-e KV_CACHE_DTYPE=$KV_CACHE_DTYPE} \
    ${MORIIO_TOY_PROXY:+-e MORIIO_TOY_PROXY=$MORIIO_TOY_PROXY} \
    ${BENCHMARK_SCRIPT:+-e BENCHMARK_SCRIPT=$BENCHMARK_SCRIPT} \
    ${BENCHMARK_SCRIPT_FILE:+-e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE} \
    ${NIAH_WORDS:+-e NIAH_WORDS="$NIAH_WORDS"} \
    ${NIAH_SEEDS:+-e NIAH_SEEDS=$NIAH_SEEDS} \
    ${NIAH_WARMUP:+-e NIAH_WARMUP=$NIAH_WARMUP} \
    ${NIAH_MAXTOK:+-e NIAH_MAXTOK=$NIAH_MAXTOK} \
    ${NIAH_TIMEOUT:+-e NIAH_TIMEOUT=$NIAH_TIMEOUT} \
    ${SHAPE_WARMUP:+-e SHAPE_WARMUP=$SHAPE_WARMUP} \
    ${USE_INDUCTOR_GRAPH_PARTITION:+-e USE_INDUCTOR_GRAPH_PARTITION=$USE_INDUCTOR_GRAPH_PARTITION} \
    ${PREFILL_CUDAGRAPH_MODE:+-e PREFILL_CUDAGRAPH_MODE=$PREFILL_CUDAGRAPH_MODE} \
    ${DECODE_CUDAGRAPH_MODE:+-e DECODE_CUDAGRAPH_MODE=$DECODE_CUDAGRAPH_MODE} \
    ${CUDAGRAPH_CAPTURE_SIZES:+-e CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES"} \
    ${MORI_RDMA_TC:+-e MORI_RDMA_TC=$MORI_RDMA_TC} \
    ${MORI_RDMA_SL:+-e MORI_RDMA_SL=$MORI_RDMA_SL} \
    ${MORI_SHMEM_HEAP_SIZE:+-e MORI_SHMEM_HEAP_SIZE=$MORI_SHMEM_HEAP_SIZE} \
    ${PREFILL_DEEPEP_BACKEND:+-e PREFILL_DEEPEP_BACKEND=$PREFILL_DEEPEP_BACKEND} \
    ${DECODE_DEEPEP_BACKEND:+-e DECODE_DEEPEP_BACKEND=$DECODE_DEEPEP_BACKEND} \
    ${ENABLE_DBO:+-e ENABLE_DBO=$ENABLE_DBO} \
    ${DBO_COMM_SMS:+-e DBO_COMM_SMS=$DBO_COMM_SMS} \
    ${ENABLE_PROFILING:+-e ENABLE_PROFILING=$ENABLE_PROFILING} \
    ${NCCL_IB_HCA:+-e NCCL_IB_HCA=$NCCL_IB_HCA} \
    ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX} \
    ${NCCL_NET_GDR_LEVEL:+-e NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL} \
    ${NCCL_CROSS_NIC:+-e NCCL_CROSS_NIC=$NCCL_CROSS_NIC} \
    ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME} \
    ${GLOO_SOCKET_IFNAME:+-e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME} \
    -e MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}} \
    ${MORI_IB_GID_INDEX:+-e MORI_IB_GID_INDEX=$MORI_IB_GID_INDEX} \
    ${MORI_RDMA_DEVICES:+-e MORI_RDMA_DEVICES=$MORI_RDMA_DEVICES} \
    ${MORI_NUM_QP_PER_PE:+-e MORI_NUM_QP_PER_PE=$MORI_NUM_QP_PER_PE} \
    ${VLLM_MORIIO_QP_PER_TRANSFER:+-e VLLM_MORIIO_QP_PER_TRANSFER=$VLLM_MORIIO_QP_PER_TRANSFER} \
    ${VLLM_MORIIO_NUM_WORKERS:+-e VLLM_MORIIO_NUM_WORKERS=$VLLM_MORIIO_NUM_WORKERS} \
    ${VLLM_MORIIO_POST_BATCH_SIZE:+-e VLLM_MORIIO_POST_BATCH_SIZE=$VLLM_MORIIO_POST_BATCH_SIZE} \
    ${GPU_MEMORY_UTILIZATION:+-e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION} \
    -e _GPU_MEM_UTIL_LAUNCHER_DEFAULT=$_GPU_MEM_UTIL_LAUNCHER_DEFAULT \
    -e GPUS_PER_NODE=${GPUS_PER_NODE:-8} \
    ${GPU_MAX_HW_QUEUES:+-e GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES} \
    ${HIP_FORCE_DEV_KERNARG:+-e HIP_FORCE_DEV_KERNARG=$HIP_FORCE_DEV_KERNARG} \
    ${HSA_NO_SCRATCH_RECLAIM:+-e HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM} \
    ${VLLM_HANDSHAKE_TIMEOUT_MINS:+-e VLLM_HANDSHAKE_TIMEOUT_MINS=$VLLM_HANDSHAKE_TIMEOUT_MINS} \
    ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S} \
    ${ROCSHMEM_HEAP_SIZE:+-e ROCSHMEM_HEAP_SIZE=$ROCSHMEM_HEAP_SIZE} \
    ${ROCSHMEM_MAX_NUM_CONTEXTS:+-e ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS} \
    ${LOG_WAIT_TIMEOUT_SECONDS:+-e LOG_WAIT_TIMEOUT_SECONDS=$LOG_WAIT_TIMEOUT_SECONDS} \
    ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG} \
    ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS} \
    ${NCCL_DEBUG_FILE:+-e NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE} \
    ${TRITON_CACHE_DIR:+-e TRITON_CACHE_DIR=$TRITON_CACHE_DIR} \
    ${VLLM_CACHE_ROOT:+-e VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT} \
    ${COMGR_CACHE_DIR:+-e COMGR_CACHE_DIR=$COMGR_CACHE_DIR} \
    ${AITER_JIT_DIR:+-e AITER_JIT_DIR=$AITER_JIT_DIR} \
    -e DISTRIBUTED_TIMEOUT_SECONDS=${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
    -e VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-300000} \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600} \
    ${CONNECTOR_ENV_ARGS} \
    ${_EXTRA_ENV} \
    ${VLLM_CUDAGRAPH_MODE:+-e VLLM_CUDAGRAPH_MODE=$VLLM_CUDAGRAPH_MODE} \
    ${CUDAGRAPH_CAPTURE_SIZES:+-e CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES"} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        set -o pipefail
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${SLURM_PROCID}.log
    "
}

# Stops and removes the container on cancel; exits with its status (node_lib.sh).
node_run_container _node_docker_run _stop_trace
