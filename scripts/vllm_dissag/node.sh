#!/bin/bash
# One node of a vLLM disaggregated run: started once per node by run_xPyD_models.slurm
# (`srun ... bash node.sh`), with SLURM_PROCID as the node rank and the launcher's
# exported settings as its environment. Starts this node's container and exits
# with the container's status.
echo "Rank $SLURM_PROCID on $(hostname)";
docker ps -q | xargs --no-run-if-empty docker stop;
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true;
fuser -k ${BARRIER_PORT:-15000}/tcp 2>/dev/null || true;
fuser -k 2222/tcp 2>/dev/null || true;
fuser -k 15000/tcp 2>/dev/null || true;
sleep 2;
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true;

# --- Create host-local compilation cache dirs (ext4, survives container restarts) ---
mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true;

# --- Persistent JIT cache mount ---
# The image points AITER_JIT_DIR/TRITON_CACHE_DIR/VLLM_CACHE_ROOT/COMGR_CACHE_DIR at
# /opt/vllm_cache. Mount a host dir there so AITER CK kernels compile ONCE and are reused
# across runs. Cold compile is ~15 min for the DeepSeek MoE/FP8 GEMM kernel set; a warm
# boot is ~1 min. Keyed by image ID so a new image starts a fresh cache and does not reuse
# stale-ABI shared objects. Host dir on local NVMe. Override JIT_CACHE_HOST, or set
# JIT_CACHE_PERSIST=0 to disable and fall back to the image empty in-container cache.
if [ "${JIT_CACHE_PERSIST:-1}" = "1" ]; then
    _IMG_RAW=$(docker image inspect --format "{{.Id}}" "$DOCKER_IMAGE_NAME" 2>/dev/null);
    _IMG_KEY=$(printf "%s" "$_IMG_RAW" | tr -cd "a-f0-9" | cut -c1-12);
    _IMG_KEY="${_IMG_KEY:-noimg}";
    _JIT_BASE="${JIT_CACHE_HOST:-${NVME_ROOT:-/mnt/m2m_nobackup}/${USER}/vllm_jit_cache/${_IMG_KEY}}";
    # TP-within-EP (EP_TP_SIZE>1): prefill/decode compile different AITER kernel variants — separate caches.
    if (( ${EP_TP_SIZE:-1} > 1 )) && [ "${JIT_CACHE_SPLIT_ROLE:-1}" = "1" ]; then
        if [ "${SLURM_PROCID}" -lt "${xP}" ]; then
            _JIT_ROLE="prefill";
        else
            _JIT_ROLE="decode";
        fi
        _JIT_CACHE_HOST="${_JIT_BASE}/${_JIT_ROLE}";
    else
        _JIT_CACHE_HOST="${_JIT_BASE}";
    fi
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true;
    # Inline, not cluster_mark_dir: node.sh runs in a separate shell on
    # the compute node where cluster.sh was never sourced, so the function is not
    # defined here and would fail with "command not found" once per node.
    if [ ! -e "$_JIT_CACHE_HOST/README.ci" ]; then
        printf "Signature: 8a477f597d28d172789f06886806bc55\n# Cache directory created by MAD CI.\n" \
            > "$_JIT_CACHE_HOST/CACHEDIR.TAG" 2>/dev/null || true
        printf "%s\n%s\n\n%s\n" \
            "What: JIT/Triton/AITER kernel cache for one docker image" \
            "Written by: MAD CI disagg launcher" \
            "Safe to delete when no job is running; it is rebuilt on demand." \
            > "$_JIT_CACHE_HOST/README.ci" 2>/dev/null || true
    fi
    # Stale AITER build locks in this cache are cleared inside the container, which
    # owns them (_clear_stale_jit_locks); JIT_CACHE_HOST / JIT_LOCK_STALE_MIN go in.
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache";
    echo "[jit-cache] persistent image ${_IMG_KEY}: ${_JIT_CACHE_HOST} to /opt/vllm_cache";
else
    _JIT_CACHE_MOUNT="";
fi

# --- Build host RDMA library mounts ---
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu

# -f, not -e: docker creates a DIRECTORY at a bind-mount source that does not
# exist, so on a node lacking one of these libs the first run leaves an empty
# directory behind and every later run mounts a directory over a file inside the
# image, which docker rejects ("not a directory"). Self-propagating and per-node,
# so it presents as an intermittent failure. Full note in
# scripts/vllm_multinode/run_multinode.slurm.
for _lib in libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1; do
    [ -f "$_LIBDIR/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/$_lib:$_LIBDIR/$_lib:ro"
done
for _vlib in $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.*; do
    [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done

for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        # Require a regular file AFTER symlink resolution: `-e` also accepts a directory
        # matching one of these globs, and docker then bind-mounts a directory onto a
        # file target -> "not a directory" -> container create exit 125.
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done

[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"
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

# Stop this node container when the task is signalled. scancel sends SIGTERM here, but
# the container belongs to the docker daemon and its PID 1 is a plain bash that ignores
# SIGTERM, so a cancelled job left its container running and holding every GPU: one
# cancelled job left its container holding 1535 GB on a node. docker run
# runs in the background so the trap fires as soon as the signal lands.
_stop_container() {
    [ -n "$_trace_pid" ] && kill "$_trace_pid" 2>/dev/null;
    docker stop -t 10 "$DOCKER_CONT_NAME" >/dev/null 2>&1
    docker rm -f "$DOCKER_CONT_NAME" >/dev/null 2>&1
}
trap "_stop_container; exit 143" TERM INT
# Site storage, bound at the same path inside: shared storage, local NVMe, and
# MODEL_DIR when it is outside both. All three come from cluster.sh or the caller.
# Only paths present on this node: docker creates a missing bind source as an empty
# directory, which would shadow the weights. Evaluated here, on each node.
_SITE_MOUNTS=""
for _p in ${SHARED_MOUNT:-/shared_inference} ${NVME_ROOT:-/mnt/m2m_nobackup} ${MODEL_DIR:-}; do
    [ -d "$_p" ] || continue
    _covered=0
    for _m in $_SITE_MOUNTS; do case "$_p/" in "${_m%%:*}"/*) _covered=1 ;; esac; done
    [ "$_covered" = 1 ] || _SITE_MOUNTS="$_SITE_MOUNTS $_p:$_p"
done
_SITE_MOUNTS="$(for _m in $_SITE_MOUNTS; do printf -- " -v %s" "$_m"; done)"
echo "[mounts] site storage:${_SITE_MOUNTS:- none}"
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
    -e MASTER_ADDR=$MASTER_ADDR \
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
    " &
_docker_pid=$!
wait "$_docker_pid"
_docker_rc=$?
[ -n "$_trace_pid" ] && kill "$_trace_pid" 2>/dev/null;
trap - TERM INT
exit "$_docker_rc"
