#!/bin/bash
# Interactive multi-node launcher (mirrors run_xPyD_models.slurm's docker run, but
# NODE_RANK is passed per node instead of being derived from SLURM_PROCID). Used by
# tests/drive_cell.sh to drive an existing allocation via `srun --overlap`.
# Usage: run_interactive.sh <NODE_RANK>
# Env expected: DOCKER_IMAGE_NAME, MODEL_NAME, MODEL_PATH, IPADDRS, MASTER_ADDR,
#   MASTER_PORT, NNODES, xP, yD, CONNECTOR, WIDE_EP, BENCHMARK_CON,
#   BENCHMARK_COMBINATIONS, SLURM_JOB_ID, plus any recipe/-e overrides.
set -u
NODE_RANK="$1"

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
# repo dir = the vllm_dissag dir containing this tests/ folder (override NIXL_REPO_DIR to relocate)
NIXL_REPO_DIR="${NIXL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_PATH="${LOG_PATH:-/shared_inference/${USER}/model_blog_logs}"
mkdir -p "$LOG_PATH" 2>/dev/null || true
DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/vllm_disagg.sh"

# cleanup any stale container/ports on this node
docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
fuser -k 2222/tcp 2>/dev/null || true
fuser -k 15000/tcp 2>/dev/null || true
fuser -k 30000/tcp 2>/dev/null || true
fuser -k 36367/tcp 2>/dev/null || true   # router discovery / moriio proxy_ping
fuser -k 20005/tcp 2>/dev/null || true   # serve port
sleep 2

mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
# Persistent JIT cache: the image points AITER_JIT_DIR/TRITON_CACHE_DIR/VLLM_CACHE_ROOT/
# COMGR_CACHE_DIR at /opt/vllm_cache. Mount a host dir there so AITER CK kernels compile
# ONCE and are reused across runs (cold compile is ~15 min; warm boot is ~1 min). Host dir
# on local NVMe for speed. Keyed by the image ID so a new image (different kernels/ABI)
# starts a fresh cache instead of reusing stale .so's; set JIT_CACHE_HOST to override, or
# JIT_CACHE_PERSIST=0 to disable and fall back to an ephemeral in-container cache.
if [[ "${JIT_CACHE_PERSIST:-1}" == "1" ]]; then
    _IMG_KEY="$(docker image inspect --format '{{.Id}}' "$DOCKER_IMAGE_NAME" 2>/dev/null | sed 's/^sha256://; s/[^a-f0-9]//g' | cut -c1-12)"
    _IMG_KEY="${_IMG_KEY:-noimg}"
    _JIT_BASE="${JIT_CACHE_HOST:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/${_IMG_KEY}}"
    # Kimi-K3: prefill and decode compile different AITER kernel variants — separate caches
    # (same logic as run_xPyD_models.slurm; missing this caused PIECEWISE decode hang — F25).
    if [[ "${MODEL_NAME}" == "Kimi-K3-MXFP4" && "${JIT_CACHE_SPLIT_K3:-1}" == "1" ]]; then
        if [[ "${NODE_RANK:-0}" -lt "${xP:-1}" ]]; then
            _JIT_ROLE="prefill"
        else
            _JIT_ROLE="decode"
        fi
        _JIT_CACHE_HOST="${_JIT_BASE}/${_JIT_ROLE}"
    else
        _JIT_CACHE_HOST="${_JIT_BASE}"
    fi
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache"
    echo "JIT cache (persistent, image ${_IMG_KEY}${_JIT_ROLE:+/${_JIT_ROLE}}): ${_JIT_CACHE_HOST} -> /opt/vllm_cache"
else
    _JIT_CACHE_MOUNT=""
fi

# host RDMA library mounts
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu
for _lib in libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1; do
    [ -f "$_LIBDIR/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/$_lib:$_LIBDIR/$_lib:ro"
done
for _vlib in $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.*; do
    [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done
for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done
[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"

docker run --rm \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME \
    -v /shared_inference:/shared_inference \
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size "${DOCKER_SHM_SIZE:-256G}" --ulimit nofile=524288:524288 --ulimit memlock=-1:-1 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    ${_JIT_CACHE_MOUNT} \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e NNODES=$NNODES \
    -e NODE_RANK=$NODE_RANK \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH \
    -e xP=$xP -e yD=$yD \
    -e USER_NAME=$USER \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_ITR=${BENCHMARK_ITR:-1} \
    -e BENCHMARK_CON="${BENCHMARK_CON}" \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}" \
    -e IPADDRS=$IPADDRS \
    -e CONNECTOR=$CONNECTOR \
    -e WIDE_EP=$WIDE_EP \
    ${EP_BACKEND:+-e EP_BACKEND=$EP_BACKEND} \
    -e PROXY_TYPE=${PROXY_TYPE:-vllm_router} \
    -e ROUTER_PORT=${ROUTER_PORT:-30000} \
    ${ROUTER_BINARY:+-e ROUTER_BINARY=$ROUTER_BINARY} \
    -e GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8} \
    -e GPUS_PER_NODE=${GPUS_PER_NODE:-8} \
    -e MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-eth0} \
    -e DISTRIBUTED_TIMEOUT_SECONDS=${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
    -e VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-300000} \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600} \
    -e PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:False} \
    -e PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:False} \
    -e HSA_ENABLE_IPC_MODE_LEGACY=${HSA_ENABLE_IPC_MODE_LEGACY:-0} \
    -e MORI_GPU_ARCHS=${MORI_GPU_ARCHS:-gfx942} \
    -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1} \
    ${DECODE_CUDAGRAPH_MODE:+-e DECODE_CUDAGRAPH_MODE=$DECODE_CUDAGRAPH_MODE} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${NODE_RANK}.log
    "
