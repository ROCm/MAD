#!/bin/bash
# One node of a colocated vLLM run: started once per node by run_multinode.slurm
# (`srun ... bash node.sh`), with SLURM_PROCID as the node rank and the launcher's
# exported settings as its environment. Starts this node's container and exits
# with the container's status.
echo "Rank $SLURM_PROCID on $(hostname)"
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true

# Host RDMA userspace libs must match the host kernel driver; mount them over the
# image copies (same approach as the disagg launcher).
# Use -f (regular file), NOT -e. Docker auto-creates a DIRECTORY at a bind-mount
# source that does not exist, so a node where one of these libs is absent ends up
# with an empty directory at e.g. /usr/lib/x86_64-linux-gnu/libionic.so.1 after the
# first run. -e is true for that directory, so every later run on that node mounts
# a directory over a file inside the image and docker refuses:
#   OCI runtime create failed: ... not a directory: Are you trying to mount a
#   directory onto a file (or vice-versa)?
# The failure is per-node and self-propagating, which makes it look intermittent:
# the container starts on a clean node and dies on a poisoned one, leaving the
# survivors stuck forever at the socket barrier waiting for a peer that never came.
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu
# The SONAMEs first, then the versioned files they point at. Both are required and
# the order does not matter, but omitting this loop is silently broken:
# libibverbs.so.1 is the name every consumer links against, and it is a symlink.
# Mounting only the versioned file (libibverbs.so.1.14.x) lands the host copy at a
# path the image symlink does not point to, so libibverbs.so.1 still resolves to
# the image version while libmlx5.so.1 -- matched by the pattern loop below and
# therefore the HOST copy -- demands a symbol only the host libibverbs exports:
#     OSError: libibverbs.so.1: version `IBVERBS_PRIVATE_57` not found
#             (required by libmlx5.so.1)
# vLLM loads its plugins during startup, so this kills the server about ten
# seconds in, with a Python traceback that names a quantization plugin and not the
# mount that caused it.
# scripts/vllm_dissag/run_xPyD_models.slurm has always had this loop; this
# launcher was written without it.
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

# Inline rather than the helper in cluster.sh: this body runs in a SEPARATE shell
# on the compute node, where cluster.sh has not been sourced, so calling the
# function here would be a "command not found". Same -d guard and same reason as
# the RDMA mounts above -- docker auto-creates a directory at a missing bind
# source and would shadow the model with an empty one. NVME_ROOT is defaulted
# rather than bare because an unset name under `set -u` is fatal, not empty.
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

# Host-local compilation cache dirs (ext4, survive container restarts).
mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true

# --- Persistent JIT cache mount (lifted from vllm_dissag/run_xPyD_models.slurm) ---
# The image points AITER_JIT_DIR/TRITON_CACHE_DIR/VLLM_CACHE_ROOT/COMGR_CACHE_DIR at
# /opt/vllm_cache, but this launcher only ever mounted /tmp/vllm_cache -- a path
# nothing reads. Every rank therefore recompiled AITER CK kernels from scratch on
# every run, inside --rm containers that threw the result away. Cold compile is
# ~15 min for the MoE/FP8 GEMM set against ~1 min warm, and it is per-node, so a
# node that compiles cold while its peer does not is exactly the startup skew the
# cross-node barrier cannot absorb.
# Keyed by image ID so a new image starts a fresh cache rather than reusing
# stale-ABI shared objects. Override JIT_CACHE_HOST, or set JIT_CACHE_PERSIST=0.
if [ "${JIT_CACHE_PERSIST:-1}" = "1" ]; then
    _IMG_RAW=$(docker image inspect --format "{{.Id}}" "$DOCKER_IMAGE_NAME" 2>/dev/null)
    _IMG_KEY=$(printf "%s" "$_IMG_RAW" | tr -cd "a-f0-9" | cut -c1-12)
    _IMG_KEY="${_IMG_KEY:-noimg}"
    _JIT_CACHE_HOST="${JIT_CACHE_HOST:-${NVME_ROOT:-/mnt/m2m_nobackup}/${USER}/vllm_jit_cache/${_IMG_KEY}}"
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
    # Inline, not cluster_mark_dir: node.sh runs in a separate shell on
    # the compute node where cluster.sh was never sourced, so the function is not
    # defined here -- it fails with "command not found" once per node. Same reason
    # the NVMe mount a few lines down is inlined.
    if [ ! -e "$_JIT_CACHE_HOST/README.ci" ]; then
        printf "Signature: 8a477f597d28d172789f06886806bc55\n# Cache directory created by MAD CI.\n" \
            > "$_JIT_CACHE_HOST/CACHEDIR.TAG" 2>/dev/null || true
        printf "%s\n%s\n\n%s\n" \
            "What: JIT/Triton/AITER kernel cache for one docker image" \
            "Written by: MAD CI colocated launcher" \
            "Safe to delete when no job is running; it is rebuilt on demand." \
            > "$_JIT_CACHE_HOST/README.ci" 2>/dev/null || true
    fi
    # Stale AITER build locks in this cache are cleared inside the container, which
    # owns them (_clear_stale_jit_locks); JIT_CACHE_HOST / JIT_LOCK_STALE_MIN go in.
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache"
    echo "[jit-cache] persistent image ${_IMG_KEY}: ${_JIT_CACHE_HOST} to /opt/vllm_cache"
else
    _JIT_CACHE_MOUNT=""
fi

# --- Forwarded env (see COLOCATED_FORWARD_ENV above) ---
# Written to a file rather than built into -e flags so values containing spaces
# survive; docker takes the whole line after = verbatim. Node-local /tmp, mode
# 600, and never archived -- unlike the wrapper scripts, which are build
# artifacts and must stay free of secrets.
_ENV_FILE=/tmp/colocated_fwd_env.${SLURM_JOB_ID}.${SLURM_PROCID}
rm -f "$_ENV_FILE"
: > "$_ENV_FILE"
chmod 600 "$_ENV_FILE"
for _v in ${COLOCATED_FORWARD_ENV//,/ }; do
    [ -n "${!_v:-}" ] && printf "%s=%s\n" "$_v" "${!_v}" >> "$_ENV_FILE"
done
echo "[fwd-env] forwarding: $(cut -d= -f1 "$_ENV_FILE" | tr "\n" " ")"

docker run --rm \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    --shm-size ${DOCKER_SHM_SIZE:-256G} \
    --ulimit nofile=524288:524288 --ulimit memlock=-1:-1 \
    -v $LOG_PATH:/run_logs \
    -v $MAD_SCRIPTS_DIR:$MAD_SCRIPTS_MOUNT \
    ${_SITE_MOUNTS} \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    $_JIT_CACHE_MOUNT \
    $_RDMA_MOUNTS \
    --env-file "$_ENV_FILE" \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e NODE_RANK=$SLURM_PROCID \
    -e NNODES=$NNODES \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e IPADDRS=$IPADDRS \
    -e MODEL_NAME=$MODEL_NAME \
    -e DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-unknown} \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    ${PERF_GPU_ARCH:+-e PERF_GPU_ARCH=$PERF_GPU_ARCH} \
    -e MODEL_PATH=$MODEL_PATH \
    -e SERVE_PORT=$SERVE_PORT \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e TP_SIZE=$TP_SIZE \
    -e PP_SIZE=$PP_SIZE \
    -e ENABLE_EP=$ENABLE_EP \
    -e CONTAINER_BARRIER_PORT=$CONTAINER_BARRIER_PORT \
    ${COLOCATED_ENV_ARGS} \
    ${ALL2ALL_BACKEND:+-e ALL2ALL_BACKEND=$ALL2ALL_BACKEND} \
    ${COLOCATED_EXTRA_ARGS:+-e COLOCATED_EXTRA_ARGS="$COLOCATED_EXTRA_ARGS"} \
    -e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE \
    ${BENCHMARK_ITR:+-e BENCHMARK_ITR=$BENCHMARK_ITR} \
    ${BENCHMARK_CON:+-e BENCHMARK_CON="$BENCHMARK_CON"} \
    ${BENCHMARK_COMBINATIONS:+-e BENCHMARK_COMBINATIONS="$BENCHMARK_COMBINATIONS"} \
    ${NIAH_WORDS:+-e NIAH_WORDS="$NIAH_WORDS"} \
    ${AITER_SITUV2_A8W4:+-e AITER_SITUV2_A8W4=$AITER_SITUV2_A8W4} \
    ${NIAH_SEEDS:+-e NIAH_SEEDS=$NIAH_SEEDS} \
    ${NIAH_WARMUP:+-e NIAH_WARMUP=$NIAH_WARMUP} \
    ${NIAH_MAXTOK:+-e NIAH_MAXTOK=$NIAH_MAXTOK} \
    ${NIAH_TIMEOUT:+-e NIAH_TIMEOUT=$NIAH_TIMEOUT} \
    ${SHAPE_WARMUP:+-e SHAPE_WARMUP=$SHAPE_WARMUP} \
    ${JIT_CACHE_HOST:+-e JIT_CACHE_HOST=$JIT_CACHE_HOST} \
    ${JIT_LOCK_STALE_MIN:+-e JIT_LOCK_STALE_MIN=$JIT_LOCK_STALE_MIN} \
    ${GPU_MEMORY_UTILIZATION:+-e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION} \
    ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN=$MAX_MODEL_LEN} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        set -o pipefail
        mkdir -p /run_logs/${SLURM_JOB_ID}
        SHARED_DIR=$MAD_SCRIPTS_MOUNT/vllm_dissag \
        bash $MAD_SCRIPTS_MOUNT/vllm_multinode/serve_colocated.sh \
          2>&1 | tee /run_logs/${SLURM_JOB_ID}/colocated_bench_NODE${SLURM_PROCID}.log
    "
_docker_rc=$?
rm -f "$_ENV_FILE"
exit "$_docker_rc"
