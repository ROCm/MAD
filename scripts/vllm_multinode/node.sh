#!/bin/bash
# One node of a colocated vLLM run: started once per node by run_multinode.slurm
# (through the dispatcher, common/dispatch/dispatch.sh: srun, a job-array task or a pod),
# with SLURM_PROCID as the node rank and the launcher's
# exported settings as its environment. Starts this node's container and exits
# with the container's status.
. "$(dirname "${BASH_SOURCE[0]}")/../common/node/node_lib.sh"
echo "Rank $SLURM_PROCID on $(hostname)"
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true

# Host RDMA userspace libs over the image copies (node_rdma_mounts, node_lib.sh).
node_rdma_mounts

node_site_mounts
# This node's part of the run (topology.py), passed in as MAD_TOPO_*.
node_topology vllm_multinode

# Host-local compilation cache dirs (ext4, survive container restarts).
mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true

# --- Persistent JIT cache (node_jit_cache, node_lib.sh) ---
node_jit_cache colocated

# --- Forwarded env (see COLOCATED_FORWARD_ENV above) ---
# Written to a file rather than built into -e flags so values containing spaces
# survive; docker takes the whole line after = verbatim. Node-local /tmp, and
# never archived -- unlike the wrapper scripts, which are build artifacts and must
# stay free of secrets. mktemp creates it new (never through an existing name or
# link another user placed there) and mode 600 before anything is written to it.
_ENV_FILE="$(mktemp "/tmp/colocated_fwd_env.${SLURM_JOB_ID}.${SLURM_PROCID}.XXXXXX")" || {
    echo "[fwd-env] ERROR: cannot create the forwarded-env file in /tmp" >&2; exit 1; }
for _v in ${COLOCATED_FORWARD_ENV//,/ }; do
    [ -n "${!_v:-}" ] && printf "%s=%s\n" "$_v" "${!_v}" >> "$_ENV_FILE"
done
echo "[fwd-env] forwarding: $(cut -d= -f1 "$_ENV_FILE" | tr "\n" " ")"

_node_docker_run() {
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
    ${_TOPO_ENV} \
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
}

# The forwarded-env file goes once the container ends or the task is signalled.
_rm_env_file() { rm -f "$_ENV_FILE"; }
# Stops and removes the container on cancel; exits with its status (node_lib.sh).
node_run_container _node_docker_run _rm_env_file
