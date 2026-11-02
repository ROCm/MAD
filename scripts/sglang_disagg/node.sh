#!/bin/bash
# One node of a SGLang disaggregated run: started once per node by run_xPyD_models.slurm
# (`srun ... bash node.sh`), with SLURM_PROCID as the node rank and the launcher's
# exported settings as its environment. Starts this node's container and exits
# with the container's status.
echo "Rank $SLURM_PROCID on $(hostname)";
docker ps -q | xargs --no-run-if-empty docker stop;
# Stop this node container when the task is signalled. scancel sends SIGTERM here, but
# the container belongs to the docker daemon and its PID 1 is a plain bash that ignores
# SIGTERM, so a cancelled job left its container running and holding every GPU: one
# cancelled job left its container holding 1535 GB on a node. docker run
# runs in the background so the trap fires as soon as the signal lands.
_stop_container() {
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
    --shm-size 64G \
    -v ${LOG_PATH}:/run_logs \
    -v $MOONCAKE_REPO_DIR:$MOONCAKE_COOKBOOK_PATH \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e MOONCAKE_COOKBOOK_PATH=$MOONCAKE_COOKBOOK_PATH \
    -e xP=$xP \
    -e yD=$yD \
    -e DP_MODE=$DP_MODE \
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_FILE=$BENCHMARK_FILE \
    -e IPADDRS=$IPADDRS \
    -e BENCHMARK_ITR=$BENCHMARK_ITR \
    -e SKIP_BENCHMARK=$SKIP_BENCHMARK \
    -e SKIP_CURL_TEST=$SKIP_CURL_TEST \
    ${KV_TRANSFER_BACKEND:+-e KV_TRANSFER_BACKEND=$KV_TRANSFER_BACKEND} \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}" \
    -e DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-unknown} \
    -e USE_CX7_NICS=${USE_CX7_NICS:-0} \
    ${SGLANG_FABRIC_ENV_ARGS} \
    ${PERF_GPU_ARCH:+-e PERF_GPU_ARCH=$PERF_GPU_ARCH} \
    ${BENCHMARK_SCRIPT:+-e BENCHMARK_SCRIPT=$BENCHMARK_SCRIPT} \
    ${BENCHMARK_SCRIPT_FILE:+-e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE} \
    -e MOONCAKE_REPO_DIR="$MOONCAKE_REPO_DIR" \
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
    ${AIPERF_PIN:+-e AIPERF_PIN=$AIPERF_PIN} \
    ${AGENTIC_UTILS_PIN:+-e AGENTIC_UTILS_PIN=$AGENTIC_UTILS_PIN} \
    ${SUITE_CORPUS_FORCE:+-e SUITE_CORPUS_FORCE=$SUITE_CORPUS_FORCE} \
    ${AGENTIC_STRICT_CONTEXT:+-e AGENTIC_STRICT_CONTEXT=$AGENTIC_STRICT_CONTEXT} \
    ${AGENTIC_HF_ISL_TAIL:+-e AGENTIC_HF_ISL_TAIL=$AGENTIC_HF_ISL_TAIL} \
    ${HF_HOME:+-e HF_HOME=$HF_HOME} \
    ${ENABLE_SERVER_METRICS:+-e ENABLE_SERVER_METRICS=$ENABLE_SERVER_METRICS} \
    ${ENABLE_RADIX_CACHE:+-e ENABLE_RADIX_CACHE=$ENABLE_RADIX_CACHE} \
    ${DRY_RUN:+-e DRY_RUN=$DRY_RUN} \
    ${AGENTIC_ALLOW_UV_INSTALL:+-e AGENTIC_ALLOW_UV_INSTALL=$AGENTIC_ALLOW_UV_INSTALL} \
    ${AGENTIC_UV_VERSION:+-e AGENTIC_UV_VERSION=$AGENTIC_UV_VERSION} \
    ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG} \
    ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS} \
    ${NCCL_DEBUG_FILE:+-e NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE} \
    --ulimit nofile=1048576:1048576 \
    --name $DOCKER_CONT_NAME \
    --entrypoint /bin/bash \
    $DOCKER_IMAGE_NAME -c "
        set -o pipefail
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_sglang_bench_serving.sh_NODE${SLURM_PROCID}.log
    " &
_docker_pid=$!
wait "$_docker_pid"
_docker_rc=$?
trap - TERM INT
exit "$_docker_rc"
