#!/bin/bash
#SBATCH --job-name=kvcache-transfer-bench
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output="/shared_inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared_inference/%u/model_blog_logs/slurm-%j.err"

set +x

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-size)
            START_SIZE="$2"
            shift 2
            ;;
        --stop-size)
            STOP_SIZE="$2"
            shift 2
            ;;
        --backends)
            BENCH_BACKENDS="$2"
            shift 2
            ;;
        --docker-image)
            DOCKER_IMAGE_ARG="$2"
            shift 2
            ;;
        --ibdevice)
            IBDEVICES="$2"
            shift 2
            ;;
        --sync-port)
            SYNC_PORT="$2"
            shift 2
            ;;
        --model-name)
            BENCH_MODEL_NAME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# The kvcache_transfer_bench directory. Under sbatch the scheduler runs a copy of this
# script from its spool directory, so its own path is not the repo: then the submit
# directory names it (`cd kvcache_transfer_bench && sbatch scripts/...`, or sbatch from
# the MAD root). madengine runs it in place, from scripts/.
_resolve_project_dir() {
    local d
    for d in "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)" \
             "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/scripts/kvcache_transfer_bench}" \
             "${SLURM_SUBMIT_DIR:-}" "$PWD" "$PWD/.."; do
        [[ -n "$d" && -f "$d/scripts/run_node.sh" ]] && { (cd "$d" && pwd); return 0; }
    done
    echo "ERROR: cannot find kvcache_transfer_bench (scripts/run_node.sh) next to this script," \
         "in SLURM_SUBMIT_DIR or in $PWD" >&2
    return 1
}
PROJECT_DIR="$(_resolve_project_dir)" || exit 1

START_SIZE=${START_SIZE:-4096} # 4KB
STOP_SIZE=${STOP_SIZE:-1073741824} # 1GB
BENCH_BACKENDS=${BENCH_BACKENDS:-all}
DOCKER_IMAGE=${DOCKER_IMAGE_ARG:-${DOCKER_IMAGE_NAME:?Set DOCKER_IMAGE_NAME (env) or pass --docker-image}}
IBDEVICES=${IBDEVICES:-${IBDEVICE:-}}
SYNC_PORT=${SYNC_PORT:-9999}
BENCH_MODEL_NAME=${BENCH_MODEL_NAME:-pyt_kvcache_transfer_bench_all}
KV_CACHE_TEST_PATH=/workspace/kvcache_transfer_bench

KV_CACHE_TEST_REPO="${PROJECT_DIR}/"
DOCKER_CONTAINER_NAME=container_kv_transfer_perf_bench

# Inside MAD, the fabric comes from common/cluster.sh, as for the other multinode
# launchers: it detects the RDMA adapters' family and fills in what the caller did not
# set. The fabric values the caller passed, recorded before cluster.sh fills in its
# fallbacks, always reach the container; on ainic / thor2 so does what cluster.sh
# derived, since this benchmark's defaults (eth0, mlx5_0) name one CX7 cluster's
# devices. On cx7 / unknown the container keeps the defaults it always ran with.
MAD_SCRIPTS_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
_FABRIC_VARS="NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME MORI_RDMA_DEVICES MORI_IB_GID_INDEX MORI_SOCKET_IFNAME RCCL_AINIC_ROCE"
_FABRIC_PASSED=""
for _v in $_FABRIC_VARS; do [ -n "${!_v:-}" ] && _FABRIC_PASSED="$_FABRIC_PASSED $_v"; done
if [ -f "${MAD_SCRIPTS_DIR}/common/cluster.sh" ]; then
    . "${MAD_SCRIPTS_DIR}/common/cluster.sh"
    . "${MAD_SCRIPTS_DIR}/common/launcher/launcher_lib.sh"
fi
if [ -z "${IBDEVICES:-}" ]; then
    case "${CLUSTER_ARCHETYPE:-}" in
        ainic|thor2) IBDEVICES="${KV_IB_DEVICE:-mlx5_0}" ;;
        *)           IBDEVICES=mlx5_0 ;;
    esac
fi
_FABRIC_FWD="$_FABRIC_PASSED"
case "${CLUSTER_ARCHETYPE:-}" in ainic|thor2) _FABRIC_FWD="$_FABRIC_VARS" ;; esac
# MoRI reads its own GID index; one fabric has one GID.
if [ -z "${MORI_IB_GID_INDEX:-}" ] && [ -n "${NCCL_IB_GID_INDEX:-}" ]; then
    case " $_FABRIC_FWD " in *" NCCL_IB_GID_INDEX "*) export MORI_IB_GID_INDEX="$NCCL_IB_GID_INDEX" ;; esac
fi
KV_ENV_ARGS=""
for _v in $_FABRIC_FWD MORI_ROOT; do
    [ -n "${!_v:-}" ] && KV_ENV_ARGS="$KV_ENV_ARGS -e $_v=${!_v}"
done
echo "[fabric] ${CLUSTER_ARCHETYPE:-unknown}; ibdevice ${IBDEVICES}; forwarded to the container:${KV_ENV_ARGS:- none (container defaults)}"

# Get node names from Slurm (recover from SLURM_JOB_ID if SLURM_NODELIST is empty)
if [ -z "${SLURM_NODELIST:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    SLURM_NODELIST=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP '(?<!\w)NodeList=\K\S+' | grep -v '(null)' | head -1)
    echo "INFO: Recovered SLURM_NODELIST='$SLURM_NODELIST' from SLURM_JOB_ID=$SLURM_JOB_ID"
fi
NODES=($(scontrol show hostname "${SLURM_NODELIST:-}" 2>/dev/null))
if [ ${#NODES[@]} -eq 0 ]; then
    echo "ERROR: Could not get nodes. SLURM_NODELIST='${SLURM_NODELIST:-}' SLURM_JOB_ID='${SLURM_JOB_ID:-}'"
    exit 1
fi
NODE1=${NODES[0]}
NODE2=${NODES[1]:-${NODES[0]}}

if [ -z "$NODE1" ]; then
    echo "ERROR: NODE1 is empty — NODES array: ${NODES[*]:-<empty>}"
    exit 1
fi
echo "INFO: NODE1=$NODE1 NODE2=$NODE2 (${#NODES[@]} node(s))"
echo "INFO: BENCH_MODEL_NAME=$BENCH_MODEL_NAME BENCH_BACKENDS=$BENCH_BACKENDS"
echo "INFO: KV_CACHE_TEST_REPO=$KV_CACHE_TEST_REPO -> $KV_CACHE_TEST_PATH"

# Each node's address. Nodes are multi-homed, and the first `hostname -I` address can
# be one a peer cannot reach (a /31 RDMA rail): inside MAD the address on
# FABRIC_SUBNET_PREFIX (common/launcher/launcher_lib.sh) is preferred and both
# containers resolve both node names to it; otherwise each node maps its own name to
# its first address, as before.
KV_NODE_IPS=""
KV_ADD_HOSTS=""
if declare -F launcher_node_ips >/dev/null; then
    SELECTED_NODES="$(printf '%s\n' "$NODE1" "$NODE2" | awk '!seen[$0]++')"
    launcher_node_ips "${FABRIC_SUBNET_PREFIX:-}"
    _i=0
    for _n in $SELECTED_NODES; do
        KV_NODE_IPS="$KV_NODE_IPS ${_n%%.*}=${IPS[$_i]}"
        KV_ADD_HOSTS="$KV_ADD_HOSTS --add-host ${_n}:${IPS[$_i]}"
        _i=$((_i + 1))
    done
fi

# The containers write the results as root. On a root-squashed share that is an
# unprivileged user, so the directories they write are created here, writable to it.
for _d in "${PROJECT_DIR}/shared/results_${SLURM_JOB_ID:-unknown}" "${PROJECT_DIR}/slurm_output/perf_csv"; do
    mkdir -p "$_d" && chmod 0777 "$_d" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Phase 1: Always pull latest image on all nodes. srun blocks until all nodes
# complete, so both nodes have the image before any proceeds to docker run.
# ---------------------------------------------------------------------------
srun --cpu-bind=none /bin/bash -c '
echo "Rank $SLURM_PROCID on $(hostname) (SLURM_NODEID=$SLURM_NODEID) — pull phase";
docker stop '"$DOCKER_CONTAINER_NAME"' 2>/dev/null || true;
docker rm '"$DOCKER_CONTAINER_NAME"' 2>/dev/null || true;
sleep 2;
echo "  [$(hostname)] Pulling '"$DOCKER_IMAGE"'...";
docker pull '"$DOCKER_IMAGE"'
echo "  [$(hostname)] Pull phase done"
'

# ---------------------------------------------------------------------------
# Phase 2: Run benchmark. All nodes have the image; srun starts all together.
# ---------------------------------------------------------------------------
DRIVER_SCRIPT="$KV_CACHE_TEST_PATH/scripts/run_node.sh"
srun --cpu-bind=none /bin/bash -c '
LOCAL_HOSTNAME=$(hostname)
LOCAL_IP=""
for _kv in '"$KV_NODE_IPS"'; do [ "${_kv%%=*}" = "${LOCAL_HOSTNAME%%.*}" ] && LOCAL_IP="${_kv#*=}"; done
[ -n "$LOCAL_IP" ] || LOCAL_IP=$(hostname -I | awk '"'"'{print $1}'"'"')

# Run metadata for to_perf_csv.py; madengine does not export it into the
# container. Slurm may report the GRES count as "gpu:8", hence the strip.
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE##*:}
[ -n "$GPUS_PER_NODE" ] || GPUS_PER_NODE=$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)

docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --hostname "$LOCAL_HOSTNAME" \
    --add-host "$LOCAL_HOSTNAME:$LOCAL_IP" \
    '"$KV_ADD_HOSTS"' \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --privileged=true \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    -v "$HOME:$HOME" \
    -v /sys:/sys \
    -v "$HOME/.ssh:/root/.ssh" \
    -v '"$KV_CACHE_TEST_REPO"':'"$KV_CACHE_TEST_PATH"' \
    --shm-size 64G \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_SUBMIT_DIR='"$SLURM_SUBMIT_DIR"' \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NODE_RANK=$SLURM_PROCID \
    -e DOCKER_IMAGE_NAME='"$DOCKER_IMAGE"' \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e MAD_SYSTEM_GPU_ARCHITECTURE='"${MAD_SYSTEM_GPU_ARCHITECTURE:-}"' \
    '"$KV_ENV_ARGS"' \
    --name '"$DOCKER_CONTAINER_NAME"' \
    '"$DOCKER_IMAGE"' /bin/bash -c "'"$DRIVER_SCRIPT"' --node1 '"${NODE1}"' --node2 '"${NODE2}"' --kv-cache-test-path '"$KV_CACHE_TEST_PATH"' --shared-folder '"$KV_CACHE_TEST_PATH"' --backends '"${BENCH_BACKENDS:-all}"' --start-size '"$START_SIZE"' --stop-size '"$STOP_SIZE"' --sync-port '"$SYNC_PORT"' --ibdevice '"$IBDEVICES"' --model-name '"$BENCH_MODEL_NAME"'"
'
BENCH_RC=$?

srun --cpu-bind=none /bin/bash -c 'docker stop '"$DOCKER_CONTAINER_NAME"' 2>/dev/null || true; docker rm '"$DOCKER_CONTAINER_NAME"' 2>/dev/null || true'

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ "$SLURM_SUBMIT_DIR" != "$PROJECT_DIR" ]; then
    PERF_SRC="$PROJECT_DIR/slurm_output/perf_csv"
    if [ -d "$PERF_SRC" ]; then
        mkdir -p "$SLURM_SUBMIT_DIR/slurm_output/perf_csv"
        cp "$PERF_SRC"/*.csv "$SLURM_SUBMIT_DIR/slurm_output/perf_csv/" 2>/dev/null \
            && echo "INFO: copied perf csv -> $SLURM_SUBMIT_DIR/slurm_output/perf_csv/"
    fi
fi

if [ "$BENCH_RC" -ne 0 ]; then
    echo "ERROR: benchmark failed on one or more nodes (srun exit $BENCH_RC)" >&2
fi
exit $BENCH_RC
