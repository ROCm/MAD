#!/bin/bash
#SBATCH --job-name=kvcache-transfer-bench
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

START_SIZE=${START_SIZE:-4096} # 4KB
STOP_SIZE=${STOP_SIZE:-1073741824} # 1GB
BENCH_BACKENDS=${BENCH_BACKENDS:-all}
DOCKER_IMAGE=${DOCKER_IMAGE_ARG:-${DOCKER_IMAGE_NAME:?Set DOCKER_IMAGE_NAME (env) or pass --docker-image}}
IBDEVICES=${IBDEVICES:-${IBDEVICE:-}}
SYNC_PORT=${SYNC_PORT:-9999}
# Cluster network settings, forwarded into the containers; run_node.sh detects unset ones
export IBDEVICES GLOO_SOCKET_IFNAME
BENCH_MODEL_NAME=${BENCH_MODEL_NAME:-pyt_kvcache_transfer_bench_all}
KV_CACHE_TEST_PATH=/workspace/kvcache_transfer_bench

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Plain sbatch runs a spooled copy of this script, so SCRIPT_DIR is not the checkout
[ -f "$PROJECT_DIR/scripts/run_node.sh" ] || PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from scripts/kvcache_transfer_bench}"
KV_CACHE_TEST_REPO="$(cd "$PROJECT_DIR" && pwd)/"
DOCKER_CONTAINER_NAME=container_kv_transfer_perf_bench
# Get node names from Slurm (recover from SLURM_JOB_ID if SLURM_NODELIST is empty)
if [ -z "${SLURM_NODELIST:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    SLURM_NODELIST=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP '(?<!\w)NodeList=\K\S+' | grep -v '(null)' | head -1)
    echo "INFO: Recovered SLURM_NODELIST='$SLURM_NODELIST' from SLURM_JOB_ID=$SLURM_JOB_ID"
fi
# Expand a hostlist without assuming one scontrol spelling. `hostnames` and
# `hostname` are aliases on stock Slurm; a given build may have only one.
# An already-expanded comma list needs neither.
NODES=()
_expanded=
for _scontrol_cmd in hostnames hostname; do
    if _expanded=$(scontrol show "$_scontrol_cmd" "${SLURM_NODELIST:-}" 2>/dev/null) && [ -n "$_expanded" ]; then
        break
    fi
    _expanded=
done
if [ -n "$_expanded" ]; then
    NODES=($_expanded)
else
    NODES=($(tr ',' ' ' <<<"${SLURM_NODELIST:-}"))
fi
if [ ${#NODES[@]} -lt 2 ]; then
    echo "ERROR: Need 2 nodes (target, initiator), got ${#NODES[@]}: '${NODES[*]:-}'. SLURM_NODELIST='${SLURM_NODELIST:-}' SLURM_JOB_ID='${SLURM_JOB_ID:-}'"
    exit 1
fi
NODE1=${NODES[0]}
NODE2=${NODES[1]}
echo "INFO: NODE1=$NODE1 NODE2=$NODE2 (${#NODES[@]} node(s))"
echo "INFO: BENCH_MODEL_NAME=$BENCH_MODEL_NAME BENCH_BACKENDS=$BENCH_BACKENDS"
echo "INFO: IBDEVICES=${IBDEVICES:-<detect>} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<detect>}"
echo "INFO: KV_CACHE_TEST_REPO=$KV_CACHE_TEST_REPO -> $KV_CACHE_TEST_PATH"

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
LOCAL_IP=$(hostname -I | awk '"'"'{print $1}'"'"')

# Run metadata for to_perf_csv.py; madengine does not export it into the
# container. Slurm may report the GRES count as "gpu:8", hence the strip.
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE##*:}
[ -n "$GPUS_PER_NODE" ] || GPUS_PER_NODE=$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)

# Default to a writable /sys. Some docker authz policies reject that and
# require the mount to be read-only; detection only reads sysfs, so either works.
SYS_VOL=/sys:/sys
_sys_probe=$(mktemp)
if ! docker run --rm --network host --entrypoint /bin/true -v /sys:/sys '"$DOCKER_IMAGE"' >"$_sys_probe" 2>&1; then
    if grep -qi "read-only" "$_sys_probe"; then
        echo "INFO: docker requires /sys mounted read-only on $(hostname)"
        SYS_VOL=/sys:/sys:ro
    else
        cat "$_sys_probe" >&2
        rm -f "$_sys_probe"
        exit 1
    fi
fi
rm -f "$_sys_probe"

docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --hostname "$LOCAL_HOSTNAME" \
    --add-host "$LOCAL_HOSTNAME:$LOCAL_IP" \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --privileged=true \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    -v "$HOME:$HOME" \
    -v "$SYS_VOL" \
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
    -e IBDEVICES \
    -e GLOO_SOCKET_IFNAME \
    ${MORI_ROOT:+-e MORI_ROOT} \
    --name '"$DOCKER_CONTAINER_NAME"' \
    '"$DOCKER_IMAGE"' /bin/bash -c "'"$DRIVER_SCRIPT"' --node1 '"${NODE1}"' --node2 '"${NODE2}"' --kv-cache-test-path '"$KV_CACHE_TEST_PATH"' --shared-folder '"$KV_CACHE_TEST_PATH"' --backends '"${BENCH_BACKENDS:-all}"' --start-size '"$START_SIZE"' --stop-size '"$STOP_SIZE"' --sync-port '"$SYNC_PORT"' --model-name '"$BENCH_MODEL_NAME"'"
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