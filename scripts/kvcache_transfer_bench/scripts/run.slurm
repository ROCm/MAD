#!/bin/bash
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
IBDEVICES=${IBDEVICES:-${IBDEVICE:-mlx5_0}}
SYNC_PORT=${SYNC_PORT:-9999}
KV_CACHE_TEST_PATH=/workspace/kvcache_transfer_bench

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KV_CACHE_TEST_REPO="$(cd "$PROJECT_DIR" && pwd)/"
DOCKER_CONTAINER_NAME=container_kv_transfer_perf_bench
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
    --name '"$DOCKER_CONTAINER_NAME"' \
    '"$DOCKER_IMAGE"' /bin/bash -c "'"$DRIVER_SCRIPT"' --node1 '"${NODE1}"' --node2 '"${NODE2}"' --kv-cache-test-path '"$KV_CACHE_TEST_PATH"' --shared-folder '"$KV_CACHE_TEST_PATH"' --backends '"${BENCH_BACKENDS:-all}"' --start-size '"$START_SIZE"' --stop-size '"$STOP_SIZE"' --sync-port '"$SYNC_PORT"' --ibdevice '"$IBDEVICES"'"
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