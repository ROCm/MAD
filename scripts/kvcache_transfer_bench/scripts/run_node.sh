#!/bin/bash
NODE1=""
NODE2=""
KV_CACHE_TEST_PATH="/workspace/kvcache_transfer_bench"
SHARED_FOLDER=""
BENCH_BACKENDS="all"
START_SIZE="4096" # 4kb
STOP_SIZE="1073741824" # 1GB
IBDEVICES="mlx5_0"
SYNC_PORT="9999"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node1)
            NODE1="$2"
            shift 2
            ;;
        --node2)
            NODE2="$2"
            shift 2
            ;;
        --kv-cache-test-path)
            KV_CACHE_TEST_PATH="$2"
            shift 2
            ;;
        --shared-folder)
            SHARED_FOLDER="$2"
            shift 2
            ;;
        --backends)
            BENCH_BACKENDS="$2"
            shift 2
            ;;
        --start-size)
            START_SIZE="$2"
            shift 2
            ;;
        --stop-size)
            STOP_SIZE="$2"
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

BENCH_BACKENDS=$(echo "$BENCH_BACKENDS" | tr ',' ' ')
if [ "$BENCH_BACKENDS" = "all" ]; then
    BENCH_BACKENDS="mooncake mori rixl"
fi

JOB_ID=${SLURM_JOB_ID:-${JOB_ID:-unknown}}
SHARED_FOLDER_BASE="${SHARED_FOLDER:-${KV_CACHE_TEST_PATH}}"
SHARED_FOLDER="${SHARED_FOLDER_BASE%/}/shared/results_${JOB_ID}"

# UCX/GLOO env for RDMA/ROCm transfers (used by mori, rixl, mooncake)
export GLOO_SOCKET_IFNAME=eth0
export UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp
export UCX_NET_DEVICES="${IBDEVICES}:1"
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp
export UCX_SOCKADDR_CM_ENABLE=y
export UCX_MEMTYPE_CACHE=y
export UCX_RDMA_CM_ENABLED=y
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=4k
export UCX_ROCM_IPC_MIN_ZCOPY=0
export HSA_ENABLE_SDMA=1
export UCX_LOG_LEVEL=error
export NIXL_LOG_LEVEL=WARN

# Require essential args
missing=
[ -z "$NODE1" ]       && missing="${missing}--node1 "
[ -z "$NODE2" ]       && missing="${missing}--node2 "

if [ -n "$missing" ]; then
    echo "ERROR: Required arguments not set: $missing" >&2
    echo "Usage: $0 --node1 HOST --node2 HOST [--kv-cache-test-path PATH] [--shared-folder PATH] [--backends BACKENDS] [--start-size N] [--stop-size N] [--sync-port N] [--ibdevice DEV]" >&2
    exit 1
fi

mkdir -p "$SHARED_FOLDER"

# Resolve current hostname on the fly (hostname command works when $HOSTNAME is unset)
CURRENT_HOST=$(hostname)
echo "CURRENT_HOST: $CURRENT_HOST"

for backend in $BENCH_BACKENDS; do
    echo "=== Starting backend: $backend on $(hostname) (NODE1=$NODE1 CURRENT_HOST=$CURRENT_HOST) ==="
    if [ "$NODE1" = "$CURRENT_HOST" ]; then
        echo "  Role: TARGET (node matches NODE1)"
        python3 $KV_CACHE_TEST_PATH/backends/$backend/target_bench.py \
            --target_node "$NODE1" \
            --initiator_node "$NODE2" \
            --shared_folder "$SHARED_FOLDER" \
            --start_size $START_SIZE \
            --end_size $STOP_SIZE \
            --sync-port "$SYNC_PORT" 2>&1
    else
        echo "  Role: INITIATOR (node does not match NODE1)"
        python3 $KV_CACHE_TEST_PATH/backends/$backend/initiator_bench.py \
            --target_node "$NODE1" \
            --initiator_node "$NODE2" \
            --shared_folder "$SHARED_FOLDER" \
            --start_size $START_SIZE \
            --end_size $STOP_SIZE \
            --sync-port "$SYNC_PORT" 2>&1
    fi
    echo "=== Backend $backend finished with exit code $? ==="
done

# Merge results and generate report (run only on the initiator node)
if [ "$NODE1" != "$CURRENT_HOST" ]; then
    echo "=== Merging results and generating report ==="
    python3 $KV_CACHE_TEST_PATH/scripts/merge_results.py \
        --input-dir "$SHARED_FOLDER" \
        --output "$SHARED_FOLDER/results_merged.json"
    echo "=== Merge complete (exit code $?) ==="

    # Copy SLURM output/error files into the shared results folder
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -n "${JOB_ID:-}" ] && [ "$JOB_ID" != "unknown" ]; then
        for ext in out err; do
            src="${SLURM_SUBMIT_DIR}/kv_cache_perf_bench_${JOB_ID}.${ext}"
            if [ -f "$src" ]; then
                cp "$src" "$SHARED_FOLDER/" && echo "Copied $(basename "$src") -> $SHARED_FOLDER/"
            fi
        done
    fi
fi

