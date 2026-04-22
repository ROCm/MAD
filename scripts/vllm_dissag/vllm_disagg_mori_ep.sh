#!/bin/bash
# VLLM Disaggregated Server Launcher - MoRI EP Configuration
# =============================================================================
# Supports multi-node xP/yD topologies with co-located proxy on NODE 0.
# Applies vLLM PR #39276 patches at runtime for multi-node DP support.
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
echo "MoRI EP topology: xP=${xP} yD=${yD} (total nodes=$((xP + yD)))"
IPADDRS="${IPADDRS:-localhost}"
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Port Configuration
# =============================================================================

RPC_PORT="${MORI_RPC_PORT:-13345}"
SERVE_PORT="${MORI_SERVE_PORT:-20005}"
KV_PORT="${MORI_KV_PORT:-9711}"
PROXY_PORT="${MORI_PROXY_PORT:-10001}"
PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}"
LOCAL_PING_PORT="${MORI_LOCAL_PING_PORT:-61555}"
HANDSHAKE_PORT="${MORI_HANDSHAKE_PORT:-8405}"
NOTIFY_PORT="${MORI_NOTIFY_PORT:-61005}"

# =============================================================================
# Node-Specific Configuration
# =============================================================================

_GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PREFILL_DP_SIZE=$((xP * _GPUS_PER_NODE))
DECODE_DP_SIZE=$((yD * _GPUS_PER_NODE))
DP_PARALLEL_SIZE_LOCAL=${_GPUS_PER_NODE}
PREFILL_DP_START_RANK=$(( NODE_RANK * _GPUS_PER_NODE ))
PREFILL_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' '{print $1}')
DECODE_DP_START_RANK=$(( (NODE_RANK - xP) * _GPUS_PER_NODE ))
DECODE_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+1)}')

echo "-----------------------------Printing node specific details ----------------------"
echo "IPADDRS = ${IPADDRS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "HOST_IP=$(hostname -I)"
echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
echo "PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK}"
echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
echo "DECODE_DP_START_RANK=${DECODE_DP_START_RANK}"
echo "DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"
host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

# =============================================================================
# Helper Functions
# =============================================================================

setup_mori_env() {
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    export VLLM_USE_AITER_TRITON_SILU_MUL=0

    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ALL2ALL_BACKEND=mori

    # Network — values from cluster profile via Docker env, or defaults
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"

    # Timeouts — generous values so AITER JIT compilation on the first run
    # doesn't trip internal watchdogs.
    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"
    export VLLM_RINGBUFFER_WARNING_INTERVAL="${VLLM_RINGBUFFER_WARNING_INTERVAL:-3600}"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300000}"

    # RDMA / NCCL tuning — cluster-specific via Docker env
    export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-3}"
    export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"

    # MoRI EP QP tuning
    export MORI_NUM_QP_PER_PE="${MORI_NUM_QP_PER_PE:-4}"
    export VLLM_MORIIO_QP_PER_TRANSFER="${VLLM_MORIIO_QP_PER_TRANSFER:-4}"
    export VLLM_MORIIO_NUM_WORKERS="${VLLM_MORIIO_NUM_WORKERS:-4}"

    # MoRIIO robustness timeouts (used by PR #39276 patches)
    export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"
    export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-1800}"
    export VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-30}"

    # Compilation caches — host-local bind-mount avoids NFS file-lock races
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vllm_cache/triton}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache/vllm}"
    export COMGR_CACHE_DIR="${COMGR_CACHE_DIR:-/tmp/vllm_cache/comgr}"
    export AITER_JIT_DIR="${AITER_JIT_DIR:-/tmp/vllm_cache/aiter_jit}"
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${COMGR_CACHE_DIR}" "${AITER_JIT_DIR}" 2>/dev/null || true

    # Pre-populate AITER tuning CSVs to prevent CSV race condition
    if [[ "${VLLM_ROCM_USE_AITER:-1}" == "1" ]]; then
        local _aiter_cfgs="/tmp/aiter_configs"
        local _aiter_src="/usr/local/lib/python3.12/dist-packages/aiter/configs"
        if [ -d "${_aiter_src}" ] && [ ! -f "${_aiter_cfgs}/a8w8_blockscale_tuned_gemm.csv" ]; then
            mkdir -p "${_aiter_cfgs}"
            cp "${_aiter_src}"/*.csv "${_aiter_cfgs}/" 2>/dev/null || true
        fi
    fi

    # GPU / ROCm tuning
    export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
    export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
    export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
    export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

    # RocSHMEM
    export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-8589934592}"
    export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-256}"
}

build_kv_transfer_config() {
    local kv_role="$1"
    echo '{"kv_connector":"MoRIIOConnector","kv_role":"'"${kv_role}"'","kv_port":"'"${KV_PORT}"'","kv_connector_extra_config":{"proxy_ip":"'"${MASTER_ADDR}"'","proxy_port":"'"${PROXY_PORT}"'","proxy_ping_port":"'"${PROXY_PING_PORT}"'","http_port":"'"${SERVE_PORT}"'","local_ping_port":"'"${LOCAL_PING_PORT}"'","handshake_port":"'"${HANDSHAKE_PORT}"'","notify_port":"'"${NOTIFY_PORT}"'"}}'
}

# Launch a vllm serve worker and set WORKER_PID to its PID.
#   $1 = dp_size       data-parallel size
#   $2 = dp_addr        data-parallel master address
#   $3 = kv_role        "kv_producer" or "kv_consumer"
#   $4 = log_prefix     "prefill" or "decode"
#   $5 = role           "master" or "child"
#   $6 = dp_start_rank  (required for child nodes)
launch_vllm_worker() {
    local dp_size="$1"
    local dp_addr="$2"
    local kv_role="$3"
    local log_prefix="$4"
    local role="$5"
    local dp_start_rank="${6:-}"

    setup_mori_env

    local extra_args=()
    local kv_args=()

    if [[ "$role" == "master" ]]; then
        extra_args+=(--api-server-count=${_GPUS_PER_NODE})
        # Fix 6: only master nodes get --kv-transfer-config.
        # Child nodes join via --headless and participate in EP all-to-all
        # but do not perform KV transfers.
        local kv_config
        kv_config=$(build_kv_transfer_config "${kv_role}")
        kv_args+=(--kv-transfer-config "${kv_config}")
    else
        extra_args+=(--data-parallel-start-rank "${dp_start_rank}" --headless)
    fi

    # Patch PyTorch's default_pg_timeout so DP Gloo groups use our timeout
    # instead of the 30-min default.
    local _timeout_s="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
    local _torch_const="/usr/local/lib/python3.12/dist-packages/torch/distributed/constants.py"
    if [ -f "$_torch_const" ]; then
        sed -i "s/default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT/default_pg_timeout: timedelta = timedelta(seconds=${_timeout_s})/" "$_torch_const" 2>/dev/null || true
    fi

    # Execution mode: prefill always uses eager; decode can optionally use
    # CUDA graphs via VLLM_CUDAGRAPH_MODE (e.g. FULL_DECODE_ONLY).
    # --enforce-eager overrides --compilation-config, so they are mutually exclusive.
    local exec_args=()
    local _cudagraph_mode="${VLLM_CUDAGRAPH_MODE:-}"
    if [[ "$log_prefix" == "decode" && -n "$_cudagraph_mode" && "$_cudagraph_mode" != "NONE" ]]; then
        local _capture_sizes="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8 16 32 64 128 256}"
        exec_args+=(--compilation-config '{"cudagraph_mode":"'"${_cudagraph_mode}"'","custom_ops":["+quant_fp8"]}')
        exec_args+=(--cudagraph-capture-sizes ${_capture_sizes})
    else
        exec_args+=(--enforce-eager)
    fi

    local profiler_args=()
    if [[ "${RUN_PROFILE:-0}" == "1" ]]; then
        local _profile_dir="/run_logs/${SLURM_JOB_ID}/profiles/${log_prefix}_NODE${NODE_RANK}"
        mkdir -p "${_profile_dir}"
        profiler_args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${_profile_dir}\"}")
        echo "Profiler enabled for ${log_prefix} ${role} NODE${NODE_RANK} → ${_profile_dir}"
    fi

    vllm serve ${MODEL_PATH} \
        -tp 1 \
        --data-parallel-size "${dp_size}" \
        --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
        --data-parallel-address "${dp_addr}" \
        --data-parallel-rpc-port ${RPC_PORT} \
        --enable-expert-parallel \
        --port ${SERVE_PORT} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.8} \
        --kv-cache-dtype fp8 \
        --block-size 1 \
        --no-enable-prefix-caching \
        --all2all-backend mori \
        --trust-remote-code \
        --distributed-timeout-seconds ${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
        "${profiler_args[@]}" \
        "${exec_args[@]}" \
        "${extra_args[@]}" \
        "${kv_args[@]}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &

    WORKER_PID=$!
}

wait_for_proxy_and_cleanup() {
    local worker_pid="$1"
    local label="$2"

    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $PROXY_PORT

    echo "Waiting until proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $PROXY_PORT

    echo "Killing the ${label} server"
    pkill -P "$worker_pid" 2>/dev/null; kill "$worker_pid" 2>/dev/null || true
}

print_node_info() {
    local role_desc="$1"
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is ${role_desc}."
}

# =============================================================================
# Container Synchronization
# =============================================================================

_BARRIER_PORT="${BARRIER_PORT_MORI:-2222}"
for _pid in $(ss -tlnp sport = ${_BARRIER_PORT} 2>/dev/null | grep -oP "pid=\K\d+"); do
    kill -9 "$_pid" 2>/dev/null
done
sleep 2

echo "Waiting at the container creation barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${_BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${_BARRIER_PORT}

# =============================================================================
# Runtime Patches — Apply vLLM PR #39276 for multi-node DP support
# =============================================================================

PATCH_SCRIPT="${NIXL_COOKBOOK_PATH:-$(dirname "$0")}/apply_moriio_2pd_patches.sh"
_PATCH_REQUIRED=0
if [ "$xP" -gt 1 ] || [ "$yD" -gt 1 ]; then
    _PATCH_REQUIRED=1
fi

if [ -f "${PATCH_SCRIPT}" ]; then
    echo "Applying runtime patches (PR #39276)..."
    if ! bash "${PATCH_SCRIPT}" 2>&1; then
        if [ "$_PATCH_REQUIRED" -eq 1 ]; then
            echo "Error: runtime patch failed but multi-node DP requires PR #39276 (xP=${xP}, yD=${yD}). Aborting."
            exit 1
        fi
        echo "Warning: runtime patch failed — continuing (1P/1D does not strictly require it)"
    fi
else
    if [ "$_PATCH_REQUIRED" -eq 1 ]; then
        echo "Error: ${PATCH_SCRIPT} not found but multi-node DP requires PR #39276 (xP=${xP}, yD=${yD}). Aborting."
        exit 1
    fi
    echo "Warning: ${PATCH_SCRIPT} not found — skipping runtime patches"
fi

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    # =================================================================
    # Rank 0: Prefill master + Proxy (co-located)
    # =================================================================
    print_node_info "Prefill master + Proxy node (co-located)"
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
    echo "PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK}"
    echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"
    echo "vLLM serve port: ${SERVE_PORT}  Proxy port: ${PROXY_PORT}"

    launch_vllm_worker "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" "kv_producer" "prefill" "master"
    local_worker_pid=$WORKER_PID

    echo "Waiting for prefill & decode servers to be ready..."
    sleep 20

    TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="Application startup complete."

    PREFILL_LOG=/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log
    DECODE_LOG=/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log

    wait_log_signal_or_fail() {
        local LOG_FILE="$1"
        local LABEL="$2"
        local ELAPSED=0
        until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}" 2>/dev/null; do
            if [ "${ELAPSED}" -ge "${TIMEOUT_SECONDS}" ]; then
                echo "Timeout (${TIMEOUT_SECONDS}s): '${SEARCH_SIGNAL}' not found in ${LABEL}: ${LOG_FILE}" \
                    | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
                exit 1
            fi
            sleep "${SLEEP_SECONDS}"
            ELAPSED=$((ELAPSED + SLEEP_SECONDS))
        done
        echo "Ready: ${LABEL} (${LOG_FILE})"
    }

    wait_log_signal_or_fail "${PREFILL_LOG}" "prefill master"
    wait_log_signal_or_fail "${DECODE_LOG}" "decode master"

    sleep 10
    python /app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py \
        2>&1 | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &

    proxy_pid=$!

    echo "Proxy server ready for benchmarking on ${host_name}:${host_ip}:${PROXY_PORT}"
    sleep 20
    curl -X POST http://127.0.0.1:${PROXY_PORT}/v1/completions -H "Content-Type: application/json" -d '{
        "prompt": "Who is AMD CEO?",
        "temperature": 0,
        "max_tokens" : 10,
        "top_k": 1
    }'

    sleep 20
    export BENCHMARK_PORT=${PROXY_PORT}
    export DECODE_MASTER_ADDR
    export SERVE_PORT
    bash $NIXL_COOKBOOK_PATH/benchmark_xPyD.sh

    echo "Killing the proxy server.."
    pkill -P $proxy_pid 2>/dev/null; kill $proxy_pid 2>/dev/null || true
    echo "Killing the prefill master server.."
    pkill -P $local_worker_pid 2>/dev/null; kill $local_worker_pid 2>/dev/null || true

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    # =================================================================
    # Prefill child (only active when xP > 1)
    # =================================================================
    print_node_info "Prefill child node"
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
    echo "PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK}"
    echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"

    launch_vllm_worker "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" "kv_producer" "prefill" "child" "${PREFILL_DP_START_RANK}"
    wait_for_proxy_and_cleanup $WORKER_PID "prefill child"

elif [ "$NODE_RANK" -eq "$xP" ]; then
    # =================================================================
    # Decode master
    # =================================================================
    print_node_info "Decode master node"
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
    echo "DECODE_DP_START_RANK=${DECODE_DP_START_RANK}"
    echo "DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"

    launch_vllm_worker "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" "kv_consumer" "decode" "master"
    wait_for_proxy_and_cleanup $WORKER_PID "decode master"

else
    # =================================================================
    # Decode child (rank > xP)
    # =================================================================
    print_node_info "Decode child node"
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
    echo "DECODE_DP_START_RANK=${DECODE_DP_START_RANK}"
    echo "DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"

    launch_vllm_worker "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" "kv_consumer" "decode" "child" "${DECODE_DP_START_RANK}"
    wait_for_proxy_and_cleanup $WORKER_PID "decode child"

fi

echo "Script completed successfully."
exit 0
