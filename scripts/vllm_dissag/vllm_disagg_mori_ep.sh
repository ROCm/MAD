#!/bin/bash
# VLLM Disaggregated Server Launcher - MoRI EP Configuration
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
if [ "$xP" -gt 1 ] || [ "$yD" -gt 1 ]; then
    echo "Error: xP > 1 or yD > 1 is not supported yet due to MoRI IO connector issues." >&2
    exit 1
fi
IPADDRS="${IPADDRS:-localhost}"
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Port Configuration
# =============================================================================

RPC_PORT=13345
SERVE_PORT=20005
KV_PORT=9711
PROXY_PORT=10001
PROXY_PING_PORT=36367
LOCAL_PING_PORT=61555
HANDSHAKE_PORT=8405
NOTIFY_PORT=61005

# =============================================================================
# Node-Specific Configuration
# =============================================================================

PREFILL_DP_SIZE=$((xP * 8))
DECODE_DP_SIZE=$((yD * 8))
DP_PARALLEL_SIZE_LOCAL=8
PREFILL_DP_START_RANK=$(( NODE_RANK * 8 ))
PREFILL_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' '{print $1}')
DECODE_DP_START_RANK=$(( (NODE_RANK - xP) * 8 ))
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
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ALL2ALL_BACKEND=mori
    export GLOO_SOCKET_IFNAME=eth0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export VLLM_RINGBUFFER_WARNING_INTERVAL=3600
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
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
    if [[ "$role" == "master" ]]; then
        extra_args+=(--api-server-count=8)
    else
        extra_args+=(--data-parallel-start-rank "${dp_start_rank}" --headless)
    fi

    local kv_config
    kv_config=$(build_kv_transfer_config "${kv_role}")

    vllm serve ${MODEL_PATH} \
        -tp 1 \
        --data-parallel-size "${dp_size}" \
        --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
        --data-parallel-address "${dp_addr}" \
        --data-parallel-rpc-port ${RPC_PORT} \
        --enable-expert-parallel \
        --port ${SERVE_PORT} \
        --gpu-memory-utilization 0.8 \
        --kv-cache-dtype fp8 \
        --block-size 1 \
        --no-enable-prefix-caching \
        --all2all-backend mori \
        --trust-remote-code \
        --enforce-eager \
        "${extra_args[@]}" \
        --kv-transfer-config "${kv_config}" \
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

for _pid in $(ss -tlnp sport = 2222 2>/dev/null | grep -oP "pid=\K\d+"); do
    kill -9 "$_pid" 2>/dev/null
done
sleep 2

echo "Waiting at the container creation barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 2222 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 2222

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

    TIMEOUT_SECONDS=4000
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
