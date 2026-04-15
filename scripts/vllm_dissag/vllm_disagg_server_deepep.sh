#!/bin/bash
# vLLM Disaggregated Server - DeepEP + DBO Configuration
# =============================================================================
# Co-located proxy topology (matches run_xPyD_models.slurm).
#
# Node roles (by NODE_RANK):
#   0               -> Prefill MASTER + Proxy  (co-located, API server + DP coordinator + router)
#   1  .. xP-1      -> Prefill CHILD           (--headless, no API server)
#   xP              -> Decode MASTER           (API server, DP coordinator)
#   xP+1 .. end     -> Decode CHILD            (--headless, no API server)
#
# Total nodes = xP + yD

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH="${MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V3}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"

echo "Listing NIXL_COOKBOOK_PATH : "
ls "${NIXL_COOKBOOK_PATH}"

# =============================================================================
# Port Configuration
# =============================================================================

SERVER_PORT=2584
RPC_PORT=13345
KV_PORT=14600
BARRIER_PORT="${BARRIER_PORT:-15000}"

PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
ROUTER_PORT="${ROUTER_PORT:-18001}"
PROXY_PORT="${ROUTER_PORT}"

if [[ "$PROXY_TYPE" != "vllm_router" && "$PROXY_TYPE" != "toy_proxy" ]]; then
    echo "Error: Invalid PROXY_TYPE='$PROXY_TYPE'. Must be 'vllm_router' or 'toy_proxy'." >&2
    exit 1
fi

# =============================================================================
# DeepEP / DBO / Profiling Configuration
# =============================================================================

PREFILL_DEEPEP_BACKEND="${PREFILL_DEEPEP_BACKEND:-deepep_high_throughput}"
DECODE_DEEPEP_BACKEND="${DECODE_DEEPEP_BACKEND:-deepep_low_latency}"
ENABLE_DBO="${ENABLE_DBO:-false}"
DBO_COMM_SMS="${DBO_COMM_SMS:-}"
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"

DBO_ARGS=""
[[ "${ENABLE_DBO}" == "true" ]] && DBO_ARGS="--enable-dbo"

# =============================================================================
# Node-Specific Configuration
# =============================================================================

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

PREFILL_DP_SIZE=$((xP * 8))
DECODE_DP_SIZE=$((yD * 8))
DP_SIZE_LOCAL=8

# Rank 0 is prefill master + proxy (co-located)
PREFILL_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' '{print $1}')
DECODE_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+1)}')
PREFILL_DP_START_RANK=$(( NODE_RANK * DP_SIZE_LOCAL ))
DECODE_DP_START_RANK=$(( (NODE_RANK - xP) * DP_SIZE_LOCAL ))

echo "============================================="
echo "DeepEP Configuration for ${MODEL_NAME}"
echo "  Prefill backend : ${PREFILL_DEEPEP_BACKEND}"
echo "  Decode backend  : ${DECODE_DEEPEP_BACKEND}"
echo "  DBO enabled     : ${ENABLE_DBO}"
echo "  DBO COMM_SMS    : ${DBO_COMM_SMS:-<envs.py default>}"
echo "  Profiling       : ${ENABLE_PROFILING}"
echo "  Server port     : ${SERVER_PORT}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  Prefill DP size : ${PREFILL_DP_SIZE}  (xP=${xP})"
echo "  Decode DP size  : ${DECODE_DP_SIZE}  (yD=${yD})"
echo "  DP size local   : ${DP_SIZE_LOCAL}"
echo "  Prefill master  : ${PREFILL_MASTER_ADDR}"
echo "  Decode master   : ${DECODE_MASTER_ADDR}"
echo "  Local IP        : ${host_ip}"
echo "  NODE_RANK       : ${NODE_RANK}"
echo "============================================="

# =============================================================================
# Helper Functions
# =============================================================================

setup_deepep_env() {
    local backend=$1

    export ROCSHMEM_HEAP_SIZE=7524589824
    export ROCSHMEM_MAX_NUM_CONTEXTS=256
    export HSA_NO_SCRATCH_RECLAIM=1

    # --- Auto-detect RocSHMEM directory ---
    if [[ -z "${ROCSHMEM_DIR}" ]]; then
        for _d in /root/rocshmem /opt/rocshmem; do
            [[ -d "$_d/lib" ]] && export ROCSHMEM_DIR="$_d" && break
        done
        export ROCSHMEM_DIR="${ROCSHMEM_DIR:-/root/rocshmem}"
    fi

    # --- Auto-detect OMPI directory ---
    if [[ -z "${OMPI_DIR}" ]]; then
        for _d in /root/install/ompi /usr/lib/x86_64-linux-gnu/openmpi /opt/ompi; do
            [[ -d "$_d" ]] && export OMPI_DIR="$_d" && break
        done
        export OMPI_DIR="${OMPI_DIR:-/root/install/ompi}"
    fi

    # --- Auto-detect UCX lib directory ---
    local _ucx_lib=""
    for _d in /root/install/ucx/lib /usr/local/ucx/lib /opt/ucx/lib; do
        [[ -f "$_d/libucp.so" ]] && _ucx_lib="$_d" && break
    done
    : "${_ucx_lib:=/root/install/ucx/lib}"

    # --- Auto-detect NIXL lib directory ---
    local _nixl_lib=""
    for _d in /opt/nixl/lib \
              /usr/local/RIXL/install/lib/x86_64-linux-gnu \
              /usr/local/lib/python3.12/dist-packages/.rixl.mesonpy.libs \
              /usr/local/nixl/lib; do
        [[ -f "$_d/libnixl.so" && -d "$_d/plugins" ]] && _nixl_lib="$_d" && break
    done
    if [[ -z "$_nixl_lib" ]]; then
        for _d in /opt/nixl/lib \
                  /usr/local/RIXL/install/lib/x86_64-linux-gnu \
                  /usr/local/lib/python3.12/dist-packages/.rixl.mesonpy.libs \
                  /root/RIXL/build/src/core \
                  /usr/local/nixl/lib; do
            [[ -f "$_d/libnixl.so" ]] && _nixl_lib="$_d" && break
        done
    fi

    # --- Fix NIXL Python bindings if missing ---
    if ! python3 -c "import nixl" 2>/dev/null; then
        local _rixl_py
        _rixl_py=$(find /root/RIXL/build -path "*/bindings/python" -type d 2>/dev/null | head -1)
        if [[ -n "$_rixl_py" ]]; then
            echo "[setup_deepep_env] NIXL Python bindings missing, installing from $_rixl_py ..."
            pip install --no-deps -e "$_rixl_py" 2>/dev/null || true
        fi
    fi

    # --- Build LD_LIBRARY_PATH ---
    local _extra_ld=""
    [[ -n "$_nixl_lib" ]] && _extra_ld+="${_nixl_lib}:"
    _extra_ld+="${_ucx_lib}:/usr/local/lib:/usr/local/lib64:/opt/rocm/lib"
    export LD_LIBRARY_PATH="${_extra_ld}:${LD_LIBRARY_PATH}"

    echo "[setup_deepep_env] ROCSHMEM_DIR=$ROCSHMEM_DIR  OMPI_DIR=$OMPI_DIR"
    echo "[setup_deepep_env] UCX_LIB=$_ucx_lib  NIXL_LIB=${_nixl_lib:-<not found>}"

    # --- vLLM runtime flags ---
    export VLLM_USE_V1=1
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_ALL2ALL_BACKEND="${backend}"
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    export VLLM_ROCM_USE_AITER_RMSNORM=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_USE_AITER_TRITON_SILU_MUL=0
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_ROCM_USE_AITER_MOE=0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600

    export VLLM_NIXL_SIDE_CHANNEL_HOST="${host_ip}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557

    # --- Network / RDMA ---
    export GLOO_SOCKET_IFNAME=eth0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=1
    export NCCL_NET_GDR_LEVEL=PHB

    export UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp
    if [[ -z "${UCX_NET_DEVICES}" ]]; then
        local available_devs
        available_devs=$(ibstat 2>/dev/null | awk '
            /^CA /{gsub(/\047/,"",$2); ca=$2}
            /Rate:/{if($2+0 >= 200) devs=devs (devs?",":"") ca":1"}
            END{print devs}')
        export UCX_NET_DEVICES="${available_devs:-mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
    fi
    if [[ -z "${NCCL_IB_HCA}" ]]; then
        local nccl_hcas
        nccl_hcas=$(ibstat 2>/dev/null | awk '
            /^CA /{gsub(/\047/,"",$2); ca=$2}
            /Rate:/{if($2+0 >= 200) devs=devs (devs?",":"") ca}
            END{print devs}')
        export NCCL_IB_HCA="${nccl_hcas:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    fi
    export UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp
    export UCX_SOCKADDR_CM_ENABLE=y
    export UCX_RDMA_CM_ENABLED=y
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-INFO}"

    [[ -n "${DBO_COMM_SMS}" ]] && export VLLM_DBO_COMM_SMS="${DBO_COMM_SMS}"

    # --- PR #39276: Fix NIXL engine_id collision in multi-node DP ---
    local _core_py
    _core_py=$(python3 -c "import vllm.v1.engine.core as m; print(m.__file__)" 2>/dev/null)
    if [[ -n "$_core_py" && -f "$_core_py" ]]; then
        if grep -q '_dp{local_dp_rank}' "$_core_py"; then
            sed -i 's/_dp{local_dp_rank}/_dp{dp_rank}/g' "$_core_py"
            echo "[setup_deepep_env] Applied PR#39276 fix: engine_id uses dp_rank (core.py)"
        fi
    fi
    local _utils_py
    _utils_py=$(python3 -c "import vllm.v1.engine.utils as m; print(m.__file__)" 2>/dev/null)
    if [[ -n "$_utils_py" && -f "$_utils_py" ]]; then
        if grep -q '_dp{local_index}' "$_utils_py"; then
            sed -i 's/_dp{local_index}/_dp{index}/g' "$_utils_py"
            echo "[setup_deepep_env] Applied PR#39276 fix: engine_id uses index (utils.py)"
        fi
    fi

    echo "[setup_deepep_env] UCX_NET_DEVICES=$UCX_NET_DEVICES"
    echo "[setup_deepep_env] NCCL_IB_HCA=$NCCL_IB_HCA"
}

build_kv_transfer_config() {
    local kv_role="$1"
    local engine_id="$2"
    local dp_size="$3"
    echo "{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${engine_id}\", \"kv_role\": \"${kv_role}\", \"kv_parallel_size\": ${dp_size}, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"${host_ip}\", \"kv_port\": ${KV_PORT}}"
}

# Launch a vllm serve worker and set WORKER_PID to its PID.
#   $1 = role           "prefill_master" | "prefill_child" | "decode_master" | "decode_child"
#   $2 = backend        DeepEP all2all backend name
#   $3 = dp_size        data-parallel size
#   $4 = dp_addr        data-parallel master address
#   $5 = kv_role        "kv_producer" or "kv_consumer"
#   $6 = engine_id      "pd-prefill" or "pd-decode"
#   $7 = log_prefix     "prefill" or "decode"
#   $8 = dp_start_rank  (only for child nodes)
launch_vllm_worker() {
    local role="$1"
    local backend="$2"
    local dp_size="$3"
    local dp_addr="$4"
    local kv_role="$5"
    local engine_id="$6"
    local log_prefix="$7"
    local dp_start_rank="${8:-}"

    setup_deepep_env "${backend}"

    local extra_args=()
    case "$role" in
        *_master)
            extra_args+=(--api-server-count=8)
            extra_args+=(--data-parallel-start-rank 0)
            ;;
        *_child)
            extra_args+=(--data-parallel-start-rank "${dp_start_rank}")
            extra_args+=(--headless)
            ;;
    esac

    # Decode roles get cudagraph; prefill uses enforce-eager
    local compile_args=()
    case "$role" in
        decode_*)
            compile_args+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}')
            compile_args+=(--cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256)
            ;;
        prefill_*)
            compile_args+=(--enforce-eager)
            ;;
    esac

    local kv_config
    kv_config=$(build_kv_transfer_config "${kv_role}" "${engine_id}" "${dp_size}")

    vllm serve "${MODEL_PATH}" \
        --port "${SERVER_PORT}" \
        --trust-remote-code \
        -tp 1 \
        --data-parallel-size "${dp_size}" \
        --data-parallel-size-local "${DP_SIZE_LOCAL}" \
        --data-parallel-address "${dp_addr}" \
        --data-parallel-rpc-port "${RPC_PORT}" \
        --master-addr "${dp_addr}" \
        "${compile_args[@]}" \
        --no-enable-prefix-caching --block-size 1 \
        --gpu-memory-utilization 0.8 \
        --kv-cache-dtype fp8 \
        --enable-expert-parallel \
        --all2all-backend "${backend}" \
        ${DBO_ARGS} \
        "${extra_args[@]}" \
        --kv-transfer-config "${kv_config}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &

    WORKER_PID=$!
}

wait_for_proxy_and_cleanup() {
    local worker_pid="$1"
    local label="$2"

    echo "Waiting for proxy server to be up..."
    python "${NIXL_COOKBOOK_PATH}/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports "${PROXY_PORT}"

    echo "Waiting until proxy server closes..."
    python "${NIXL_COOKBOOK_PATH}/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port "${PROXY_PORT}"

    echo "Killing the ${label} server"
    pkill -P "${worker_pid}" 2>/dev/null; kill "${worker_pid}" 2>/dev/null || true
}

print_node_info() {
    local role_desc="$1"
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is ${role_desc}."
    echo "========================================="
}

# =============================================================================
# Container Synchronization
# =============================================================================

for _pid in $(ss -tlnp sport = "${BARRIER_PORT}" 2>/dev/null | grep -oP "pid=\K\d+"); do
    kill -9 "$_pid" 2>/dev/null
done
sleep 2

echo "Waiting at the container creation barrier on ${host_name}"
python "${NIXL_COOKBOOK_PATH}/socket_barrier.py" \
    --local-ip "${host_ip}" \
    --local-port "${BARRIER_PORT}" \
    --enable-port \
    --node-ips "${IPADDRS}" \
    --node-ports "${BARRIER_PORT}"

sleep 3

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

PREFILL_MASTER_IP="${IP_ARRAY[0]}"
DECODE_MASTER_IP="${IP_ARRAY[$xP]}"
MASTER_IPS="${PREFILL_MASTER_IP},${DECODE_MASTER_IP}"

if [ "$NODE_RANK" -eq 0 ]; then
    # =================================================================
    # Rank 0: Prefill MASTER + Proxy (co-located)
    # =================================================================
    print_node_info "Prefill master + Proxy node (co-located)"
    echo "Prefill master IP : ${PREFILL_MASTER_IP}"
    echo "Decode master IP  : ${DECODE_MASTER_IP}"
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}  PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
    echo "vLLM serve port: ${SERVER_PORT}  Proxy port: ${PROXY_PORT}"

    launch_vllm_worker "prefill_master" "${PREFILL_DEEPEP_BACKEND}" \
        "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" \
        "kv_producer" "pd-prefill" "prefill"

    local_worker_pid="${WORKER_PID}"

    echo "Waiting for prefill & decode master servers to start..."

    TIMEOUT_SECONDS=4000
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="Application startup complete."

    PREFILL_LOG="/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log"
    DECODE_LOG="/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log"

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

    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        echo "Starting vLLM Router (Production Proxy) on port ${PROXY_PORT}..."
        [ -f /root/.cargo/env ] && source /root/.cargo/env

        PREFILL_URLS="--prefill http://${PREFILL_MASTER_IP}:${SERVER_PORT}"
        DECODE_URLS="--decode http://${DECODE_MASTER_IP}:${SERVER_PORT}"

        UCX_TLS=tcp,self,shm VLLM_USE_V1=1 \
        vllm-router \
            --host 0.0.0.0 \
            --port "${ROUTER_PORT}" \
            --vllm-pd-disaggregation \
            $PREFILL_URLS \
            $DECODE_URLS \
            --policy round_robin \
            --prefill-policy round_robin \
            --decode-policy round_robin \
            --intra-node-data-parallel-size 1 \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
    else
        echo "Starting Toy Proxy Server on port ${PROXY_PORT}..."

        UCX_TLS=tcp,self,shm NCCL_UCX_TLS=tcp VLLM_USE_V1=1 \
        python3 "/app/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
            --host 0.0.0.0 \
            --port "${PROXY_PORT}" \
            --prefiller-hosts "${PREFILL_MASTER_IP}" \
            --prefiller-ports "${SERVER_PORT}" \
            --decoder-hosts "${DECODE_MASTER_IP}" \
            --decoder-ports "${SERVER_PORT}" \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    python "${NIXL_COOKBOOK_PATH}/socket_barrier.py" \
        --node-ips "${host_ip}" \
        --node-ports "${PROXY_PORT}"

    echo "Proxy ready for benchmarking on ${host_name}:${host_ip}:${PROXY_PORT}"

    sleep 10
    export BENCHMARK_PORT="${PROXY_PORT}"
    bash "${NIXL_COOKBOOK_PATH}/benchmark_xPyD.sh"

    echo "Killing proxy server"
    pkill -P "${proxy_pid}" 2>/dev/null; kill "${proxy_pid}" 2>/dev/null || true
    echo "Killing prefill master server"
    pkill -P "${local_worker_pid}" 2>/dev/null; kill "${local_worker_pid}" 2>/dev/null || true

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    # =================================================================
    # Prefill CHILD (--headless, no API server; only when xP > 1)
    # =================================================================
    print_node_info "Prefill child node"
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}  PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK}"

    launch_vllm_worker "prefill_child" "${PREFILL_DEEPEP_BACKEND}" \
        "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" \
        "kv_producer" "pd-prefill" "prefill" "${PREFILL_DP_START_RANK}"

    wait_for_proxy_and_cleanup "${WORKER_PID}" "prefill child"

elif [ "$NODE_RANK" -eq "$xP" ]; then
    # =================================================================
    # Decode MASTER (API server + DP coordinator)
    # =================================================================
    print_node_info "Decode master node"
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}  DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"

    launch_vllm_worker "decode_master" "${DECODE_DEEPEP_BACKEND}" \
        "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" \
        "kv_consumer" "pd-decode" "decode"

    wait_for_proxy_and_cleanup "${WORKER_PID}" "decode master"

else
    # =================================================================
    # Decode CHILD (--headless, no API server; rank > xP)
    # =================================================================
    print_node_info "Decode child node"
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}  DECODE_DP_START_RANK=${DECODE_DP_START_RANK}"

    launch_vllm_worker "decode_child" "${DECODE_DEEPEP_BACKEND}" \
        "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" \
        "kv_consumer" "pd-decode" "decode" "${DECODE_DP_START_RANK}"

    wait_for_proxy_and_cleanup "${WORKER_PID}" "decode child"

fi

echo "Script completed successfully"
exit 0
