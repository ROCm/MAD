#!/bin/bash
# VLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask

#trap 'echo "Error occurred. Cleaning up..."; exit 0' ERR

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)
SERVER_PORT=2584
# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"full_cuda_graph\": false, \"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"full_cuda_graph\": false, \"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_ENVS=(
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["Llama-3.1-405B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["DeepSeek-V3"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0 "
    ["gpt-oss-120b"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0 "
)

# =============================================================================
# Configuration Selection Functions
# =============================================================================

get_model_config() {
    local mode="$1"
    local model_name="$2"
    
    if [[ "$mode" == "prefill" ]]; then
        if [[ -n "${MODEL_PREFILL_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_PREFILL_CONFIGS[$model_name]}"
        else
            echo "--tp-size 4"
        fi
    elif [[ "$mode" == "decode" ]]; then
        if [[ -n "${MODEL_DECODE_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_DECODE_CONFIGS[$model_name]}"
        else
            echo "--tp-size 4"
        fi
    fi
}

get_model_envs() {
    local model_name="$1"
    
    if [[ -n "${MODEL_ENVS[$model_name]}" ]]; then
        echo "${MODEL_ENVS[$model_name]}"
    else
        echo ""
    fi
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "Warning: MODEL_NAME not set, using default configurations"
    echo "ERROR: please set MODEL_NAME"
    exit 0
else
    PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
    DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
    PREFILL_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
    DECODE_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000

# =============================================================================
# ETCD Server Setup
# =============================================================================

echo "Proceeding to start etcd server on $host_name"

${NIXL_COOKBOOK_PATH}/start_etcd.sh > /dev/null &
etcd_pid=$!

echo "Waiting at etcd server barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --node-ips ${IPADDRS} \
    --node-ports 2379

echo "All etcd servers are up : $host_name"
sleep 3

echo "etcd endpoint health=================="
/usr/local/bin/etcd//etcdctl endpoint health
echo "======================================"

echo "etcd member list======================"
/usr/local/bin/etcd//etcdctl member list
echo "======================================"

echo "etcd status======================"
/usr/local/bin/etcd//etcdctl endpoint status --write-out=table
echo "======================================"


echo "Waiting at etcd server barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py --node-ips ${IPADDRS} --node-ports 2379
# END SECTION===========================================================================

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""
PREFILL_PORTS=""
DECODE_PORTS=""

# Loop through for `--prefill` (IPs from index 0 to N-1)
for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
    PREFILL_PORTS+="$SERVER_PORT "
done

# Loop through for `--decode` (IPs from N onward)
for ((i=xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
    DECODE_PORTS+="$SERVER_PORT "
done

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node"
    echo "${PREFILL_ARGS} are Proxy's Prefill"
    echo "${DECODE_ARGS} are Proxy's Decode"
    echo "================================================"
    

    # =============================================================================
    # Wait for PD servers
    # =============================================================================
    PD_IPADDRS="${IPADDRS#*,}"
    echo "Waiting for all prefill and decode servers to be up . . ."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${PD_IPADDRS} \
        --node-ports $SERVER_PORT

    if [[ -z "$UCX_NET_DEVICES" ]]; then
        echo "Error: UCX_NET_DEVICES is empty" >&2
        exit 1
    fi
    
    if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
        echo "Error: NCCL_SOCKET_IFNAME is empty" >&2
        exit 1
    fi

    UCX_TLS=tcp,self,shm NCCL_UCX_TLS=tcp VLLM_USE_V1=1 \
    python3 "/app/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
            --host 0.0.0.0 \
            --port $SERVER_PORT \
            --prefiller-hosts ${PREFILL_ARGS} \
            --prefiller-ports ${PREFILL_PORTS} \
            --decoder-hosts ${DECODE_ARGS} \
            --decoder-ports ${DECODE_PORTS} 2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    proxy_pid=$!
    
    echo "Waiting for proxy server to be up . . ."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${host_ip} \
        --node-ports $SERVER_PORT

    echo "Proxy Server Ready for benchmarking on ${host_name}:${host_ip}"


    sleep 10;
    bash $NIXL_COOKBOOK_PATH/benchmark_xPyD.sh

    echo "Killing the proxy server"
    kill $proxy_pid

elif  [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"

    PREFILL_CMD="LD_LIBRARY_PATH=/app/install/nixl/lib/x86_64-linux-gnu/:/app/install/ucx/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH \
    ${PREFILL_MODEL_ENVS} \
    VLLM_USE_V1=1 \
    VLLM_SERVER_DEV_MODE=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${host_ip} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
    UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp \
    UCX_NET_DEVICES=mlx5_0:1 \
    UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp \
    UCX_SOCKADDR_CM_ENABLE=y \
    UCX_RDMA_CM_ENABLED=y \
    UCX_MEMTYPE_CACHE=y \
    UCX_RNDV_SCHEME=get_zcopy \
    UCX_RNDV_THRESH=4k \
    UCX_ROCM_IPC_MIN_ZCOPY=0 \
    HSA_ENABLE_SDMA=1 \
    UCX_LOG_LEVEL=info \
    NIXL_LOG_LEVEL=DEBUG \
    HSA_ENABLE_SDMA=1 \
    vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"pd-run\", \"kv_role\": \"kv_producer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${host_ip}\"'\", \"kv_port\": 14600}'"

    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi

    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &

    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $SERVER_PORT

    echo "Waiting untill proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $SERVER_PORT

    echo "Killing the prefill server"
    kill $prefill_pid

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"


    DECODE_CMD="LD_LIBRARY_PATH=/app/install/nixl/lib/x86_64-linux-gnu/:/app/install/ucx/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH \
    ${DECODE_MODEL_ENVS} \
    VLLM_USE_V1=1 \
    VLLM_SERVER_DEV_MODE=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${host_ip} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
    UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp \
    UCX_NET_DEVICES=mlx5_0:1 \
    UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp \
    UCX_SOCKADDR_CM_ENABLE=y \
    UCX_RDMA_CM_ENABLED=y \
    UCX_MEMTYPE_CACHE=y \
    UCX_RNDV_SCHEME=get_zcopy \
    UCX_RNDV_THRESH=4k \
    UCX_ROCM_IPC_MIN_ZCOPY=0 \
    HSA_ENABLE_SDMA=1 \
    UCX_LOG_LEVEL=info \
    NIXL_LOG_LEVEL=DEBUG \
    HSA_ENABLE_SDMA=1 \
    vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"llama8b-run\", \"kv_role\": \"kv_consumer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${host_ip}\"'\", \"kv_port\": 14600}'"

    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG"
    fi

    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &

    decode_pid=$!

    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $SERVER_PORT

    echo "Waiting untill proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $SERVER_PORT

    echo "Killing the decode server"
    kill $decode_pid

fi

echo "Killing the etcd server"
kill $etcd_pid 

echo "Script completed successfully"
exit 0
