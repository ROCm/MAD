#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
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
KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-mooncake}"

# KV_TRANSFER_BACKEND is interpolated into an eval'd launch command below; restrict
# it to known-good values to avoid invalid backends and shell-token injection.
case "$KV_TRANSFER_BACKEND" in
    mori|mooncake|nixl) ;;
    *)
        echo "ERROR: unsupported KV_TRANSFER_BACKEND='$KV_TRANSFER_BACKEND' (expected: mori|mooncake|nixl)" >&2
        exit 1
        ;;
esac

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask

# AINIC/ionic RoCE fix: MOONCAKE_COOKBOOK's set_env_vars.sh assumes a Mellanox/mlx5
# fabric. On ionic-based RoCE clusters (e.g. mia1) it (1) hardcodes mlx5 IB device
# names, (2) forces NCCL_IB_DISABLE=1, and (3) mangles NCCL_SOCKET_IFNAME via an
# `ip route` awk parse. Save the transport iface set by the launcher, then re-assert
# the correct ionic values after sourcing so RDMA over ionic actually engages.
_SAVED_NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

source $MOONCAKE_COOKBOOK_PATH/set_env_vars.sh
#trap 'echo "Error occurred. Cleaning up..."; exit 0' ERR

# Re-assert RDMA/socket env clobbered by set_env_vars.sh (see note above).
export NCCL_IB_DISABLE=0
[[ -n "${IB_DEVICES:-}" ]] && export IBDEVICES="${IB_DEVICES}"
export NCCL_SOCKET_IFNAME="${_SAVED_NCCL_SOCKET_IFNAME:-eno0}"
export GLOO_SOCKET_IFNAME="${_SAVED_NCCL_SOCKET_IFNAME%%,*}"

# This node's IP is already resolved by run.sh and handed to us (rank-ordered,
# post-rendezvous) via IPADDRS. Reuse IPADDRS[NODE_RANK] so the address we bind
# and advertise (--host, socket_barrier --local-ip) matches exactly what peers
# registered for us in the router/barrier; re-deriving it here can pick a
# different NIC on multi-homed nodes and desync the two.
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"
host_ip="${IP_ARRAY[NODE_RANK]:-$(hostname -I | awk '{print $1}')}"
host_name=$(hostname)
unset _SAVED_NCCL_SOCKET_IFNAME

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["Qwen3-32B"]="--tp-size 8"
    ["Mixtral-8x7B-v0.1"]="--tp-size 8"
    ["Llama-3.1-8B-Instruct"]="--tp-size 8"
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tp-size 8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tp-size 8"
    ["DeepSeek-V3"]="--tp-size 8"
    ["DeepSeek-R1"]="--tp-size 8"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Qwen3-32B"]="--tp-size 8"
    ["Mixtral-8x7B-v0.1"]="--tp-size 8"
    ["Llama-3.1-8B-Instruct"]="--tp-size 8"
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tp-size 8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tp-size 8"
    ["DeepSeek-V3"]="--tp-size 8"
    ["DeepSeek-R1"]="--tp-size 8"
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

if [[ -z "$MODEL_NAME" ]]; then
    echo "Warning: MODEL_NAME not set, using default configurations"
    PREFILL_MODEL_CONFIG="--tp-size 4"
    DECODE_MODEL_CONFIG="--tp-size 4"
else
    PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
    DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 4342 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 4342

# =============================================================================
# Cluster Topology Configuration
# =============================================================================

# IP_ARRAY was already parsed from IPADDRS above (where host_ip is derived).

# Colocated topology (matches sglang_disagg_mori_io_ep.sh): the router/proxy runs
# on NODE_RANK=0 alongside the first prefill server, so no dedicated proxy node is
# needed. Total nodes = xP + yD.
#   IP_ARRAY[0..xP-1]     -> prefill nodes (NODE_RANK 0..xP-1)
#   IP_ARRAY[xP..xP+yD-1] -> decode nodes  (NODE_RANK xP..xP+yD-1)
# Backend sglang servers listen on SERVER_PORT; the router listens on ROUTER_PORT.
# They must differ so prefill0 and the router can share NODE_RANK=0.
SERVER_PORT="${SERVER_PORT:-3000}"
ROUTER_PORT="${ROUTER_PORT:-2322}"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=0; i<$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+=" --prefill http://${IP_ARRAY[$i]}:${SERVER_PORT} "
done

for ((i=$xP; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+=" --decode  http://${IP_ARRAY[$i]}:${SERVER_PORT} "
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
    echo "${host_name}:${host_ip} is Prefill Node 0 + Proxy/Router (co-located)"
    echo "${PREFILL_ARGS} are Proxy's Prefill"
    echo "${DECODE_ARGS} are Proxy's Decode"
    echo "================================================"

    # Launch the first prefill server on this node, co-located with the router.
    PREFILL_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host ${host_ip} \
        --port ${SERVER_PORT} \
        --stream-output \
        --trust-remote-code \
        --disaggregation-transfer-backend ${KV_TRANSFER_BACKEND}"

    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi

    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &

    node0_prefill_pid=$!

    echo "Proxy server is waiting for prefill and decode nodes to be ready ..." \
	    | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
    sleep 20
    TIMEOUT_SECONDS=4000
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="The server is fired up and ready to roll!"
    SECONDS=0
    for ((i=0; i<$xP && i<${#IP_ARRAY[@]}; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/prefill_NODE${i}.log
         #wait until prefill nodes get ready
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
            if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
                echo "FATAL: awaited ${SECONDS}s; readiness signal not found for prefill ${i} (${LOG_FILE}). Aborting before launching the router." \
			| tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >&2
                exit 1
            fi
            sleep $SLEEP_SECONDS
	    SECONDS=$(( SECONDS + SLEEP_SECONDS))
	 done      
    done

    for ((i=$xP; i<${#IP_ARRAY[@]}; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/decode_NODE${i}.log
         #wait until decode nodes get ready         
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
            if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
               echo "FATAL: awaited ${SECONDS}s; readiness signal not found for decode ${i} (${LOG_FILE}). Aborting before launching the router." \
		       | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >&2
                exit 1
            fi
            sleep $SLEEP_SECONDS
            SECONDS=$(( SECONDS + SLEEP_SECONDS))
         done
    done

    sleep 10

    python -m sglang_router.launch_router \
    	--pd-disaggregation \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS} \
        --host 0.0.0.0 \
        --port ${ROUTER_PORT} \
        2>&1 | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    
    proxy_pid=$!
    
    echo "Waiting for all prefill and decode servers to be up . . ."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports ${SERVER_PORT}

    echo "Proxy Server Ready for benchmarking on ${host_name}:${host_ip}"

    sleep 10
    cd /opt/mooncake-cookbook
    bash /opt/mooncake-cookbook/benchmark_xPyD.sh

    echo "Killing the proxy server"
    kill $proxy_pid

    echo "Killing the co-located prefill server"
    kill $node0_prefill_pid

elif [ "$NODE_RANK" -ge 1 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Using KV transfer backend: ${KV_TRANSFER_BACKEND}"
    
    PREFILL_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host ${host_ip} \
        --port ${SERVER_PORT} \
        --stream-output \
        --trust-remote-code \
        --disaggregation-transfer-backend ${KV_TRANSFER_BACKEND}"
    
    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi
    
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    
    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports ${ROUTER_PORT}
    
    echo "Waiting until proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port ${ROUTER_PORT}

    echo "Killing the prefill server"
    kill $prefill_pid

elif [ "$NODE_RANK" -ge "$xP" ] && [ "$NODE_RANK" -le "$((xP + yD - 1))" ]; then
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    echo "Using KV transfer backend: ${KV_TRANSFER_BACKEND}"
    
    DECODE_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host ${host_ip} \
        --port ${SERVER_PORT} \
        --stream-output \
        --trust-remote-code \
        --disaggregation-transfer-backend ${KV_TRANSFER_BACKEND}"
    
    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG"
    fi
    
    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
    
    decode_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports ${ROUTER_PORT}
    
    echo "Waiting until proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port ${ROUTER_PORT}

    echo "Killing the decode server"
    kill $decode_pid

else
    echo "ERROR: NODE_RANK=${NODE_RANK} out of range (expected 0..$((xP + yD - 1))) for xP=${xP} yD=${yD}" >&2
    exit 1
fi

# =============================================================================
# Cleanup
# =============================================================================

echo "Killing the etcd server"
kill $etcd_pid 

echo "Script completed successfully"
exit 0
