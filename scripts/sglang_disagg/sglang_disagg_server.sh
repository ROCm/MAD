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

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask

source $MOONCAKE_COOKBOOK_PATH/set_env_vars.sh
#trap 'echo "Error occurred. Cleaning up..."; exit 0' ERR

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

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
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Qwen3-32B"]="--tp-size 8"
    ["Mixtral-8x7B-v0.1"]="--tp-size 8"
    ["Llama-3.1-8B-Instruct"]="--tp-size 8"
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tp-size 8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tp-size 8"
    ["DeepSeek-V3"]="--tp-size 8"
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

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+=" --prefill http://${IP_ARRAY[$i]}:2322 "
done

for ((i=$xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+=" --decode  http://${IP_ARRAY[$i]}:2322 "
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
    echo "Proxy server is waiting for prefile and decode nodes to be ready ..." \
	    | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
    sleep 20
    TIMEOUT_SECONDS=4000
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="The server is fired up and ready to roll!"
    SECONDS=0
    for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/prefill_NODE${i}.log
         #wait until prefill nodes get ready
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
            if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
                echo "Awaited ${SECONDS} seconds. Timeout reached. Signal not found in prefill ${i} file" \
			| tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
	        	
            fi
            sleep $SLEEP_SECONDS
	    SECONDS=$(( SECONDS + SLEEP_SECONDS))
	 done      
    done

    for ((i=$xP+1; i<${#IP_ARRAY[@]}; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/decode_NODE${i}.log
         #wait until decode nodes get ready         
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
            if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
               echo "Awaited ${SECONDS} seconds. Timeout reached. Signal not found in decode ${i} file" \
		       | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
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
        --port 2322 \
        2>&1 | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    
    proxy_pid=$!
    
    echo "Waiting for all prefill and decode servers to be up . . ."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports 2322

    echo "Proxy Server Ready for benchmarking on ${host_name}:${host_ip}"

    sleep 10
    cd /opt/mooncake-cookbook
    bash /opt/mooncake-cookbook/benchmark_xPyD.sh

    echo "Killing the proxy server"
    kill $proxy_pid

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    
    PREFILL_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host ${host_ip} \
        --port 2322 \
        --stream-output \
        --trust-remote-code \
        --disaggregation-transfer-backend mooncake"
    
    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi
    
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    
    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 2322
    
    echo "Waiting untill proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 2322

    echo "Killing the prefill server"
    kill $prefill_pid

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    
    DECODE_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host ${host_ip} \
        --port 2322 \
        --stream-output \
        --trust-remote-code \
        --disaggregation-transfer-backend mooncake"
    
    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG"
    fi
    
    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
    
    decode_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 2322
    
    echo "Waiting untill proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 2322

    echo "Killing the decode server"
    kill $decode_pid

fi

# =============================================================================
# Cleanup
# =============================================================================

echo "Killing the etcd server"
kill $etcd_pid 

echo "Script completed successfully"
exit 0
