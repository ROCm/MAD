#!/bin/bash
#SBATCH --job-name=sglang-disagg   # Specify a custom string for your slurm batch job
#SBATCH -N 2            # CHECK this to be right in batch jobs
#SBATCH -n 2          # CHECK this to be right in batch jobs
#SBATCH --ntasks-per-node=1
# --spread-job intentionally removed: it spreads nodes across IB rails, which causes
# NCCL ibv_modify_qp timeouts on cross-rail node pairs.
#
# --switches=2: allow up to 2 IB switches (rails). select_same_rail_nodes() below
# enforces per-role rail consistency. Set --switches=1 to force single rail.
#SBATCH --switches=2
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="/shared_inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared_inference/%u/model_blog_logs/slurm-%j.err"


# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# -----------------------------------------------------------------------------
# DP_MODE=1 allowlist — must match MORI_DP_MODE1_ALLOWED_MODELS in sglang_disagg_mori_io_ep.sh
# -----------------------------------------------------------------------------
MORI_DP_MODE1_ALLOWED_MODELS=(
    "DeepSeek-V3"
    "DeepSeek-R1"
)

mori_model_allows_dp_mode_one() {
    local name="$1"
    local m
    for m in "${MORI_DP_MODE1_ALLOWED_MODELS[@]}"; do
        [[ "$name" == "$m" ]] && return 0
    done
    return 1
}

mori_dp_mode1_allowed_models_lines() {
    local m
    for m in "${MORI_DP_MODE1_ALLOWED_MODELS[@]}"; do
        printf '  - %s\n' "$m"
    done
}

# MoRI IO EP (`sglang_disagg_mori_io_ep.sh`): most models use DP_MODE=0 (tp only).
MORI_EP_VALID_MODELS=( \
    "Qwen3-32B" \
    "Mixtral-8x7B-Instruct-v0.1" \
    "Llama-3.1-8B-Instruct" \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
    "DeepSeek-R1" \
)

# Define valid model names
VALID_MODELS=( \
    "Qwen3-32B" \
    "Mixtral-8x7B-Instruct-v0.1" \
    "Llama-3.1-8B-Instruct" \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
    "DeepSeek-R1" \
)

model_allows_mori_ep() {
    local m="$1"
    for x in "${MORI_EP_VALID_MODELS[@]}"; do
        [[ "$m" == "$x" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Run file: always use sglang_disagg_mori_io_ep.sh (unified launcher).
# RUN_MORI=0: sets KV_TRANSFER_BACKEND=mooncake (Mooncake).
# RUN_MORI=1: uses mori IO (default).
# ---------------------------------------------------------------------------
RUN_FILE="sglang_disagg_mori_io_ep.sh"
_run_mori="${RUN_MORI:-0}"
if [[ "$_run_mori" == "1" ]]; then
    echo "RUN_MORI=1: using $RUN_FILE with MoRI IO for model '$MODEL_NAME'"
else
    export KV_TRANSFER_BACKEND="mooncake"
    echo "RUN_MORI=0: using $RUN_FILE with Mooncake (KV_TRANSFER_BACKEND=mooncake)"
fi
if [[ ! -f "$RUN_FILE" ]]; then
    echo "Error: '$RUN_FILE' not found in $(pwd)." >&2
    exit 1
fi

if test -z "${DOCKER_IMAGE_NAME:-}"; then
  echo "Error: DOCKER_IMAGE_NAME is not set. Pass it via env before sbatch."
  exit 1
fi

# Set current directory to be REPO directory with all relevant scripts
export MOONCAKE_REPO_DIR=$(pwd)
LOG_PATH="${LOG_PATH:-/shared_inference/${USER}/model_blog_logs}"

xP="${xP:-1}" #-> Number of Prefill Servers
yD="${yD:-1}" #-> Number of Decode Servers
DP_MODE="${DP_MODE:-0}" #-> mori_io_ep: 1=dp flags + --dp-size/--ep-size; 0=tp flags, omit dp/ep CLI flags

# xP>1 or yD>1 with DP_MODE=0: each node runs an independent TP server; proxy registers all IPs.
# sglang_disagg_mori_io_ep.sh (RUN_MORI=1): set to 1 to skip phases; pass from wrapper before sbatch
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_CURL_TEST="${SKIP_CURL_TEST:-0}"

# Agentic replay shorthand: AGENTIC=1 selects the agentic benchmark hook. The
# in-container launcher (sglang_disagg_mori_io_ep.sh) then auto-enables server
# metrics + radix prefix cache for BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh.
if [[ "${AGENTIC:-0}" == "1" ]]; then
    export BENCHMARK_SCRIPT_FILE="${BENCHMARK_SCRIPT_FILE:-benchmark_agentic.sh}"
    echo "AGENTIC=1: BENCHMARK_SCRIPT_FILE=${BENCHMARK_SCRIPT_FILE} (server metrics + radix cache auto-enabled)"
fi

MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-"/shared_inference/models_blog/"}"

validate_model_name() {
  local is_valid_model=false

  for model in "${VALID_MODELS[@]}"; do
    if [[ "$MODEL_NAME" == "$model" ]]; then
      is_valid_model=true
      break
    fi
  done

  if ! $is_valid_model; then
    printf "Error: Invalid MODEL_NAME: '%s'\nValid models are:\n" "$MODEL_NAME"
    for model in "${VALID_MODELS[@]}"; do
      printf "  - %s\n" "$model"
    done
    exit 1
  fi

  echo "MODEL_NAME '$MODEL_NAME' is valid."
  return 0
}

validate_model_name "${MODEL_NAME}"

# ------------------------
# Model path validation and selection across all nodes
# ------------------------
ORIGINAL_MODEL_PATH="/shared_inference/models_blog/${MODEL_NAME}"

echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

# Get all allocated nodes
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

# Function to check model path on all nodes
check_model_path() {
    local path=$1
    local check_name=$2
    
    echo "Checking $check_name: $path"
    
    # Run check on all nodes in parallel
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
        if [ -d '$path' ]; then 
            echo \"\$(hostname): ✓ Found $path\"
            exit 0
        else 
            echo \"\$(hostname): ✗ Missing $path\"
            exit 1
        fi
    "
    
    # Check if all nodes succeeded (exit code 0)
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $check_name available on ALL nodes"
        return 0
    else
        echo "✗ $check_name NOT available on all nodes"
        return 1
    fi
}

# Check /mnt/m2m_nobackup/models_blog first
MODEL_PATH_1="/mnt/m2m_nobackup/models_blog/$MODEL_NAME"
if check_model_path "$MODEL_PATH_1" "/mnt/m2m_nobackup/models_blog"; then
    MODEL_PATH="$MODEL_PATH_1"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
# Check /shared_inference/models_blog
elif check_model_path "/shared_inference/models_blog/$MODEL_NAME" "/shared_inference/models_blog"; then
    MODEL_PATH="/shared_inference/models_blog/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
elif check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
else
    echo ""
    echo "✗ FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in either:"
    echo "  - /mnt/m2m_nobackup/models_blog/$MODEL_NAME"
    echo "  - /shared_inference/models_blog/$MODEL_NAME"
    echo ""
    echo "Model must be accessible from all nodes for distributed execution."
    echo "Please ensure the model is available on all allocated nodes."
    exit 1
fi

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""


# Calculate NUM_NODES based on xP and yD
# The proxy/router runs on the first prefill node (NODE_RANK=0), so no extra node is needed.
NUM_NODES=$((xP + yD))
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD) DP_MODE=$DP_MODE"

# ------------------------
# Extract NUM_NODES from SLURM allocation and select nodes
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
AVAILABLE_COUNT=$(echo "$FULL_NODELIST" | wc -l)

if (( AVAILABLE_COUNT < NUM_NODES )); then
    echo "ERROR: Need ${NUM_NODES} nodes (xP=${xP} + yD=${yD}) but only ${AVAILABLE_COUNT} allocated." >&2
    exit 1
fi

# Pick the first NUM_NODES nodes from the allocation and sort alphabetically.
# srun assigns PROCID=0 to the alphabetically-first hostname regardless of --nodelist order.
# IPADDRS[0..xP-1] = prefill, IPADDRS[xP..] = decode, so the sorted order must match.
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n "$NUM_NODES" | sort)
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

# Create new nodelist in SLURM format
# This is a simplified approach - for complex ranges, you might need more sophisticated parsing
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/ravgupta}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-m2m}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-${xP}p${yD}d_bench-serving}"

echo ""
echo "Updated SLURM Environment Variables:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_QOS: $SLURM_JOB_QOS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_CONF: $SLURM_CONF"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# Node information
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'hostname -I')
# Prefer the 10.158.x fabric IP: nodes are multi-homed and picking the first
# hostname -I address can land on a different subnet, which hangs the socket
# barrier on mixed-subnet coordination. Cluster-specific (OCI useocpm2m fabric).
# Fall back to the first address if no 10.158.x is present; fail fast if empty.
MASTER_ADDR_RAW="$MASTER_ADDR"
MASTER_ADDR=$(echo "$MASTER_ADDR_RAW" | tr ' ' '\n' | grep -E '^10\.158\.' | head -1)
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(echo "$MASTER_ADDR_RAW" | awk '{print $1}')
fi
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: could not resolve MASTER_ADDR for node '$MASTER_NODE' from 'hostname -I' output: '$MASTER_ADDR_RAW'" >&2
    exit 1
fi
MASTER_PORT=39566  # Choose an open port

IPS=()

for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'hostname -I')
    IP_RAW="$IP"
    IP=$(echo "$IP_RAW" | tr ' ' '\n' | grep -E '^10\.158\.' | head -1)
    if [ -z "$IP" ]; then
        IP=$(echo "$IP_RAW" | awk '{print $1}')
    fi
    if [ -z "$IP" ]; then
        echo "ERROR: could not resolve IP for node '$NODE' from 'hostname -I' output: '$IP_RAW'" >&2
        exit 1
    fi
    IPS+=("$IP")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

MOONCAKE_COOKBOOK_PATH="/opt/mooncake-cookbook"
BENCHMARK_FILE="${BENCHMARK_FILE:-$MOONCAKE_COOKBOOK_PATH/disagg_blog_p1/models_blog/run_xPyD_models.slurm}"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "MASTER_ADDR is ${MASTER_ADDR}"
echo "MASTER_PORT is ${MASTER_PORT}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${MOONCAKE_REPO_DIR}"
echo "SKIP_BENCHMARK=${SKIP_BENCHMARK} SKIP_CURL_TEST=${SKIP_CURL_TEST} (sglang_disagg_mori_io_ep.sh NODE_RANK=0)"

if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Created directory: $LOG_PATH"
else
    echo "Directory already exists: $LOG_PATH"
fi

export LOG_PATH=$LOG_PATH
export MOONCAKE_REPO_DIR=$MOONCAKE_REPO_DIR
export MOONCAKE_COOKBOOK_PATH=$MOONCAKE_COOKBOOK_PATH
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export MODEL_PATH=$MODEL_PATH
export xP=$xP
export yD=$yD
export DP_MODE=$DP_MODE
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export BENCHMARK_ITR=${BENCHMARK_ITR:-2}
export SKIP_BENCHMARK
export SKIP_CURL_TEST

export DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$MOONCAKE_COOKBOOK_PATH/${RUN_FILE}"

# Use only the selected nodes for srun execution
SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
docker ps -q | xargs --no-run-if-empty docker stop;
docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    -v /shared_inference:/shared_inference \
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G \
    -v ${LOG_PATH}:/run_logs \
    -v $MOONCAKE_REPO_DIR:$MOONCAKE_COOKBOOK_PATH \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e MOONCAKE_COOKBOOK_PATH=$MOONCAKE_COOKBOOK_PATH \
    -e xP=$xP \
    -e yD=$yD \
    -e DP_MODE=$DP_MODE \
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_FILE=$BENCHMARK_FILE \
    -e IPADDRS=$IPADDRS \
    -e BENCHMARK_ITR=$BENCHMARK_ITR \
    -e SKIP_BENCHMARK=$SKIP_BENCHMARK \
    -e SKIP_CURL_TEST=$SKIP_CURL_TEST \
    ${KV_TRANSFER_BACKEND:+-e KV_TRANSFER_BACKEND=$KV_TRANSFER_BACKEND} \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}" \
    -e DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-unknown} \
    -e USE_CX7_NICS=${USE_CX7_NICS:-0} \
    ${BENCHMARK_SCRIPT_FILE:+-e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE} \
    -e MOONCAKE_REPO_DIR="$MOONCAKE_REPO_DIR" \
    ${AGENTIC_CONC:+-e AGENTIC_CONC=$AGENTIC_CONC} \
    ${DURATION:+-e DURATION=$DURATION} \
    ${AGENTIC_PORT:+-e AGENTIC_PORT=$AGENTIC_PORT} \
    ${MODEL_PREFIX:+-e MODEL_PREFIX=$MODEL_PREFIX} \
    ${WEKA_LOADER_OVERRIDE:+-e WEKA_LOADER_OVERRIDE=$WEKA_LOADER_OVERRIDE} \
    ${AIPERF_UNSAFE_OVERRIDE:+-e AIPERF_UNSAFE_OVERRIDE=$AIPERF_UNSAFE_OVERRIDE} \
    ${AGENTIC_CACHE_WARMUP_DURATION:+-e AGENTIC_CACHE_WARMUP_DURATION=$AGENTIC_CACHE_WARMUP_DURATION} \
    ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN=$MAX_MODEL_LEN} \
    ${AGENTIC_CONFIG:+-e AGENTIC_CONFIG=$AGENTIC_CONFIG} \
    ${AGENTIC_WORKLOAD:+-e AGENTIC_WORKLOAD=$AGENTIC_WORKLOAD} \
    ${SUITE_CORPUS_DIR:+-e SUITE_CORPUS_DIR=$SUITE_CORPUS_DIR} \
    ${AIPERF_PIN:+-e AIPERF_PIN=$AIPERF_PIN} \
    ${AGENTIC_UTILS_PIN:+-e AGENTIC_UTILS_PIN=$AGENTIC_UTILS_PIN} \
    ${SUITE_CORPUS_FORCE:+-e SUITE_CORPUS_FORCE=$SUITE_CORPUS_FORCE} \
    ${AGENTIC_STRICT_CONTEXT:+-e AGENTIC_STRICT_CONTEXT=$AGENTIC_STRICT_CONTEXT} \
    ${AGENTIC_HF_ISL_TAIL:+-e AGENTIC_HF_ISL_TAIL=$AGENTIC_HF_ISL_TAIL} \
    ${HF_HOME:+-e HF_HOME=$HF_HOME} \
    ${ENABLE_SERVER_METRICS:+-e ENABLE_SERVER_METRICS=$ENABLE_SERVER_METRICS} \
    ${ENABLE_RADIX_CACHE:+-e ENABLE_RADIX_CACHE=$ENABLE_RADIX_CACHE} \
    ${DRY_RUN:+-e DRY_RUN=$DRY_RUN} \
    ${AGENTIC_ALLOW_UV_INSTALL:+-e AGENTIC_ALLOW_UV_INSTALL=$AGENTIC_ALLOW_UV_INSTALL} \
    ${AGENTIC_UV_VERSION:+-e AGENTIC_UV_VERSION=$AGENTIC_UV_VERSION} \
    --ulimit nofile=1048576:1048576 \
    --name $DOCKER_CONT_NAME \
    --entrypoint /bin/bash \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_sglang_bench_serving.sh_NODE${SLURM_PROCID}.log
    "
'
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'docker stop $DOCKER_CONT_NAME; docker rm -f $DOCKER_CONT_NAME'

