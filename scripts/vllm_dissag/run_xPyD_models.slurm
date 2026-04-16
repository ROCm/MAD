#!/bin/bash
#SBATCH --job-name=vllm-pd    # Specify a custom string for your slurm batch job
#SBATCH -N 2           # Request xP + yD nodes (proxy co-located on prefill master)
#SBATCH -n 2          # Request xP + yD total tasks
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="/shared_inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared_inference/%u/model_blog_logs/slurm-%j.err"


# ------------------------
# Auto-detect and cd to the script directory so that $(pwd) always
# points to the folder containing the server scripts, regardless of
# where the user called sbatch from.
# Under Slurm, $0 points to a spool copy, so prefer SLURM_SUBMIT_DIR.
# ------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
    SCRIPT_DIR="${SCRIPT_DIR:-.}"
fi
cd "$SCRIPT_DIR" || { echo "Error: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

REQUIRED_FILES=("vllm_disagg_server.sh" "vllm_disagg_mori_ep.sh" "vllm_disagg_server_deepep.sh" "benchmark_xPyD.sh")
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file '$f' not found in $(pwd)." >&2
        echo "Please run sbatch from the scripts/vllm_dissag/ directory, e.g.:" >&2
        echo "  cd MAD/scripts/vllm_dissag && sbatch run_xPyD_models.slurm" >&2
        exit 1
    fi
done
echo "Running from: $(pwd)"

# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# Define valid model names
VALID_MODELS=( \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "gpt-oss-120b" \
    "DeepSeek-R1" \
)

# Models allowed for vllm_disagg_mori_ep.sh when RUN_MORI=1
MORI_EP_VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "DeepSeek-R1" \
)

# Models allowed for vllm_disagg_server_deepep.sh when RUN_DEEPEP=1
DEEPEP_VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "DeepSeek-R1" \
)

MODEL_NAME="${MODEL_NAME:-None}"

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

model_allows_mori_ep() {
  local m="$1"
  for x in "${MORI_EP_VALID_MODELS[@]}"; do
    [[ "$m" == "$x" ]] && return 0
  done
  return 1
}

model_allows_deepep() {
  local m="$1"
  for x in "${DEEPEP_VALID_MODELS[@]}"; do
    [[ "$m" == "$x" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Run-mode selection: exactly one of RUN_MORI / RUN_DEEPEP may be "1".
# ---------------------------------------------------------------------------
_run_mori="${RUN_MORI:-0}"
_run_deepep="${RUN_DEEPEP:-0}"

if [[ "$_run_mori" == "1" && "$_run_deepep" == "1" ]]; then
  echo "Error: Both RUN_MORI and RUN_DEEPEP are set to 1. Set only one." >&2
  exit 1
fi

if [[ "$_run_mori" == "1" ]]; then
  if (( xP != 1 || yD != 1 )); then
    echo "Error: RUN_MORI=1 requires xP=1 and yD=1 (got xP=$xP, yD=$yD). Multi-node MoRI EP is not yet supported." >&2
    exit 1
  fi
  if model_allows_mori_ep "$MODEL_NAME"; then
    RUN_FILE="vllm_disagg_mori_ep.sh"
    echo "RUN_MORI=1: using $RUN_FILE for model '$MODEL_NAME'"
  else
    echo "Error: RUN_MORI=1 but MODEL_NAME '$MODEL_NAME' is not in MORI_EP_VALID_MODELS"
    printf "MoRI EP allowed models:\n"
    for m in "${MORI_EP_VALID_MODELS[@]}"; do
      printf "  - %s\n" "$m"
    done
    exit 1
  fi
elif [[ "$_run_deepep" == "1" ]]; then
  if model_allows_deepep "$MODEL_NAME"; then
    RUN_FILE="vllm_disagg_server_deepep.sh"
    echo "RUN_DEEPEP=1: using $RUN_FILE for model '$MODEL_NAME'"
  else
    echo "Error: RUN_DEEPEP=1 but MODEL_NAME '$MODEL_NAME' is not in DEEPEP_VALID_MODELS"
    printf "DeepEP allowed models:\n"
    for m in "${DEEPEP_VALID_MODELS[@]}"; do
      printf "  - %s\n" "$m"
    done
    exit 1
  fi
else
  RUN_FILE="vllm_disagg_server.sh"
  echo "RUN_MORI/RUN_DEEPEP not set: using $RUN_FILE"
fi

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
  echo "Error: DOCKER_IMAGE_NAME is not set. Please export DOCKER_IMAGE_NAME before running."
  exit 1
fi
export DOCKER_IMAGE_NAME

# Set current directory to be REPO directory with all relevant scripts
NIXL_REPO_DIR=$(pwd)
LOG_PATH="${LOG_PATH:-/shared_inference/${USER}/model_blog_logs}"

xP="${xP:-1}" #-> Number of Prefill Servers
yD="${yD:-1}" #-> Number of Decode Servers

MODEL_DIR="${MODEL_DIR:-"/shared_inference/models_blog/"}"


# ------------------------
# Model path validation and selection across all nodes
# ------------------------
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
    srun --overlap --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
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

# Check MODEL_DIR first (user-provided), then standard paths
if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
elif check_model_path "/mnt/m2m_nobackup/models_blog/$MODEL_NAME" "/mnt/m2m_nobackup/models_blog"; then
    MODEL_PATH="/mnt/m2m_nobackup/models_blog/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
elif check_model_path "/shared_inference/models_blog/$MODEL_NAME" "/shared_inference/models_blog"; then
    MODEL_PATH="/shared_inference/models_blog/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
else
    echo ""
    echo "✗ FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in either:"
    echo "  - $MODEL_DIR/$MODEL_NAME"
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
NUM_NODES=$((xP + yD))
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD, proxy co-located on prefill master)"

# DeepEP configuration (only exported when RUN_DEEPEP=1)
if [[ "$_run_deepep" == "1" ]]; then
    export PREFILL_DEEPEP_BACKEND="${PREFILL_DEEPEP_BACKEND:-deepep_high_throughput}"
    export DECODE_DEEPEP_BACKEND="${DECODE_DEEPEP_BACKEND:-deepep_low_latency}"
    export ENABLE_DBO="${ENABLE_DBO:-false}"
    export DBO_COMM_SMS="${DBO_COMM_SMS:-}"
    export ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
    echo "DeepEP config: PREFILL_BACKEND=$PREFILL_DEEPEP_BACKEND DECODE_BACKEND=$DECODE_DEEPEP_BACKEND DBO=$ENABLE_DBO"
fi

# ------------------------
# Extract first NUM_NODES from SLURM allocation and update SLURM variables
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Get the full nodelist and extract first NUM_NODES
FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd,)

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"

export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-1p1d_bench-serving}"

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
#echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# Node information
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
# Prefer 10.x.x.x overlay IPs for inter-node communication.
_pick_overlay_ip() {
    local raw
    raw=$(srun --overlap --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$1" bash -c 'hostname -I')
    local ip
    for ip in $raw; do
        case "$ip" in 10.*) echo "$ip"; return ;; esac
    done
    echo "$raw" | awk 'NR==1 {print $1}'
}

MASTER_ADDR=$(_pick_overlay_ip "$MASTER_NODE")
MASTER_PORT=39566

IPS=()
for NODE in $SELECTED_NODES; do
    IPS+=("$(_pick_overlay_ip "$NODE")")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
BENCHMARK_CON="${BENCHMARK_CON:-}"
BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-}"

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "MASTER_ADDR is ${MASTER_ADDR}"
echo "MASTER_PORT is ${MASTER_PORT}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${NIXL_REPO_DIR}"

if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Created directory: $LOG_PATH"
else
    echo "Directory already exists: $LOG_PATH"
fi

export LOG_PATH=$LOG_PATH
export NIXL_REPO_DIR=$NIXL_REPO_DIR
export NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export MODEL_PATH=$MODEL_PATH
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export BENCHMARK_ITR=$BENCHMARK_ITR
export BENCHMARK_CON="${BENCHMARK_CON}"
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}"

export DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/${RUN_FILE}"

# Use only the selected nodes for srun execution
SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

srun --overlap --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
podman ps -q | xargs --no-run-if-empty podman stop 2>/dev/null || true;
fuser -k 5000/tcp 2>/dev/null || true;
fuser -k 2222/tcp 2>/dev/null || true;
fuser -k 15000/tcp 2>/dev/null || true;
fuser -k 20005/tcp 2>/dev/null || true;
fuser -k 10001/tcp 2>/dev/null || true;
sleep 2;

# Set VLLM_HOST_IP to this node'\''s overlay IP so MoRIIO uses routable addresses
_VLLM_HOST_IP=$(echo "$IPADDRS" | tr "," "\n" | sed -n "$((SLURM_PROCID + 1))p")
echo "[vllm-host-ip] SLURM_PROCID=$SLURM_PROCID  VLLM_HOST_IP=$_VLLM_HOST_IP"

# --- Build host RDMA library mounts ---
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu
for _lib in libmlx5.so libmlx5.so.1 libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1 libionic.so libionic.so.1; do
    [ -e "$_LIBDIR/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/$_lib:$_LIBDIR/$_lib:ro"
done
for _vlib in $_LIBDIR/libmlx5.so.1.* $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.* $_LIBDIR/libionic.so.1.*; do
    [ -e "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done
[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"

# Ionic RDMA: if libibverbs dir is already mounted from host, the ionic
# provider inside it is included automatically; no extra mounts needed.
echo "[host-rdma] mounts: $_RDMA_MOUNTS"

podman run --rm \
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
    $([ -d /shared_inference ] && echo "-v /shared_inference:/shared_inference") \
    $([ -d /shared/amdgpu ] && echo "-v /shared/amdgpu:/shared/amdgpu") \
    $([ -d /mnt/m2m_nobackup ] && echo "-v /mnt/m2m_nobackup:/mnt/m2m_nobackup") \
    -v /sys:/sys:ro \
    -v $HOME/.ssh:/root/.ssh \
    --ulimit nofile=524288:524288 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH \
    -e xP=$xP \
    -e yD=$yD \
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_ITR=$BENCHMARK_ITR \
    -e BENCHMARK_CON="${BENCHMARK_CON}" \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}" \
    -e IPADDRS=$IPADDRS \
    -e PROXY_TYPE=${PROXY_TYPE:-vllm_router} \
    -e ROUTER_PORT=${ROUTER_PORT:-18001} \
    -e BENCHMARK_PORT=${BENCHMARK_PORT:-18001} \
    -e PREFILL_DEEPEP_BACKEND=${PREFILL_DEEPEP_BACKEND:-} \
    -e DECODE_DEEPEP_BACKEND=${DECODE_DEEPEP_BACKEND:-} \
    -e ENABLE_DBO=${ENABLE_DBO:-} \
    -e DBO_COMM_SMS=${DBO_COMM_SMS:-} \
    -e ENABLE_PROFILING=${ENABLE_PROFILING:-} \
    -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} \
    -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
    -e MORI_GPU_ARCHS=${MORI_GPU_ARCHS:-gfx950} \
    -e MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-}" \
    -e MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-}" \
    -e MORI_IO_LOG_LEVEL="${MORI_IO_LOG_LEVEL:-INFO}" \
    -e VLLM_HOST_IP=$_VLLM_HOST_IP \
    -e PYTHONUNBUFFERED=1 \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        rm -rf /root/.triton/cache/ /tmp/triton* /tmp/torchinductor_root/
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${SLURM_PROCID}.log
    "
'
srun --overlap --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'podman stop $DOCKER_CONT_NAME 2>/dev/null || true; podman rm $DOCKER_CONT_NAME 2>/dev/null || true'

