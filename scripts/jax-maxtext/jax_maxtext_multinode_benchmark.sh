#!/bin/bash
#SBATCH --nodes=8
#SBATCH --exclusive             # exclusive node access
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # all mem avail
#SBATCH --ntasks-per-node=1     # n tasks per machine (one task per gpu)
#SBATCH --overcommit

set -e

#################################################################################
# Unified MaxText Training Script
#
# Usage: sbatch jax_maxtext_multinode_benchmark.sh <config_file.yml> [docker_image]
#
# Arguments:
#   config_file.yml - Required. Path to model config YAML file (absolute or relative)
#                     Examples: llama2-7b.gpu.yml, /path/to/model.gpu.yml, ../configs/my-model.yml
#   docker_image    - Optional. Docker image to use (default: rocm/jax-training:maxtext-v26.3)
#
# Examples:
#   sbatch jax_maxtext_multinode_benchmark.sh llama2-7b.gpu.yml
#   sbatch jax_maxtext_multinode_benchmark.sh /mnt/vast/araina/configs/llama3-70b.gpu.yml
#   sbatch jax_maxtext_multinode_benchmark.sh ../models/custom-model.gpu.yml my-docker-image:tag
#################################################################################

mkdir -p -v outputs; chmod a+w outputs

LOOKUP_USER="${USER:-}"

# ------- Parse command line arguments -------
DEFAULT_DOCKER_IMAGE="rocm/jax-training:latest"

if [[ $# -eq 0 ]]; then
    echo "ERROR: No config file provided!"
    echo ""
    echo "Usage: sbatch jax_maxtext_multinode_benchmark.sh <config_file.yml> [docker_image]"
    echo ""
    echo "Arguments:"
    echo "  config_file.yml - Required. Path to model config YAML file (absolute or relative)"
    echo "  docker_image    - Optional. Default: $DEFAULT_DOCKER_IMAGE"
    echo ""
    echo "Examples:"
    echo "  sbatch jax_maxtext_multinode_benchmark.sh llama2-7b.gpu.yml"
    echo "  sbatch jax_maxtext_multinode_benchmark.sh /path/to/model.gpu.yml"
    echo "  sbatch jax_maxtext_multinode_benchmark.sh ../configs/my-model.yml"
    exit 1
fi

CONFIG_FILE="$1"
DOCKER_IMAGE="${2:-$DEFAULT_DOCKER_IMAGE}"
EXP_TAG=""

# Convert to absolute path for consistency
if [[ "$CONFIG_FILE" != /* ]]; then
    # Relative path - convert to absolute
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
fi

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Extract model name from config filename (remove path and extension)
CONFIG_BASENAME=$(basename "$CONFIG_FILE")
MODEL_NAME="${CONFIG_BASENAME%.gpu.yml}"
MODEL_NAME="${MODEL_NAME%.yml}"

# Get the directory containing the config file (for mounting)
CONFIG_DIR=$(dirname "$CONFIG_FILE")

echo "CONFIG_FILE=$CONFIG_FILE"
echo "CONFIG_DIR=$CONFIG_DIR"
echo "CONFIG_BASENAME=$CONFIG_BASENAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "DOCKER_IMAGE=$DOCKER_IMAGE"

# ------- date command detection (GNU date or gdate) -------
DATE_CMD="date"
if ! date -d '1970-01-01 00:00:00' +%s >/dev/null 2>&1; then
  if command -v gdate >/dev/null 2>&1; then
    DATE_CMD="gdate"
  else
    echo "WARNING: Your 'date' doesn't support -d. Reservation time parsing may fail; falling back to first active match." >&2
  fi
fi

# ------- function to pick reservation -------
get_reservation_for_user() {
  local uname="${1}"
  local datecmd="${2}"
  local now_epoch
  now_epoch="$("$datecmd" +%s)"
  scontrol show reservation -o 2>/dev/null | \
  awk -v user="$uname" -v now="$now_epoch" -v datecmd="$datecmd" '
    function to_epoch(ts,   cmd, epoch_str) {
      gsub(/T/, " ", ts)
      if (ts == "" || ts == "Unknown") return 0
      cmd = datecmd " -d \"" ts "\" +%s"
      epoch_str = ""
      cmd | getline epoch_str
      close(cmd)
      if (epoch_str ~ /^[0-9]+$/) return epoch_str + 0
      return 0
    }
    {
      name=""; users=""; start_s=""; end_s=""
      if (match($0, /ReservationName=([^ ]+)/, m)) name=m[1]
      if (match($0, /Users=([^ ]+)/, mu))          users=mu[1]
      if (match($0, /StartTime=([^ ]+)/, ms))      start_s=ms[1]
      if (match($0, /EndTime=([^ ]+)/, me))        end_s=me[1]
      n = split(users, arr, ",")
      ok=0
      for (i=1; i<=n; i++) if (arr[i] == user) { ok=1; break }
      if (!ok) next
      start = to_epoch(start_s)
      end   = to_epoch(end_s)
      if (start==0 || end==0) {
        start=1; end=now+1
      }
      if (start <= now && now <= end) {
        printf("%d\t%s\n", start, name)
      }
    }
  ' | sort -nr | awk 'NR==1 { print $2 }'
}

RESERVATION_NAME="$(get_reservation_for_user "${LOOKUP_USER}" "${DATE_CMD}")"
if [[ -n "${RESERVATION_NAME}" ]]; then
  echo "Using reservation for user '${LOOKUP_USER}': ${RESERVATION_NAME}"
else
  echo "No active reservation found for user '${LOOKUP_USER}'. Submitting without --reservation."
fi

# Config file already validated above
echo "EXP_TAG=$EXP_TAG"

# ------- Build job name -------
JOB_NAME="JAX-${MODEL_NAME}"
if [[ -n "$EXP_TAG" ]]; then
    JOB_NAME="${JOB_NAME}-${EXP_TAG}"
fi
echo "JOB_NAME=$JOB_NAME"

# ------- Setup for distributed execution -------
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"

# Determine coordinator IP (first node in the job)
COORDINATOR_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Coordinator node: $COORDINATOR_NODE"

# Get the IP of the coordinator node
if [[ -n "${SLURM_LAUNCH_NODE_IPADDR:-}" ]]; then
    JAX_COORDINATOR_IP=$SLURM_LAUNCH_NODE_IPADDR
    echo "Using JAX_COORDINATOR_IP=$JAX_COORDINATOR_IP (from SLURM_LAUNCH_NODE_IPADDR)"
else
    # Resolve the first node's IP address
    JAX_COORDINATOR_IP=$(srun --nodes=1 --ntasks=1 -w "$COORDINATOR_NODE" hostname -I | awk '{print $1}')
    echo "Using JAX_COORDINATOR_IP=$JAX_COORDINATOR_IP (resolved from $COORDINATOR_NODE)"
fi

JAX_PORT=$((20000 + $RANDOM % 40000))
echo "JAX_PORT=$JAX_PORT"

# ------- Git summary -------
echo "=== GIT SUMMARY BEGIN ==="
echo "[BRANCH]"
git status --branch --short 2>/dev/null || echo "Not a git repository"
echo
echo "[LAST COMMIT]"
git --no-pager log -1 --pretty=format:"%h %s (%ad) <%an>" 2>/dev/null || echo "No commits"
echo
echo "=== GIT SUMMARY END ==="

# ============================================================================
# Node Setup Script (heredoc for readability - no escaping needed)
# This function outputs the script that runs on each node before docker run.
# Handles: docker detection, GPU cleanup, image pull, NCCL setup
# ============================================================================
write_node_setup_script() {
cat << 'NODE_SETUP_EOF'
#!/bin/bash
set -e

echo "=== Node $SLURM_NODEID: Starting ==="

# ------- Get docker binary -------
if command -v podman >/dev/null 2>&1; then
    runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    if [ -d "$runtime_dir" ] && podman info >/dev/null 2>&1; then
        DOCKER_BIN="podman"
    fi
fi
if [ -z "${DOCKER_BIN:-}" ] && command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        DOCKER_BIN="docker"
    else
        DOCKER_BIN="sudo docker"
    fi
fi
if [ -z "${DOCKER_BIN:-}" ]; then
    echo "ERROR: No docker/podman found" >&2
    exit 1
fi
echo "DOCKER_BIN=$DOCKER_BIN"

# ------- GPU cleanup -------
echo "=== GPU cleanup ==="
GPU_PIDS=$(rocm-smi --showpids 2>/dev/null | grep -oP "^\d+" | grep -v "^$" || true)
if [ -n "$GPU_PIDS" ]; then
    echo "Found GPU processes: $GPU_PIDS"
    ALL_CONTAINERS=$($DOCKER_BIN ps -q 2>/dev/null || true)
    if [ -n "$ALL_CONTAINERS" ]; then
        echo "Stopping containers..."
        echo "$ALL_CONTAINERS" | xargs $DOCKER_BIN stop -t 10 || true
        sleep 20
    fi
    USE_SUDO=""
    [[ "$DOCKER_BIN" == sudo* ]] && USE_SUDO="sudo"
    for PID in $GPU_PIDS; do
        $USE_SUDO kill -9 $PID 2>/dev/null || true
    done
    sleep 10
fi
# ------- Docker image pull -------
# DOCKER_IMAGE is passed via environment variable
if ! $DOCKER_BIN image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "[INFO] Pulling $DOCKER_IMAGE ..."
    $DOCKER_BIN pull "$DOCKER_IMAGE"
fi

# ------- NCCL setup -------
NCCL_IB_HCA=$(ls /sys/class/infiniband 2>/dev/null | tr "\n" "," | sed "s/,$//" || true)
candidates=$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $2, $4}' | sort -k1,1)
NCCL_SOCKET_IFNAME=$(echo "$candidates" | awk '$2 ~ /^10\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')
[ -z "$NCCL_SOCKET_IFNAME" ] && NCCL_SOCKET_IFNAME=$(echo "$candidates" | awk '$2 ~ /^172\.(1[6-9]|2[0-9]|3[0-1])\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')
[ -z "$NCCL_SOCKET_IFNAME" ] && NCCL_SOCKET_IFNAME=$(echo "$candidates" | awk '$2 ~ /^192\.168\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')

# Export variables for the caller (DOCKER_IMAGE comes from parent environment)
export DOCKER_BIN NCCL_IB_HCA NCCL_SOCKET_IFNAME
NODE_SETUP_EOF
}

# ============================================================================
# Inner Docker Script (heredoc for readability - no escaping needed)
# This function outputs the script that runs inside the container.
# ============================================================================
write_inner_script() {
cat << INNER_SCRIPT_EOF
#!/bin/bash
set -ex

cd /workspace/maxtext
MAXTEXT_SRC_DIR=.
[[ -d ./src ]] && MAXTEXT_SRC_DIR=./src
export PYTHONPATH="${MAXTEXT_SRC_DIR}:${PYTHONPATH}"
cd "${MAXTEXT_SRC_DIR}"

# ------- Output directory setup -------
export OUTPUT_PATH="/dockerx/outputs/\${JOB_ID_AND_NAME}"
mkdir -p \$OUTPUT_PATH

# ------- NCCL Configuration -------
export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=INFO
export TF_CPP_MIN_LOG_LEVEL=2

# ------- JAX/XLA Configuration -------
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export JAX_HIP_GRAPH_LOWERING=false
# Note: XLA_FLAGS is passed via docker --env to override container's baked-in value

# ------- NCCL Performance Tuning -------
export NCCL_CROSS_NIC=2
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=8
export NCCL_IB_QPS_PER_CONNECTION=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPU_MAX_HW_QUEUES=2

# ------- HIP/ROCm Configuration -------
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1

# ------- Transformer Engine Configuration -------
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_FRAMEWORK=jax
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_FUSED_ATTN_CK=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_ROCM=1
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_CK_IS_V3_ATOMIC_FP32=0

# ------- RCCL/NCCL IB Tuning -------
export IONIC_LOCKFREE=all
export NCCL_GDR_COPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_ECE_ENABLE=0
export NCCL_IB_FIFO_TC=184
export NCCL_IB_GID_INDEX=1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=96
export NCCL_IB_USE_INLINE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PXN_DISABLE=0
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=1

# ------- Run MaxText Training -------
echo "Starting MaxText training with model: \${MODEL_NAME}"
echo "Output path: \${OUTPUT_PATH}"

# Use config file path - if it's in a different directory, it will be mounted separately
python3 -m MaxText.train /configs/${CONFIG_BASENAME} base_output_directory=\${OUTPUT_PATH}
INNER_SCRIPT_EOF
}

# ============================================================================
# Launch on all nodes via srun
# ============================================================================
echo "==STARTING JOBS ON ALL NODES=="

# Export variables that srun needs
export JAX_COORDINATOR_IP
export JAX_PORT  
export MODEL_NAME
export CONFIG_BASENAME
export CONFIG_DIR
export DOCKER_IMAGE

# XLA_FLAGS must be passed via docker --env to override the container's baked-in value
# (the container's profile scripts set XLA_FLAGS with 'FALSE' which fails parsing)
export XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
--xla_gpu_enable_command_buffer='' \
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_cublaslt=true \
--xla_gpu_autotune_level=4 \
--xla_gpu_enable_all_gather_combine_by_dim=false"

# Write the setup scripts (defined above via heredoc for readability)
NODE_SETUP_SCRIPT=".maxtext_node_setup_$$.sh"
INNER_SCRIPT=".maxtext_inner_$$.sh"
write_node_setup_script > "$NODE_SETUP_SCRIPT"
write_inner_script > "$INNER_SCRIPT"
chmod +x "$NODE_SETUP_SCRIPT" "$INNER_SCRIPT"
trap "rm -f '$NODE_SETUP_SCRIPT' '$INNER_SCRIPT'" EXIT

srun -l bash -c '
# Source the node setup script (docker detection, GPU cleanup, image pull, NCCL setup)
source "'"$NODE_SETUP_SCRIPT"'"

# Docker run
echo "==Starting container on node $SLURM_NODEID=="
EXTRA_GIDS=$(id -G)
GROUP_ADD_ARGS=""
for gid in $EXTRA_GIDS; do GROUP_ADD_ARGS="$GROUP_ADD_ARGS --group-add $gid"; done

$DOCKER_BIN run --rm --cap-add=SYS_PTRACE --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --env JAX_COORDINATOR_IP=$JAX_COORDINATOR_IP \
    --env JAX_COORDINATOR_PORT=$JAX_PORT \
    --env JOB_ID_AND_NAME="${SLURM_JOB_ID}-${SLURM_JOB_NAME}" \
    --env MODEL_NAME=$MODEL_NAME \
    --env NCCL_IB_HCA=$NCCL_IB_HCA \
    --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    --env NNODES=$SLURM_NNODES \
    --env NODE_RANK=$SLURM_NODEID \
    --env XLA_FLAGS="$XLA_FLAGS" \
    --security-opt seccomp=unconfined --privileged $GROUP_ADD_ARGS \
    -v /boot:/boot:ro -v $PWD:/dockerx -v $CONFIG_DIR:/configs:ro -w /dockerx \
    $DOCKER_IMAGE /bin/bash -lc "source /dockerx/'"$INNER_SCRIPT"'"
'

echo "==DONE=="
