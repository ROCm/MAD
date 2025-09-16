#!/bin/bash
#SBATCH --job-name=1p1d_bench-serving    # Specify a custom string for your slurm batch job
#SBATCH -N 3            # CHECK this to be right in batch jobs 
#SBATCH -n 3          # CHECK this to be right in batch jobs
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="/shared-inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared-inference/%u/model_blog_logs/slurm-%j.err"


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
    "Qwen3-32B" \
    "Mixtral-8x7B-v0.1" \
    "Llama-3.1-8B-Instruct" \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
)


# Each model has an associated run file - Set it here
declare -A MODEL_RUNFILES=(
    ["Qwen3-32B"]="sglang_disagg_server.sh"
    ["Mixtral-8x7B-v0.1"]="sglang_disagg_server.sh"
    ["Llama-3.1-8B-Instruct"]="sglang_disagg_server.sh"
    ["Llama-3.1-405B-Instruct-FP8-KV"]="sglang_disagg_server.sh"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="sglang_disagg_server.sh"
    ["DeepSeek-V3"]="sglang_disagg_server.sh"
)

# Check if MODEL_NAME exists and fetch runfile
if [[ -n "${MODEL_RUNFILES[$MODEL_NAME]}" ]]; then
    RUN_FILE="${MODEL_RUNFILES[$MODEL_NAME]}"
    echo "Model found: $MODEL_NAME"
    echo "Runfile set: $RUN_FILE"
else
    echo "Error: Model '$MODEL_NAME' not found in MODEL_RUNFILES"
    echo "Available models: ${!MODEL_RUNFILES[@]}"
    exit 1
fi

export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-lmsysorg/sglang:v0.5.2rc1-rocm700-mi30x}"
if test -z "${DOCKER_IMAGE_NAME}"; then
  echo "Error: DOCKER_IMAGE_NAME is not set or empty."
  exit 1
fi

# Set current directory to be REPO directory with all relevant scripts
export MOONCAKE_REPO_DIR=$(pwd)
LOG_PATH="${LOG_PATH:-/shared-inference/${USER}/model_blog_logs}"

xP="${xP:-1}" #-> Number of Prefill Servers
yD="${yD:-1}" #-> Number of Decode Servers

MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-"/shared-inference/models_blog/"}"

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
ORIGINAL_MODEL_PATH="/shared-inference/models_blog/${MODEL_NAME}"

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
# Check /shared-inference/models_blog
elif check_model_path "/shared-inference/models_blog/$MODEL_NAME" "/shared-inference/models_blog"; then
    MODEL_PATH="/shared-inference/models_blog/$MODEL_NAME"
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
    echo "  - /shared-inference/models_blog/$MODEL_NAME"
    echo ""
    echo "Model must be accessible from all nodes for distributed execution."
    echo "Please ensure the model is available on all allocated nodes."
    exit 1
fi

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""


# Calculate NUM_NODES based on xP and yD
NUM_NODES=$((xP + yD + 1))
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD + 1)"
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD + 1)"

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
MASTER_ADDR=$(echo "$MASTER_ADDR" | awk 'NR==1 {print $1}')
MASTER_PORT=39566  # Choose an open port

IPS=()

for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'hostname -I')
    IP=$(echo "$IP" | awk 'NR==1 {print $1}')
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
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"

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
    -v $HOME:$HOME \
    -v /shared-inference:/shared-inference \
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
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_FILE=$BENCHMARK_FILE \
    -e IPADDRS=$IPADDRS \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME bash -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_sglang_bench_serving.sh_NODE${SLURM_PROCID}.log
    "
'
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'docker rm -f $DOCKER_CONT_NAME'

