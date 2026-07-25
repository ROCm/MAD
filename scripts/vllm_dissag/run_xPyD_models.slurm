#!/bin/bash
#SBATCH --job-name=vllm-pd    # Specify a custom string for your slurm batch job
#SBATCH -N 2                  # Default 2 nodes (1P/1D); override with sbatch -N for larger topologies
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
#
# Priority:
# 1. BASH_SOURCE — works for direct invocation (bash script.sh)
# 2. SLURM_SUBMIT_DIR + script path — when sbatch submits from repo root,
#    SLURM_SUBMIT_DIR is the CWD, not the script dir. Append the relative
#    path from the SBATCH command to get the actual script directory.
# 3. SLURM_SUBMIT_DIR alone — last resort (assumes sbatch was run from
#    the script directory).
# ------------------------
_resolve_script_dir() {
    # Try BASH_SOURCE first (works for direct invocation)
    if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
        local _d
        _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
        if [[ -n "$_d" && -f "$_d/vllm_disagg.sh" ]]; then
            echo "$_d"
            return 0
        fi
    fi

    # Try SLURM_SUBMIT_DIR + relative script path
    # When madengine does: sbatch scripts/vllm_dissag/run_xPyD_models.slurm
    # SLURM_SUBMIT_DIR = repo root, so we need to append the dirname
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        local _candidate="$SLURM_SUBMIT_DIR/scripts/vllm_dissag"
        if [[ -f "$_candidate/vllm_disagg.sh" ]]; then
            echo "$_candidate"
            return 0
        fi
        # Maybe they ran sbatch from the script dir itself
        if [[ -f "$SLURM_SUBMIT_DIR/vllm_disagg.sh" ]]; then
            echo "$SLURM_SUBMIT_DIR"
            return 0
        fi
    fi

    # Fallback
    echo "."
    return 1
}

SCRIPT_DIR="$(_resolve_script_dir)"
cd "$SCRIPT_DIR" || { echo "Error: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

REQUIRED_FILES=("vllm_disagg.sh" "parallelism.sh" "connectors/rixl.sh" "connectors/moriio.sh" "models.yaml" "benchmark_xPyD.sh" "parse_to_csv.py" "socket_barrier.py" "socket_wait.py" "connectors/moriio.env" "connectors/rixl.env")
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file '$f' not found in $(pwd)." >&2
        echo "Please run sbatch from the scripts/vllm_dissag/ directory, e.g.:" >&2
        echo "  cd MAD/scripts/vllm_dissag && sbatch run_xPyD_models.slurm" >&2
        exit 1
    fi
done
echo "Running from: $(pwd)"

# ------------------------------------------------------------------------------
# models.yaml env precedence: capture which recipe knobs the USER explicitly set
# at submit time. The driver (vllm_disagg.sh) uses this to let models.yaml `env:`
# OVERRIDE image-baked ENV defaults (e.g. a DeepSeek-tuned image bakes
# KV_BLOCK_SIZE=16 / VLLM_ROCM_USE_AITER_MLA=0, which would otherwise shadow a
# model's own recipe defined in models.yaml env:), while a genuine
# submit-time `-e VAR=...` still wins. Precedence: image-baked < models.yaml < submit -e.
# Captured HERE (before the slurm sets any defaults) so it reflects user intent only.
_RECIPE_ENV_KEYS="VLLM_USE_V1 VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_RMSNORM VLLM_ROCM_USE_AITER_MLA KV_BLOCK_SIZE KV_CACHE_DTYPE KV_CACHE_MEMORY_BYTES GPU_MEMORY_UTILIZATION VLLM_CUDAGRAPH_MODE PREFILL_CUDAGRAPH_MODE DECODE_CUDAGRAPH_MODE CUDAGRAPH_CAPTURE_SIZES VLLM_ALL2ALL_BACKEND PREFILL_MORI_BACKEND DECODE_MORI_BACKEND MORI_SHMEM_HEAP_SIZE"
MODELS_YAML_PROTECT=""
for _k in $_RECIPE_ENV_KEYS; do
    [ -n "${!_k+x}" ] && MODELS_YAML_PROTECT="${MODELS_YAML_PROTECT} ${_k}"
done
export MODELS_YAML_PROTECT="${MODELS_YAML_PROTECT# }"
echo "models.yaml protect-list (submit-time overrides): '${MODELS_YAML_PROTECT}'"

# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# Define valid model names (must have a models.yaml entry)
VALID_MODELS=( \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "gpt-oss-120b" \
    "DeepSeek-R1" \
    "Qwen3-32B" \
    "Qwen3-30B-A3B" \
)

# Models allowed for CONNECTOR=moriio WIDE_EP=1 (MoRI-EP; legacy RUN_MORI=1)
MORI_EP_VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "DeepSeek-R1" \
)

# Models allowed for CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep (legacy RUN_DEEPEP=1)
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
# Axis selection -> single launcher (vllm_disagg.sh).
# Two axes: CONNECTOR={rixl|moriio} x WIDE_EP={0=TP|1=wideEP}; EP_BACKEND only
# when WIDE_EP=1 (rixl->deepep, moriio->mori). Legacy RUN_MORI / RUN_DEEPEP are
# still honored via a back-compat shim. The launcher itself does the final
# CONNECTOR/WIDE_EP/EP_BACKEND validation; here we resolve them + gate the model
# against the existing allowlists.
# ---------------------------------------------------------------------------
RUN_FILE="vllm_disagg.sh"
_run_mori="${RUN_MORI:-0}"
_run_deepep="${RUN_DEEPEP:-0}"

if [[ "$_run_mori" == "1" && "$_run_deepep" == "1" ]]; then
  echo "Error: Both RUN_MORI and RUN_DEEPEP are set to 1. Set only one." >&2
  exit 1
fi

# Back-compat: legacy flags map onto the axes when CONNECTOR is not set explicitly.
if [[ -z "${CONNECTOR:-}" ]]; then
  if [[ "$_run_mori" == "1" ]]; then
    CONNECTOR=moriio; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-mori}"
  elif [[ "$_run_deepep" == "1" ]]; then
    CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-deepep}"
  else
    # Default keeps the historical "no flags" behavior: rixl + TP.
    CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-0}"
  fi
fi
WIDE_EP="${WIDE_EP:-0}"

# Models that ONLY run wideEP (DP/EP), never TP: the DeepSeek family is served with
# the MoRI-EP / DeepEP recipe (block=16, MLA off, per-role cudagraph). Running them
# in TP mode is unsupported — the TP argv would double the model's own
# --compilation-config and drop the mandatory +quant_fp8 op. Reject early.
WIDE_EP_ONLY_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" )
model_is_wide_ep_only() {
  local m="$1"
  for x in "${WIDE_EP_ONLY_MODELS[@]}"; do [[ "$m" == "$x" ]] && return 0; done
  return 1
}

# Model allowlist gate (kept, sglang-style). wideEP modes use the per-backend
# allowlists; TP uses the global VALID_MODELS (already validated above) minus the
# wideEP-only models.
if [[ "$WIDE_EP" == "0" ]]; then
  if model_is_wide_ep_only "$MODEL_NAME"; then
    echo "Error: MODEL_NAME '$MODEL_NAME' is wideEP-only (set WIDE_EP=1). TP mode is not supported for it." >&2
    printf "wideEP-only models:\n"; for m in "${WIDE_EP_ONLY_MODELS[@]}"; do printf "  - %s\n" "$m"; done
    exit 1
  fi
elif [[ "$WIDE_EP" == "1" && "${CONNECTOR}" == "moriio" ]]; then
  if ! model_allows_mori_ep "$MODEL_NAME"; then
    echo "Error: CONNECTOR=moriio WIDE_EP=1 but MODEL_NAME '$MODEL_NAME' is not in MORI_EP_VALID_MODELS" >&2
    printf "MoRI EP allowed models:\n"; for m in "${MORI_EP_VALID_MODELS[@]}"; do printf "  - %s\n" "$m"; done
    exit 1
  fi
elif [[ "$WIDE_EP" == "1" && "${CONNECTOR}" == "rixl" ]]; then
  if ! model_allows_deepep "$MODEL_NAME"; then
    echo "Error: CONNECTOR=rixl WIDE_EP=1 (deepep) but MODEL_NAME '$MODEL_NAME' is not in DEEPEP_VALID_MODELS" >&2
    printf "DeepEP allowed models:\n"; for m in "${DEEPEP_VALID_MODELS[@]}"; do printf "  - %s\n" "$m"; done
    exit 1
  fi
fi

export CONNECTOR WIDE_EP EP_BACKEND
echo "Launcher: $RUN_FILE  (CONNECTOR=${CONNECTOR} WIDE_EP=${WIDE_EP} EP_BACKEND=${EP_BACKEND:-<n/a>}) for model '$MODEL_NAME'"

# ---------------------------------------------------------------------------
# Connector platform env: per-connector <connector>.env holds the ROCm-7.2.3
# runtime env that MUST reach the container at PID 1 (e.g. expandable_segments:False
# for GPU-RDMA registration). Source the resolved connector's file and collect its
# KEY=VALUE lines into CONNECTOR_ENV_ARGS as `-e KEY=${KEY:-VALUE}` pairs, so a
# submit-time export of the same name still overrides. Forwarded in the docker run.
# ---------------------------------------------------------------------------
CONNECTOR_ENV_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.env"
CONNECTOR_ENV_ARGS=""
if [[ -f "$CONNECTOR_ENV_FILE" ]]; then
  echo "Loading connector platform env: $CONNECTOR_ENV_FILE"
  while IFS= read -r _line; do
    [[ "$_line" =~ ^[[:space:]]*# || -z "${_line// }" ]] && continue
    _k="${_line%%=*}"; _v="${_line#*=}"
    CONNECTOR_ENV_ARGS+=" -e ${_k}=${!_k:-$_v}"   # submit-time export of $_k wins
  done < "$CONNECTOR_ENV_FILE"
else
  echo "WARN: connector env file not found: $CONNECTOR_ENV_FILE" >&2
fi

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
  echo "Error: DOCKER_IMAGE_NAME is not set. Please export DOCKER_IMAGE_NAME before running." >&2
  echo "  There is no public prebuilt image. Build your own from the provided Dockerfile:" >&2
  echo "    docker build -f docker/vllm_disagg_inference.ubuntu.amd.Dockerfile \\" >&2
  echo "      -t <your-registry>/vllm-disagg:local .   # all connectors (WITH_NIXL=1 default)" >&2
  echo "    (add --build-arg WITH_NIXL=0 for a lean MoRI-EP-only image)" >&2
  echo "  then: export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:local" >&2
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

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
BENCHMARK_CON="${BENCHMARK_CON:-}"
BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-}"

# Benchmark script selector: BENCHMARK_SCRIPT tag -> file run by the launcher.
#   sweep        (default) -> benchmark_xPyD.sh         (general concurrency sweep)
#   long_context           -> benchmark_long_context.sh (per-shape warmup, c=1-first)
#   niah                   -> benchmark_niah.sh         (needle-in-haystack accuracy)
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-sweep}"
case "$BENCHMARK_SCRIPT" in
    sweep)        BENCHMARK_SCRIPT_FILE="benchmark_xPyD.sh" ;;
    long_context) BENCHMARK_SCRIPT_FILE="benchmark_long_context.sh" ;;
    niah)         BENCHMARK_SCRIPT_FILE="benchmark_niah.sh" ;;
    *) echo "Error: invalid BENCHMARK_SCRIPT='$BENCHMARK_SCRIPT' (valid: sweep, long_context, niah)" >&2; exit 1 ;;
esac
if [[ ! -f "$BENCHMARK_SCRIPT_FILE" ]]; then
    echo "Error: selected benchmark script '$BENCHMARK_SCRIPT_FILE' not found in $(pwd)." >&2
    exit 1
fi
echo "BENCHMARK_SCRIPT='$BENCHMARK_SCRIPT' -> $BENCHMARK_SCRIPT_FILE"

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

export CONNECTOR_ENV_ARGS="$CONNECTOR_ENV_ARGS"
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
export BENCHMARK_SCRIPT_FILE="${BENCHMARK_SCRIPT_FILE}"

export DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/${RUN_FILE}"

# Use only the selected nodes for srun execution
SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
docker ps -q | xargs --no-run-if-empty docker stop;
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true;
fuser -k 5000/tcp 2>/dev/null || true;
fuser -k 2222/tcp 2>/dev/null || true;
fuser -k 15000/tcp 2>/dev/null || true;
sleep 2;
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true;

# --- Create host-local compilation cache dirs (ext4, survives container restarts) ---
mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true;

# --- Persistent JIT cache mount ---
# The image points AITER_JIT_DIR/TRITON_CACHE_DIR/VLLM_CACHE_ROOT/COMGR_CACHE_DIR at
# /opt/vllm_cache. Mount a host dir there so AITER CK kernels compile ONCE and are reused
# across runs. Cold compile is ~15 min for the DeepSeek MoE/FP8 GEMM kernel set; a warm
# boot is ~1 min. Keyed by image ID so a new image starts a fresh cache and does not reuse
# stale-ABI shared objects. Host dir on local NVMe. Override JIT_CACHE_HOST, or set
# JIT_CACHE_PERSIST=0 to disable and fall back to the image empty in-container cache.
# NOTE: this whole section runs inside a single-quoted `srun bash -c '...'`, so avoid
# single quotes here; the image-id hash is extracted with tr, not sed.
if [ "${JIT_CACHE_PERSIST:-1}" = "1" ]; then
    _IMG_RAW=$(docker image inspect --format "{{.Id}}" "$DOCKER_IMAGE_NAME" 2>/dev/null);
    _IMG_KEY=$(printf "%s" "$_IMG_RAW" | tr -cd "a-f0-9" | cut -c1-12);
    _IMG_KEY="${_IMG_KEY:-noimg}";
    _JIT_CACHE_HOST="${JIT_CACHE_HOST:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/${_IMG_KEY}}";
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true;
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache";
    echo "[jit-cache] persistent image ${_IMG_KEY}: ${_JIT_CACHE_HOST} to /opt/vllm_cache";
else
    _JIT_CACHE_MOUNT="";
fi

# --- Build host RDMA library mounts ---
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu

for _lib in libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1; do
    [ -e "$_LIBDIR/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/$_lib:$_LIBDIR/$_lib:ro"
done
for _vlib in $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.*; do
    [ -e "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done

for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        [ -e "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done

[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"
echo "[host-rdma] mounts: $_RDMA_MOUNTS"

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
    --shm-size ${DOCKER_SHM_SIZE:-256G} \
    --ulimit nofile=524288:524288 \
    --ulimit memlock=-1:-1 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    ${_JIT_CACHE_MOUNT} \
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
    ${BENCHMARK_PORT:+-e BENCHMARK_PORT=$BENCHMARK_PORT} \
    ${PROXY_TYPE:+-e PROXY_TYPE=$PROXY_TYPE} \
    ${ROUTER_PORT:+-e ROUTER_PORT=$ROUTER_PORT} \
    -e IPADDRS=$IPADDRS \
    ${CONNECTOR:+-e CONNECTOR=$CONNECTOR} \
    ${WIDE_EP:+-e WIDE_EP=$WIDE_EP} \
    ${EP_BACKEND:+-e EP_BACKEND=$EP_BACKEND} \
    ${RUN_MORI:+-e RUN_MORI=$RUN_MORI} \
    ${RUN_DEEPEP:+-e RUN_DEEPEP=$RUN_DEEPEP} \
    ${VLLM_ALL2ALL_BACKEND:+-e VLLM_ALL2ALL_BACKEND=$VLLM_ALL2ALL_BACKEND} \
    ${PREFILL_MORI_BACKEND:+-e PREFILL_MORI_BACKEND=$PREFILL_MORI_BACKEND} \
    ${DECODE_MORI_BACKEND:+-e DECODE_MORI_BACKEND=$DECODE_MORI_BACKEND} \
    -e MODELS_YAML_PROTECT="${MODELS_YAML_PROTECT:-}" \
    ${KV_BLOCK_SIZE:+-e KV_BLOCK_SIZE=$KV_BLOCK_SIZE} \
    ${KV_CACHE_MEMORY_BYTES:+-e KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES} \
    ${VLLM_ROCM_USE_AITER_MLA:+-e VLLM_ROCM_USE_AITER_MLA=$VLLM_ROCM_USE_AITER_MLA} \
    ${ROUTER_BINARY:+-e ROUTER_BINARY=$ROUTER_BINARY} \
    ${KV_CACHE_DTYPE:+-e KV_CACHE_DTYPE=$KV_CACHE_DTYPE} \
    ${MORIIO_TOY_PROXY:+-e MORIIO_TOY_PROXY=$MORIIO_TOY_PROXY} \
    ${BENCHMARK_SCRIPT_FILE:+-e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE} \
    ${PREFILL_CUDAGRAPH_MODE:+-e PREFILL_CUDAGRAPH_MODE=$PREFILL_CUDAGRAPH_MODE} \
    ${DECODE_CUDAGRAPH_MODE:+-e DECODE_CUDAGRAPH_MODE=$DECODE_CUDAGRAPH_MODE} \
    ${CUDAGRAPH_CAPTURE_SIZES:+-e CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES"} \
    ${MORI_RDMA_TC:+-e MORI_RDMA_TC=$MORI_RDMA_TC} \
    ${MORI_RDMA_SL:+-e MORI_RDMA_SL=$MORI_RDMA_SL} \
    ${MORI_SHMEM_HEAP_SIZE:+-e MORI_SHMEM_HEAP_SIZE=$MORI_SHMEM_HEAP_SIZE} \
    ${PREFILL_DEEPEP_BACKEND:+-e PREFILL_DEEPEP_BACKEND=$PREFILL_DEEPEP_BACKEND} \
    ${DECODE_DEEPEP_BACKEND:+-e DECODE_DEEPEP_BACKEND=$DECODE_DEEPEP_BACKEND} \
    ${ENABLE_DBO:+-e ENABLE_DBO=$ENABLE_DBO} \
    ${DBO_COMM_SMS:+-e DBO_COMM_SMS=$DBO_COMM_SMS} \
    ${ENABLE_PROFILING:+-e ENABLE_PROFILING=$ENABLE_PROFILING} \
    ${NCCL_IB_HCA:+-e NCCL_IB_HCA=$NCCL_IB_HCA} \
    ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX} \
    ${NCCL_NET_GDR_LEVEL:+-e NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL} \
    ${NCCL_CROSS_NIC:+-e NCCL_CROSS_NIC=$NCCL_CROSS_NIC} \
    ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME} \
    ${GLOO_SOCKET_IFNAME:+-e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME} \
    -e MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-eth0} \
    ${MORI_IB_GID_INDEX:+-e MORI_IB_GID_INDEX=$MORI_IB_GID_INDEX} \
    ${MORI_RDMA_DEVICES:+-e MORI_RDMA_DEVICES=$MORI_RDMA_DEVICES} \
    ${MORI_NUM_QP_PER_PE:+-e MORI_NUM_QP_PER_PE=$MORI_NUM_QP_PER_PE} \
    ${VLLM_MORIIO_QP_PER_TRANSFER:+-e VLLM_MORIIO_QP_PER_TRANSFER=$VLLM_MORIIO_QP_PER_TRANSFER} \
    ${VLLM_MORIIO_NUM_WORKERS:+-e VLLM_MORIIO_NUM_WORKERS=$VLLM_MORIIO_NUM_WORKERS} \
    -e GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8} \
    -e GPUS_PER_NODE=${GPUS_PER_NODE:-8} \
    ${GPU_MAX_HW_QUEUES:+-e GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES} \
    ${HIP_FORCE_DEV_KERNARG:+-e HIP_FORCE_DEV_KERNARG=$HIP_FORCE_DEV_KERNARG} \
    ${HSA_NO_SCRATCH_RECLAIM:+-e HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM} \
    ${VLLM_HANDSHAKE_TIMEOUT_MINS:+-e VLLM_HANDSHAKE_TIMEOUT_MINS=$VLLM_HANDSHAKE_TIMEOUT_MINS} \
    ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S} \
    ${ROCSHMEM_HEAP_SIZE:+-e ROCSHMEM_HEAP_SIZE=$ROCSHMEM_HEAP_SIZE} \
    ${ROCSHMEM_MAX_NUM_CONTEXTS:+-e ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS} \
    ${LOG_WAIT_TIMEOUT_SECONDS:+-e LOG_WAIT_TIMEOUT_SECONDS=$LOG_WAIT_TIMEOUT_SECONDS} \
    ${TRITON_CACHE_DIR:+-e TRITON_CACHE_DIR=$TRITON_CACHE_DIR} \
    ${VLLM_CACHE_ROOT:+-e VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT} \
    ${COMGR_CACHE_DIR:+-e COMGR_CACHE_DIR=$COMGR_CACHE_DIR} \
    ${AITER_JIT_DIR:+-e AITER_JIT_DIR=$AITER_JIT_DIR} \
    -e DISTRIBUTED_TIMEOUT_SECONDS=${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
    -e VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-300000} \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600} \
    ${CONNECTOR_ENV_ARGS} \
    ${VLLM_CUDAGRAPH_MODE:+-e VLLM_CUDAGRAPH_MODE=$VLLM_CUDAGRAPH_MODE} \
    ${CUDAGRAPH_CAPTURE_SIZES:+-e CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES"} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${SLURM_PROCID}.log
    "
'
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'docker stop $DOCKER_CONT_NAME 2>/dev/null || true; docker rm $DOCKER_CONT_NAME 2>/dev/null || true'

