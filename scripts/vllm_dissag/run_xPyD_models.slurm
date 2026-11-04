#!/bin/bash
#SBATCH --job-name=vllm-pd    # Specify a custom string for your slurm batch job
#SBATCH -N 2                  # Default 2 nodes (1P/1D); override with sbatch -N for larger topologies
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="/shared_inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared_inference/%u/model_blog_logs/slurm-%j.err"


# This launcher's directory. Under sbatch the scheduler runs a copy of this script
# from its spool directory (SLURM's slurmd, Spur's /var/spool/spur), so the script's
# own path is not the repo: then the submit directory names it, as
# `cd <MAD>/scripts/vllm_dissag && sbatch ...` or `sbatch scripts/vllm_dissag/...` from the root.
_resolve_launcher_dir() {
    local _d
    for _d in "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" \
              "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/scripts/vllm_dissag}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$_d" && -f "$_d/vllm_disagg.sh" ]] && { (cd "$_d" && pwd); return 0; }
    done
    echo "Error: cannot find scripts/vllm_dissag (looked next to this script, in SLURM_SUBMIT_DIR and in $PWD);" \
         "run sbatch from MAD/scripts/vllm_dissag or from the MAD root" >&2
    return 1
}
SCRIPT_DIR="$(_resolve_launcher_dir)" || exit 1
cd "$SCRIPT_DIR" || { echo "Error: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

# Site config: filesystem roots, partition, fabric device, ports, timeouts.
# Sourced, so `environment > these defaults`, which is what lets plain
# sbatch and madengine drive this script identically -- both speak only
# environment variables, and this file just fills what neither of them set.
# Model-specific performance flags stay in models.yaml, not here.
MAD_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Configuration layers (common/config/layers.sh): what was set at submit time is
# recorded first, so the recipe below can tell a chosen value from a site default.
. "${MAD_SCRIPTS_DIR}/common/config/layers.sh"
config_begin
# A topology spec file, when given (MAD_TOPOLOGY), sets this launcher's settings
# before the defaults below; see common/topology/apply.sh.
. "${MAD_SCRIPTS_DIR}/common/topology/apply.sh" vllm_dissag
config_layer topology
if [ ! -f "${MAD_SCRIPTS_DIR}/common/cluster.sh" ]; then
    echo "Error: missing ${MAD_SCRIPTS_DIR}/common/cluster.sh" >&2
    exit 1
fi
. "${MAD_SCRIPTS_DIR}/common/cluster.sh"
config_layer site
. "${MAD_SCRIPTS_DIR}/common/launcher/launcher_lib.sh"

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
    "GLM-5.1-FP8" \
    "Kimi-K3" \
    "Kimi-K3-MXFP4" \
    "Kimi-K3-MXFP4-MI355X" \
)

# Models allowed for CONNECTOR=moriio WIDE_EP=1 (MoRI-EP; legacy RUN_MORI=1)
MORI_EP_VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "DeepSeek-R1" \
    "GLM-5.1-FP8" \
    "Kimi-K3" \
    "Kimi-K3-MXFP4" \
    "Kimi-K3-MXFP4-MI355X" \
)

# Models allowed for CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep (legacy RUN_DEEPEP=1)
DEEPEP_VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-5layer" \
    "DeepSeek-R1" \
)


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
# the MoRI-EP / DeepEP recipe (block=16, MLA off, per-role cudagraph); on rixl its TP
# argv would double the model's own --compilation-config and drop the mandatory
# +quant_fp8 op. Reject early.
# GLM-5.1-FP8 (MLA+DSA) is validated only under MoRI-EP wideEP; moriio+TP is untested.
# Kimi-K3 and Kimi-K3-MXFP4 are wideEP-only as well (see their models.yaml blocks).
WIDE_EP_ONLY_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" "GLM-5.1-FP8" "Kimi-K3" "Kimi-K3-MXFP4" "Kimi-K3-MXFP4-MI355X" )
# The exception: the DeepSeek family also runs TP on the moriio connector, from the tp:
# blocks in models.yaml (the TP baseline next to wideEP). Their TP flags are written for
# that connector, which emits the one --compilation-config with +quant_fp8 itself; rixl
# TP stays rejected.
MORIIO_TP_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" )
model_is_wide_ep_only() {
  local m="$1"
  if [[ "${CONNECTOR:-}" == "moriio" ]]; then
    for x in "${MORIIO_TP_MODELS[@]}"; do [[ "$m" == "$x" ]] && return 1; done
  fi
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
# The recipe's env: block first, over site and connector defaults and under what was
# submitted or set by the topology spec (common/config/layers.sh).
config_apply_recipe "${SCRIPT_DIR}/models.yaml" "$MODEL_NAME" "${SCRIPT_DIR}/connectors/${CONNECTOR}.env"
CONNECTOR_ENV_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.env"
CONNECTOR_ENV_ARGS=""
if [[ -f "$CONNECTOR_ENV_FILE" ]]; then
  echo "Loading connector platform env: $CONNECTOR_ENV_FILE"
  while IFS= read -r _line; do
    [[ "$_line" =~ ^[[:space:]]*# || -z "${_line// }" ]] && continue
    _k="${_line%%=*}"; _v="${_line#*=}"
    [[ -z "${!_k:-}" ]] && config_note "$_k" "$_v" connector-env
    CONNECTOR_ENV_ARGS+=" -e ${_k}=${!_k:-$_v}"   # submit-time export of $_k wins
  done < "$CONNECTOR_ENV_FILE"
else
  echo "WARN: connector env file not found: $CONNECTOR_ENV_FILE" >&2
fi

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
  echo "Error: DOCKER_IMAGE_NAME is not set. Please export DOCKER_IMAGE_NAME before running." >&2
  echo "  There is no public prebuilt image. Build your own from the card's pin set (docker/pins/):" >&2
  echo "    docker build -f docker/di_vllm.ubuntu.amd.Dockerfile \\" >&2
  echo "      \$(python3 docker/pins/pins.py build-args vllm-mori121 --arch gfx942) \\" >&2
  echo "      -t <your-registry>/di-vllm:vllm-mori121-gfx942 .   # from the MAD root" >&2
  echo "  then: export DOCKER_IMAGE_NAME=<your-registry>/di-vllm:vllm-mori121-gfx942" >&2
  exit 1
fi
export DOCKER_IMAGE_NAME

# Set current directory to be REPO directory with all relevant scripts
NIXL_REPO_DIR=$(pwd)
LOG_PATH="${LOG_PATH:-${SHARED_MOUNT:-/shared_inference}/${USER}/model_blog_logs}"

xP="${xP:-1}" #-> Number of Prefill NODES (a pool may span >1 node)
yD="${yD:-1}" #-> Number of Decode  NODES (a pool may span >1 node)
# ROUTER_NODE=1: the router (proxy) and the benchmark client get a node of their own,
# after the decode nodes, instead of sharing the first prefill node.
export ROUTER_NODE="${ROUTER_NODE:-0}"   # node.sh resolves each rank with it
case "$ROUTER_NODE" in 0|1) ;; *) echo "Error: ROUTER_NODE=${ROUTER_NODE} (0 or 1)" >&2; exit 1 ;; esac
# MODEL_DIR / MODEL_NAME defaults come from cluster.sh.
# EP_TP_SIZE: tensor-parallel degree within each DP rank on the wideEP path. Default 1
# keeps the historical one-DP-rank-per-GPU layout; >1 only when the replicated
# (non-expert) weights do not fit a single GPU. See vllm_disagg.sh topology math.
# Not TP_SIZE: cluster.sh (sourced above) defaults that to GPUS_PER_NODE for the
# colocated launcher.
# Resolve it from the recipe (TP within each EP pool; unset/1 = plain wideEP) so
# submit-time topology/JIT decisions match the in-container launcher. env/-e still wins.
#
# The recipe's value (config_recipe, common/config/layers.sh) when nothing set it: the
# in-container launcher exports the whole env: block itself; the few keys below are
# needed earlier, on the batch node, to size the job and gate it.
[[ -z "${EP_TP_SIZE:-}" ]] && EP_TP_SIZE="$(config_recipe EP_TP_SIZE)"
export EP_TP_SIZE="${EP_TP_SIZE:-1}"

# GPU_ARCHS: the architectures this model's recipe is written for (models.yaml).
# Checked against the allocated nodes before anything expensive happens -- see
# cluster_require_gpu_arch in cluster.sh. The model card's skip_gpu_arch states the
# same thing for orchestrators; tests/argv_assert.sh keeps the two consistent.
[[ -z "${GPU_ARCHS:-}" ]] && GPU_ARCHS="$(config_recipe GPU_ARCHS)"
export GPU_ARCHS

# TP-within-EP (EP_TP_SIZE>1) is only valid on the wideEP MoRIIO path. Per-pool sizing is
# validated in vllm_disagg.sh (EP_TP_SIZE must divide GPUS_PER_NODE). Recommended shape: 2P/2D.
if (( ${EP_TP_SIZE:-1} > 1 )); then
  if [[ "$WIDE_EP" != "1" || "$CONNECTOR" != "moriio" ]]; then
    echo "Error: EP_TP_SIZE=${EP_TP_SIZE} (TP-within-EP) requires CONNECTOR=moriio WIDE_EP=1." >&2
    exit 1
  fi
fi



# ------------------------
# Model path validation and selection across all nodes
# ------------------------
echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

# Get all allocated nodes
ALL_NODES=$(cluster_expand_nodelist "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

# Function to check model path on all nodes

# Before the weights probe: the wrong GPU should cost seconds, not a checkpoint read.
cluster_require_gpu_arch "${MODEL_NAME}" "${GPU_ARCHS}" || exit 1
cluster_report_rdma_rails
# MoRI JIT-compiles its EP kernels for MORI_GPU_ARCHS. connectors/moriio.env pins it to
# gfx942, which on MI355X builds kernels for the wrong GPU. Follow the detected arch
# unless the submitter set it (the later -e wins in docker run).
if [ -z "${MORI_GPU_ARCHS:-}" ] && [ -n "${MAD_GPU_ARCH:-}" ]; then
  CONNECTOR_ENV_ARGS+=" -e MORI_GPU_ARCHS=${MAD_GPU_ARCH}"
fi

# Walk the candidate list from cluster.sh: local NVMe, then shared NFS, then an
# explicit MODEL_DIR. Requires the weights on EVERY allocated node -- all ranks
# load from one path, so a partial hit fails the run deep into bring-up.
cluster_resolve_model_path || exit 1

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""


# Calculate NUM_NODES based on xP and yD
NUM_NODES=$((xP + yD + ROUTER_NODE))
if [[ "$ROUTER_NODE" == "1" ]]; then
    echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD + 1 router node)"
else
    echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD, proxy co-located on prefill master)"
fi
if [[ "${WIDE_EP:-0}" == "1" ]]; then
    echo "Parallelism: EP_TP_SIZE=$EP_TP_SIZE -> DP ranks/node = ${GPUS_PER_NODE:-8}/$EP_TP_SIZE (wideEP)"
else
    echo "Parallelism: TP (WIDE_EP=0) -> degree from the model's tp: flags in models.yaml"
fi

# DeepEP configuration (only exported when RUN_DEEPEP=1)
if [[ "$_run_deepep" == "1" ]]; then
    export PREFILL_DEEPEP_BACKEND="${PREFILL_DEEPEP_BACKEND:-deepep_high_throughput}"
    export DECODE_DEEPEP_BACKEND="${DECODE_DEEPEP_BACKEND:-deepep_low_latency}"
    export ENABLE_DBO="${ENABLE_DBO:-false}"
    export DBO_COMM_SMS="${DBO_COMM_SMS:-}"
    export ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
    echo "DeepEP config: PREFILL_BACKEND=$PREFILL_DEEPEP_BACKEND DECODE_BACKEND=$DECODE_DEEPEP_BACKEND DBO=$ENABLE_DBO"
fi

# The first NUM_NODES nodes of the allocation, the SLURM_* view narrowed to them, and
# their addresses (common/launcher/launcher_lib.sh).
launcher_select_nodes "$NUM_NODES" "1p1d_bench-serving"
USER_NAME=$(whoami)
MASTER_PORT=39566  # Choose an open port
launcher_node_ips

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
BENCHMARK_CON="${BENCHMARK_CON:-}"
BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-}"

# Benchmark script selector: BENCHMARK_SCRIPT tag -> file run by the launcher.
#   sweep        (default) -> benchmark_xPyD.sh         (general concurrency sweep)
#   long_context           -> benchmark_long_context.sh (per-shape warmup, c=1-first)
#   agentic                -> benchmark_agentic.sh      (aiperf agentic trace replay)
# The launcher/connectors auto-enable prefix caching + server /metrics when
# BENCHMARK_SCRIPT=agentic.
#   niah                   -> benchmark_niah.sh         (needle-in-a-haystack; NIAH_WORDS)
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-sweep}"
case "$BENCHMARK_SCRIPT" in
    sweep)        BENCHMARK_SCRIPT_FILE="benchmark_xPyD.sh" ;;
    long_context) BENCHMARK_SCRIPT_FILE="benchmark_long_context.sh" ;;
    agentic)      BENCHMARK_SCRIPT_FILE="benchmark_agentic.sh" ;;
    niah)         BENCHMARK_SCRIPT_FILE="benchmark_niah.sh" ;;
    *) echo "Error: invalid BENCHMARK_SCRIPT='$BENCHMARK_SCRIPT' (valid: sweep, long_context, agentic, niah)" >&2; exit 1 ;;
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
cluster_mark_dir "$LOG_PATH" "per-job logs from vLLM disaggregated runs (one directory per SLURM job id)"
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
export IPADDRS          # launcher_node_ips
export BENCHMARK_ITR=$BENCHMARK_ITR
export BENCHMARK_CON="${BENCHMARK_CON}"
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}"
export BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT}"
export BENCHMARK_SCRIPT_FILE="${BENCHMARK_SCRIPT_FILE}"

export DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
# Container-creation barrier ports (connectors/rixl.sh: BARRIER_PORT, moriio.sh:
# BARRIER_PORT_MORI), one per job unless set (cluster_barrier_port). They used to be
# fixed (15000, 2222), and each node freed them first with `fuser -k`, which killed
# whatever else -- another job of this user on a shared node -- was listening there.
export BARRIER_PORT="${BARRIER_PORT:-$(cluster_barrier_port)}"
export BARRIER_PORT_MORI="${BARRIER_PORT_MORI:-$BARRIER_PORT}"
export RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/${RUN_FILE}"

# Every selected node runs node.sh, then the containers are removed; rank 0 gives the
# job's verdict (common/launcher/launcher_lib.sh, cluster_job_verdict).
launcher_run_nodes "${SCRIPT_DIR}/node.sh" \
    'docker stop $DOCKER_CONT_NAME 2>/dev/null || true; docker rm $DOCKER_CONT_NAME 2>/dev/null || true'
launcher_verdict || exit $?

