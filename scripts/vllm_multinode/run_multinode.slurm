#!/bin/bash
#SBATCH --job-name=vllm-colocated
#SBATCH -N 2                    # Default 2 nodes; override with sbatch -N
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output="/shared_inference/%u/model_blog_logs/slurm-%j.out"
#SBATCH --error="/shared_inference/%u/model_blog_logs/slurm-%j.err"
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################
# COLOCATED multi-node vLLM serving (one instance spanning N nodes).
# =============================================================================
# The counterpart to scripts/vllm_dissag/run_xPyD_models.slurm:
#
#   vllm_dissag     prefill/decode DISAGGREGATED, KV connector, xP+yD nodes
#                   -> highest concurrent throughput
#   vllm_multinode  COLOCATED single instance, no KV connector, N nodes
#                   -> lowest single-request latency (one request uses all GPUs)
#
# Needed whenever a model does not fit one node: e.g. Kimi-K3 on MI300X, where
# the ~1.5 TB MXFP4 checkpoint cannot fit 8x192 GB, so TP8 within each node and
# PP across nodes puts ~102 GB/GPU on a 2-node allocation.
#
# This launcher deliberately owns very little. Node discovery, the container
# launch and the per-node role split live here; everything else is REUSED from
# scripts/vllm_dissag/ (models.yaml recipe, socket_barrier.py, benchmark_xPyD.sh,
# benchmark_niah.sh, parse_to_csv.py), which is why scripts/ is mounted whole.
#
# Usage:
#   cd MAD/scripts/vllm_multinode && sbatch -N 2 run_multinode.slurm
# or via madengine (see models.json entries tagged vllm_multinode).
#
# Key env (madengine passes these through env_vars):
#   MODEL_NAME              key into scripts/vllm_dissag/models.yaml
#   DOCKER_IMAGE_NAME       image to run
#   TP_SIZE                 tensor parallel within a node   (default GPUS_PER_NODE)
#   PP_SIZE                 pipeline parallel across nodes  (default NNODES)
#   ENABLE_EP               1 to add --enable-expert-parallel
#   ALL2ALL_BACKEND         e.g. allgather_reducescatter | mori_low_latency
#   COLOCATED_EXTRA_ARGS    extra serve flags (JSON values must be self-quoted)
#   BENCHMARK_SCRIPT        sweep (default) | long_context | niah
# =============================================================================
set -euo pipefail

# This launcher's directory. Under sbatch the scheduler runs a copy of this script
# from its spool directory (SLURM's slurmd, Spur's /var/spool/spur), so the script's
# own path is not the repo: then the submit directory names it, as
# `cd <MAD>/scripts/vllm_multinode && sbatch ...` or `sbatch scripts/vllm_multinode/...` from the root.
_resolve_launcher_dir() {
    local _d
    for _d in "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" \
              "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/scripts/vllm_multinode}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$_d" && -f "$_d/serve_colocated.sh" ]] && { (cd "$_d" && pwd); return 0; }
    done
    echo "Error: cannot find scripts/vllm_multinode (looked next to this script, in SLURM_SUBMIT_DIR and in $PWD);" \
         "run sbatch from MAD/scripts/vllm_multinode or from the MAD root" >&2
    return 1
}
SCRIPT_DIR="$(_resolve_launcher_dir)" || exit 1
MAD_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Configuration layers (common/config/layers.sh): what was set at submit time is
# recorded first, so later layers can tell a chosen value from a default.
. "${MAD_SCRIPTS_DIR}/common/config/layers.sh"
config_begin
# A topology spec file, when given (MAD_TOPOLOGY), sets this launcher's settings
# before the defaults below; see common/topology/apply.sh.
. "${MAD_SCRIPTS_DIR}/common/topology/apply.sh" vllm_multinode
config_layer topology
SHARED_DIR="${MAD_SCRIPTS_DIR}/vllm_dissag"

REQUIRED_FILES=(
    "${MAD_SCRIPTS_DIR}/common/cluster.sh"
    "${SCRIPT_DIR}/serve_colocated.sh"
    "${SHARED_DIR}/models.yaml"
    "${SHARED_DIR}/socket_barrier.py"
    "${SHARED_DIR}/benchmark_xPyD.sh"
    "${SHARED_DIR}/parse_to_csv.py"
)
for f in "${REQUIRED_FILES[@]}"; do
    [[ -f "$f" ]] || { echo "Error: required file missing: $f" >&2; exit 1; }
done

: "${DOCKER_IMAGE_NAME:?set DOCKER_IMAGE_NAME to the vLLM image to run}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME (key into scripts/vllm_dissag/models.yaml)}"
# Site config: filesystem roots, partition, fabric device, ports, timeouts.
# Sourced, so `environment > these defaults`, which is what lets plain
# sbatch and madengine drive this script identically -- both speak only
# environment variables, and this file just fills what neither of them set.
# Model-specific performance flags stay in models.yaml, not here.
#
# It leaves MODEL_PATH empty unless someone set it: the NVMe-vs-NFS choice needs
# the node list, which is not known until node discovery below.
# Fabric settings the caller passed (card env_vars, madengine, sbatch --export), recorded
# BEFORE cluster.sh fills in its fallbacks. See COLOCATED_ENV_ARGS.
_FABRIC_VARS="NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME RCCL_AINIC_ROCE RDMAV_DRIVERS MORI_RDMA_DEVICES MORI_SOCKET_IFNAME MORI_IB_GID_INDEX"
_FABRIC_PASSED=""
for _v in $_FABRIC_VARS; do [ -n "${!_v:-}" ] && _FABRIC_PASSED="$_FABRIC_PASSED $_v"; done
. "${MAD_SCRIPTS_DIR}/common/cluster.sh"
config_layer site
. "${MAD_SCRIPTS_DIR}/common/launcher/launcher_lib.sh"
# The recipe's env: block over site defaults, under what was submitted
# (common/config/layers.sh); serve_colocated.sh applies the rest in the container.
config_apply_recipe "${SHARED_DIR}/models.yaml" "$MODEL_NAME"
USER_NAME="${USER:-$(id -un)}"
LOG_PATH="${LOG_PATH:-${SHARED_MOUNT:-/shared_inference}/${USER_NAME}/model_blog_logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SERVE_PORT="${SERVE_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_NODES="${SLURM_NNODES:-1}"
TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"
PP_SIZE="${PP_SIZE:-${NUM_NODES}}"
ENABLE_EP="${ENABLE_EP:-0}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-}"

# Benchmark selector — same tags and same scripts as the disagg launcher.
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-sweep}"
case "$BENCHMARK_SCRIPT" in
    sweep)        BENCHMARK_SCRIPT_FILE="benchmark_xPyD.sh" ;;
    long_context) BENCHMARK_SCRIPT_FILE="benchmark_long_context.sh" ;;
    niah)         BENCHMARK_SCRIPT_FILE="benchmark_niah.sh" ;;
    *) echo "Error: invalid BENCHMARK_SCRIPT='$BENCHMARK_SCRIPT' (valid: sweep, long_context, niah)" >&2; exit 1 ;;
esac

echo "=========================== colocated multi-node ==========================="
echo "  model      : ${MODEL_NAME}  (${MODEL_PATH})"
echo "  image      : ${DOCKER_IMAGE_NAME}"
echo "  nodes      : ${NUM_NODES} x ${GPUS_PER_NODE} GPU"
echo "  parallelism: TP=${TP_SIZE} PP=${PP_SIZE} EP=${ENABLE_EP} all2all=${ALL2ALL_BACKEND:-<none>}"
echo "  benchmark  : ${BENCHMARK_SCRIPT} -> ${BENCHMARK_SCRIPT_FILE}"
echo "============================================================================"

if [ -n "${MAD_TOPOLOGY_NNODES:-}" ] && [ "$MAD_TOPOLOGY_NNODES" != "$NUM_NODES" ]; then
    echo "Error: MAD_TOPOLOGY describes ${MAD_TOPOLOGY_NNODES} nodes but the allocation has ${NUM_NODES}." >&2
    exit 1
fi
if [ $(( TP_SIZE * PP_SIZE )) -ne $(( NUM_NODES * GPUS_PER_NODE )) ]; then
    echo "Error: TP_SIZE*PP_SIZE (${TP_SIZE}x${PP_SIZE}=$((TP_SIZE*PP_SIZE))) != total GPUs (${NUM_NODES}x${GPUS_PER_NODE}=$((NUM_NODES*GPUS_PER_NODE)))." >&2
    exit 1
fi

# ------------------------
# Node discovery: rank-ordered IPs, master = first node
# ------------------------
# The whole allocation is the run (NUM_NODES = the allocation), so the SLURM_* view is
# not narrowed; the addresses come from common/launcher/launcher_lib.sh.
SELECTED_NODES=$(cluster_expand_nodelist "$SLURM_JOB_NODELIST" | head -n "$NUM_NODES")
SELECTED_NODELIST=$(echo "$SELECTED_NODES" | paste -sd,)
launcher_node_ips

# ------------------------
# Weight location: local NVMe first, NFS second
# ------------------------
# Shared with the two disagg launchers via cluster.sh. This launcher used to
# hardcode the NFS path and never probe, so a colocated run pulled its checkpoint
# over NFS even on nodes that already held a local copy.
# The card's GPU_ARCHS names the GPUs this recipe is written for; checked before the
# weights probe so the wrong GPU costs seconds (cluster_require_gpu_arch, cluster.sh).
cluster_require_gpu_arch "${MODEL_NAME}" "${GPU_ARCHS:-}" || exit 1
cluster_report_rdma_rails

cluster_resolve_model_path || exit 1
echo "  weights    : ${MODEL_PATH}"

mkdir -p "$LOG_PATH"
cluster_mark_dir "$LOG_PATH" "per-job logs from multinode vLLM runs (one directory per SLURM job id)"

export DOCKER_CONT_NAME="colocated_${MODEL_NAME}_${SLURM_JOB_ID}"
export MODEL_NAME MODEL_PATH MASTER_ADDR MASTER_PORT IPADDRS SERVE_PORT
export TP_SIZE PP_SIZE ENABLE_EP ALL2ALL_BACKEND GPUS_PER_NODE
export BENCHMARK_SCRIPT_FILE LOG_PATH USER_NAME
export NNODES="$NUM_NODES"
export MAD_SCRIPTS_MOUNT=/mad_scripts
export MAD_SCRIPTS_DIR   # node.sh mounts the scripts tree from it

# Only mount local NVMe when it is actually there. Docker auto-creates a DIRECTORY
# at a bind-mount source that does not exist, so an unconditional -v would shadow
# the path with an empty directory on a node that has no such mount -- turning a
# clean NFS fallback into a confusing empty-model-dir failure. Evaluated per node
# in node.sh, on each node, not here on the batch host.

# Knobs CI needs to tune (timeouts, pre-warm) are not worth a new -e line and a
# MAD PR each. COLOCATED_FORWARD_ENV names them, comma-separated, and the srun
# body below forwards whatever it names into the container.
# Anything arriving via sbatch --export=ALL is already exported; re-export here
# so a value set inside this script also survives into srun.
#
# Default it to empty FIRST. `export NAME` on an unset variable marks the name for
# export without giving it a value, so the variable stays unset -- and this script
# runs under `set -u`, which makes the expansion below a fatal error rather than an
# empty loop. Nothing sets this name: not the pipeline, not any model card. It is
# an optional knob, so its absence is the normal case, and the launcher died on
# that normal case before it reached a single line of workload setup.
: "${COLOCATED_FORWARD_ENV:=}"
export COLOCATED_FORWARD_ENV
for _v in ${COLOCATED_FORWARD_ENV//,/ }; do export "$_v" 2>/dev/null || true; done

# Container env this launcher passes, by the same rules as the SGLang disagg launcher:
#   - every fabric setting the caller passed (rails, GID index, socket interfaces), on
#     any cluster. Without them NCCL and Gloo in the container pick their own
#     interfaces for the cross-node PP traffic, whatever the run asked for;
#   - on ainic / thor2, also what cluster.sh derived for the fabric. On cx7 / unknown,
#     cluster.sh-only values (one cluster's fallbacks) are not forwarded, so those runs
#     keep the autodetection they ran with;
#   - with a MoRI all2all backend, the MoRI platform env from
#     scripts/vllm_dissag/connectors/moriio.env (expandable_segments:False on both
#     allocators, HSA_ENABLE_IPC_MODE_LEGACY=0, the RDMA tuning), which PyTorch and
#     MoRI read at start-up. A value exported at submit time wins over the file, and
#     MORI_GPU_ARCHS follows the detected GPU unless it was set (the file pins gfx942).
COLOCATED_ENV_ARGS=""
case "${ALL2ALL_BACKEND}" in
    mori*)
        _MORI_ENV_FILE="${SHARED_DIR}/connectors/moriio.env"
        echo "Loading MoRI platform env: ${_MORI_ENV_FILE}"
        while IFS= read -r _line; do
            [[ "$_line" =~ ^[[:space:]]*# || -z "${_line// }" ]] && continue
            _k="${_line%%=*}"; _v="${_line#*=}"
            [ "$_k" = MORI_GPU_ARCHS ] && [ -z "${MORI_GPU_ARCHS:-}" ] && [ -n "${MAD_GPU_ARCH:-}" ] && _v="$MAD_GPU_ARCH"
            COLOCATED_ENV_ARGS+=" -e ${_k}=${!_k:-$_v}"
        done < "$_MORI_ENV_FILE"
        ;;
esac
_FABRIC_FWD="$_FABRIC_PASSED"
case "${CLUSTER_ARCHETYPE:-}" in ainic|thor2) _FABRIC_FWD="$_FABRIC_VARS" ;; esac
for _v in $_FABRIC_FWD; do
    case " $COLOCATED_ENV_ARGS " in *" -e ${_v}="*) continue ;; esac
    [ -n "${!_v:-}" ] && COLOCATED_ENV_ARGS+=" -e ${_v}=${!_v}"
done
export COLOCATED_ENV_ARGS
echo "[env] to the container:${COLOCATED_ENV_ARGS:- none}"

# The container barrier port (serve_colocated.sh). One per job, not a fixed 2223: the
# launcher used to free 2223 with `fuser -k`, which on a node shared with another job
# killed whatever that job had listening there (cluster_barrier_port).
export CONTAINER_BARRIER_PORT="${CONTAINER_BARRIER_PORT:-$(cluster_barrier_port)}"

# Tear the containers down however the job ends, installed BEFORE the blocking srun.
#
# `docker run` here is attached, not detached, but the container belongs to the
# docker daemon rather than to this job: scancel kills this script and the srun
# under it, and the container carries on holding its GPU memory. The node then
# looks occupied to the next job with no SLURM job on it to explain why, which is
# what the next job hit --
#     RuntimeError: The memory capacity is unbalanced.
#                   Some GPUs may be occupied by other processes.
# -- after the earlier job had been cancelled at the 4 hour wall clock.
#
# The trap used to be installed after the srun returned, so a cancel during the run
# -- the case it exists for -- never reached it. srun runs in the background and the
# script waits on it, so a TERM is handled as soon as it lands rather than when srun
# exits. Idempotent, so the explicit call after the run and the traps can all fire.
_colocated_teardown() {
    dispatch_run teardown "$SELECTED_NODELIST" bash -c \
        'docker stop $DOCKER_CONT_NAME 2>/dev/null || true; docker rm $DOCKER_CONT_NAME 2>/dev/null || true' \
        2>/dev/null || true
}
trap _colocated_teardown EXIT
trap 'trap - EXIT; _colocated_teardown; exit 143' INT TERM

# Each node runs node.sh: srun starts it once per node. It used to be a single-quoted
# `bash -c` body here, where one apostrophe -- even in a comment -- silently truncated
# the script; as a file it has no such hazard. One rule remains for its docker run:
# no comments inside a backslash continuation. The backslash splices the comment onto
# the command, ending it there; a run died that way, with docker run losing its image
# argument. Checked by scripts/common/check_continuations.py.
#
# On the -e list in node.sh: DOCKER_IMAGE_NAME and SLURM_JOB_NODELIST are provenance for
# the perf CSV. Both parse_to_csv.py implementations read them (_get_run_metadata
# for docker_image and machine_name) but run inside the container, so a name the
# list omits is simply absent and the column comes out blank.
dispatch_is_leader && config_write "${LOG_PATH}/${SLURM_JOB_ID}/config.resolved.json"
dispatch_run node "$SELECTED_NODELIST" bash "${SCRIPT_DIR}/node.sh" &
_srun_pid=$!
LAUNCHER_RC=0
wait "$_srun_pid" || LAUNCHER_RC=$?

_colocated_teardown

# Rank 0 gives the job's verdict (common/launcher/launcher_lib.sh, cluster_job_verdict).
# A run with no results must not pass: a hard engine crash (seen with the gfx942 MoE
# LLVM abort, which died in determine_available_memory and still exited 0) would
# otherwise read as a clean run.
launcher_verdict || {
    _rc=$?
    echo "Colocated multi-node run FAILED. Logs: ${LOG_PATH}/${SLURM_JOB_ID}" >&2
    exit "$_rc"
}

echo "Colocated multi-node run complete. Logs: ${LOG_PATH}/${SLURM_JOB_ID}"
