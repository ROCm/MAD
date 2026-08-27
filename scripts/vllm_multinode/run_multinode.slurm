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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MAD_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_DIR="${MAD_SCRIPTS_DIR}/vllm_dissag"

REQUIRED_FILES=(
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
MODEL_DIR="${MODEL_DIR:-/shared_inference/models_blog/}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR%/}/${MODEL_NAME}}"
USER_NAME="${USER:-$(id -un)}"
LOG_PATH="${LOG_PATH:-/shared_inference/${USER_NAME}/model_blog_logs}"
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

if [ $(( TP_SIZE * PP_SIZE )) -ne $(( NUM_NODES * GPUS_PER_NODE )) ]; then
    echo "Error: TP_SIZE*PP_SIZE (${TP_SIZE}x${PP_SIZE}=$((TP_SIZE*PP_SIZE))) != total GPUs (${NUM_NODES}x${GPUS_PER_NODE}=$((NUM_NODES*GPUS_PER_NODE)))." >&2
    exit 1
fi

# ------------------------
# Node discovery: rank-ordered IPs, master = first node
# ------------------------
SELECTED_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n "$NUM_NODES")
SELECTED_NODELIST=$(echo "$SELECTED_NODES" | paste -sd,)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'hostname -I' | awk '{print $1}')

IPS=()
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'hostname -I' | awk '{print $1}')
    IPS+=("$IP")
    echo "  node ${NODE} -> ${IP}"
done
IPADDRS=$(echo "${IPS[*]}" | tr ' ' ',')

mkdir -p "$LOG_PATH"

export DOCKER_CONT_NAME="colocated_${MODEL_NAME}_${SLURM_JOB_ID}"
export MODEL_NAME MODEL_PATH MASTER_ADDR MASTER_PORT IPADDRS SERVE_PORT
export TP_SIZE PP_SIZE ENABLE_EP ALL2ALL_BACKEND GPUS_PER_NODE
export BENCHMARK_SCRIPT_FILE LOG_PATH USER_NAME
export NNODES="$NUM_NODES"
export MAD_SCRIPTS_MOUNT=/mad_scripts

srun --nodelist="$SELECTED_NODELIST" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)"
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true
fuser -k 2223/tcp 2>/dev/null || true
sleep 2
docker pull $DOCKER_IMAGE_NAME 2>/dev/null || true

# Host RDMA userspace libs must match the host kernel driver; mount them over the
# image copies (same approach as the disagg launcher).
# Use -f (regular file), NOT -e. Docker auto-creates a DIRECTORY at a bind-mount
# source that does not exist, so a node where one of these libs is absent ends up
# with an empty directory at e.g. /usr/lib/x86_64-linux-gnu/libionic.so.1 after the
# first run. -e is true for that directory, so every later run on that node mounts
# a directory over a file inside the image and docker refuses:
#   OCI runtime create failed: ... not a directory: Are you trying to mount a
#   directory onto a file (or vice-versa)?
# The failure is per-node and self-propagating, which makes it look intermittent:
# the container starts on a clean node and dies on a poisoned one, leaving the
# survivors stuck forever at the socket barrier waiting for a peer that never came.
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu
for _vlib in $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.*; do
    [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done
for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done
[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"

docker run --rm \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    --shm-size ${DOCKER_SHM_SIZE:-256G} \
    --ulimit nofile=524288:524288 --ulimit memlock=-1:-1 \
    -v $LOG_PATH:/run_logs \
    -v '"$MAD_SCRIPTS_DIR"':$MAD_SCRIPTS_MOUNT \
    -v /shared_inference:/shared_inference \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e NODE_RANK=$SLURM_PROCID \
    -e NNODES=$NNODES \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e IPADDRS=$IPADDRS \
    -e MODEL_NAME=$MODEL_NAME \
    -e MODEL_PATH=$MODEL_PATH \
    -e SERVE_PORT=$SERVE_PORT \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e TP_SIZE=$TP_SIZE \
    -e PP_SIZE=$PP_SIZE \
    -e ENABLE_EP=$ENABLE_EP \
    ${ALL2ALL_BACKEND:+-e ALL2ALL_BACKEND=$ALL2ALL_BACKEND} \
    ${COLOCATED_EXTRA_ARGS:+-e COLOCATED_EXTRA_ARGS="$COLOCATED_EXTRA_ARGS"} \
    -e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE \
    ${BENCHMARK_ITR:+-e BENCHMARK_ITR=$BENCHMARK_ITR} \
    ${BENCHMARK_CON:+-e BENCHMARK_CON="$BENCHMARK_CON"} \
    ${BENCHMARK_COMBINATIONS:+-e BENCHMARK_COMBINATIONS="$BENCHMARK_COMBINATIONS"} \
    ${NIAH_WORDS:+-e NIAH_WORDS="$NIAH_WORDS"} \
    ${NIAH_MAXTOK:+-e NIAH_MAXTOK=$NIAH_MAXTOK} \
    ${NIAH_TIMEOUT:+-e NIAH_TIMEOUT=$NIAH_TIMEOUT} \
    ${GPU_MEMORY_UTILIZATION:+-e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION} \
    ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN=$MAX_MODEL_LEN} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        SHARED_DIR=$MAD_SCRIPTS_MOUNT/vllm_dissag \
        bash $MAD_SCRIPTS_MOUNT/vllm_multinode/serve_colocated.sh \
          2>&1 | tee /run_logs/${SLURM_JOB_ID}/colocated_bench_NODE${SLURM_PROCID}.log
    "
'

srun --nodelist="$SELECTED_NODELIST" bash -c 'docker stop $DOCKER_CONT_NAME 2>/dev/null || true; docker rm $DOCKER_CONT_NAME 2>/dev/null || true'

# Publish the perf CSV where madengine collects it. The benchmark writes
# /run_logs/$SLURM_JOB_ID/perf.csv inside the container (= $LOG_PATH on the host),
# but madengine resolves a model's `multiple_results` against the JOB DIRECTORY it
# launched us from - so copy it there under the declared name.
_PERF_SRC="${LOG_PATH}/${SLURM_JOB_ID}/perf.csv"
_PERF_DST="$(pwd)/perf_${MODEL_NAME}.csv"
if [ -f "$_PERF_SRC" ]; then
    cp "$_PERF_SRC" "$_PERF_DST"
    echo "perf CSV -> ${_PERF_DST} ($(( $(wc -l < "$_PERF_DST") - 1 )) rows)"
else
    # Exit non-zero. A run that produced no parseable results is a failure, and
    # reporting it as success is the worst outcome for a benchmark: madengine
    # records exit_code=0 in its completion marker and the collector reports
    # "0 successful, 0 failed", so a hard engine crash is indistinguishable from
    # a clean run. Observed with the gfx942 MoE LLVM abort, which died in
    # determine_available_memory and still exited 0.
    echo "ERROR: no perf CSV at ${_PERF_SRC}; the benchmark produced no parseable results." >&2
    echo "Colocated multi-node run FAILED. Logs: ${LOG_PATH}/${SLURM_JOB_ID}" >&2
    exit 1
fi

echo "Colocated multi-node run complete. Logs: ${LOG_PATH}/${SLURM_JOB_ID}"
