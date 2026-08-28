#!/bin/bash
# Generic interactive-allocation driver for one test cell.
# Drives an existing Slurm allocation via `srun --overlap` (no sbatch), one
# container per node, NODE_RANK assigned per node.
#
# Required env:
#   JOBID           slurm job to --overlap onto
#   NODES           comma list of node names (or NODE_PREFIX-relative suffixes) — len = xP+yD
#   IPS             comma list of node IPs, same order as NODES
#   MODEL_NAME, MODEL_PATH
#   CONNECTOR (rixl|moriio), WIDE_EP (0|1); EP_BACKEND for wideEP
#   xP, yD
#   TAG             short run tag (log dir suffix)
#   DOCKER_IMAGE_NAME  the image to run (build from docker/*_mori_ep_fullsource*.Dockerfile)
# Optional: NODE_PREFIX (prepended to each NODES entry; default empty = full names),
#           BENCHMARK_CON, BENCHMARK_COMBINATIONS, PROXY_TYPE, ROUTER_BINARY,
#           ROUTER_PORT, RUN_MORI, RUN_DEEPEP
set -u
: "${JOBID:?} ${NODES:?} ${IPS:?} ${MODEL_NAME:?} ${MODEL_PATH:?} ${CONNECTOR:?} ${WIDE_EP:?} ${xP:?} ${yD:?} ${TAG:?} ${DOCKER_IMAGE_NAME:?}"

NODE_PREFIX="${NODE_PREFIX:-}"          # e.g. "mycluster-node-"; empty => NODES are full names
BENCHMARK_CON="${BENCHMARK_CON:-8 16}"
BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-512/512}"
BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
# Path to a built vllm-router binary (from vllm-project/router PR#181). Required only if
# the image doesn't already ship vllm-router on PATH.
ROUTER_BINARY="${ROUTER_BINARY:-}"
EP_BACKEND="${EP_BACKEND:-}"

IFS=',' read -ra NODE_ARR <<< "$NODES"
IFS=',' read -ra IP_ARR   <<< "$IPS"
NNODES=$(( xP + yD ))
IPADDRS="$IPS"
MASTER_ADDR="${IP_ARR[0]}"
MASTER_PORT=39566
RUN_TAG="int${JOBID}_${TAG}"
HOSTRUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_interactive.sh"

# env forwarded into each srun step (SLURM_JOB_ID set to RUN_TAG only INSIDE bash -c).
FWD="DOCKER_IMAGE_NAME MODEL_NAME MODEL_PATH CONNECTOR WIDE_EP EP_BACKEND xP yD NNODES IPADDRS MASTER_ADDR MASTER_PORT BENCHMARK_CON BENCHMARK_COMBINATIONS BENCHMARK_ITR PROXY_TYPE ROUTER_PORT ROUTER_BINARY RUN_MORI RUN_DEEPEP DECODE_CUDAGRAPH_MODE LOG_PATH"
EXPORTS=""
for v in $FWD; do EXPORTS="$EXPORTS $v=\"${!v:-}\""; done
EXPORTS="$EXPORTS SLURM_JOB_ID=\"$RUN_TAG\""

echo "=== cell $TAG: $CONNECTOR WIDE_EP=$WIDE_EP ${EP_BACKEND:+EP_BACKEND=$EP_BACKEND} model=$MODEL_NAME xP=$xP yD=$yD ==="
echo "    nodes=$NODES  ips=$IPS  con='$BENCHMARK_CON'  combos='$BENCHMARK_COMBINATIONS'"

pids=()
# launch decode + child nodes first (ranks 1..N-1), then prefill master (rank 0) last
for (( r=NNODES-1; r>=1; r-- )); do
  node="${NODE_PREFIX}${NODE_ARR[$r]}"
  srun --jobid="$JOBID" --overlap -w "$node" bash -c "export $EXPORTS; $HOSTRUN $r" \
      > "/tmp/${RUN_TAG}_rank${r}.log" 2>&1 &
  pids+=($!)
done
sleep 5
srun --jobid="$JOBID" --overlap -w "${NODE_PREFIX}${NODE_ARR[0]}" bash -c "export $EXPORTS; $HOSTRUN 0" \
    > "/tmp/${RUN_TAG}_rank0.log" 2>&1 &
master_pid=$!

echo "logs: /tmp/${RUN_TAG}_rank*.log   container logs: /shared_inference/${USER}/model_blog_logs/${RUN_TAG}/"
wait "$master_pid"
echo "=== $TAG: rank0 exited; stopping others ==="
for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
echo "DONE $TAG"
