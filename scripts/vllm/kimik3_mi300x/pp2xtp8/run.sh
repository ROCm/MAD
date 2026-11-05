#!/bin/bash
# Kimi-K3 (MXFP4) on MI300X / gfx942 -- PP2xTP8 baseline (no expert parallelism).
# TP8 within each node, PP2 across 2 nodes -> each node holds half the layers
# (~102 GB/GPU); a single 8-GPU node cannot fit the model + KV. Simplest, lowest-
# latency K3 serve on MI300X. Colocated (single instance; no P/D disaggregation).
#
# Usage (worker FIRST, then head):
#   ROLE=worker MASTER=<head eth0 IP> bash run.sh   # on node1 (rank1)
#   ROLE=head   MASTER=<head eth0 IP> bash run.sh   # on node0 (rank0, serves API)
set -euo pipefail

IMAGE="${IMAGE:-amdsiloai/vllm:kimi-k3-mi325x-release-v2}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=<path to Kimi-K3-MXFP4 weights>}"
ROLE="${ROLE:?set ROLE=head|worker}"
MASTER="${MASTER:?set MASTER=<head eth0 IP>}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
CONTAINER="k3_pp2tp8_${ROLE}"
LOGHOST="${LOGHOST:-$HOME/k3run/logs}"; mkdir -p "$LOGHOST"

# Proven fabric env (from cluster_rdma_env_recommender.py)
BOOT_NIC="eth0"
IB_HCA="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
GID_INDEX=3

if [ "$ROLE" = "head" ]; then
  NODE_RANK=0; SERVE_EXTRA="--port ${PORT}"; HEADLESS=""
else
  NODE_RANK=1; SERVE_EXTRA=""; HEADLESS="--headless"
fi

echo "[pp2tp8] node=$(hostname -s) role=$ROLE rank=$NODE_RANK master=$MASTER"
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || { echo "ERROR: model missing at $MODEL_DIR"; exit 1; }
docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d --name "$CONTAINER" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --shm-size 128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_ROCM_USE_AITER_MLA=0 \
  -e NCCL_SOCKET_IFNAME=$BOOT_NIC -e GLOO_SOCKET_IFNAME=$BOOT_NIC \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$GID_INDEX \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  -v "$MODEL_DIR":/model:ro -v "$LOGHOST":/logs \
  --entrypoint bash \
  "$IMAGE" -c "
    vllm serve /model --served-model-name kimi-k3 \
      --tensor-parallel-size 8 --pipeline-parallel-size 2 \
      --distributed-executor-backend mp \
      --nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER} --master-port 29500 ${HEADLESS} \
      --trust-remote-code --reasoning-parser kimi_k3 --mm-encoder-tp-mode data \
      --safetensors-load-strategy prefetch \
      --max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS} \
      --gpu-memory-utilization ${GPU_UTIL} ${SERVE_EXTRA} 2>&1 | tee /logs/vllm_pp2tp8_${ROLE}.log
  "
echo "[pp2tp8] $ROLE started. log: $LOGHOST/vllm_pp2tp8_${ROLE}.log"
[ "$ROLE" = head ] && echo "[pp2tp8] health: curl http://${MASTER}:${PORT}/v1/models"
