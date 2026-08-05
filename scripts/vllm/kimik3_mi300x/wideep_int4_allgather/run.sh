#!/bin/bash
# Kimi-K3 (MXFP4) on MI300X / gfx942 -- Wide expert-parallel, generic all2all.
# PP2xTP8 across 2 nodes for weight fit (~102 GB/GPU) PLUS --enable-expert-parallel
# so the 896 experts split 8-way across each node's 8 GPUs (112/GPU), replicated per
# --all2all-backend allgather_reducescatter (see ../wideep_int4_moriep for the true
# MoRI-EP kernels). AITER_SITUV2_A8W4=1 selects the a8w4 (fp8-act x int4-wt) SiTU
# MoE path. Colocated (single instance; no prefill/decode disaggregation).
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
CONTAINER="k3_wideepint4_${ROLE}"
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
  -e AITER_SITUV2_A8W4=1 \
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
      --enable-expert-parallel --all2all-backend allgather_reducescatter \
      --nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER} --master-port 29500 ${HEADLESS} \
      --trust-remote-code --reasoning-parser kimi_k3 --mm-encoder-tp-mode data \
      --safetensors-load-strategy prefetch \
      --max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS} \
      --gpu-memory-utilization ${GPU_UTIL} ${SERVE_EXTRA} 2>&1 | tee /logs/vllm_wideepint4_${ROLE}.log
  "
echo "[pp2tp8] $ROLE started. log: $LOGHOST/vllm_wideepint4_${ROLE}.log"
[ "$ROLE" = head ] && echo "[pp2tp8] health: curl http://${MASTER}:${PORT}/v1/models"
