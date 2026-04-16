#!/bin/bash
# launch_mori_2p2d.sh — Example launcher for 2-Prefill / 2-Decode MoRI EP
# disaggregated inference over Ionic AINIC RDMA on AMD MI355X clusters.
#
# Prerequisites:
#   1. 4-node SLURM allocation with full CPUs:
#      sbatch --wrap="sleep infinity" -N4 --cpus-per-task=256 \
#        --nodelist=<node0>,<node1>,<node2>,<node3> ...
#   2. Docker image loaded on all nodes:
#      localhost/mad-mori-ep:gfx950-v2
#   3. Model downloaded to shared storage.
#
# Usage: Edit JOBID and NODE0-3 below, then:
#   bash launch_mori_2p2d.sh
set -euo pipefail

# ── Cluster-specific settings (edit these) ───────────────────────────────────
JOBID=${JOBID:-1346}
NODE0=${NODE0:-mi355-gpu-39}   # Prefill master + proxy
NODE1=${NODE1:-mi355-gpu-40}   # Prefill child
NODE2=${NODE2:-mi355-gpu-51}   # Decode master
NODE3=${NODE3:-mi355-gpu-55}   # Decode child

export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-localhost/mad-mori-ep:gfx950-v2}"
export MODEL_NAME="${MODEL_NAME:-DeepSeek-V3-5layer}"
export MODEL_DIR="${MODEL_DIR:-/shared/amdgpu/home/ravgupta_qle/models}"

# ── Disaggregated inference topology ─────────────────────────────────────────
export xP=2
export yD=2
export RUN_MORI=1
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-512/512}"
export BENCHMARK_CON="${BENCHMARK_CON:-8 16}"
export BENCHMARK_ITR="${BENCHMARK_ITR:-1}"
export LOG_PATH="${LOG_PATH:-/shared/amdgpu/home/ravgupta_qle/logs_mori}"

# ── Network interface for TCP control plane ──────────────────────────────────
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp193s0f1np1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp193s0f1np1}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-enp193s0f1np1}

# ── NCCL/RCCL multi-node IB transport over Ionic AINICs ─────────────────────
# Exclude Broadcom frontend NICs whose GID[1] is fe80:: (link-local, not routable)
export NCCL_IB_HCA="${NCCL_IB_HCA:-^rocep193s0f0,rocep193s0f1}"
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── MORI IO RDMA configuration ──────────────────────────────────────────────
export MORI_IB_GID_INDEX=${MORI_IB_GID_INDEX:-1}
export MORI_IO_LOG_LEVEL=${MORI_IO_LOG_LEVEL:-INFO}

# ── SLURM environment (synthesized for the launcher) ─────────────────────────
export SLURM_JOB_ID=$JOBID
export SLURM_JOB_NODELIST="${NODE0},${NODE1},${NODE2},${NODE3}"
export SLURM_NNODES=4
export SLURM_NTASKS=4
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Launching MORI EP 2P/2D ==="
echo "Nodes: ${NODE0} (prefill master+proxy), ${NODE1} (prefill child),"
echo "       ${NODE2} (decode master),        ${NODE3} (decode child)"
echo "Model: $MODEL_NAME"
echo "Image: $DOCKER_IMAGE_NAME"
echo ""

bash "$(dirname "$0")/run_xPyD_models.slurm" 2>&1 | tee log_mori_${MODEL_NAME}_2p2d.log
