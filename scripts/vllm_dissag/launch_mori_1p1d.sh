#!/bin/bash
set -euo pipefail

JOBID=1300
NODE0=mi355-gpu-39
NODE1=mi355-gpu-40

export DOCKER_IMAGE_NAME="localhost/mad-mori-ep:gfx950-v2"
export MODEL_NAME="DeepSeek-V3"
export MODEL_DIR="/shared/amdgpu/home/ravgupta_qle/models"
export xP=1
export yD=1
export RUN_MORI=1
export BENCHMARK_COMBINATIONS="1024/1024 8192/1024 1024/8192"
export BENCHMARK_CON="8 16 32 64 128 256 512"
export BENCHMARK_ITR=1
export LOG_PATH="/shared/amdgpu/home/ravgupta_qle/logs_mori"
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1

# MORI IO RDMA: use global IPv6 GID for routed RoCE over Ionic AINICs
export MORI_IB_GID_INDEX=1
export MORI_IO_LOG_LEVEL=INFO

export SLURM_JOB_ID=$JOBID
export SLURM_JOB_NODELIST="${NODE0},${NODE1}"
export SLURM_NNODES=2
export SLURM_NTASKS=2
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Launching MORI EP 1P/1D ==="
echo "Nodes: $NODE0, $NODE1"
echo "Model: $MODEL_NAME"
echo "Image: $DOCKER_IMAGE_NAME"
echo ""

bash "$(dirname "$0")/run_xPyD_models.slurm" 2>&1 | tee log_mori_${MODEL_NAME}_1p1d.log
