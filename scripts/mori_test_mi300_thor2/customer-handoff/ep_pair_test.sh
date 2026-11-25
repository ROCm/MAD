#!/bin/bash
# MoRI-EP internode async_ll test for one node pair. Run INSIDE mori_host.
# Usage: ep_pair_test.sh <node_rank 0|1> <master_ip> <rdma_dev> <port>
set -u
RANK="$1"; MASTER="$2"; DEV="${3:-rdma3}"; PORT="${4:-29100}"
export PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
export MORI_RDMA_DEVICES=$DEV MORI_GPU_ARCHS=gfx942 GPU_PER_NODE=1
# RoCE fabric values — override for your fabric (from the ClusterSphere recommender).
export MORI_RDMA_SL=${MORI_RDMA_SL:-3} MORI_RDMA_TC=${MORI_RDMA_TC:-104} MORI_IB_GID_INDEX=${MORI_IB_GID_INDEX:-3}
export HSA_NO_SCRATCH_RECLAIM=1 MORI_SHMEM_HEAP_SIZE=16G
export PYTORCH_ALLOC_CONF=expandable_segments:False PYTORCH_HIP_ALLOC_CONF=expandable_segments:False HSA_ENABLE_IPC_MODE_LEGACY=0
# Mgmt/OOB interface for torchrun rendezvous. Override for your host: SOCKET_IFNAME=<your-mgmt-nic>
IFACE="${SOCKET_IFNAME:-eno8303}"
export GLOO_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE
export PYTHONPATH=/tmp/mori-src:${PYTHONPATH:-}
cd /tmp/mori-src
exec torchrun --nnodes=2 --node_rank=$RANK --nproc_per_node=1 --master_addr=$MASTER --master_port=$PORT \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --cmd test --dtype bf16 --max-tokens 128 --num-qp 2 --kernel-type async_ll
