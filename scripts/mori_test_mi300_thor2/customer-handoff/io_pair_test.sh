#!/bin/bash
# MoRI-IO internode CPU-mem write sweep for one node pair. Run INSIDE mori_host.
# Usage: io_pair_test.sh <node_rank 0|1> <master_ip> <own_ip> <rdma_dev> <port>
set -u
RANK="$1"; MASTER="$2"; OWNIP="$3"; DEV="${4:-rdma3}"; PORT="${5:-29500}"
export PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
export MORI_RDMA_DEVICES=$DEV MORI_GPU_ARCHS=gfx942
# RoCE fabric values — override for your fabric (from the ClusterSphere recommender).
export MORI_RDMA_SL=${MORI_RDMA_SL:-3} MORI_RDMA_TC=${MORI_RDMA_TC:-104} MORI_IB_GID_INDEX=${MORI_IB_GID_INDEX:-3}
export HSA_NO_SCRATCH_RECLAIM=1
# Mgmt/OOB interface for torchrun rendezvous. Override for your host: SOCKET_IFNAME=<your-mgmt-nic>
IFACE="${SOCKET_IFNAME:-eno8303}"
export GLOO_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE
export PYTHONPATH=/tmp/mori-src:${PYTHONPATH:-}
cd /tmp/mori-src
exec torchrun --nnodes=2 --node_rank=$RANK --nproc_per_node=1 --master_addr=$MASTER --master_port=$PORT \
  tests/python/io/benchmark.py --host=$OWNIP --backend rdma --mem-type cpu --op-type write \
  --all --sweep-start-size 8 --sweep-max-size 67108864 --enable-sess --enable-batch-transfer \
  --num-qp-per-transfer 2 --num-initiator-dev 1 --num-target-dev 1 --transfer-batch-size 1
