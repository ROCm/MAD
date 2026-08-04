#!/bin/bash
export PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
export MORI_RDMA_DEVICES=rdma3 MORI_GPU_ARCHS=gfx942 GPU_PER_NODE=1
export MORI_RDMA_SL=3 MORI_RDMA_TC=104 MORI_IB_GID_INDEX=3
export HSA_NO_SCRATCH_RECLAIM=1 MORI_SHMEM_HEAP_SIZE=16G
export PYTORCH_ALLOC_CONF=expandable_segments:False PYTORCH_HIP_ALLOC_CONF=expandable_segments:False HSA_ENABLE_IPC_MODE_LEGACY=0
export GLOO_SOCKET_IFNAME=eno8303 MORI_SOCKET_IFNAME=eno8303 NCCL_SOCKET_IFNAME=eno8303
export PYTHONPATH=/tmp/mori-src:/tmp/mori-src/python:$PYTHONPATH
cd /tmp/mori-src
exec torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=192.0.2.10 --master_port=29000   examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd test --dtype bf16 --max-tokens 128 --num-qp 2 --kernel-type async_ll
