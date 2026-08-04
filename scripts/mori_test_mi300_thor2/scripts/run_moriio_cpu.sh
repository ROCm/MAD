#!/bin/bash
export PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
export MORI_RDMA_DEVICES=rdma3 MORI_GPU_ARCHS=gfx942
export MORI_RDMA_SL=3 MORI_RDMA_TC=104 MORI_IB_GID_INDEX=3
export HSA_NO_SCRATCH_RECLAIM=1
export GLOO_SOCKET_IFNAME=eno8303 MORI_SOCKET_IFNAME=eno8303
export PYTHONPATH=/tmp/mori-src:$PYTHONPATH
cd /tmp/mori-src
exec torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=192.0.2.10 --master_port=29500   tests/python/io/benchmark.py --host=192.0.2.10 --backend rdma --mem-type cpu --op-type write   --all --sweep-start-size 8 --sweep-max-size 67108864 --enable-sess --enable-batch-transfer   --num-qp-per-transfer 2 --num-initiator-dev 1 --num-target-dev 1 --transfer-batch-size 1
