#!/usr/bin/bash

set -ex
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_70b.toml"}

GPUS_PER_NODE=8
echo "getting MASTER_ADDR:" ${MASTER_ADDR}
echo "getting MASTER_PORT:" ${MASTER_PORT}
echo "PROCID:$PROCID"
echo "NODEID:$NODEID"
echo "NNODES: $NNODES"
NNODES=$NNODES
NODE_RANK=${NODEID}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node=${NGPU} \
         --rdzv_backend c10d \
         --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
         --local-ranks-filter ${LOG_RANK} \
         --role rank --tee 3 \
         torchtitan/train.py --job.config_file ${CONFIG_FILE}
