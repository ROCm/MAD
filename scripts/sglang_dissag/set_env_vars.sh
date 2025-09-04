#!/bin/bash

# export TORCH_NCCL_HIGH_PRIORITY=1
# export NCCL_CHECKS_DISABLE=1

# use ibv_devinfo
# export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

# export NCCL_CROSS_NIC=0

export NCCL_IGNORE_CPU_AFFINITY=1

# use <ip addr> command to get the thernetname
# use <ls /sys/class/net> to see all NICs
export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1


# Automatically Fetch the default interface instead of Hard coding.
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME},mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export IBDEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9


# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export RCCL_MSCCL_ENABLE=0
# export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
# export RCCL_MSCCLPP_ENABLE=0
# export HSA_ENABLE_IPC_MODE_LEGACY=1
