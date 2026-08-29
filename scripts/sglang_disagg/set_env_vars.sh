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


# Auto-detect the default interface (the `dev` field, NOT $NF which is the route
# metric on some hosts). Respect any pre-set NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME.
_DEFAULT_IFACE=$(ip route | awk '/^default/{for(i=1;i<=NF;i++)if($i=="dev")print $(i+1)}' | head -n 1)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$_DEFAULT_IFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$_DEFAULT_IFACE}
# NOTE: do NOT append mlx5_* here — that is Mellanox-only and breaks bnxt/other
# fabrics. RDMA NIC selection is handled by mori_ep_env.sh (IB_DEVICES / MORI_RDMA_DEVICES).


# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export RCCL_MSCCL_ENABLE=0
# export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
# export RCCL_MSCCLPP_ENABLE=0
# export HSA_ENABLE_IPC_MODE_LEGACY=1

# -----------------------------------------------------------------------------
# DeepSeek-V4-Flash-FP8 load-bearing env (applied only for this model).
# DSV4-Flash needs the aiter indexer/compress path + the dsv4 fp8 MoE kernels;
# without these the CK MoE dispatch fails on gfx942. FP4 experts OFF (our weights
# are block-FP8 E4M3, not FP4). ROCM700A=0 per validated DSV4-Flash runs.
# -----------------------------------------------------------------------------
if [[ "${MODEL_NAME:-}" == "DeepSeek-V4-Flash-FP8" ]]; then
  export SGLANG_USE_AITER=1
  export SGLANG_USE_ROCM700A=0
  export SGLANG_DSV4_FP4_EXPERTS=false
  export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  export AITER_BF16_FP8_MOE_BOUND=0
  export SGLANG_OPT_USE_AITER_INDEXER=true
  export SGLANG_OPT_USE_TILELANG_INDEXER=false
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  export SGLANG_OPT_FP8_WO_A_GEMM=false
  export SGLANG_OPT_USE_TOPK_V2=false
  export SGLANG_OPT_USE_FUSED_COMPRESS=true
  export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
  # Cap host thread pools: at high DP the per-rank OpenBLAS/OMP pools otherwise
  # exhaust RLIMIT_NPROC ("can't start new thread") during weight load.
  export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 GOTO_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "[set_env_vars] applied DeepSeek-V4-Flash-FP8 env block"
fi
