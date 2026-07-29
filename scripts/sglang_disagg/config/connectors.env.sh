#!/bin/bash
# Connectors bucket: MoRI RDMA config + DP_MODE=1 MoRI-EP tuning.
# Depends on _DEFAULT_IB from nic-selection.env.sh (source that first).
# Requires DP_MODE to be set before sourcing (entrypoint sets it at line 44).
# See ../CONFIG.md section 3 (Connectors).

# =============================================================
# MORI CONFIGURATION - Use all CX7 NICs for expert parallelism
# =============================================================
# MoRI uses the same NIC set as NCCL
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_DEFAULT_IB}}"

# Match NCCL GID index for consistency
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-3}"

# QPs per connection for MoRI (similar reasoning as NCCL)
export MORI_QPS_PER_CONNECTION="${MORI_QPS_PER_CONNECTION:-4}"

# =============================================================
# SOCKET INTERFACE - Keep mlx5_1 IP-side for control plane
# =============================================================
# MoRI control-plane socket; defaults to the GLOO socket iface (set in framework.env.sh)
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}

# MoRI EP tuning (DP_MODE=1 only)
if [[ "${DP_MODE:-}" == "1" ]]; then
    export SGLANG_MORI_FP8_DISP="${SGLANG_MORI_FP8_DISP:-True}"
    [[ "${MODEL_NAME:-}" == *mxfp4* ]] && export SGLANG_MORI_FP8_DISP=False
    export SGLANG_MORI_FP4_DISP="${SGLANG_MORI_FP4_DISP:-False}"
    export SGLANG_MORI_FP8_COMB="${SGLANG_MORI_FP8_COMB:-False}"
    export MORI_MAX_DISPATCH_TOKENS_DECODE="${MORI_MAX_DISPATCH_TOKENS_DECODE:-160}"
    export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"
    # MoRI shmem heap: 4 GiB default is too small for EP>=32; use 16 GiB.
    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"
fi
