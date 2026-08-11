#!/bin/bash
# NCCL/MORI RDMA configuration for multi-NIC nodes.
# Toggle USE_CX7_NICS to switch between CX7 rail NICs and mlx5_1 management NIC.

# =============================================================
# NIC MODE SELECTION
# =============================================================
# USE_CX7_NICS=1 — use all 8 CX7 400G rail NICs (default)
# USE_CX7_NICS=0 — use mlx5_1 100G management NIC
USE_CX7_NICS="${USE_CX7_NICS:-1}"

if [[ "$USE_CX7_NICS" == "1" ]]; then
    _DEFAULT_IB="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
    echo "[mori_ep_env] NIC mode: CX7 multi-rail (${_DEFAULT_IB}) — same-rail nodes required"
else
    _DEFAULT_IB="mlx5_1"
    echo "[mori_ep_env] NIC mode: mlx5_1 management NIC — cross-rail safe"
fi

# =============================================================
# CORE DEVICE LIST
# =============================================================
export IB_DEVICES="${IB_DEVICES:-${_DEFAULT_IB}}"

# =============================================================
# NCCL CONFIGURATION
# =============================================================
export NCCL_IB_HCA="${NCCL_IB_HCA:-${_DEFAULT_IB}}"

# Critical: RoCE v2 with IPv4 routing (GID index 3 is typical)
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

# Allow NCCL to use multiple NICs per GPU when beneficial
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

# GPUDirect RDMA - level 3 enables PXB (PCI Bridge) crossing
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"

# Use IB transport (force IB over Socket)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# Multiple QPs per connection for CX7 - improves parallelism
# CX7 NDR can saturate with more QPs (default is 1, recommend 4-8)
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"

# Split data across QPs for better load balancing
export NCCL_IB_SPLIT_DATA_ON_QPS="${NCCL_IB_SPLIT_DATA_ON_QPS:-1}"

# Increase chunk size to better utilize CX7 bandwidth (default 128KB)
# For NDR 400G, larger chunks reduce overhead
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"   # 8MB buffer

# Timeout for IB operations (increase for large clusters)
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"

# Service Level for QoS (if your fabric uses SL-based QoS)
export NCCL_IB_SL="${NCCL_IB_SL:-0}"

# Traffic Class for RoCE QoS (DSCP 26 = AF31, common for RDMA)
export NCCL_IB_TC="${NCCL_IB_TC:-106}"  # = DSCP 26 << 2

# PCIe relaxed ordering for better performance
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"

# Adaptive routing (if fabric supports it)
export NCCL_IB_ADAPTIVE_ROUTING="${NCCL_IB_ADAPTIVE_ROUTING:-1}"

# Enable NCCL topology auto-detection
export NCCL_TOPO_DUMP_FILE="${NCCL_TOPO_DUMP_FILE:-/tmp/nccl_topo.xml}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"

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
# Default route NIC is typically the management interface
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}

# =============================================================
# TIMEOUTS - Adjusted for larger fabric
# =============================================================
export GLOO_TIMEOUT_MS="${GLOO_TIMEOUT_MS:-300000}"
export TORCH_DIST_INIT_BARRIER_TIMEOUT="${TORCH_DIST_INIT_BARRIER_TIMEOUT:-300}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1200}"

# =============================================================
# RUNTIME OPTIMIZATIONS
# =============================================================
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

if [[ -d /sgl-workspace/aiter ]]; then
    export PYTHONPATH="/sgl-workspace/aiter:${PYTHONPATH:-}"
fi

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
