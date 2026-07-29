#!/bin/bash
# Framework bucket: NCCL / GLOO / PyTorch / aiter runtime tuning.
# Depends on _DEFAULT_IB from nic-selection.env.sh (source that first).
# See ../CONFIG.md section 2 (Framework).

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
# SOCKET INTERFACE - Keep mlx5_1 IP-side for control plane
# =============================================================
# Default route NIC is typically the management interface
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}

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
