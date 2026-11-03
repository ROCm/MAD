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

# The constant defaults (NCCL tuning, MoRI QPs, timeouts, runtime switches) are data:
# connectors/mori.env, applied as `${KEY:-VALUE}` like the lines around it.
_MORI_ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/connectors/mori.env"
if [[ -f "$_MORI_ENV_FILE" ]]; then
    while IFS= read -r _line; do
        [[ "$_line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        _k="${_line%%=*}"
        [[ -n "${!_k:-}" ]] || printf -v "$_k" '%s' "${_line#*=}"
        export "${_k?}"
    done < "$_MORI_ENV_FILE"
    unset _line _k
else
    echo "[mori_ep_env] WARN: ${_MORI_ENV_FILE} not found; NCCL/MoRI tuning defaults not applied" >&2
fi

# =============================================================
# MORI CONFIGURATION - Use all CX7 NICs for expert parallelism
# =============================================================
# MoRI uses the same NIC set as NCCL
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_DEFAULT_IB}}"

# Match NCCL GID index for consistency
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-${NCCL_IB_GID_INDEX}}"

# =============================================================
# SOCKET INTERFACE - Keep mlx5_1 IP-side for control plane
# =============================================================
# Default route NIC is typically the management interface
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}

# Timeouts and runtime switches: connectors/mori.env (above).

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
