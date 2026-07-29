#!/bin/bash
# NIC selection prelude (NOT a CONFIG.md bucket).
# Sourced first to set _DEFAULT_IB for Framework (NCCL_IB_HCA) and Connectors (IB_DEVICES).
# Boundary: USE_CX7_NICS (Cluster) -> IB_DEVICES (Connectors) + NCCL_IB_HCA (Framework).

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
