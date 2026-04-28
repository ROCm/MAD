#!/bin/bash
# MoRI / SGLang multi-node defaults aligned with SemiAnalysis InferenceX amd_utils.
# Upstream reference: https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/multi_node/amd_utils/env.sh
#
# Expects MODEL_NAME to be set when sourced (for mxfp4-related toggles).
# IBDEVICES: prefer pre-set value; else reuse IB_DEVICES from the caller; else a safe default.

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

export IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"

# NCCL / RCCL IB HCAs (same list as disaggregation IB device list in typical setups)
export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICES}"

# Socket NIC: honor IFNAME / existing exports; else default route (InferenceX-style)
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}

export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1200}"

# MoRI / MoE EP tuning: only apply when DP_MODE=1 (data-parallel MoRI path).
if [[ "${DP_MODE:-}" == "1" ]]; then
    #export MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"

    # Symmetric heap for mori.shmem (MoE EP scales with world size / tokens). Mori default ~4G is often too small
    # for multi-GPU or multi-node; errors look like "Out of static heap memory" / HIP invalid argument after OOM.
    #export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"
    export SGLANG_MORI_FP8_DISP="${SGLANG_MORI_FP8_DISP:-True}"
    if [[ "${MODEL_NAME:-}" == *mxfp4* ]]; then
        export SGLANG_MORI_FP8_DISP=False
    fi

    export SGLANG_MORI_FP4_DISP="${SGLANG_MORI_FP4_DISP:-False}"
    export SGLANG_MORI_FP8_COMB="${SGLANG_MORI_FP8_COMB:-False}"

    #if [[ "${MODEL_NAME:-}" == *mxfp4* ]]; then
    #    export MORI_MAX_DISPATCH_TOKENS_PREFILL="${MORI_MAX_DISPATCH_TOKENS_PREFILL:-12288}"
    #else
    #    export MORI_MAX_DISPATCH_TOKENS_PREFILL="${MORI_MAX_DISPATCH_TOKENS_PREFILL:-16384}"
    #fi
    #export MORI_MAX_DISPATCH_TOKENS_DECODE="${MORI_MAX_DISPATCH_TOKENS_DECODE:-160}"

    ##export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"

    #export MORI_EP_LAUNCH_CONFIG_MODE="${MORI_EP_LAUNCH_CONFIG_MODE:-AUTO}"
    #export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-16384}"
    #export MORI_IO_QP_MAX_CQE="${MORI_IO_QP_MAX_CQE:-32768}"
    #export MORI_IO_QP_MAX_SGE="${MORI_IO_QP_MAX_SGE:-4}"

    #export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-INFO}"
fi

export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

if [[ -d /sgl-workspace/aiter ]]; then
    export PYTHONPATH="/sgl-workspace/aiter:${PYTHONPATH:-}"
fi
