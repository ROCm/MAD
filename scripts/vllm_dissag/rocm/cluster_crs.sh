#!/bin/bash
# =============================================================================
# cluster_crs.sh — Crusoe (crsuse2-m2m-v3: MI355X + AMD Pensando AINIC) config
# for the vLLM disaggregated (P/D) launcher.
# -----------------------------------------------------------------------------
# Same layout and knobs as cluster.sh (the original CI cluster); only the site
# values differ. Select it at submit time with
#       CLUSTER_ENV_IN=/vllm-workspace/.buildkite/amd-disagg/cluster_crs.sh
# and vllm_disagg.sh sources it in place of cluster.sh.
#
# Every value is `${VAR:-default}`, so the environment always wins:
#       environment variable  >  built-in default below
#
# So you can override anything inline:
#   PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 ./vllm_disagg.sh prefill
#
# Model-SPECIFIC perf flags live in models.yaml, NOT here.
# =============================================================================

_CLUSTER_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------- model / mode
# MODEL_NAME indexes into models.yaml; MODEL_DIR is the parent dir holding it.
# MODEL_PATH (resolved by the launcher) = ${MODEL_DIR}/${MODEL_NAME}.   [site]
export MODEL_NAME="${MODEL_NAME:-DeepSeek-V3}"
export MODEL_DIR="${MODEL_DIR:-/shared_nfs/models/deepseek-ai}"

# Shared NFS root (visible on every node and the login node): model weights +
# per-run logs. [site]
export SHARED_MOUNT="${SHARED_MOUNT:-/shared_nfs}"

# Parallelism mode (launcher derives PARALLEL_MODE tp|ep from this):
#   WIDE_EP_MODE=0  tp : each node is an independent TP server (TP8, 1P1D)
#   WIDE_EP_MODE=1  ep : data-parallel + expert-parallel across xP/yD nodes (wideep)
export WIDE_EP_MODE="${WIDE_EP_MODE:-0}"

# ----------------------------------------------------------------- topology
# xP prefill nodes + yD decode nodes. IPADDRS is the ordered, comma-separated
# node list (prefill IPs first, then decode IPs). NODE_RANK is this node's global
# 0-based rank; under SLURM it defaults to $SLURM_PROCID. Leave IPADDRS empty to
# use the PREFILL_IP/DECODE_IP fallback defaults below (1P1D only).
export xP="${xP:-1}"
export yD="${yD:-1}"
export IPADDRS="${IPADDRS:-}"
export NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-}}"
export PREFILL_IP="${PREFILL_IP:-10.0.0.1}"
export DECODE_IP="${DECODE_IP:-10.0.0.2}"

# Per-node GPU count and TP degree
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"

# ----------------------------------------------------------------- ports
# TP-mode (WIDE_EP_MODE=0) server ports.
export PREFILL_PORT="${PREFILL_PORT:-8100}"
export DECODE_PORT="${DECODE_PORT:-8200}"

# EP-mode (WIDE_EP_MODE=1) ports: API serve port, DP RPC port, KV transfer port, and
# per-node MoRIIO local ping port (must differ from PROXY_PING_PORT).
export SERVE_PORT="${SERVE_PORT:-20005}"
export RPC_PORT="${RPC_PORT:-13345}"
export KV_PORT="${KV_PORT:-9711}"
export LOCAL_PING_PORT="${LOCAL_PING_PORT:-61555}"

# MoRIIO proxy: HTTP port clients/benchmark hit, plus the connector control ports.
# PROXY_PING_PORT MUST be 36367 — the proxy hardcodes its zmq service-discovery
# socket on that port; prefill/decode register to PROXY_IP:PROXY_PING_PORT.
export PROXY_IP="${PROXY_IP:-${PREFILL_IP}}"
export PROXY_PORT="${PROXY_PORT:-10001}"
export PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"
export HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"
export NOTIFY_PORT="${NOTIFY_PORT:-61005}"

export PROXY_SCRIPT="${PROXY_SCRIPT:-/app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py}"

# MoRIIO KV transfer direction (injected into --kv-transfer-config by the launcher):
#   0 -> omit read_mode     (default; MoRIIO write mode: prefill pushes to decode)
#   1 -> "read_mode": true  (decode pulls KV from prefill; matches upstream disagg)
export MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"

# ----------------------------------------------------------------- router / gateway
# Selection for client (bench/accuracy) traffic:
#   proxy       -> the in-container MoRIIO proxy started by the launcher
#   vllm-router -> an external `vllm/vllm-router` container started by the SLURM job
#                  on the rank-0 node (default)
# Both use the SAME MoRIIO discovery mechanism (prefill/decode register to
# PROXY_IP:PROXY_PING_PORT=36367); only the client HTTP front door differs.
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
export ROUTER_PORT="${ROUTER_PORT:-30000}"
export ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
# Single client-facing port bench/accuracy target: the router port when routing,
# else the proxy port. Env override always wins.
if [[ "${ROUTER_TYPE}" == "vllm-router" ]]; then
    export GATEWAY_PORT="${GATEWAY_PORT:-${ROUTER_PORT}}"
else
    export GATEWAY_PORT="${GATEWAY_PORT:-${PROXY_PORT}}"
fi

# Where per-run logs / benchmark results are written. A $SLURM_JOB_ID subdir is
# appended so each CI run is self-scoped (falls back to 'local' off-SLURM).  [site]
_LOG_BASE="${LOG_BASE:-${SHARED_MOUNT}/${USER:-$(id -un)}/disagg_logs}"
export LOG_PATH="${LOG_PATH:-${_LOG_BASE}/${SLURM_JOB_ID:-local}}"

# ----------------------------------------------------------------- vLLM runtime
# Engine/platform/transport-level env (NOT model-specific). Model-architecture
# AITER kernel toggles live in models.yaml under each model's `env:` block.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

#export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
#export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
#
#export HOME=/tmp
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
# Remote-code modules are rewritten by every rank on every node at import; keep
# those writes off the shared NFS cache.
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
# Engine startup gets the same budget as the P/D health gate.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-${HEALTH_TIMEOUT_S:-5400}}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

# Extra `vllm serve` flags for every role (appended after the models.yaml flags).
# amdgpu 6.16 on these nodes counts VRAM pinned for an RDMA NIC a second time
# against the KFD limit, and MoRI-IO pins the whole KV cache when it opens its
# first session. Once weights + activations + CUDA graphs + 2x KV exceed VRAM,
# HIP reports 0 bytes free and the next allocation fails
# (HSA_STATUS_ERROR_OUT_OF_RESOURCES loading a kernel, or a torch OOM) although
# VRAM is physically free. Sizing the KV cache from --gpu-memory-utilization 0.85
# hits that for R1 and V4-Pro; 64 GiB of KV leaves them ~40 GiB.
# EP mode also pins MoRI's 16 GiB symmetric heap, and each EP rank holds a full copy of
# the non-expert weights, so EP8 DeepSeek-V4-Pro / GLM-5.2 need a smaller KV cache.
if [[ "${WIDE_EP_MODE}" == "1" ]]; then _KV_GIB=32; else _KV_GIB=64; fi
export CLUSTER_SERVE_FLAGS="${CLUSTER_SERVE_FLAGS:---kv-cache-memory-bytes $((_KV_GIB << 30))}"
# EP ranks run TP1, so each one holds a whole sequence's KV: MiniMax-M3's default
# 1M-token context needs 74 GiB of it. gsm8k needs a few thousand tokens.
if [[ "${WIDE_EP_MODE}" == "1" && "${CLUSTER_SERVE_FLAGS}" != *--max-model-len* ]]; then
    CLUSTER_SERVE_FLAGS+=" --max-model-len 131072"
fi
# MoRI hands the experts its whole receive buffer (max_num_batched_tokens x EP ranks
# rows), and the AITER MXFP8/MXFP4 experts compute every row: they ignore the
# received-token count. At the 16384 default MiniMax-M3's EP8 decode step takes ~1.4 s.
# vLLM requires max_num_seqs <= max_num_batched_tokens.
if [[ "${WIDE_EP_MODE}" == "1" && "${CLUSTER_SERVE_FLAGS}" != *--max-num-batched-tokens* ]]; then
    CLUSTER_SERVE_FLAGS+=" --max-num-batched-tokens 1024 --max-num-seqs 256"
fi

# ----------------------------------------------------------------- scale
# DP/EP group formation timeout across nodes (seconds).
export DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"

# ----------------------------------------------------------------- RDMA / NCCL
# AMD Pensando AINIC (400G) RoCE fabric: 8 NICs exposed as ionic_0..7 (netdevs
# enP2p0s9-12 on NUMA 0, enP3p0s9-12 on NUMA 1). The rails carry no IPv4; GID
# index 1 is their RoCE v2 (IPv6) GID. The VPC NIC ens3 is an RDMA device too
# (mlx5_0, 200G): keep it out of every device list below.
_IB_DEVICES="${IB_DEVICES:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export IB_DEVICES="${IB_DEVICES:-${_IB_DEVICES}}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-${_IB_DEVICES}}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${_IB_GID_INDEX}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# In-box RCCL net transport (no external plugin); pin bootstrap to the VPC iface.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
# RCCL 2.32 otherwise selects its IB-CAST transport on AINIC, which fails
# cross-node (DP16+EP16 2P2D). IB needs an RDMA device in the container.
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_NET_DISABLE_INTRA="${NCCL_NET_DISABLE_INTRA:-1}"
export NCCL_IB_TC="${NCCL_IB_TC:-104}"
export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-192}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-12}"

# MoRI uses the same NIC set as NCCL.
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_IB_DEVICES}}"
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-${_IB_GID_INDEX}}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"

# Rail MTU is not uniform: crsuse2-m2m-v3-081/082 run netdev MTU 9000 (RoCE 4096),
# the other nodes 1500 (RoCE 1024). MoRI-IO takes each QP's path MTU from the local
# port, so a mixed pair drops every packet. 1024 works for any pair; raise it to
# 4096 only when every node in the job is at 9000.
export MORI_IB_PATH_MTU="${MORI_IB_PATH_MTU:-1024}"

# Pin to gfx950 to avoid jit compilation failures with other archs on this cluster.
export MORI_GPU_ARCHS="gfx950"

# Required with the AINIC (ionic) NICs; without the host proxy the EP dispatch
# path fails to come up.
export MORI_ENABLE_HOST_PROXY="${MORI_ENABLE_HOST_PROXY:-1}"

# NIXL UCX defaults: RC on ionic_0, GPU-direct RDMA.
if [[ "${KV_CONNECTOR:-}" == Nixl* ]]; then
    export UCX_TLS="${UCX_TLS:-rc_verbs,ud_verbs,tcp,self,rocm}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-ionic_0:1,lo}"
    export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-${_IB_GID_INDEX}}"
    export UCX_UD_VERBS_ETH_DGID_CHECK="${UCX_UD_VERBS_ETH_DGID_CHECK:-n}"
    export UCX_IB_GPU_DIRECT_RDMA="${UCX_IB_GPU_DIRECT_RDMA:-yes}"
    export UCX_RMA_PPLN_ENABLE="${UCX_RMA_PPLN_ENABLE:-y}"
    export UCX_PROTO_EMULATION_ENABLE="${UCX_PROTO_EMULATION_ENABLE:-n}"
    export UCX_KEEPALIVE_INTERVAL="${UCX_KEEPALIVE_INTERVAL:-inf}"
    export UCX_UD_VERBS_RX_QUEUE_LEN="${UCX_UD_VERBS_RX_QUEUE_LEN:-1024}"
    export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"
    export ACCURACY_NUM_CONCURRENT="${ACCURACY_NUM_CONCURRENT:-1}"
fi

# ----------------------------------------------------------------- benchmark
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/128 2048/128}"
export BENCHMARK_CON="${BENCHMARK_CON:-32 64}"
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-2}"
export BENCHMARK_MIN_PROMPTS="${BENCHMARK_MIN_PROMPTS:-32}"

# ----------------------------------------------------------------- accuracy
# `accuracy` role runs lm_eval (local-completions backend, local-chat-completions
# with ACCURACY_CHAT=1) against the proxy.
export ACCURACY_TASKS="${ACCURACY_TASKS:-gsm8k}"
export ACCURACY_NUM_CONCURRENT="${ACCURACY_NUM_CONCURRENT:-64}"
export ACCURACY_MAX_RETRIES="${ACCURACY_MAX_RETRIES:-3}"
export ACCURACY_METRIC="${ACCURACY_METRIC:-exact_match}"
export ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD:-0.90}"

# ----------------------------------------------------------------- SLURM (submit)
# Used by run-slurm-disagg-test.sh on the login node (harmless to export here). [site]
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aig-sw-shared}"
