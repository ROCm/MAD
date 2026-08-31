#!/bin/bash
# =============================================================================
# madengine-native entrypoint for SGLang Disaggregated P/D inference.
#
# This is the `scripts` target referenced by models.json (sglang_disagg_*).
# madengine's built-in `sglang-disagg` SLURM/K8s launcher
# (src/madengine/deployment/slurm.py::_generate_sglang_disagg_command) runs ONE
# container per node and exports the cluster topology as env vars. This script
# bridges those into the env contract of the proven MAD-private disagg
# launchers and execs the right one (MoRI EP, or Mooncake/NIXL).
#
# madengine -> proven env mapping:
#   SGLANG_NODE_RANK            -> NODE_RANK   (0=proxy, 1..xP=prefill, rest=decode)
#   SGLANG_DISAGG_PREFILL_NODES -> xP
#   SGLANG_DISAGG_DECODE_NODES  -> yD
#   SGLANG_NODE_IPS             -> IPADDRS     (comma-separated, rank order)
#   SGLANG_TP_SIZE              -> IO_EP_TP_SIZE / per-server --tp-size
#   MASTER_PORT                 -> MASTER_PORT
#
# Model identity + transport come from deployment env_vars / models.json args:
#   MODEL_NAME   short catalog key in models.yaml (e.g. DeepSeek-R1)   [required]
#   MODEL_PATH   weights dir mounted into the container                [required]
#   RUN_MORI     1 = MoRI EP all-to-all (default), 0 = Mooncake/NIXL
#   KV_TRANSFER_BACKEND  prefill->decode KV transfer backend (mori/mooncake/nixl)
#   DP_MODE      1 = DP-attention EP (DeepSeek-V3/R1 inter-node), 0 = TP
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- parse optional CLI args (models.json "args"), env takes precedence ------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name|--model_name|--model_repo) MODEL_NAME="${MODEL_NAME:-$2}"; shift 2;;
        --model-path|--model_path)              MODEL_PATH="${MODEL_PATH:-$2}"; shift 2;;
        --run-mori)                             RUN_MORI="${RUN_MORI:-$2}";     shift 2;;
        --dp-mode)                              DP_MODE="${DP_MODE:-$2}";       shift 2;;
        --xp|--xP|--prefill-nodes)              xP="${xP:-$2}";                 shift 2;;
        --yd|--yD|--decode-nodes)               yD="${yD:-$2}";                 shift 2;;
        --ipaddrs|--node-ips)                   IPADDRS="${IPADDRS:-$2}";       shift 2;;
        --master-port)                          MASTER_PORT="${MASTER_PORT:-$2}"; shift 2;;
        --gpus-per-node)                        GPUS_PER_NODE="${GPUS_PER_NODE:-$2}"; shift 2;;
        --kv-transfer-backend|--transfer-backend) KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-$2}"; shift 2;;
        *) shift;;
    esac
done

# --- bridge madengine sglang-disagg topology -> proven launcher env ---------
export NODE_RANK="${NODE_RANK:-${SGLANG_NODE_RANK:-${SLURM_PROCID:-0}}}"
# Persist run.sh output to the shared /run_logs mount so failures are
# diagnosable from the submission node (madengine discards model stdout).
# The dir is created from inside the container (root); make it world-writable
# so an ordinary user outside the container can clean it up afterwards.
#
# /run_logs MUST be a shared (NFS-backed) mount passed through the docker env:
# the disaggregated launchers write per-node readiness logs
# (prefill_NODE*/decode_NODE*/proxy_NODE*) to /run_logs/${SLURM_JOB_ID} and the
# rank-0 proxy greps them across nodes. Disaggregated P/D always spans >=2 nodes
# (rank-0 prefill/proxy + at least one decode node), so a node-local fallback
# would silently break that cross-node rendezvous. Require /run_logs and fail
# fast if it is missing or not writable.
RUN_LOG_DIR="/run_logs/${SLURM_JOB_ID:-local}"
if ! mkdir -p "${RUN_LOG_DIR}" 2>/dev/null || [[ ! -w "${RUN_LOG_DIR}" ]]; then
    echo "FATAL: /run_logs is not writable. It must be a shared (NFS-backed)" >&2
    echo "       mount passed through the docker env: the disaggregated launchers" >&2
    echo "       write per-node readiness logs to /run_logs/\${SLURM_JOB_ID} and the" >&2
    echo "       rank-0 proxy greps them across nodes. Mount it and re-run." >&2
    exit 1
fi
chmod 1777 /run_logs 2>/dev/null || true
chmod -R 0777 "${RUN_LOG_DIR}" 2>/dev/null || true
export RUN_LOG_DIR
exec > >(tee -a "${RUN_LOG_DIR}/run_sh_rank${NODE_RANK}.log") 2>&1
export xP="${xP:-${SGLANG_DISAGG_PREFILL_NODES:-1}}"
export yD="${yD:-${SGLANG_DISAGG_DECODE_NODES:-1}}"
export NNODES="${NNODES:-${SGLANG_DISAGG_TOTAL_NODES:-${SLURM_JOB_NUM_NODES:-$((xP + yD))}}}"
export MASTER_PORT="${MASTER_PORT:-23731}"
export IO_EP_TP_SIZE="${IO_EP_TP_SIZE:-${SGLANG_TP_SIZE:-8}}"
# madengine forwards MASTER_ADDR (rank-0 host) into the container; use it as the
# rendezvous address for in-container IP discovery below.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# IPADDRS = comma-separated node IPs in rank order. It may be supplied
# explicitly via env/arg or SGLANG_NODE_IPS. madengine's container runner does
# NOT forward SGLANG_NODE_IPS into the container, so when it is absent we
# self-discover the node IPs in-container using only the forwarded
# MASTER_ADDR/NODE_RANK/NNODES, mirroring scripts/vllm_dissag/run.sh.
export IPADDRS="${IPADDRS:-${SGLANG_NODE_IPS:-}}"

# Prefer the routable transport interface (NCCL_SOCKET_IFNAME, eth0 here);
# hostname -I order is not deterministic and may surface link-local addrs.
# NCCL_SOCKET_IFNAME may be a comma-separated list (see set_env_vars.sh), so
# take only the first interface before handing it to `ip addr show`.
_HOST_IFACE="${NCCL_SOCKET_IFNAME:-eth0}"; _HOST_IFACE="${_HOST_IFACE%%,*}"
HOST_IP="$(ip -4 -o addr show "${_HOST_IFACE}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1)"
[[ -z "${HOST_IP}" ]] && HOST_IP="$(hostname -I | awk '{print $1}')"
# Derive the rendezvous port from the job id. Use a wide modulo so two
# concurrent jobs whose ids happen to share a rank-0 host are very unlikely to
# collide (a narrow 10000-wide range collides whenever ids differ by a multiple
# of 10000). The value is deterministic across nodes (depends only on the job
# id, which is identical on every node).
IP_SYNC_PORT="${IP_SYNC_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"

# rank-0 acts as a tiny TCP rendezvous server: peers connect to MASTER_ADDR and
# report their host IP; rank-0 aggregates the rank-ordered list and broadcasts.
# The implementation lives in the sibling ip_rendezvous.py so it can be unit tested.
_tcp_discover_ipaddrs() {
  python3 "$SCRIPT_DIR/ip_rendezvous.py" \
    "${NODE_RANK}" "${NNODES}" "${HOST_IP}" "${MASTER_ADDR}" "${IP_SYNC_PORT}" "${SLURM_JOB_ID:-0}"
}

# Method 1: SLURM nodelist (only if scontrol/getent exist inside the container).
if [[ -z "${IPADDRS}" && -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1 && command -v getent >/dev/null 2>&1; then
    _ips=""; _cnt=0
    while IFS= read -r _n; do
        [[ -z "${_n}" ]] && continue
        _ip="$(getent ahostsv4 "${_n}" | awk 'NR==1{print $1}')"
        [[ -n "${_ip}" ]] && { _ips="${_ips:+${_ips},}${_ip}"; _cnt=$((_cnt + 1)); }
    done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    [[ "${_cnt}" == "${NNODES}" && -n "${_ips}" ]] && IPADDRS="${_ips}"
fi

# Method 2: TCP rendezvous via forwarded MASTER_ADDR (primary path).
if [[ -z "${IPADDRS}" ]]; then
    _tcp="$(_tcp_discover_ipaddrs || true)"
    [[ -n "${_tcp}" ]] && IPADDRS="${_tcp}"
fi

if [[ -z "${IPADDRS}" ]]; then
    echo "ERROR: unable to determine IPADDRS (node IP list);" \
         "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} NODE_RANK=${NODE_RANK}." >&2
    exit 1
fi
export IPADDRS
# rank-0 IP is the master/load-balancer host.
export MASTER_ADDR="$(echo "$IPADDRS" | cut -d',' -f1)"

# --- model + transport defaults --------------------------------------------
export MODEL_NAME="${MODEL_NAME:-}"
export MODEL_PATH="${MODEL_PATH:-}"
export RUN_MORI="${RUN_MORI:-1}"
export DP_MODE="${DP_MODE:-0}"
# Default the KV transfer backend from RUN_MORI so the non-MoRI path does not
# silently inherit an invalid `mori` backend (mooncake/NIXL use mooncake).
if [[ "$RUN_MORI" == "1" ]]; then
    export KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-mori}"
else
    export KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-mooncake}"
fi

# GPT-OSS's mxfp4 MoE must be driven by one stack end to end. models.yaml pins
# --moe-runner-backend triton because aiter's CK MXFP4 GEMM is not in this build, but with
# aiter still enabled sglang's mxfp4 layer hands that triton runner an AiterMoeQuantInfo and
# the first forward dies with "'AiterMoeQuantInfo' object has no attribute 'use_mxfp8'"
# (srt/layers/moe/moe_runner/triton.py:183, job 25796). Pairing it here rather than in the
# manifest means a run cannot be launched having forgotten one half; still overridable.
if [[ "$MODEL_NAME" == GPT-OSS-* ]]; then
    export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
fi

# Helper scripts (socket_barrier.py, set_env_vars.sh, benchmark_xPyD.sh, ...)
# live alongside this script; point the Mooncake-path lookups at them.
export MOONCAKE_COOKBOOK_PATH="${MOONCAKE_COOKBOOK_PATH:-$SCRIPT_DIR}"
mkdir -p "${RUN_LOG_DIR}" 2>/dev/null || true

if [[ -z "$MODEL_NAME" || -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_NAME and MODEL_PATH must be set (via deployment env_vars or args)." >&2
    echo "  MODEL_NAME='$MODEL_NAME'  MODEL_PATH='$MODEL_PATH'" >&2
    exit 1
fi

# --- stage model weights if missing (rank-0 gated, NFS-shared MODEL_PATH) -----
# run.sh historically assumed pre-staged weights. Stage from HF when MODEL_PATH
# is incomplete. Only NODE_RANK 0 downloads; peers wait on a sentinel.
# Treat weights as complete only if our own download sentinel is present, OR a
# config.json plus at least one weight shard exists (covers manually pre-staged
# dirs). A partial earlier download (config.json present, shards missing, no
# sentinel) must NOT be mistaken for a complete stage.
_weights_complete() {
    [[ -f "${MODEL_PATH}/.stage_done" ]] && return 0
    [[ -f "${MODEL_PATH}/config.json" ]] || return 1
    compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null 2>&1 && return 0
    compgen -G "${MODEL_PATH}/*.bin" >/dev/null 2>&1 && return 0
    return 1
}
if ! _weights_complete; then
    _MODEL_REPO="${MODEL_REPO:-}"
    if [[ -z "${_MODEL_REPO}" ]]; then
        case "${MODEL_NAME}" in
            DeepSeek-R1)    _MODEL_REPO="deepseek-ai/DeepSeek-R1-0528" ;;
            Llama-3.1-70B-Instruct) _MODEL_REPO="meta-llama/Llama-3.1-70B-Instruct" ;;
            Qwen3-Next-80B) _MODEL_REPO="Qwen/Qwen3-Next-80B-A3B-Instruct" ;;
            GPT-OSS-120B)   _MODEL_REPO="openai/gpt-oss-120b" ;;
            Kimi-K2-Instruct) _MODEL_REPO="moonshotai/Kimi-K2-Instruct-0905" ;;
            # Non-gated mirror: the official meta-llama Llama-4 repos need a
            # per-account license grant, which an unattended run cannot obtain.
            Llama-4-Scout-17B-16E-Instruct) _MODEL_REPO="unsloth/Llama-4-Scout-17B-16E-Instruct" ;;
            *) echo "ERROR: weights missing at ${MODEL_PATH}; set MODEL_REPO for ${MODEL_NAME}" >&2; exit 1 ;;
        esac
    fi
    _SENTINEL="${MODEL_PATH}/.stage_done"
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        echo "[stage_weights] NODE_RANK=0 downloading ${_MODEL_REPO} -> ${MODEL_PATH}"
        mkdir -p "${MODEL_PATH}"
        # huggingface-cli is deprecated and refuses to run on newer huggingface_hub
        # (it prints "use `hf` instead" and exits non-zero); prefer the `hf` CLI and
        # fall back to huggingface-cli only on older images that lack `hf`.
        if command -v hf >/dev/null 2>&1; then
            _HF_DL=(hf download)
        else
            _HF_DL=(huggingface-cli download)
        fi
        HF_TOKEN="${MAD_SECRETS_HFTOKEN:-${HF_TOKEN:-}}" "${_HF_DL[@]}" "${_MODEL_REPO}" --local-dir "${MODEL_PATH}" || { echo "[stage_weights] download failed" >&2; exit 1; }
        touch "${_SENTINEL}"
    else
        echo "[stage_weights] NODE_RANK=${NODE_RANK} waiting for weights at ${MODEL_PATH}"
        for _i in $(seq 1 720); do
            _weights_complete && break
            sleep 10
        done
        _weights_complete || { echo "[stage_weights] timed out waiting for weights" >&2; exit 1; }
    fi
fi


echo "=============================================================="
echo " SGLang Disaggregated (madengine bridge)"
echo "   MODEL_NAME=$MODEL_NAME  MODEL_PATH=$MODEL_PATH"
echo "   NODE_RANK=$NODE_RANK  xP=$xP  yD=$yD  TP=$IO_EP_TP_SIZE"
echo "   RUN_MORI=$RUN_MORI  DP_MODE=$DP_MODE  KV_TRANSFER_BACKEND=$KV_TRANSFER_BACKEND"
echo "   SGLANG_USE_AITER=${SGLANG_USE_AITER:-<unset, defaults to 1 downstream>}"
echo "   MASTER_ADDR=$MASTER_ADDR  IPADDRS=$IPADDRS"
echo "=============================================================="

# sglang_disagg_mori_io_ep.sh is a functional superset of the retired
# sglang_disagg_server.sh (per-model config from models.yaml, mori/mooncake/nixl
# via KV_TRANSFER_BACKEND, DP_MODE support) — route both RUN_MORI values there.
exec bash "$SCRIPT_DIR/sglang_disagg_mori_io_ep.sh"
