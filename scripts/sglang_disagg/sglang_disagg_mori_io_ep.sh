#!/bin/bash
# MoRI EP PD entrypoint (used when RUN_MORI=1 in run_xPyD_models.slurm).
# Customize for MoRI expert-parallel + disaggregated launch; until then this
# delegates to the standard Mooncake PD launcher.

_MORI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${_MORI_SCRIPT_DIR}"

# -----------------------------------------------------------------------------
# DP_MODE=1 allowlist (MoRI IO EP). Must stay in sync with run_xPyD_models.slurm.
# -----------------------------------------------------------------------------
MORI_DP_MODE1_ALLOWED_MODELS=(
    "DeepSeek-V3"
    "DeepSeek-R1"
    "Kimi-K2-Instruct"
    "Kimi-K2-Instruct-MoRI-AB"
    "Kimi-K2-Instruct-DeepEP-AB"
)

mori_model_allows_dp_mode_one() {
    local name="$1"
    local m
    for m in "${MORI_DP_MODE1_ALLOWED_MODELS[@]}"; do
        [[ "$name" == "$m" ]] && return 0
    done
    return 1
}

mori_dp_mode1_allowed_models_lines() {
    local m
    for m in "${MORI_DP_MODE1_ALLOWED_MODELS[@]}"; do
        printf '  - %s\n' "$m"
    done
}

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
DP_MODE="${DP_MODE:-0}"

# PARALLEL_MODE is derived strictly from DP_MODE for models.yaml (tp vs dp flags).
# NOTE: DP_MODE=1 with yD>1 (e.g. 1P2D) was previously limited due to a sglang
# detokenizer deadlock (multi-node disaggregated decode, nnodes=2, dp=16).
# That guard is now lifted to re-validate — DeepSeek-V3/R1 only.
if [[ "$DP_MODE" == "1" ]]; then
    PARALLEL_MODE=dp
else
    PARALLEL_MODE=tp
fi
echo "PARALLEL_MODE=${PARALLEL_MODE} (DP_MODE=${DP_MODE})"

if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "ERROR: MODEL_NAME not set, exiting"
    exit 1
fi

if [[ "$DP_MODE" == "1" ]] && ! mori_model_allows_dp_mode_one "$MODEL_NAME"; then
    echo "ERROR: DP_MODE=1 is not supported for model '${MODEL_NAME}'. Allowed models:" >&2
    mori_dp_mode1_allowed_models_lines >&2
    echo "Use DP_MODE=0 for other models." >&2
    exit 1
fi

IPADDRS="${IPADDRS:-localhost}"
BARRIER_PORT="${BARRIER_PORT:-4342}"
# IB_DEVICES controls --disaggregation-ib-device (RDMA NICs for KV-cache transfer).
# Default is set in mori_ep_env.sh (sourced below). Override: set IB_DEVICES env var.
# Note: CX7 rail NICs require same-rail nodes; mlx5_1 (mgmt NIC) is cross-rail safe.

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

# === Model-Specific Configuration from YAML ===
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

if ! python3 -c "import yaml" >/dev/null 2>&1; then
    echo "[stage_deps] PyYAML not found; installing at runtime (expected on the" >&2
    echo "             base, non-overlay image) ..." >&2
    python3 -m pip install --no-cache-dir --quiet pyyaml || true
fi
if ! python3 -c "import yaml" >/dev/null 2>&1; then
    echo "ERROR: PyYAML is not installed and could not be installed at runtime," >&2
    echo "       but is required to parse ${MODELS_YAML}. Use an image built" >&2
    echo "       from docker/sglang_disagg_inference_full_overlay*.Dockerfile" >&2
    echo "       (which installs pyyaml at build time), or ensure network" >&2
    echo "       access for 'python3 -m pip install pyyaml'." >&2
    exit 1
fi

export MODELS_YAML MODEL_NAME PARALLEL_MODE xP GPUS_PER_NODE
eval "$(python3 - <<'PY'
import os
import re
import shlex
import sys
import yaml

config_path = os.environ["MODELS_YAML"]
model_name = os.environ["MODEL_NAME"]
mode = os.environ["PARALLEL_MODE"]
xP = int(os.environ.get("xP", "1"))
gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))


def q(v):
    return shlex.quote(str(v if v is not None else ""))


with open(config_path, "r", encoding="utf-8") as f:
    models = yaml.safe_load(f) or {}

if model_name not in models:
    # Quote the interpolated values: this string is eval'd by the shell, so an
    # unescaped model_name / config_path could break the launcher or inject
    # shell tokens.
    msg = f"ERROR: Model {model_name} not found in {config_path}"
    print(f"echo {q(msg)}; exit 1")
    sys.exit(0)

cfg = models[model_name] or {}
prefill = cfg.get("prefill", {}) or {}
decode = cfg.get("decode", {}) or {}


prefill_flags = prefill.get(mode, "") or ""

exports = {
    "MODEL_BASE_FLAGS": cfg.get("base_flags", ""),
    "MODEL_MODE_FLAGS": cfg.get(f"{mode}_flags", ""),
    "MODEL_PREFILL_FLAGS": prefill_flags,
    "MODEL_DECODE_FLAGS": decode.get(mode, ""),
    "MODEL_EXPERIMENTAL_FLAGS": cfg.get("experimental_flags", ""),
}

# Optional `arch_flags:` map, keyed by GPU-arch prefix, applied by the launcher's
# arch gate below (longest matching prefix wins). Keys must be valid shell
# identifiers because they become variable-name suffixes.
arch_flags = cfg.get("arch_flags", {}) or {}
if not isinstance(arch_flags, dict):
    print(f"echo {q(f'ERROR: arch_flags for {model_name} must be a mapping of arch-prefix to flags')}; exit 1")
    sys.exit(0)
for key in arch_flags:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
        print(f"echo {q(f'ERROR: arch_flags key {key!r} is not a valid identifier')}; exit 1")
        sys.exit(0)
exports["MODEL_ARCH_FLAG_KEYS"] = " ".join(str(k) for k in arch_flags)
for key, value in arch_flags.items():
    exports[f"MODEL_ARCH_FLAGS__{key}"] = value or ""

for key, value in exports.items():
    print(f"{key}={q(value)}")
PY
)"

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

if [[ "$PARALLEL_MODE" != "dp" && "$PARALLEL_MODE" != "tp" ]]; then
    echo "ERROR: PARALLEL_MODE must be 'dp' or 'tp' (got: ${PARALLEL_MODE})"
    exit 1
fi

# =============================================================================
# Parallelism Settings
# =============================================================================

# DP_MODE=0: CLI --tp-size is IO_EP_TP_SIZE (default 8) on every worker; PREFILL_EP_SIZE/DECODE_EP_SIZE
# still scale with xP/yD×GPUS_PER_NODE for MoRI env (not passed as CLI unless DP_MODE=1).
# DP_MODE=1: --tp-size scales with cluster; --dp-size/--ep-size on CLI (same total degree as Nnodes×GPUS_PER_NODE).
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GENERIC_TP_SIZE="${GENERIC_TP_SIZE:-8}"

if [[ "$DP_MODE" == "1" ]]; then
    PREFILL_TP_SIZE=$((xP * GPUS_PER_NODE))
    DECODE_TP_SIZE=$((yD * GPUS_PER_NODE))
else
    PREFILL_TP_SIZE="${GENERIC_TP_SIZE}"
    DECODE_TP_SIZE="${GENERIC_TP_SIZE}"
fi


if [[ "$DP_MODE" == "1" ]]; then
    PREFILL_EP_SIZE=$((xP * GPUS_PER_NODE))
    DECODE_EP_SIZE=$((yD * GPUS_PER_NODE))
    PREFILL_DP_SIZE=$((xP * GPUS_PER_NODE))
    DECODE_DP_SIZE=$((yD * GPUS_PER_NODE))
    export PREFILL_DP_SIZE DECODE_DP_SIZE PREFILL_EP_SIZE DECODE_EP_SIZE
else
    unset PREFILL_DP_SIZE DECODE_DP_SIZE PREFILL_EP_SIZE DECODE_EP_SIZE 2>/dev/null || true
fi
export PREFILL_TP_SIZE DECODE_TP_SIZE

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================


# --- arch-specific flags ----------------------------------------------------
# Some settings are correct on one GPU generation and actively harmful on another, so
# models.yaml can carry an optional `arch_flags:` map keyed by GPU-arch prefix. The
# launcher detects the arch and appends the entry whose key is the LONGEST matching
# prefix, so `gfx95:` covers the whole gfx95x family while `gfx942:` can still say
# something different. Appended after base_flags, so an arch entry can also override a
# base value (sglang parses with argparse: the last occurrence of a flag wins).
#
# Motivating case, Llama-4-Scout, where the two arches need opposite things:
#   gfx95  needs --moe-runner-backend triton (AITER MoE segfaults during warmup),
#   gfx942 must NOT have it -- it corrupts generation while every harness signal stays
#          green (a 4-node sweep reported 8/8 points at 2501 tok/s emitting only token
#          id 0), and then needs a higher --mem-fraction-static than gfx95, because the
#          AITER MoE runner it falls back to needs more device memory on a 192 GB card.
#
# Set SGLANG_GFX_ARCH to override detection (e.g. to exercise another arch's path).
# When the arch cannot be detected, NO arch flags are applied -- deliberately, because
# the failure modes are not symmetric: missing a kernel workaround fails loudly at
# warmup, while applying one on the wrong arch fails silently with corrupted output
# that benchmarks still score as a pass. Prefer the loud failure.
_detect_gfx_arch() {
    local a="${SGLANG_GFX_ARCH:-${MAD_SYSTEM_GPU_ARCHITECTURE:-}}"
    [[ -n "$a" ]] && { echo "$a"; return; }
    a="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [[ -z "$a" ]] && a="$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName)' 2>/dev/null | cut -d: -f1 || true)"
    echo "$a"
}
GFX_ARCH="$(_detect_gfx_arch)"
MODEL_ARCH_FLAGS=""
if [[ -n "${MODEL_ARCH_FLAG_KEYS:-}" ]]; then
    if [[ -z "$GFX_ARCH" ]]; then
        echo "WARNING: GPU arch not detected; NOT applying any arch_flags (keys: ${MODEL_ARCH_FLAG_KEYS}). Set SGLANG_GFX_ARCH to force." >&2
    else
        _arch_key=""
        for _k in ${MODEL_ARCH_FLAG_KEYS}; do
            if [[ "$GFX_ARCH" == "${_k}"* ]] && (( ${#_k} > ${#_arch_key} )); then _arch_key="$_k"; fi
        done
        if [[ -n "$_arch_key" ]]; then
            _v="MODEL_ARCH_FLAGS__${_arch_key}"; MODEL_ARCH_FLAGS="${!_v}"
            echo "Arch gate: ${GFX_ARCH} matches arch_flags[${_arch_key}] -> applying: ${MODEL_ARCH_FLAGS}"
        else
            echo "Arch gate: ${GFX_ARCH} matches no arch_flags key (${MODEL_ARCH_FLAG_KEYS}) -> applying none"
        fi
    fi
fi

PREFILL_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_ARCH_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_PREFILL_FLAGS} ${MODEL_EXPERIMENTAL_FLAGS}"
DECODE_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_ARCH_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_DECODE_FLAGS} ${MODEL_EXPERIMENTAL_FLAGS}"
echo "Using model-specific configuration for: $MODEL_NAME (mode=${PARALLEL_MODE})"

# Agentic gating: the default concurrency sweep keeps the perf-tuned config
# (models.yaml base_flags, radix cache off). Only the agentic trace-replay path
# (BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh) — or an explicit override — turns
# on the radix prefix cache and server-side Prometheus metrics. This keeps the
# default path byte-for-byte unchanged.
AGENTIC_METRICS_ENABLED=0
if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_SERVER_METRICS:-0}" == "1" ]]; then
    AGENTIC_METRICS_ENABLED=1
fi
SERVER_METRICS_FLAGS=""
if [[ "${AGENTIC_METRICS_ENABLED}" == "1" ]]; then
    SERVER_METRICS_FLAGS="--enable-metrics --enable-metrics-for-all-schedulers"
fi
if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_RADIX_CACHE:-0}" == "1" ]]; then
    PREFILL_MODEL_CONFIG="${PREFILL_MODEL_CONFIG//--disable-radix-cache/}"
    DECODE_MODEL_CONFIG="${DECODE_MODEL_CONFIG//--disable-radix-cache/}"
    echo "[radix] radix prefix cache ENABLED (stripped --disable-radix-cache) for agentic/ENABLE_RADIX_CACHE"
fi

export PREFILL_MODEL_CONFIG DECODE_MODEL_CONFIG MODEL_EXPERIMENTAL_FLAGS SERVER_METRICS_FLAGS

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mori_ep_env.sh"

# KV transfer backend: default mori, switchable to mooncake or nixl.
# Kept out of models.yaml so model config is backend-agnostic.
_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-mori}"

# _TRANSFER_BACKEND is interpolated into an eval'd launch command below; restrict
# it to known-good values to avoid invalid backends and shell-token injection.
case "$_TRANSFER_BACKEND" in
    mori|mooncake|nixl) ;;
    *)
        echo "ERROR: unsupported KV_TRANSFER_BACKEND='$_TRANSFER_BACKEND' (expected: mori|mooncake|nixl)" >&2
        exit 1
        ;;
esac

PREFILL_MODEL_CONFIG+=" --disaggregation-transfer-backend ${_TRANSFER_BACKEND}"
DECODE_MODEL_CONFIG+=" --disaggregation-transfer-backend ${_TRANSFER_BACKEND}"

# A profiled run needs a measurable configuration, which differs from the tuned one on two counts,
# and both belong here: node 0 assembles its prefill command separately from the other prefill
# nodes, so anything added at a launch site lands on some nodes only -- job 25824 profiled three
# nodes with RCCL and one without.
#
# --disable-custom-all-reduce: sglang does the intra-node TP exchange with its own all-reduce
# kernel, which never enters torch.distributed, so it emits neither RCCL debug lines nor
# record_param_comms events. Job 25815 profiled 1 call and 4 B per rank -- the startup barrier and
# nothing else. Routing TP through RCCL makes that traffic visible and comparable with Primus.
#
# --disable-cuda-graph on both roles: a replayed HIP graph dispatches one packet, so the
# collectives captured inside it reach no profiler, and rocprofv3 aborts the capture itself with
# HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (job 25806 died at bs=1). Decode alone was not enough:
# a models.yaml entry can capture graphs in its DP prefill block too (`Kimi-K2-Instruct` does,
# with `--cuda-graph-bs $(seq 1 3)`), silently hiding the collectives being profiled. Set here
# rather than in models.yaml so a tuned configuration is not permanently degraded.
#
# Both cost performance, so throughput has to be read from a run without PROFILE_ENABLE.
if [[ "${PROFILE_ENABLE:-0}" == "1" ]]; then
    PREFILL_MODEL_CONFIG+=" --disable-custom-all-reduce --disable-cuda-graph"
    DECODE_MODEL_CONFIG+=" --disable-custom-all-reduce --disable-cuda-graph"
    echo "[profile] measurement flags added: TP through RCCL, no HIP graphs in either role"
fi

# Give each server process its own RCCL log instead of eight of them interleaving in the node's
# stdout. RCCL_LOG_DIR is set by the deployment (a shared mount, since the analysis runs outside the
# container); without it nothing changes and RCCL keeps writing to stdout.
#
# At NCCL_DEBUG=INFO the ranks of a node overwrite each other mid-record: about 0.5% of collective
# records and a good part of the topology lines arrive spliced, and every one of them has to be
# detected and discarded downstream. NCCL_DEBUG_FILE takes the interleaving away at the source --
# %h is the host and %p the pid, and each file is line buffered.
#
# Set here rather than at a launch site: node 0 assembles its prefill command separately from the
# other prefill nodes and decode has a third site, which is how job 25824 ended up with three nodes
# profiled and one not.
#
# RCCL_LOG_DIR=auto puts them beside this job's server logs, which is where the analysis looks by
# default; a manifest cannot spell that path itself, since it does not know the job id.
#
# A directory RCCL cannot write to is worse than none: it drops the output instead of falling back
# to stdout, and the run finishes with no RCCL data at all. Check before trusting it.
if [[ -n "${RCCL_LOG_DIR:-}" ]]; then
    [[ "${RCCL_LOG_DIR}" == "auto" ]] && RCCL_LOG_DIR="/run_logs/${SLURM_JOB_ID:-0}/rccl"
    if (( NODE_RANK < xP )); then _rccl_role="prefill"; else _rccl_role="decode"; fi
    mkdir -p "${RCCL_LOG_DIR}" 2>/dev/null && chmod 0777 "${RCCL_LOG_DIR}" 2>/dev/null
    if [[ -w "${RCCL_LOG_DIR}" ]]; then
        export NCCL_DEBUG_FILE="${RCCL_LOG_DIR}/${_rccl_role}_NODE${NODE_RANK}.%h.%p.log"
        echo "[profile] RCCL logs go to ${NCCL_DEBUG_FILE} (one file per process)"
    else
        unset NCCL_DEBUG_FILE
        echo "[profile] WARNING: RCCL_LOG_DIR=${RCCL_LOG_DIR} is not writable from the container;" >&2
        echo "[profile]          RCCL keeps logging to stdout" >&2
    fi
fi

# Adapter counters, sampled for the whole life of this node's server. The all-to-all reaches no
# RCCL log and a trace names its kernels without saying what they put on the wire, so this is the
# only channel that says how much crossed the fabric. It counts verbs, not causality: reads and
# atomics in quantity show a protocol that waits, but their absence shows nothing, since a reply
# can itself be a write. Reading sysfs perturbs nothing, so RDMA_COUNTERS=1 without
# PROFILE_ENABLE measures a configuration anyone would actually deploy.
#
# Defaults on while profiling, since that is where the question is being asked anyway.
RDMA_COUNTERS="${RDMA_COUNTERS:-${PROFILE_ENABLE:-0}}"
_rdma_sampler_pid=""
_start_rdma_counters() {
    [[ "${RDMA_COUNTERS}" == "1" ]] || return 0
    # Same fallback as the server logs. `local` here put the samples beside no run:
    # off SLURM the logs go to /run_logs/0 and the channel could not be found from
    # the directory holding them.
    local _dir="${RDMA_COUNTERS_DIR:-/run_logs/${SLURM_JOB_ID:-0}/rdma}"
    local _role
    if (( NODE_RANK < xP )); then _role="prefill"; else _role="decode"; fi
    mkdir -p "${_dir}" 2>/dev/null && chmod 0777 "${_dir}" 2>/dev/null
    if [[ ! -w "${_dir}" ]]; then
        echo "[profile] WARNING: RDMA counter directory ${_dir} is not writable; channel skipped" >&2
        return 0
    fi
    # Only the adapters this run was given: a node here has ten and the deployment names eight,
    # and summing in another job's traffic would add noise to a channel that measures a difference.
    "${MOONCAKE_COOKBOOK_PATH}/rdma_counters.sh" \
        --out "${_dir}/${_role}_NODE${NODE_RANK}.csv" \
        --devices "${IB_DEVICES:-}" \
        --interval "${RDMA_COUNTERS_INTERVAL:-30}" &
    _rdma_sampler_pid=$!
    echo "[profile] RDMA counters -> ${_dir}/${_role}_NODE${NODE_RANK}.csv (pid ${_rdma_sampler_pid})"
}

# SIGTERM rather than SIGKILL: the sampler takes one last sample on it, so the window closes where
# the server stopped.
_stop_rdma_counters() {
    [[ -n "${_rdma_sampler_pid}" ]] || return 0
    kill -TERM "${_rdma_sampler_pid}" 2>/dev/null || true
    wait "${_rdma_sampler_pid}" 2>/dev/null || true
    _rdma_sampler_pid=""
}

# Everything this script started, on every exit and not only the orderly one: the readiness
# timeouts and worker failure paths leave before their `_shutdown_server` call, so the container
# teardown used to hard-kill the servers, and a killed server writes no profiler output. Servers
# go first and the sampler last, so its window covers them.
#
# Idempotent: `_shutdown_server` on a dead pid returns at once and `_stop_rdma_counters` clears
# the pid it holds.
_running_servers=()
_cleanup_on_exit() {
    local _pid
    for _pid in "${_running_servers[@]}"; do
        [[ -n "$_pid" ]] && _shutdown_server "$_pid" "server ${_pid} (exit cleanup)"
    done
    _stop_rdma_counters
}
trap _cleanup_on_exit EXIT

export PREFILL_MODEL_CONFIG DECODE_MODEL_CONFIG

if [[ "${_TRANSFER_BACKEND}" != "mori" ]]; then
    echo "[override] Transfer backend: ${_TRANSFER_BACKEND}"
fi


# =============================================================================
# Cluster Topology (dist-init endpoints)
# =============================================================================

# Proxy/router runs on NODE_RANK=0 (first prefill node); no extra proxy node needed.
# IP layout: IP_ARRAY[0..xP-1] = prefill nodes, IP_ARRAY[xP..xP+yD-1] = decode nodes.
IP_FIRST_PREFILL=$(echo "$IPADDRS" | awk -F',' '{print $1}')
IP_FIRST_DECODE=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+1)}')

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

DIST_INIT_PORT="${DIST_INIT_PORT:-5757}"
PREFILL_NNODES="${xP}"
DECODE_NNODES="${yD}"
PREFILL_DIST_INIT_ADDR="${IP_FIRST_PREFILL}:${DIST_INIT_PORT}"
DECODE_DIST_INIT_ADDR="${IP_FIRST_DECODE}:${DIST_INIT_PORT}"

if [[ "$DP_MODE" == "1" ]]; then
    _expected_ip_slots=$((xP + yD))
    if [[ -z "$IP_FIRST_PREFILL" || -z "$IP_FIRST_DECODE" ]]; then
        echo "ERROR: DP_MODE=1 requires non-empty IP_FIRST_PREFILL and IP_FIRST_DECODE (from IPADDRS=${IPADDRS})" >&2
        exit 1
    fi
    if ((${#IP_ARRAY[@]} < _expected_ip_slots)); then
        echo "ERROR: DP_MODE=1 expects at least ${_expected_ip_slots} comma-separated hosts in IPADDRS (xP=${xP} prefill + yD=${yD} decode); got ${#IP_ARRAY[@]}" >&2
        exit 1
    fi
    echo "[debug] DP_MODE=1 topology:"
    echo "[debug]   xP=${xP} yD=${yD} GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "[debug]   PREFILL_TP_SIZE=${PREFILL_TP_SIZE}  PREFILL_DP_SIZE=${PREFILL_DP_SIZE}  PREFILL_EP_SIZE=${PREFILL_EP_SIZE}  PREFILL_NNODES=${PREFILL_NNODES}"
    echo "[debug]   DECODE_TP_SIZE=${DECODE_TP_SIZE}  DECODE_DP_SIZE=${DECODE_DP_SIZE}  DECODE_EP_SIZE=${DECODE_EP_SIZE}  DECODE_NNODES=${DECODE_NNODES}"
    echo "[debug]   PREFILL_DIST_INIT_ADDR=${PREFILL_DIST_INIT_ADDR}"
    echo "[debug]   DECODE_DIST_INIT_ADDR=${DECODE_DIST_INIT_ADDR}"
    echo "[debug]   IP_FIRST_PREFILL=${IP_FIRST_PREFILL}  IP_FIRST_DECODE=${IP_FIRST_DECODE}"
    echo "[debug]   MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE}"
fi

PREFILL_ARGS=""
DECODE_ARGS=""

# Router backend URLs:
# DP_MODE=0: each node is an independent HTTP worker → register all xP prefill and all yD decode nodes.
# DP_MODE=1: only the master node (NODE_RANK=0 for prefill, NODE_RANK=xP for decode) exposes an HTTP
#   server; secondary nodes are pure compute workers with a dummy health-check placeholder on port 3000
#   that returns 404 on all real endpoints. Registering secondary nodes causes the proxy to mark them
#   unhealthy and skip all traffic, so only the master IPs are registered.
if [[ "$DP_MODE" == "1" ]]; then
    PREFILL_ARGS=" --prefill http://${IP_ARRAY[0]}:3000"
    DECODE_ARGS=" --decode http://${IP_ARRAY[$xP]}:3000"
else
    for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
        PREFILL_ARGS+=" --prefill http://${IP_ARRAY[$i]}:3000"
    done
    for ((i=xP; i<${#IP_ARRAY[@]}; i++)); do
        DECODE_ARGS+=" --decode http://${IP_ARRAY[$i]}:3000"
    done
fi

echo "PREFILL_ARGS: $PREFILL_ARGS"
echo "DECODE_ARGS: $DECODE_ARGS"


# =============================================================================
# Container Synchronization
# =============================================================================

# Time-to-first-ready scales with checkpoint size, so the default is per model rather
# than one number for every card. 4000s is fine for the ~600 GB class (DeepSeek-R1
# reached ready in well under half of it), but Kimi-K2 is ~1 TB of FP8 weights: a
# 4-node run measured 15m15s of weight load on an idle NFS and 24m48s on a busy one,
# and the slow case blew past 4000s at 4003s -- run.sh then tore down healthy servers
# that were still initialising. Give the 1 TB class 3h; an explicit
# ROUTER_READY_TIMEOUT_SECONDS still overrides both.
#
# Computed here, before the per-rank branching, because every rank needs it: rank 0
# polls the server logs with it, and the other ranks wait for rank 0's router with it.
_router_ready_default=4000
case "${MODEL_NAME:-}" in
    *Kimi-K2*|*kimi-k2*) _router_ready_default=10800 ;;
esac
ROUTER_READY_TIMEOUT_SECONDS="${ROUTER_READY_TIMEOUT_SECONDS:-$_router_ready_default}"

# The two barrier waits below are not the same kind of wait and must not share a
# timeout. The container-creation barrier is peers starting a container: minutes.
# The router wait further down is ranks 1..N waiting for rank 0 to finish loading
# weights and bring every server up: hours. A single 3600s default for both killed a
# healthy 4-node run at exactly 3600s ("barrier timed out ... 10.158.x:2322"), and
# would also have killed the validated Kimi run, which took 72 minutes to reach ready.
# So the router wait is bounded by the readiness timeout itself, plus a margin, and can
# never fire before the thing it is waiting for has been given up on.
BARRIER_TIMEOUT_SECONDS="${BARRIER_TIMEOUT_SECONDS:-3600}"
ROUTER_BARRIER_TIMEOUT_SECONDS="${ROUTER_BARRIER_TIMEOUT_SECONDS:-$(( ROUTER_READY_TIMEOUT_SECONDS + 1800 ))}"

echo "Waiting at the container creation barrier on $host_name (timeout ${BARRIER_TIMEOUT_SECONDS}s)"
if ! python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${BARRIER_PORT} \
    --timeout "${BARRIER_TIMEOUT_SECONDS}"; then
    # Checked, unlike before: an unchecked barrier turns a hard failure into
    # "Script completed successfully" and madengine then reports the run as passed.
    echo "ERROR: container-creation barrier failed on ${host_name}; aborting this rank" >&2
    exit 1
fi


# =============================================================================
# Prepared sglang launch commands
# =============================================================================
# NODE_RANK 0        : first prefill node + router/proxy (co-located); supports xP>=1, yD>=1 for both DP_MODE=0 and DP_MODE=1.
# NODE_RANK 1..xP-1  : remaining prefill workers (PREFILL_NODE_RANK = NODE_RANK).
# NODE_RANK xP..xP+yD-1: decode workers (DECODE_NODE_RANK = NODE_RANK - xP).
# NODE_RANK 0 (DP_MODE=0): waits for all worker logs; (DP_MODE=1): waits for master prefill (NODE 0) + master decode (NODE xP) only.
# Workers sync on MASTER_ADDR:2322 and wait for proxy shutdown.

cd /sgl-workspace/sglang || {
    echo "ERROR: cd /sgl-workspace/sglang failed"
    exit 1
}

unset PREFILL_CMD DECODE_CMD ROUTER_CMD 2>/dev/null || true

setup_sglang_worker_env() {
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${IFNAME:-eth0}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFNAME:-eth0}}"
    export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
    # Kimi-K2 MLA: keep the non-fused decode path. This matches upstream, which
    # defaults the flag off (srt/environ.py: SGLANG_ROCM_FUSED_DECODE_MLA =
    # EnvBool(False)) and sets it to 0 explicitly in its own Kimi-K2 ROCm CI
    # recipes (scripts/ci/slurm/recipes/mi355x-fp8/kimik26/**.yaml). Other MLA
    # models (e.g. DeepSeek) keep whatever the image chose.
    #
    # This deliberately does NOT test "is the variable unset?". The rocm/sgl-dev
    # images bake SGLANG_ROCM_FUSED_DECODE_MLA=1 into the image Config.Env, so on
    # every sgl-dev base the variable is already non-empty and an unset-test never
    # fires -- Kimi then silently ran the fused path, the opposite of the intent
    # here (observed on a 4-node 2P2D gfx942 run: the live server process had
    # =1). An image-level default is not an operator decision and does not get to
    # win; set SGLANG_KEEP_FUSED_DECODE_MLA=1 to opt back in deliberately.
    case "${MODEL_NAME:-}" in
        *Kimi-K2*|*kimi-k2*)
            if [[ "${SGLANG_KEEP_FUSED_DECODE_MLA:-0}" != "1" ]]; then
                export SGLANG_ROCM_FUSED_DECODE_MLA=0
            fi
            ;;
    esac
    export SGLANG_MORI_FP8_DISP="${SGLANG_MORI_FP8_DISP:-True}"
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1200}"
}

# _dbg: timestamped debug trace (NODE_RANK + wall-clock prefix).
_dbg() { echo "[debug $(date +%T) NODE${NODE_RANK}] $*"; }

# _launch_server CMD LOG_PATH ROLE
# Start one sglang server in the background and leave its **own** pid in `_launched_pid`.
#
# The pid matters: these used to start as `eval "$CMD" | tee -a "$LOG" &`, and for a pipeline `$!`
# is the *last* command -- the `tee`. `_shutdown_server` then SIGINTed tee, saw the pid gone and
# reported a clean stop, while the server died on a broken pipe without unwinding.
#
# Three shapes were measured, each with a child that prints when its SIGINT handler fires:
#   `eval "$CMD" | tee -a log &`  -> $! is tee; the server never hears the signal
#   `eval "$CMD" >> log &`        -> $! is a bash subshell, which ignores SIGINT as an async job
#   `eval "exec $CMD" >> log &`   -> $! is the server; it ran its handler and exited
# The third is what runs now; the tee was redundant anyway, its stdout went to /dev/null.
#
# `set -m` for the launch alone: without job control the background job joins this shell's process
# group and only the one pid we hold can be signalled. Monitor mode gives the job its own group,
# with id equal to the pid, so `_shutdown_server` can signal the server and its forked children.
_launch_server() {
    local _cmd="$1" _log="$2" _role="$3"
    set -m
    eval "exec ${_cmd}" >> "$_log" 2>&1 &
    _launched_pid=$!
    set +m
    # Registered for the exit trap, which covers the paths that never reach `_shutdown_server`.
    _running_servers+=("$_launched_pid")
}

# _shutdown_server PID [LABEL]
# Stop a server and do not return until it is actually gone. SIGINT first, because sglang's
# launch_server unwinds its scheduler subprocesses on it, which is what lets a profiler flush:
# a bare `kill` returns immediately, this script exits, and madengine tears the container down
# mid-write, which is why profiled runs produced empty rocprof directories while the logs showed
# "[rocprofv3] finalizing after signal" (job 25815). Escalation keeps a wedged server from
# hanging the job. Flushing kernel-trace stats takes far longer than a plain exit, hence the
# wider budget while profiling.
# _server_died LOG -- true when a server's log shows it will never print the ready signal.
# The barrier below polls for that signal and nothing else, so a server that died in its first
# minutes was waited on until the router timeout and then the wall clock: job 241090 spent four
# hours on a prefill that was gone at two. The remote roles' pids are on another node, so the log
# is what both cases have in common.
_SERVER_DEATH_SIGNS="${_SERVER_DEATH_SIGNS:-scheduler died during initialization|torch.OutOfMemoryError|The memory capacity is unbalanced|CUDA error: out of memory}"
_server_died() {
    [[ -f "$1" ]] || return 1
    grep -qE "${_SERVER_DEATH_SIGNS}" "$1" 2>/dev/null
}

# _server_gone PID -- true only when neither the process group nor the pid still exists.
_server_gone() {
    kill -0 -- "-$1" 2>/dev/null && return 1
    kill -0 "$1" 2>/dev/null && return 1
    return 0
}

_shutdown_server() {
    local _pid="$1" _label="${2:-server}" _grace _i
    if [[ "${PROFILE_ENABLE:-0}" == "1" ]]; then
        _grace="${SHUTDOWN_GRACE_S:-300}"
    else
        _grace="${SHUTDOWN_GRACE_S:-30}"
    fi

    # The process group, falling back to the pid: `_launch_server` gives each server its own group
    # so this reaches the whole tree, and the fallback covers a pid that is not a group leader.
    kill -INT -- "-${_pid}" 2>/dev/null \
        || kill -INT "$_pid" 2>/dev/null \
        || { _dbg "${_label} (pid=${_pid}) already gone"; return 0; }
    # "Gone" means the whole group, not just its leader: a scheduler child outliving the leader is
    # exactly the case the escalation below is for.
    for ((_i = 1; _i <= _grace; _i++)); do
        if _server_gone "$_pid"; then
            _dbg "${_label} (pid=${_pid}) exited on SIGINT after ${_i}s"
            return 0
        fi
        sleep 1
    done

    _dbg "${_label} (pid=${_pid}) still alive after ${_grace}s, sending SIGTERM"
    kill -TERM -- "-${_pid}" 2>/dev/null || kill -TERM "$_pid" 2>/dev/null
    # A profiled process writes its output after the last child is gone, and a validation run was
    # SIGKILLed one second short of that, so the SIGTERM wait is wider while profiling.
    local _term_grace=30
    [[ "${PROFILE_ENABLE:-0}" == "1" ]] && _term_grace=180
    for ((_i = 1; _i <= _term_grace; _i++)); do
        _server_gone "$_pid" && { _dbg "${_label} exited on SIGTERM"; return 0; }
        sleep 1
    done

    _dbg "${_label} (pid=${_pid}) ignored SIGTERM, sending SIGKILL"
    kill -KILL -- "-${_pid}" 2>/dev/null || kill -KILL "$_pid" 2>/dev/null
}

# _wait_for_tcp HOST PORT [TIMEOUT_S] [LABEL]
# Polls TCP connectivity; returns 0 when port is open, 1 on timeout.
# Uses /dev/tcp; falls back to nc if available. Logs progress every 10s.
_wait_for_tcp() {
    local _host="$1" _port="$2"
    local _timeout_s="${3:-120}"
    local _label="${4:-${_host}:${_port}}"
    local _start_ts _elapsed _last_log
    _start_ts=$(date +%s)
    _last_log=$(date +%s)
    _dbg "_wait_for_tcp: checking ${_label} (timeout ${_timeout_s}s) ..."
    while true; do
        if bash -c "exec 3<>/dev/tcp/${_host}/${_port} 2>/dev/null && exec 3<&- && exec 3>&-" 2>/dev/null; then
            _elapsed=$(( $(date +%s) - _start_ts ))
            _dbg "_wait_for_tcp: ${_label} is reachable (${_elapsed}s elapsed)"
            return 0
        fi
        _elapsed=$(( $(date +%s) - _start_ts ))
        if (( _elapsed >= _timeout_s )); then
            _dbg "ERROR: _wait_for_tcp: ${_label} not reachable after ${_elapsed}s" >&2
            return 1
        fi
        if (( $(date +%s) - _last_log >= 10 )); then
            _dbg "_wait_for_tcp: still waiting for ${_label} (${_elapsed}s / ${_timeout_s}s) ..."
            _last_log=$(date +%s)
        fi
        sleep 2
    done
}

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "${host_name}:${host_ip} is Prefill Node 0 + Router/proxy (NODE_RANK=0, co-located)"

    mkdir -p "/run_logs/${SLURM_JOB_ID:-0}"

    # --- Launch first prefill server on this node (co-located with router) ---
    setup_sglang_worker_env
    PREFILL_NODE_RANK=0

    # DP_MODE=1 with dp-size>1: router must use follow_bootstrap_room so each
    # request is pinned to the DP rank that owns its bootstrap slot, preventing
    # KV-transfer rank mismatches that cause requests to hang indefinitely.
    _prefill_lb_method="round_robin"
    [[ "$DP_MODE" == "1" ]] && _prefill_lb_method="follow_bootstrap_room"

    _prefill_cmd="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode prefill \
        --load-balance-method ${_prefill_lb_method} \
        --prefill-round-robin-balance \
        --disaggregation-ib-device ${IB_DEVICES} \
        --host ${host_ip} \
        --port 3000 \
        --trust-remote-code \
        --tp-size ${PREFILL_TP_SIZE}"

    if [[ "$DP_MODE" == "1" ]]; then
        _prefill_cmd+=" \
            --dp-size ${PREFILL_DP_SIZE} \
            --ep-size ${PREFILL_EP_SIZE} \
            --dist-init-addr ${PREFILL_DIST_INIT_ADDR} \
            --nnodes ${PREFILL_NNODES} \
            --node-rank ${PREFILL_NODE_RANK}"
    fi

    _prefill_cmd+=" \
        --decode-log-interval 1 \
        ${PREFILL_MODEL_CONFIG} \
        --log-level-http warning"

    export _prefill_cmd PREFILL_NODE_RANK

    PREFILL_LOG="/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log"
    {
        echo "========== PREFILL_CMD (NODE_RANK=0, PREFILL_NODE_RANK=0, co-located with router) =========="
        echo "$_prefill_cmd"
        echo ""
    } | tee "$PREFILL_LOG"
    _dbg "launching prefill server (PREFILL_NODE_RANK=0, PREFILL_TP_SIZE=${PREFILL_TP_SIZE})"
    set -x
    _start_rdma_counters
    _launch_server "$_prefill_cmd" "$PREFILL_LOG" prefill
    set +x
    _node0_prefill_pid=$_launched_pid
    _dbg "prefill server started pid=${_node0_prefill_pid}"

    # DP_MODE=0: wait for SEARCH_SIGNAL in every prefill (NODE 0..xP-1) and decode (NODE xP..xP+yD-1) log.
    # DP_MODE=1: wait for master prefill NODE 0 + master decode NODE xP only.
    # Requires shared /run_logs across nodes.
    SEARCH_SIGNAL="${SEARCH_SIGNAL:-The server is fired up and ready to roll!}"
    ROUTER_POLL_SLEEP_SECONDS="${ROUTER_POLL_SLEEP_SECONDS:-10}"
    _wait_start_ts=$(date +%s)
    _runlog="/run_logs/${SLURM_JOB_ID:-0}"

    if [[ "$DP_MODE" == "0" ]]; then
        echo "Waiting for all ${xP} prefill + ${yD} decode servers (grep: ${SEARCH_SIGNAL}) before starting router..."
        for ((i = 0; i < xP; i++)); do
            LOG_FILE="${_runlog}/prefill_NODE${i}.log"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if _server_died "$LOG_FILE"; then
                    echo "ERROR: prefill NODE${i} died before becoming ready (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                _elapsed=$(( $(date +%s) - _wait_start_ts ))
                if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                    echo "ERROR: Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for prefill NODE${i} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
            done
            echo "Prefill NODE${i} ready."
        done
        for ((i = xP; i <= xP + yD - 1; i++)); do
            LOG_FILE="${_runlog}/decode_NODE${i}.log"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if _server_died "$LOG_FILE"; then
                    echo "ERROR: decode NODE${i} died before becoming ready (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                _elapsed=$(( $(date +%s) - _wait_start_ts ))
                if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                    echo "ERROR: Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for decode NODE${i} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
            done
            echo "Decode NODE${i} ready."
        done
    else
        _master_prefill_log="${_runlog}/prefill_NODE0.log"
        _master_decode_log="${_runlog}/decode_NODE${xP}.log"
        echo "Waiting for master prefill (NODE 0) + master decode (NODE ${xP}) — grep: ${SEARCH_SIGNAL} — DP_MODE=${DP_MODE}"
        for _label_and_file in "master prefill|${_master_prefill_log}" "master decode|${_master_decode_log}"; do
            IFS='|' read -r _log_label LOG_FILE <<< "${_label_and_file}"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if _server_died "$LOG_FILE"; then
                    echo "ERROR: ${_log_label} died before becoming ready (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                _elapsed=$(( $(date +%s) - _wait_start_ts ))
                if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                    echo "ERROR: Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for ${_log_label} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
            done
            echo "${_log_label} ready (${LOG_FILE})."
        done
    fi

    echo "Prefill/decode backends report ready per logs; starting sglang_router (proxy)."

    # Build and launch only after worker logs confirm servers are up (avoids the proxy probing backends too early).
    # 0.0.0.0 so bench_serving (--host 127.0.0.1 in benchmark_xPyD.sh) can connect; binding only ${host_ip} rejects loopback.
    # PREFILL_ARGS / DECODE_ARGS already contain --prefill/--decode for all xP/yD nodes.
    # Use them for both DP_MODE=0 and DP_MODE=1 so xP>1 or yD>1 registers all backends.
    ROUTER_CMD="python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS} \
        --host 0.0.0.0 \
        --port 2322"
    export ROUTER_CMD

    echo "========== ROUTER_CMD (NODE_RANK=0, DP_MODE=${DP_MODE}) ==========" >&2
    echo "$ROUTER_CMD"
    set -x
    # Through `_launch_server` like the two roles: the router was the last pipeline launch, so
    # `$!` was its `tee` and `_shutdown_server` signalled that instead -- job 239268 shows "proxy
    # still alive after 300s" while the tee had died in the first second.
    _proxy_log="/run_logs/${SLURM_JOB_ID:-0}/proxy_NODE${NODE_RANK}.log"
    # Truncated here, because `_launch_server` appends and nothing else writes this file. The two
    # roles get theirs emptied by the `tee` that writes their command header; the router has no
    # header, so without this a requeue under the same job id -- or any local run, where the id is
    # absent and the path is always the same -- splices the next attempt onto the last one.
    : > "$_proxy_log"
    _launch_server "$ROUTER_CMD" "$_proxy_log" proxy
    set +x
    proxy_pid=$_launched_pid
    echo "Router (sglang_router) started pid=${proxy_pid} (DP_MODE=${DP_MODE})"

    #_bench_host="${BENCHMARK_ROUTER_HOST:-127.0.0.1}"
    #bench_port="${BENCHMARK_ROUTER_PORT:-2322}"
    #_wait_per_s="${ROUTER_LISTEN_POLL_SECONDS:-2}"
    #_wait_max_s="${ROUTER_LISTEN_TIMEOUT_SECONDS:-120}"
    #_waited=0
    #_router_up=0
    #echo "Waiting for router on ${_bench_host}:${_bench_port} (timeout ${_wait_max_s}s)..."
    #while ((_waited < _wait_max_s)); do
    #    if command -v nc >/dev/null 2>&1 && nc -z "${_bench_host}" "${_bench_port}" 2>/dev/null; then
    #        _router_up=1
    #        break
    #    fi
    #    if bash -c "exec 3<>/dev/tcp/${_bench_host}/${_bench_port}; exec 3<&-; exec 3>&-" 2>/dev/null; then
    #        _router_up=1
    #        break
    #    fi
    #    sleep "${_wait_per_s}"
    #    _waited=$((_waited + _wait_per_s))
    #done
    #if [[ "${_router_up}" -eq 0 ]]; then
    #    echo "ERROR: Router not listening on ${_bench_host}:${_bench_port} within ${_wait_max_s}s (proxy pid=${proxy_pid}). Check proxy_NODE0.log" >&2
    #    exit 1
    #fi
    #echo "Proxy ready for benchmarking on ${host_name}:${host_ip} (${_bench_host}:${_bench_port})"
    echo "Proxy ready for benchmarking on ${host_name}:${host_ip}"

    # Wait for proxy to register both prefill and decode workers before smoke test / benchmark.
    # The proxy initializes workers in the background after startup — hitting it too early
    # returns "No decode workers available" even though both SGLang servers are running.
    # /v1/models returns 200 immediately (before workers activate), so we poll with a real
    # completions request that exercises the full PD path.
    _proxy_base="${ROUTER_HTTP_BASE:-http://127.0.0.1:2322}"
    _proxy_wait_max=300
    _proxy_waited=0
    _proxy_interval=10
    echo "Waiting up to ${_proxy_wait_max}s for proxy PD path to become ready ..."
    while [[ ${_proxy_waited} -lt ${_proxy_wait_max} ]]; do
        _probe=$(curl -sS --max-time 30 "${_proxy_base}/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"probe","prompt":"hi","max_tokens":1,"temperature":0}' 2>/dev/null || echo "")
        # Success: response contains "choices" (actual completion). Failure: "error" with
        # "No decode workers" or "No available prefill workers" or connection refused.
        if echo "${_probe}" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'choices' in d else 1)" 2>/dev/null; then
            echo "Proxy PD path ready after ${_proxy_waited}s"
            break
        fi
        sleep "${_proxy_interval}"
        _proxy_waited=$((_proxy_waited + _proxy_interval))
    done
    if [[ ${_proxy_waited} -ge ${_proxy_wait_max} ]]; then
        echo "WARN: Proxy PD path not ready after ${_proxy_wait_max}s. Proceeding anyway." >&2
    fi

    # Smoke test: OpenAI-compatible completions before bench_serving (skip with SKIP_CURL_TEST=1).
    if [[ "${SKIP_CURL_TEST:-0}" != "1" ]]; then
        _curl_base="${ROUTER_HTTP_BASE:-http://127.0.0.1:2322}"
        export CURL_TEST_MODEL="${CURL_TEST_MODEL:-${MODEL_PATH}}"
        echo "========== CURL smoke test: POST ${_curl_base}/v1/completions (CURL_TEST_MODEL=\${MODEL_PATH} by default) =========="
        _curl_json=$(MODEL_PATH="${MODEL_PATH}" CURL_TEST_MODEL="${CURL_TEST_MODEL}" python3 - <<'PY'
import json, os
m = (os.environ.get("CURL_TEST_MODEL") or os.environ.get("MODEL_PATH") or "").strip()
print(json.dumps({"model": m, "prompt": "Who is AMD CEO?", "temperature": 0, "top_k": 1}))
PY
)
        if ! curl -sS "${_curl_base}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "${_curl_json}" \
            | tee "${_runlog}/curl_smoke_NODE${NODE_RANK}.log"; then
            echo "WARN: CURL smoke test failed (e.g. connection). See ${_runlog}/curl_smoke_NODE${NODE_RANK}.log" >&2
        fi
        echo ""
    fi

    benchmark_status=0
    # Server-side Prometheus metrics for the agentic replay: point aiperf at the
    # sglang servers' :3000/metrics directly (the router exposes Prometheus on a
    # separate port and lacks gpu_cache_hit_rate). Gated on the agentic path so
    # the default sweep is unaffected. AGENTIC_SERVER_METRICS is consumed by
    # scripts/common/agentic_lib.sh (build_replay_cmd -> aiperf --server-metrics).
    if [[ "${AGENTIC_METRICS_ENABLED}" == "1" \
          && -n "${IP_FIRST_PREFILL:-}" && -n "${IP_FIRST_DECODE:-}" ]]; then
        export AGENTIC_SERVER_METRICS="${AGENTIC_SERVER_METRICS:-${IP_FIRST_PREFILL}:3000 ${IP_FIRST_DECODE}:3000}"
        echo "[metrics] AGENTIC_SERVER_METRICS=${AGENTIC_SERVER_METRICS}"
        echo "=== server /metrics reachability check ==="
        curl -sf "http://${IP_FIRST_PREFILL}:3000/metrics" | head -3 || echo "PREFILL metrics UNREACHABLE"
        curl -sf "http://${IP_FIRST_DECODE}:3000/metrics" | head -3 || echo "DECODE metrics UNREACHABLE"
    fi
    if [[ "${SKIP_BENCHMARK:-0}" != "1" ]] && [[ -n "${MOONCAKE_COOKBOOK_PATH:-}" ]]; then
        # Benchmark hook is selectable: default random-sweep benchmark_xPyD.sh,
        # or set BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh for aiperf agentic
        # trace replay (see scripts/common/agentic_lib.sh).
        _bench_file="${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}"
        if [[ -f "${MOONCAKE_COOKBOOK_PATH}/${_bench_file}" ]]; then
            echo "Running ${MOONCAKE_COOKBOOK_PATH}/${_bench_file}"
            (
                cd "${MOONCAKE_COOKBOOK_PATH}" || exit 1
                bash "${_bench_file}"
            ) || benchmark_status=$?
        else
            echo "WARN: ${_bench_file} not found under MOONCAKE_COOKBOOK_PATH=${MOONCAKE_COOKBOOK_PATH}" >&2
            benchmark_status=1
        fi
    fi

    echo "Stopping the proxy server (pid=${proxy_pid})"
    _shutdown_server "${proxy_pid}" "proxy"

    echo "Stopping the co-located prefill server (pid=${_node0_prefill_pid})"
    _shutdown_server "${_node0_prefill_pid}" "prefill NODE${NODE_RANK}"
    _stop_rdma_counters

    if [[ "${benchmark_status}" -ne 0 ]]; then
        echo "ERROR: benchmark failed with status ${benchmark_status}" >&2
        exit "${benchmark_status}"
    fi

elif [[ "$NODE_RANK" -ge 1 && "$NODE_RANK" -lt "$xP" ]]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-default})"
    # NODE_RANK 0..xP-1 map directly to PREFILL_NODE_RANK 0..xP-1 (proxy co-located on NODE_RANK=0).
    PREFILL_NODE_RANK=$((NODE_RANK))
    setup_sglang_worker_env
    #if [[ "$DP_MODE" == "0" ]]; then
    #    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    #    echo "DP_MODE=0 prefill SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
    #fi

    _prefill_lb_method="round_robin"
    [[ "$DP_MODE" == "1" ]] && _prefill_lb_method="follow_bootstrap_room"

    PREFILL_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode prefill \
        --load-balance-method ${_prefill_lb_method} \
        --prefill-round-robin-balance \
        --disaggregation-ib-device ${IB_DEVICES} \
        --host ${host_ip} \
        --port 3000 \
        --trust-remote-code \
        --tp-size ${PREFILL_TP_SIZE}"

    if [[ "$DP_MODE" == "1" ]]; then
        PREFILL_CMD+=" \
            --dp-size ${PREFILL_DP_SIZE} \
            --ep-size ${PREFILL_EP_SIZE} \
            --dist-init-addr ${PREFILL_DIST_INIT_ADDR} \
            --nnodes ${PREFILL_NNODES} \
            --node-rank ${PREFILL_NODE_RANK}"
    fi
   
    PREFILL_CMD+=" \
        --decode-log-interval 1 \
        ${PREFILL_MODEL_CONFIG} \
        ${SERVER_METRICS_FLAGS} \
        --log-level-http warning"

    export PREFILL_CMD PREFILL_NODE_RANK

    # NOTE: do NOT gate PREFILL_NODE_RANK>=1 on port 5757 being open.
    # Same rendezvous deadlock as decode: all prefill nodes must launch concurrently.
    if [[ "$DP_MODE" == "1" ]] && (( PREFILL_NODE_RANK >= 1 )); then
        _dbg "PREFILL_NODE_RANK=${PREFILL_NODE_RANK}: launching without gating on dist-init port (rendezvous requires all ranks concurrent)"
    fi

    PREFILL_LOG="/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log"
    mkdir -p "$(dirname "$PREFILL_LOG")"
    {
        echo "========== PREFILL_CMD (NODE_RANK=${NODE_RANK}, PREFILL_NODE_RANK=${PREFILL_NODE_RANK}) =========="
        echo "$PREFILL_CMD"
        echo ""
    } | tee "$PREFILL_LOG"
    _dbg "launching prefill server (PREFILL_NODE_RANK=${PREFILL_NODE_RANK}, PREFILL_TP_SIZE=${PREFILL_TP_SIZE})"
    set -x
    _start_rdma_counters
    _launch_server "$PREFILL_CMD" "$PREFILL_LOG" prefill
    set +x
    prefill_pid=$_launched_pid
    _dbg "prefill server started pid=${prefill_pid}"

    _dbg "waiting for proxy server to be up (MASTER_ADDR=${MASTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up (timeout ${ROUTER_BARRIER_TIMEOUT_SECONDS}s)..."
    if ! python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322 \
        --timeout "${ROUTER_BARRIER_TIMEOUT_SECONDS}"; then
        echo "ERROR: proxy on ${MASTER_ADDR}:2322 never came up within ${ROUTER_BARRIER_TIMEOUT_SECONDS}s" >&2
        _shutdown_server "${prefill_pid}" "prefill NODE${NODE_RANK}"
        exit 1
    fi

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Stopping the prefill server"
    _shutdown_server "${prefill_pid}" "prefill NODE${NODE_RANK}"
    _stop_rdma_counters

elif [[ "$NODE_RANK" -ge $xP && "$NODE_RANK" -le $((xP + yD - 1)) ]]; then
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-default})"
    DECODE_NODE_RANK=$((NODE_RANK - xP))
    setup_sglang_worker_env
    _dbg "decode node: DECODE_NODE_RANK=${DECODE_NODE_RANK} DECODE_TP_SIZE=${DECODE_TP_SIZE} DECODE_DP_SIZE=${DECODE_DP_SIZE:-n/a} DECODE_EP_SIZE=${DECODE_EP_SIZE:-n/a}"
    _dbg "DECODE_DIST_INIT_ADDR=${DECODE_DIST_INIT_ADDR} DECODE_NNODES=${DECODE_NNODES}"

    if [[ "$DP_MODE" == "1" ]]; then
        #export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
        #echo "DP_MODE=0 decode SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"
        export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_DECODE}"
        _dbg "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
        _dbg "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD}"
        _dbg "MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE}"

        # NOTE: do NOT gate DECODE_NODE_RANK>=1 on port 5757 being open.
        # torch init_process_group uses a rendezvous: rank 0 opens the port only
        # AFTER all ranks have joined. Waiting for rank 0's port before launching
        # rank 1 creates a permanent deadlock. All decode nodes must launch concurrently.
        _dbg "DECODE_NODE_RANK=${DECODE_NODE_RANK}: launching without gating on dist-init port (rendezvous requires all ranks concurrent)"
    fi

    _decode_lb_method="round_robin"
    [[ "$DP_MODE" == "1" ]] && _decode_lb_method="follow_bootstrap_room"

    DECODE_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode decode \
        --load-balance-method ${_decode_lb_method} \
        --prefill-round-robin-balance \
        --disaggregation-ib-device ${IB_DEVICES} \
        --host ${host_ip} \
        --port 3000 \
        --trust-remote-code \
        --tp-size ${DECODE_TP_SIZE}"

    if [[ "$DP_MODE" == "1" ]]; then
        DECODE_CMD+=" \
            --dp-size ${DECODE_DP_SIZE} \
            --ep-size ${DECODE_EP_SIZE} \
            --dist-init-addr ${DECODE_DIST_INIT_ADDR} \
            --nnodes ${DECODE_NNODES} \
            --node-rank ${DECODE_NODE_RANK}"
    fi

    DECODE_CMD+=" \
        --decode-log-interval 1 \
        ${DECODE_MODEL_CONFIG} \
        ${SERVER_METRICS_FLAGS} \
        --log-level-http warning"

    export DECODE_CMD DECODE_NODE_RANK

    DECODE_LOG="/run_logs/${SLURM_JOB_ID:-0}/decode_NODE${NODE_RANK}.log"
    mkdir -p "$(dirname "$DECODE_LOG")"
    {
        echo "========== DECODE_CMD (NODE_RANK=${NODE_RANK}, DECODE_NODE_RANK=${DECODE_NODE_RANK}) =========="
        echo "$DECODE_CMD"
        echo ""
    } | tee "$DECODE_LOG"
    _dbg "launching decode server (DECODE_NODE_RANK=${DECODE_NODE_RANK}, DECODE_TP_SIZE=${DECODE_TP_SIZE})"
    set -x
    _start_rdma_counters
    _launch_server "$DECODE_CMD" "$DECODE_LOG" decode
    set +x
    decode_pid=$_launched_pid
    _dbg "decode server started pid=${decode_pid}"

    _dbg "waiting for proxy server to be up (MASTER_ADDR=${MASTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up (timeout ${ROUTER_BARRIER_TIMEOUT_SECONDS}s)..."
    if ! python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322 \
        --timeout "${ROUTER_BARRIER_TIMEOUT_SECONDS}"; then
        echo "ERROR: proxy on ${MASTER_ADDR}:2322 never came up within ${ROUTER_BARRIER_TIMEOUT_SECONDS}s" >&2
        _shutdown_server "${decode_pid}" "decode NODE${NODE_RANK}"
        exit 1
    fi

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Stopping the decode server"
    _shutdown_server "${decode_pid}" "decode NODE${NODE_RANK}"
    _stop_rdma_counters

else
    echo "ERROR: NODE_RANK=${NODE_RANK} out of range (expected 0..$((xP + yD))) for xP=${xP} yD=${yD}" >&2
    exit 1
fi

echo "Script completed successfully"
exit 0
