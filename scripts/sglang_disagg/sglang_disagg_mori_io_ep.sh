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
    "DeepSeek-V4-Flash-FP8"
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

pip install py-spy
pip install --ignore-installed --force-reinstall flask
pip install pyyaml


# host_ip must match the IPADDRS scope (fabric on skyRiver, not the mgmt default route).
# Pick the local IPv4 that is present in IPADDRS; fall back to the default-route src.
host_ip=""
for _cand in $(hostname -I 2>/dev/null); do
    case ",${IPADDRS}," in *",${_cand},"*) host_ip="$_cand"; break;; esac
done
[[ -z "$host_ip" ]] && host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
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

MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
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

with open(config_path, "r", encoding="utf-8") as f:
    models = yaml.safe_load(f) or {}

if model_name not in models:
    print(f'echo "ERROR: Model {model_name} not found in {config_path}"; exit 1')
    sys.exit(0)

cfg = models[model_name] or {}
prefill = cfg.get("prefill", {}) or {}
decode = cfg.get("decode", {}) or {}


def q(v):
    return shlex.quote(str(v if v is not None else ""))


prefill_flags = prefill.get(mode, "") or ""

exports = {
    "MODEL_BASE_FLAGS": cfg.get("base_flags", ""),
    "MODEL_MODE_FLAGS": cfg.get(f"{mode}_flags", ""),
    "MODEL_PREFILL_FLAGS": prefill_flags,
    "MODEL_DECODE_FLAGS": decode.get(mode, ""),
    "MODEL_EXPERIMENTAL_FLAGS": cfg.get("experimental_flags", ""),
}

for key, value in exports.items():
    print(f"{key}={q(value)}")
PY
)"

PREFILL_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_PREFILL_FLAGS} ${MODEL_EXPERIMENTAL_FLAGS}"
DECODE_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_DECODE_FLAGS} ${MODEL_EXPERIMENTAL_FLAGS}"
echo "Using model-specific configuration for: $MODEL_NAME (mode=${PARALLEL_MODE})"

export PREFILL_MODEL_CONFIG DECODE_MODEL_CONFIG MODEL_EXPERIMENTAL_FLAGS

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mori_ep_env.sh"
# Also source per-model env (set_env_vars.sh) so MODEL_NAME-guarded blocks (e.g. DSV4
# thread caps + aiter flags) apply on the MoRI-EP path, not just the mooncake server path.
[[ -f "${SCRIPT_DIR}/set_env_vars.sh" ]] && source "${SCRIPT_DIR}/set_env_vars.sh"

# KV transfer backend: default mori, switchable to mooncake (Mooncake).
# Kept out of models.yaml so model config is backend-agnostic.
_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-mori}"
PREFILL_MODEL_CONFIG+=" --disaggregation-transfer-backend ${_TRANSFER_BACKEND}"
DECODE_MODEL_CONFIG+=" --disaggregation-transfer-backend ${_TRANSFER_BACKEND}"
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

echo "Waiting at the container creation barrier on $host_name"
python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${BARRIER_PORT}


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
    export SGLANG_MORI_FP8_DISP="${SGLANG_MORI_FP8_DISP:-True}"
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1200}"
}

# _dbg: timestamped debug trace (NODE_RANK + wall-clock prefix).
_dbg() { echo "[debug $(date +%T) NODE${NODE_RANK}] $*"; }

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
    eval "$_prefill_cmd" 2>&1 | tee -a "$PREFILL_LOG" >/dev/null &
    set +x
    _node0_prefill_pid=$!
    _dbg "prefill server started pid=${_node0_prefill_pid}"

    # DP_MODE=0: wait for SEARCH_SIGNAL in every prefill (NODE 0..xP-1) and decode (NODE xP..xP+yD-1) log.
    # DP_MODE=1: wait for master prefill NODE 0 + master decode NODE xP only.
    # Requires shared /run_logs across nodes.
    SEARCH_SIGNAL="${SEARCH_SIGNAL:-The server is fired up and ready to roll!}"
    ROUTER_READY_TIMEOUT_SECONDS="${ROUTER_READY_TIMEOUT_SECONDS:-4000}"
    ROUTER_POLL_SLEEP_SECONDS="${ROUTER_POLL_SLEEP_SECONDS:-10}"
    _wait_start_ts=$(date +%s)
    _runlog="/run_logs/${SLURM_JOB_ID:-0}"

    if [[ "$DP_MODE" == "0" ]]; then
        echo "Waiting for all ${xP} prefill + ${yD} decode servers (grep: ${SEARCH_SIGNAL}) before starting router..."
        for ((i = 0; i < xP; i++)); do
            LOG_FILE="${_runlog}/prefill_NODE${i}.log"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
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
        # Readiness gate. Prefer a network /health poll (FS-agnostic: works on
        # non-SLURM / no-shared-FS clusters where a peer's log file is not visible
        # on the router node). Fall back to the local-log grep when the peer HTTP
        # endpoint is not reachable (e.g. same-node master, or ROUTER_READY_HTTP=0).
        _master_prefill_ip="${IP_ARRAY[0]}"
        _master_decode_ip="${IP_ARRAY[$xP]}"
        _master_prefill_log="${_runlog}/prefill_NODE0.log"
        _master_decode_log="${_runlog}/decode_NODE${xP}.log"
        echo "Waiting for master prefill (${_master_prefill_ip}) + master decode (${_master_decode_ip}) — DP_MODE=${DP_MODE}"
        for _label_and_ep in "master prefill|${_master_prefill_ip}:3000|${_master_prefill_log}" "master decode|${_master_decode_ip}:3000|${_master_decode_log}"; do
            IFS='|' read -r _log_label _http_ep LOG_FILE <<< "${_label_and_ep}"
            until { [[ "${ROUTER_READY_HTTP:-1}" == "1" ]] && curl -s -m 3 "http://${_http_ep}/health" >/dev/null 2>&1; } \
                  || { [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; }; do
                _elapsed=$(( $(date +%s) - _wait_start_ts ))
                if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                    echo "ERROR: Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for ${_log_label} (http://${_http_ep}/health or ${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
            done
            echo "${_log_label} ready (${_http_ep})."
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
    eval "$ROUTER_CMD" 2>&1 | tee "/run_logs/${SLURM_JOB_ID:-0}/proxy_NODE${NODE_RANK}.log" >/dev/null &
    set +x
    proxy_pid=$!
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

    if [[ "${SKIP_BENCHMARK:-0}" != "1" ]] && [[ -n "${MOONCAKE_COOKBOOK_PATH:-}" ]]; then
        if [[ -f "${MOONCAKE_COOKBOOK_PATH}/benchmark_xPyD.sh" ]]; then
            echo "Running ${MOONCAKE_COOKBOOK_PATH}/benchmark_xPyD.sh"
            (
                cd "${MOONCAKE_COOKBOOK_PATH}" || exit 1
                bash benchmark_xPyD.sh
            )
        else
            echo "WARN: benchmark_xPyD.sh not found under MOONCAKE_COOKBOOK_PATH=${MOONCAKE_COOKBOOK_PATH}" >&2
        fi
    fi

    # KEEP_ALIVE=1: leave router + servers running (for external NIAH/perf/manual
    # testing) instead of tearing down. Block on the server pids so the container
    # stays up until stopped.
    if [[ "${KEEP_ALIVE:-0}" == "1" ]]; then
        echo "KEEP_ALIVE=1: router (pid=${proxy_pid}) + prefill (pid=${_node0_prefill_pid}) left running. Router on ${host_ip}:2322. Ctrl-C / docker stop to end."
        wait "${_node0_prefill_pid}" "${proxy_pid}"
    else
        echo "Killing the proxy server (pid=${proxy_pid})"
        kill "${proxy_pid}"

        echo "Killing the co-located prefill server (pid=${_node0_prefill_pid})"
        kill "${_node0_prefill_pid}"
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
    eval "$PREFILL_CMD" 2>&1 | tee -a "$PREFILL_LOG" >/dev/null &
    set +x
    prefill_pid=$!
    _dbg "prefill server started pid=${prefill_pid}"

    _dbg "waiting for proxy server to be up (MASTER_ADDR=${MASTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Killing the prefill server"
    kill "${prefill_pid}"

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
    eval "$DECODE_CMD" 2>&1 | tee -a "$DECODE_LOG" >/dev/null &
    set +x
    decode_pid=$!
    _dbg "decode server started pid=${decode_pid}"

    _dbg "waiting for proxy server to be up (MASTER_ADDR=${MASTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Killing the decode server"
    kill "${decode_pid}"

else
    echo "ERROR: NODE_RANK=${NODE_RANK} out of range (expected 0..$((xP + yD))) for xP=${xP} yD=${yD}" >&2
    exit 1
fi

echo "Script completed successfully"
exit 0
