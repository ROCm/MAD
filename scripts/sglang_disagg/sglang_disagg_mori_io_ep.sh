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

if [[ "$DP_MODE" == "1" ]] && { [[ "${xP:-1}" -gt 1 ]] || [[ "${yD:-1}" -gt 1 ]]; }; then
    echo "ERROR: DP_MODE=1 is not supported when xP>1 or yD>1 (got xP=${xP} yD=${yD}). Use DP_MODE=0 for multi-prefill or multi-decode (xPyD) topologies." >&2
    exit 1
fi

IPADDRS="${IPADDRS:-localhost}"
BARRIER_PORT="${BARRIER_PORT:-4342}"
IB_DEVICES=${IB_DEVICES:-"mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask
pip install pyyaml

## Temporary workaround for OCI-CX7 to install latest RDMA core.
git clone --branch v62.0 --depth 1 https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core && \
    cd /tmp/rdma-core && \
    mkdir -p build && cd build && \
    cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr -DNO_MAN_PAGES=1 .. && \
    ninja &&     ninja install &&     ldconfig &&     rm -rf /tmp/rdma-core

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

MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

export MODELS_YAML MODEL_NAME PARALLEL_MODE
eval "$(python3 - <<'PY'
import os
import shlex
import sys
import yaml

config_path = os.environ["MODELS_YAML"]
model_name = os.environ["MODEL_NAME"]
mode = os.environ["PARALLEL_MODE"]

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


exports = {
    "MODEL_BASE_FLAGS": cfg.get("base_flags", ""),
    "MODEL_MODE_FLAGS": cfg.get(f"{mode}_flags", ""),
    "MODEL_PREFILL_FLAGS": prefill.get(mode, ""),
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

# =============================================================================
# Cluster Topology (dist-init endpoints)
# =============================================================================

IP_FIRST_PREFILL=$(echo "$IPADDRS" | awk -F',' '{print $2}')
IP_FIRST_DECODE=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+2)}')

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

DIST_INIT_PORT="${DIST_INIT_PORT:-5757}"
PREFILL_NNODES="${xP}"
DECODE_NNODES="${yD}"
PREFILL_DIST_INIT_ADDR="${IP_FIRST_PREFILL}:${DIST_INIT_PORT}"
DECODE_DIST_INIT_ADDR="${IP_FIRST_DECODE}:${DIST_INIT_PORT}"

if [[ "$DP_MODE" == "1" ]]; then
    _expected_ip_slots=$((1 + xP + yD))
    if [[ -z "$IP_FIRST_PREFILL" || -z "$IP_FIRST_DECODE" ]]; then
        echo "ERROR: DP_MODE=1 requires non-empty IP_FIRST_PREFILL and IP_FIRST_DECODE (from IPADDRS=${IPADDRS})" >&2
        exit 1
    fi
    if ((${#IP_ARRAY[@]} < _expected_ip_slots)); then
        echo "ERROR: DP_MODE=1 expects at least ${_expected_ip_slots} comma-separated hosts in IPADDRS (1 router + xP=${xP} prefill + yD=${yD} decode); got ${#IP_ARRAY[@]}" >&2
        exit 1
    fi
fi

PREFILL_ARGS=""
DECODE_ARGS=""

# Router (DP_MODE=0): one --prefill / --decode URL per worker (see sglang_disagg_server.sh).
for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+=" --prefill http://${IP_ARRAY[$i]}:3000"
done

for ((i=$xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+=" --decode http://${IP_ARRAY[$i]}:3000"
done

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
# NODE_RANK 1..xP: prefill workers start first; NODE_RANK xP+1..xP+yD: decode workers.
# NODE_RANK 0: DP_MODE=0 wait all worker logs; DP_MODE=1 wait master prefill/decode only; then router, benchmark_xPyD.sh, stop router.
# Workers then sync on MASTER_ADDR:2322 and wait for proxy shutdown.

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

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "${host_name}:${host_ip} is Router / proxy (NODE_RANK=0)"

    mkdir -p "/run_logs/${SLURM_JOB_ID:-0}"

    # DP_MODE=0: wait for SEARCH_SIGNAL in every prefill (NODE 1..xP) and decode (NODE xP+1 .. xP+yD) log.
    # DP_MODE=1: wait for master prefill NODE 1 + master decode NODE xP+1 only.
    # Requires shared /run_logs across nodes.
    SEARCH_SIGNAL="${SEARCH_SIGNAL:-The server is fired up and ready to roll!}"
    ROUTER_READY_TIMEOUT_SECONDS="${ROUTER_READY_TIMEOUT_SECONDS:-4000}"
    ROUTER_POLL_SLEEP_SECONDS="${ROUTER_POLL_SLEEP_SECONDS:-10}"
    SECONDS=0
    _runlog="/run_logs/${SLURM_JOB_ID:-0}"

    if [[ "$DP_MODE" == "0" ]]; then
        echo "Waiting for all ${xP} prefill + ${yD} decode servers (grep: ${SEARCH_SIGNAL}) before starting router..."
        for ((i = 1; i <= xP; i++)); do
            LOG_FILE="${_runlog}/prefill_NODE${i}.log"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if ((SECONDS >= ROUTER_READY_TIMEOUT_SECONDS)); then
                    echo "ERROR: Timeout waiting for prefill NODE${i} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
                SECONDS=$((SECONDS + ROUTER_POLL_SLEEP_SECONDS))
            done
            echo "Prefill NODE${i} ready."
        done
        for ((i = xP + 1; i <= xP + yD; i++)); do
            LOG_FILE="${_runlog}/decode_NODE${i}.log"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if ((SECONDS >= ROUTER_READY_TIMEOUT_SECONDS)); then
                    echo "ERROR: Timeout waiting for decode NODE${i} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
                SECONDS=$((SECONDS + ROUTER_POLL_SLEEP_SECONDS))
            done
            echo "Decode NODE${i} ready."
        done
    else
        _master_prefill_log="${_runlog}/prefill_NODE1.log"
        _master_decode_log="${_runlog}/decode_NODE$((xP + 1)).log"
        echo "Waiting for master prefill (NODE 1) + master decode (NODE $((xP + 1))) — grep: ${SEARCH_SIGNAL} — DP_MODE=${DP_MODE}"
        for _label_and_file in "master prefill|${_master_prefill_log}" "master decode|${_master_decode_log}"; do
            IFS='|' read -r _log_label LOG_FILE <<< "${_label_and_file}"
            until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                if ((SECONDS >= ROUTER_READY_TIMEOUT_SECONDS)); then
                    echo "ERROR: Timeout waiting for ${_log_label} (${LOG_FILE})" >&2
                    tail -n 40 "$LOG_FILE" 2>/dev/null || true
                    exit 1
                fi
                sleep "${ROUTER_POLL_SLEEP_SECONDS}"
                SECONDS=$((SECONDS + ROUTER_POLL_SLEEP_SECONDS))
            done
            echo "${_log_label} ready (${LOG_FILE})."
        done
    fi

    echo "Prefill/decode backends report ready per logs; starting sglang_router (proxy)."

    # Build and launch only after worker logs confirm servers are up (avoids the proxy probing backends too early).
    # 0.0.0.0 so bench_serving (--host 127.0.0.1 in benchmark_xPyD.sh) can connect; binding only ${host_ip} rejects loopback.
    if [[ "$DP_MODE" == "1" ]]; then
        ROUTER_CMD="python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill http://${IP_FIRST_PREFILL}:3000 \
            --decode http://${IP_FIRST_DECODE}:3000 \
            --host 0.0.0.0 \
            --port 2322"
    else
        ROUTER_CMD="python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            ${PREFILL_ARGS} \
            ${DECODE_ARGS} \
            --host 0.0.0.0 \
            --port 2322"
    fi
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

    sleep 90;

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

    echo "Killing the proxy server (pid=${proxy_pid})"
    kill "${proxy_pid}"

elif [[ "$NODE_RANK" -ge 1 && "$NODE_RANK" -le "$xP" ]]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-default})"
    PREFILL_NODE_RANK=$((NODE_RANK - 1))
    setup_sglang_worker_env
    #if [[ "$DP_MODE" == "0" ]]; then
    #    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    #    echo "DP_MODE=0 prefill SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
    #fi

    PREFILL_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mori \
        --load-balance-method round_robin \
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

    PREFILL_LOG="/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log"
    mkdir -p "$(dirname "$PREFILL_LOG")"
    {
        echo "========== PREFILL_CMD (NODE_RANK=${NODE_RANK}, PREFILL_NODE_RANK=${PREFILL_NODE_RANK}) =========="
        echo "$PREFILL_CMD"
        echo ""
    } | tee "$PREFILL_LOG"
    set -x
    eval "$PREFILL_CMD" 2>&1 | tee -a "$PREFILL_LOG" >/dev/null &
    set +x
    prefill_pid=$!

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

elif [[ "$NODE_RANK" -ge $((xP + 1)) && "$NODE_RANK" -le $((xP + yD)) ]]; then
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-default})"
    DECODE_NODE_RANK=$((NODE_RANK - xP - 1))
    setup_sglang_worker_env

    if [[ "$DP_MODE" == "1" ]]; then
        #export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
        #echo "DP_MODE=0 decode SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"
        export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    fi

    DECODE_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mori \
        --load-balance-method round_robin \
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
    set -x
    eval "$DECODE_CMD" 2>&1 | tee -a "$DECODE_LOG" >/dev/null &
    set +x
    decode_pid=$!

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
