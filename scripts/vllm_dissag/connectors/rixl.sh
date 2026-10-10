#!/bin/bash
# Connector profile: RIXL / NIXL (NixlConnector KV transfer).
# =============================================================================
# Sourced by vllm_disagg.sh. Connector hook contract:
#   connector_init, connector_setup_env, connector_runtime_patch,
#   connector_launch_worker, connector_wait_workers_ready, connector_start_proxy
#
# WIDE_EP=0 (rixl + TP):              byte-identical to legacy vllm_disagg_server.sh.
# WIDE_EP=1 (deepep, EP_BACKEND=deepep): byte-identical to legacy vllm_disagg_server_deepep.sh.
#
# Per-model FLAGS come from models.yaml via the driver ($MODEL_CONFIG_<ROLE>).
# Per-model ENV is exported by the driver (yaml env:) BEFORE connector_setup_env.
# =============================================================================

connector_init() {
    if parallelism_is_wide_ep; then
        # deepep ports
        SERVER_PORT=2584; SERVE_PORT="${SERVER_PORT}"
        RPC_PORT=13345; KV_PORT=14600
        CONTAINER_BARRIER_PORT="${BARRIER_PORT:-15000}"
        DP_SIZE_LOCAL=8
        PREFILL_DEEPEP_BACKEND="${PREFILL_DEEPEP_BACKEND:-deepep_high_throughput}"
        DECODE_DEEPEP_BACKEND="${DECODE_DEEPEP_BACKEND:-deepep_low_latency}"
        ENABLE_DBO="${ENABLE_DBO:-false}"
        DBO_COMM_SMS="${DBO_COMM_SMS:-}"
        ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
        DBO_ARGS=""; [[ "${ENABLE_DBO}" == "true" ]] && DBO_ARGS="--enable-dbo"
    else
        # rixl/TP ports
        SERVER_PORT=2584; SERVE_PORT="${SERVER_PORT}"
        KV_PORT=14600
        # Container-creation barrier port. Env-overridable (BARRIER_PORT) so it can
        # be moved off the collision-prone default 5000: the launcher's `fuser -k`
        # cleanup targets this port on the host (host networking), so a stale host
        # service on 5000 would otherwise be killed. Residual risk: the host-side
        # fuser still kills whatever holds this port for the launching user.
        CONTAINER_BARRIER_PORT="${BARRIER_PORT:-5000}"
    fi

    PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
    ROUTER_PORT="${ROUTER_PORT:-18001}"
    PROXY_PORT="${ROUTER_PORT}"
    if [[ "$PROXY_TYPE" != "vllm_router" && "$PROXY_TYPE" != "toy_proxy" ]]; then
        echo "Error: Invalid PROXY_TYPE='$PROXY_TYPE'. Must be 'vllm_router' or 'toy_proxy'." >&2
        exit 1
    fi
}

# connector_setup_env [ep_backend]
connector_setup_env() {
    local ep_backend="${1:-}"
    if parallelism_is_wide_ep; then
        _rixl_setup_env_deepep "${ep_backend}"
    else
        _rixl_setup_env_tp
    fi
}

# ---- TP env (was vllm_disagg_server.sh inline VAR= prefix; now exported) ----
_rixl_setup_env_tp() {
    export LD_LIBRARY_PATH="/app/install/nixl/lib/x86_64-linux-gnu/:/app/install/ucx/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    # Per-model envs (yaml env:) are already exported by the driver.
    if [[ -z "${UCX_NET_DEVICES:-}" ]]; then
        UCX_NET_DEVICES=$(ibstat 2>/dev/null | awk '
            /^CA /{gsub(/\047/,"",$2); ca=$2}
            /Rate:/{if($2+0 >= 200) devs=devs (devs?",":"") ca":1"}
            END{print devs}')
        export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
    fi
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export VLLM_USE_V1=1
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${host_ip}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557
    export UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp
    export UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp
    export UCX_SOCKADDR_CM_ENABLE=y
    export UCX_RDMA_CM_ENABLED=y
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export NIXL_LOG_LEVEL=DEBUG
}

# ---- deepep env (was setup_deepep_env) ----
_rixl_setup_env_deepep() {
    local backend="$1"
    export ROCSHMEM_HEAP_SIZE=7524589824
    export ROCSHMEM_MAX_NUM_CONTEXTS=256
    export HSA_NO_SCRATCH_RECLAIM=1

    if [[ -z "${ROCSHMEM_DIR:-}" ]]; then
        for _d in /root/rocshmem /opt/rocshmem; do
            [[ -d "$_d/lib" ]] && export ROCSHMEM_DIR="$_d" && break
        done
        export ROCSHMEM_DIR="${ROCSHMEM_DIR:-/root/rocshmem}"
    fi
    if [[ -z "${OMPI_DIR:-}" ]]; then
        for _d in /root/install/ompi /usr/lib/x86_64-linux-gnu/openmpi /opt/ompi; do
            [[ -d "$_d" ]] && export OMPI_DIR="$_d" && break
        done
        export OMPI_DIR="${OMPI_DIR:-/root/install/ompi}"
    fi
    local _ucx_lib=""
    for _d in /root/install/ucx/lib /usr/local/ucx/lib /opt/ucx/lib; do
        [[ -f "$_d/libucp.so" ]] && _ucx_lib="$_d" && break
    done
    : "${_ucx_lib:=/root/install/ucx/lib}"
    local _nixl_lib=""
    for _d in /opt/nixl/lib \
              /usr/local/RIXL/install/lib/x86_64-linux-gnu \
              /usr/local/lib/python3.12/dist-packages/.rixl.mesonpy.libs \
              /usr/local/nixl/lib; do
        [[ -f "$_d/libnixl.so" && -d "$_d/plugins" ]] && _nixl_lib="$_d" && break
    done
    if [[ -z "$_nixl_lib" ]]; then
        for _d in /opt/nixl/lib \
                  /usr/local/RIXL/install/lib/x86_64-linux-gnu \
                  /usr/local/lib/python3.12/dist-packages/.rixl.mesonpy.libs \
                  /root/RIXL/build/src/core \
                  /usr/local/nixl/lib; do
            [[ -f "$_d/libnixl.so" ]] && _nixl_lib="$_d" && break
        done
    fi
    if ! python3 -c "import nixl" 2>/dev/null; then
        local _rixl_py
        _rixl_py=$(find /root/RIXL/build -path "*/bindings/python" -type d 2>/dev/null | head -1)
        if [[ -n "$_rixl_py" ]]; then
            echo "[setup_deepep_env] NIXL Python bindings missing, installing from $_rixl_py ..."
            pip install --no-deps -e "$_rixl_py" 2>/dev/null || true
        fi
    fi
    local _extra_ld=""
    [[ -n "$_nixl_lib" ]] && _extra_ld+="${_nixl_lib}:"
    _extra_ld+="${_ucx_lib}:/usr/local/lib:/usr/local/lib64:/opt/rocm/lib"
    export LD_LIBRARY_PATH="${_extra_ld}:${LD_LIBRARY_PATH}"
    echo "[setup_deepep_env] ROCSHMEM_DIR=$ROCSHMEM_DIR  OMPI_DIR=$OMPI_DIR"
    echo "[setup_deepep_env] UCX_LIB=$_ucx_lib  NIXL_LIB=${_nixl_lib:-<not found>}"

    export VLLM_USE_V1=1
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_ALL2ALL_BACKEND="${backend}"
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    export VLLM_ROCM_USE_AITER_RMSNORM=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_USE_AITER_TRITON_SILU_MUL=0
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_ROCM_USE_AITER_MOE=0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${host_ip}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557
    export GLOO_SOCKET_IFNAME=eth0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=1
    export NCCL_NET_GDR_LEVEL=PHB
    export UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp
    if [[ -z "${UCX_NET_DEVICES:-}" ]]; then
        local available_devs
        available_devs=$(ibstat 2>/dev/null | awk '
            /^CA /{gsub(/\047/,"",$2); ca=$2}
            /Rate:/{if($2+0 >= 200) devs=devs (devs?",":"") ca":1"}
            END{print devs}')
        export UCX_NET_DEVICES="${available_devs:-mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
    fi
    if [[ -z "${NCCL_IB_HCA:-}" ]]; then
        local nccl_hcas
        nccl_hcas=$(ibstat 2>/dev/null | awk '
            /^CA /{gsub(/\047/,"",$2); ca=$2}
            /Rate:/{if($2+0 >= 200) devs=devs (devs?",":"") ca}
            END{print devs}')
        export NCCL_IB_HCA="${nccl_hcas:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    fi
    export UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp
    export UCX_SOCKADDR_CM_ENABLE=y
    export UCX_RDMA_CM_ENABLED=y
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-INFO}"
    [[ -n "${DBO_COMM_SMS}" ]] && export VLLM_DBO_COMM_SMS="${DBO_COMM_SMS}"

    # PR #39276 inline engine_id fix
    local _core_py
    _core_py=$(python3 -c "import vllm.v1.engine.core as m; print(m.__file__)" 2>/dev/null)
    if [[ -n "$_core_py" && -f "$_core_py" ]]; then
        if grep -q '_dp{local_dp_rank}' "$_core_py"; then
            sed -i 's/_dp{local_dp_rank}/_dp{dp_rank}/g' "$_core_py"
            echo "[setup_deepep_env] Applied PR#39276 fix: engine_id uses dp_rank (core.py)"
        fi
    fi
    local _utils_py
    _utils_py=$(python3 -c "import vllm.v1.engine.utils as m; print(m.__file__)" 2>/dev/null)
    if [[ -n "$_utils_py" && -f "$_utils_py" ]]; then
        if grep -q '_dp{local_index}' "$_utils_py"; then
            sed -i 's/_dp{local_index}/_dp{index}/g' "$_utils_py"
            echo "[setup_deepep_env] Applied PR#39276 fix: engine_id uses index (utils.py)"
        fi
    fi
    echo "[setup_deepep_env] UCX_NET_DEVICES=$UCX_NET_DEVICES"
    echo "[setup_deepep_env] NCCL_IB_HCA=$NCCL_IB_HCA"
}

_rixl_kv_config_tp() {  # $1 kv_role  $2 engine_id
    echo "{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"$2\", \"kv_role\": \"$1\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"${host_ip}\", \"kv_port\": ${KV_PORT}}"
}
_rixl_kv_config_deepep() {  # $1 kv_role  $2 engine_id  $3 dp_size
    echo "{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"$2\", \"kv_role\": \"$1\", \"kv_parallel_size\": $3, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"${host_ip}\", \"kv_port\": ${KV_PORT}}"
}

connector_runtime_patch() {
    # deepep applies its PR#39276 fix inline in setup_env; TP path needs nothing.
    :
}

# connector_launch_worker <role> <dp_size> <dp_addr> <kv_role> <log_prefix> [dp_start_rank]
connector_launch_worker() {
    local role="$1" dp_size="$2" dp_addr="$3" kv_role="$4" log_prefix="$5" dp_start_rank="${6:-}"

    if parallelism_is_wide_ep; then
        _rixl_launch_deepep "$role" "$dp_size" "$dp_addr" "$kv_role" "$log_prefix" "$dp_start_rank"
    else
        _rixl_launch_tp "$role" "$dp_size" "$dp_addr" "$kv_role" "$log_prefix" "$dp_start_rank"
    fi
}

# ---- rixl + TP (byte-identical to vllm_disagg_server.sh) ----
_rixl_launch_tp() {
    local role="$1" dp_size="$2" dp_addr="$3" kv_role="$4" log_prefix="$5"
    connector_setup_env

    local engine_id
    if [[ "$log_prefix" == "prefill" ]]; then engine_id="pd-run"; else engine_id="pd-decode"; fi
    local kv_config; kv_config=$(_rixl_kv_config_tp "${kv_role}" "${engine_id}")

    # Per-model config string (from models.yaml; tokenized as the legacy eval did)
    local cfg_args=()
    local _mc; if [[ "$log_prefix" == "prefill" ]]; then _mc="${MODEL_CONFIG_PREFILL:-}"; else _mc="${MODEL_CONFIG_DECODE:-}"; fi
    [[ -n "$_mc" ]] && eval "cfg_args=(${_mc})"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        _dryrun_emit "rixl" "${log_prefix}" "${role}" \
            vllm serve "${MODEL_PATH}" \
                --port "${SERVER_PORT}" \
                --trust-remote-code \
                --kv-transfer-config "${kv_config}" \
                "${cfg_args[@]}"
        WORKER_PID=0; return 0
    fi

    vllm serve "${MODEL_PATH}" \
        --port "${SERVER_PORT}" \
        --trust-remote-code \
        --kv-transfer-config "${kv_config}" \
        "${cfg_args[@]}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
    WORKER_PID=$!
}

# ---- rixl + wideEP=deepep (byte-identical to vllm_disagg_server_deepep.sh) ----
_rixl_launch_deepep() {
    local role="$1" dp_size="$2" dp_addr="$3" kv_role="$4" log_prefix="$5" dp_start_rank="${6:-}"

    local backend engine_id
    if [[ "$log_prefix" == "prefill" ]]; then
        backend="${PREFILL_DEEPEP_BACKEND}"; engine_id="pd-prefill"
    else
        backend="${DECODE_DEEPEP_BACKEND}"; engine_id="pd-decode"
    fi

    connector_setup_env "${backend}"

    # Agentic gating: the default sweep keeps prefix caching OFF via the hardcoded
    # --no-enable-prefix-caching below. The agentic trace-replay path
    # (BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh) — or ENABLE_PREFIX_CACHE=1 —
    # STRIPS that flag so prefix caching is ON. Gated so the default (non-agentic)
    # sweep argv is byte-for-byte unchanged.
    local _prefix_cache_flag="--no-enable-prefix-caching"
    if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_PREFIX_CACHE:-0}" == "1" ]]; then
        _prefix_cache_flag=""
    fi

    local extra_args=()
    if [[ "$role" == "master" ]]; then
        extra_args+=(--api-server-count=8 --data-parallel-start-rank 0)
    else
        extra_args+=(--data-parallel-start-rank "${dp_start_rank}" --headless)
    fi

    local compile_args=()
    if [[ "$log_prefix" == "decode" ]]; then
        compile_args+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}')
        compile_args+=(--cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256)
    else
        compile_args+=(--enforce-eager)
    fi

    local kv_config; kv_config=$(_rixl_kv_config_deepep "${kv_role}" "${engine_id}" "${dp_size}")

    # Per-model dp: flags from models.yaml (driver-exported). Empty for the DeepSeek
    # deepep entries today, so this is currently a no-op; kept so future per-model
    # dp: tuning is actually honored on the deepep path.
    local model_args=()
    local _mc; if [[ "$log_prefix" == "prefill" ]]; then _mc="${MODEL_CONFIG_PREFILL:-}"; else _mc="${MODEL_CONFIG_DECODE:-}"; fi
    [[ -n "$_mc" ]] && eval "model_args=(${_mc})"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        _dryrun_emit "deepep" "${log_prefix}" "${role}" \
            vllm serve "${MODEL_PATH}" \
                --port "${SERVER_PORT}" \
                --trust-remote-code \
                -tp 1 \
                --data-parallel-size "${dp_size}" \
                --data-parallel-size-local "${DP_SIZE_LOCAL}" \
                --data-parallel-address "${dp_addr}" \
                --data-parallel-rpc-port "${RPC_PORT}" \
                --master-addr "${dp_addr}" \
                "${compile_args[@]}" \
                ${_prefix_cache_flag} --block-size 1 \
                --gpu-memory-utilization 0.8 \
                --kv-cache-dtype fp8 \
                --enable-expert-parallel \
                --all2all-backend "${backend}" \
                ${DBO_ARGS} \
                "${extra_args[@]}" \
                "${model_args[@]}" \
                --kv-transfer-config "${kv_config}"
        WORKER_PID=0; return 0
    fi

    vllm serve "${MODEL_PATH}" \
        --port "${SERVER_PORT}" \
        --trust-remote-code \
        -tp 1 \
        --data-parallel-size "${dp_size}" \
        --data-parallel-size-local "${DP_SIZE_LOCAL}" \
        --data-parallel-address "${dp_addr}" \
        --data-parallel-rpc-port "${RPC_PORT}" \
        --master-addr "${dp_addr}" \
        "${compile_args[@]}" \
        ${_prefix_cache_flag} --block-size 1 \
        --gpu-memory-utilization 0.8 \
        --kv-cache-dtype fp8 \
        --enable-expert-parallel \
        --all2all-backend "${backend}" \
        ${DBO_ARGS} \
        "${extra_args[@]}" \
        "${model_args[@]}" \
        --kv-transfer-config "${kv_config}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
    WORKER_PID=$!
}

connector_wait_workers_ready() {
    if parallelism_is_wide_ep; then
        echo "Waiting for prefill & decode master servers to start..."
        local TIMEOUT_SECONDS=4000 SLEEP_SECONDS=10 SEARCH_SIGNAL="Application startup complete."
        _wait_log_signal_or_fail "/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log" "prefill master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
        _wait_log_signal_or_fail "/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log" "decode master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
    else
        echo "Waiting for all prefill and decode servers to be up . . ."
        python $NIXL_COOKBOOK_PATH/socket_barrier.py --node-ips ${IPADDRS} --node-ports $SERVER_PORT
    fi
}

connector_start_proxy() {
    local PREFILL_ARGS="" DECODE_ARGS="" PREFILL_PORTS="" DECODE_PORTS="" i
    # Agentic replay: point aiperf at the backend vLLM servers' /metrics (SERVER_PORT)
    # for prefill+decode masters. Gated on the agentic path so the default sweep is
    # unaffected. Consumed by scripts/common/agentic_lib.sh (aiperf --server-metrics).
    if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_SERVER_METRICS:-0}" == "1" ]]; then
        export AGENTIC_SERVER_METRICS="${AGENTIC_SERVER_METRICS:-${PREFILL_MASTER_ADDR}:${SERVER_PORT} ${DECODE_MASTER_ADDR}:${SERVER_PORT}}"
        echo "[metrics] AGENTIC_SERVER_METRICS=${AGENTIC_SERVER_METRICS}"
    fi
    for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do PREFILL_ARGS+="${IP_ARRAY[$i]} "; PREFILL_PORTS+="$SERVER_PORT "; done
    for ((i=xP; i<${#IP_ARRAY[@]}; i++)); do DECODE_ARGS+="${IP_ARRAY[$i]} "; DECODE_PORTS+="$SERVER_PORT "; done

    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        echo "Starting vLLM Router (Production Proxy) on port ${PROXY_PORT}..."
        [ -f /root/.cargo/env ] && source /root/.cargo/env
        # Resolve the router binary: ROUTER_BINARY (site override, e.g. a shared-FS
        # build) wins, else the one on PATH. Images that don't ship a router (e.g.
        # the mori121 runtime image) require ROUTER_BINARY to be set.
        local ROUTER_BIN="${ROUTER_BINARY:-$(command -v vllm-router 2>/dev/null || true)}"
        if [ -z "${ROUTER_BIN}" ] || [ ! -x "${ROUTER_BIN}" ]; then
            echo "Error: vllm-router not found. Set ROUTER_BINARY=<path>, or PROXY_TYPE=toy_proxy." \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            exit 1
        fi
        echo "Using vllm-router binary: ${ROUTER_BIN}"
        local PREFILL_URLS="" DECODE_URLS="" ip
        for ip in ${PREFILL_ARGS}; do PREFILL_URLS+="--prefill http://${ip}:${SERVER_PORT} "; done
        for ip in ${DECODE_ARGS};  do DECODE_URLS+="--decode http://${ip}:${SERVER_PORT} "; done
        # deepep uses a trimmed router invocation (no retry/prometheus); TP uses the full one.
        if parallelism_is_wide_ep; then
            UCX_TLS=tcp,self,shm VLLM_USE_V1=1 \
            "${ROUTER_BIN}" --host 0.0.0.0 --port "${ROUTER_PORT}" --vllm-pd-disaggregation \
                $PREFILL_URLS $DECODE_URLS \
                --policy round_robin --prefill-policy round_robin --decode-policy round_robin \
                --intra-node-data-parallel-size 1 \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null &
        else
            UCX_TLS=tcp,self,shm VLLM_USE_V1=1 \
            "${ROUTER_BIN}" --host 0.0.0.0 --port $ROUTER_PORT --vllm-pd-disaggregation \
                $PREFILL_URLS $DECODE_URLS \
                --policy round_robin --prefill-policy round_robin --decode-policy round_robin \
                --intra-node-data-parallel-size 1 --retry-max-retries 3 --prometheus-port 29000 \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null &
        fi
        proxy_pid=$!
    else
        echo "Starting Toy Proxy Server on port ${PROXY_PORT}..."
        UCX_TLS=tcp,self,shm NCCL_UCX_TLS=tcp VLLM_USE_V1=1 \
        python3 "/app/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
                --host 0.0.0.0 --port $PROXY_PORT \
                --prefiller-hosts ${PREFILL_ARGS} --prefiller-ports ${PREFILL_PORTS} \
                --decoder-hosts ${DECODE_ARGS} --decoder-ports ${DECODE_PORTS} \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
    fi

    echo "Waiting for proxy server to be up . . ."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py --node-ips ${host_ip} --node-ports $PROXY_PORT
    echo "Proxy Server ($PROXY_TYPE) Ready for benchmarking on ${host_name}:${host_ip}:${PROXY_PORT}"
    sleep 10
}
