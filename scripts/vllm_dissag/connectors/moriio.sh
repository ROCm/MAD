#!/bin/bash
# Connector profile: MoRIIO (MoRIIOConnector KV transfer).
# =============================================================================
# Sourced by vllm_disagg.sh. Connector hook contract:
#   connector_init, connector_setup_env, connector_runtime_patch,
#   connector_launch_worker, connector_wait_workers_ready, connector_start_proxy
#
# WIDE_EP=1 (MoriEP wideEP): byte-identical to legacy vllm_disagg_mori_ep.sh.
# WIDE_EP=0 (moriio + TP):   NEW cell, no legacy precedent (Stage B).
#
# Per-model FLAGS come from models.yaml via the driver ($MODEL_CONFIG_<ROLE>).
# Per-model ENV is exported by the driver (yaml env:) BEFORE connector_setup_env,
# so the ${VAR:-default} fallbacks below yield to any model/site override.
# =============================================================================

connector_init() {
    RPC_PORT="${MORI_RPC_PORT:-13345}"
    SERVE_PORT="${MORI_SERVE_PORT:-20005}"
    KV_PORT="${MORI_KV_PORT:-9711}"
    PROXY_PORT="${MORI_PROXY_PORT:-10001}"
    PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}"
    LOCAL_PING_PORT="${MORI_LOCAL_PING_PORT:-61555}"
    HANDSHAKE_PORT="${MORI_HANDSHAKE_PORT:-8405}"
    NOTIFY_PORT="${MORI_NOTIFY_PORT:-61005}"
    CONTAINER_BARRIER_PORT="${BARRIER_PORT_MORI:-2222}"

    # Proxy: vllm_router (default) or moriio_toy. vllm_router carries the DP-rank
    # KV-notify dpfix REQUIRED for wideEP DP — the toy proxy can't route the notify
    # and decode hangs ("remote blocks never arrived"). Default to vllm_router.
    PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
    if [ "$PROXY_TYPE" != "vllm_router" ] && [ "$PROXY_TYPE" != "moriio_toy" ]; then
        echo "Error: invalid PROXY_TYPE='${PROXY_TYPE}' (expected 'vllm_router' or 'moriio_toy')." >&2
        exit 1
    fi
    ROUTER_PORT="${ROUTER_PORT:-${VLLM_ROUTER_HTTP_PORT:-30000}}"
    [ "$PROXY_TYPE" == "vllm_router" ] && PROXY_PORT="${ROUTER_PORT}"

    # Per-role MoRI all2all backend (wideEP only). Newer vLLM images (the v1.2.0
    # MoRI-EP image) split the kernel: prefill=high_throughput (InterNodeV1),
    # decode=low_latency (InterNodeV1LL) and REJECT the bare "mori" alias. Default
    # to the per-role names; override via PREFILL_MORI_BACKEND/DECODE_MORI_BACKEND
    # (or VLLM_ALL2ALL_BACKEND for the prefill side).
    PREFILL_MORI_BACKEND="${PREFILL_MORI_BACKEND:-${VLLM_ALL2ALL_BACKEND:-mori_high_throughput}}"
    DECODE_MORI_BACKEND="${DECODE_MORI_BACKEND:-mori_low_latency}"
}

# connector_setup_env [ep_backend]  (ep_backend only meaningful when WIDE_EP=1; mori path uses "mori")
connector_setup_env() {
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    # MLA default on, but DeepSeek-V3 needs it OFF (block=16 + Triton MLA) to avoid
    # the fp8 decode-MLA kernel GPU-fault; respect an override from env/models.yaml.
    export VLLM_ROCM_USE_AITER_MLA="${VLLM_ROCM_USE_AITER_MLA:-1}"
    export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    export VLLM_USE_AITER_TRITON_SILU_MUL=0

    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ALL2ALL_BACKEND=mori

    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"

    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"
    export VLLM_RINGBUFFER_WARNING_INTERVAL="${VLLM_RINGBUFFER_WARNING_INTERVAL:-3600}"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300000}"

    export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-3}"
    export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    # MoRI RDMA QoS — TC=41 (RoCE DSCP for lossless RDMA), SL=0. Override per-cluster.
    export MORI_RDMA_TC="${MORI_RDMA_TC:-41}"
    export MORI_RDMA_SL="${MORI_RDMA_SL:-0}"

    export MORI_NUM_QP_PER_PE="${MORI_NUM_QP_PER_PE:-4}"
    export VLLM_MORIIO_QP_PER_TRANSFER="${VLLM_MORIIO_QP_PER_TRANSFER:-4}"
    export VLLM_MORIIO_NUM_WORKERS="${VLLM_MORIIO_NUM_WORKERS:-4}"

    export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"
    export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-1800}"
    export VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-30}"

    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vllm_cache/triton}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache/vllm}"
    export COMGR_CACHE_DIR="${COMGR_CACHE_DIR:-/tmp/vllm_cache/comgr}"
    export AITER_JIT_DIR="${AITER_JIT_DIR:-/tmp/vllm_cache/aiter_jit}"
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${COMGR_CACHE_DIR}" "${AITER_JIT_DIR}" 2>/dev/null || true

    if [[ "${VLLM_ROCM_USE_AITER:-1}" == "1" ]]; then
        local _aiter_cfgs="/tmp/aiter_configs"
        local _aiter_src="/usr/local/lib/python3.12/dist-packages/aiter/configs"
        if [ -d "${_aiter_src}" ] && [ ! -f "${_aiter_cfgs}/a8w8_blockscale_tuned_gemm.csv" ]; then
            mkdir -p "${_aiter_cfgs}"
            cp "${_aiter_src}"/*.csv "${_aiter_cfgs}/" 2>/dev/null || true
        fi
    fi

    export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
    export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
    export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
    export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

    # NOTE: the ROCm-7.2.3 platform env that MUST reach the container at PID 1
    # (PYTORCH_ALLOC_CONF/PYTORCH_HIP_ALLOC_CONF=expandable_segments:False,
    # HSA_ENABLE_IPC_MODE_LEGACY=0, MORI_GPU_ARCHS) lives in connectors/moriio.env
    # and is forwarded by the slurm via `docker -e`. It is NOT exported here — a late
    # shell export is too late (PyTorch reads alloc-conf at import, before setup_env).

    export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-8589934592}"
    export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-256}"
    # MoRI shmem heap: 4 GiB default too small for EP>=32; 16 GiB (matches #324).
    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"
}

_moriio_build_kv_transfer_config() {
    local kv_role="$1"
    echo '{"kv_connector":"MoRIIOConnector","kv_role":"'"${kv_role}"'","kv_port":"'"${KV_PORT}"'","kv_connector_extra_config":{"proxy_ip":"'"${MASTER_ADDR}"'","proxy_port":"'"${PROXY_PORT}"'","proxy_ping_port":"'"${PROXY_PING_PORT}"'","http_port":"'"${SERVE_PORT}"'","local_ping_port":"'"${LOCAL_PING_PORT}"'","handshake_port":"'"${HANDSHAKE_PORT}"'","notify_port":"'"${NOTIFY_PORT}"'"}}'
}

connector_runtime_patch() {
    # No-op: the MoRIIO multi-node disagg fixes (vLLM PR#39276 notify-path, #41751 LL
    # split, DP-rank hash-failsafe) are committed in-source in the vLLM the image is
    # built from (see the Dockerfile VLLM_REF). There is no runtime .py patcher — that
    # would be a drifting duplicate of fixes that already live upstream in the fork.
    # If you ever run an image WITHOUT these fixes baked, use an image that has them
    # (rebuild from the pinned VLLM_REF) rather than patching a stock image at runtime.
    return 0
}

# connector_launch_worker <role> <dp_size> <dp_addr> <kv_role> <log_prefix> [dp_start_rank]
connector_launch_worker() {
    local role="$1" dp_size="$2" dp_addr="$3" kv_role="$4" log_prefix="$5" dp_start_rank="${6:-}"

    connector_setup_env "${EP_BACKEND:-mori}"

    # Patch PyTorch default_pg_timeout (DP Gloo groups) — wideEP only.
    if parallelism_is_wide_ep; then
        local _timeout_s="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
        local _torch_const="/usr/local/lib/python3.12/dist-packages/torch/distributed/constants.py"
        if [ -f "$_torch_const" ]; then
            sed -i "s/default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT/default_pg_timeout: timedelta = timedelta(seconds=${_timeout_s})/" "$_torch_const" 2>/dev/null || true
        fi
    fi

    # Per-role execution mode. Ported from #324: NEVER use bare --enforce-eager.
    # On these AITER images an enforce-eager worker (no +quant_fp8 custom op) routes
    # fp8 quant through an AITER op whose signature mismatches the build
    # (dynamic_per_token_scaled_quant: out aiter_tensor_t) -> engine-init crash.
    # So even "no cudagraph" is expressed as cudagraph_mode:NONE WITH +quant_fp8.
    # Per-role mode: DECODE_CUDAGRAPH_MODE / PREFILL_CUDAGRAPH_MODE, falling back to
    # the global VLLM_CUDAGRAPH_MODE (back-compat).
    local exec_args=()
    local _cudagraph_mode="${VLLM_CUDAGRAPH_MODE:-}"
    if [[ "$log_prefix" == "decode" ]]; then
        _cudagraph_mode="${DECODE_CUDAGRAPH_MODE:-$_cudagraph_mode}"
    else
        _cudagraph_mode="${PREFILL_CUDAGRAPH_MODE:-$_cudagraph_mode}"
    fi
    if [[ -n "$_cudagraph_mode" && "$_cudagraph_mode" != "NONE" ]]; then
        local _capture_sizes="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8 16 32 64 128 256}"
        exec_args+=(--compilation-config '{"cudagraph_mode":"'"${_cudagraph_mode}"'","custom_ops":["+quant_fp8"]}')
        exec_args+=(--cudagraph-capture-sizes ${_capture_sizes})
    else
        exec_args+=(--compilation-config '{"cudagraph_mode":"NONE","custom_ops":["+quant_fp8"]}')
    fi

    # Per-model flags from models.yaml (driver-exported; empty if none).
    local model_args=()
    local _mc; if [[ "$log_prefix" == "prefill" ]]; then _mc="${MODEL_CONFIG_PREFILL:-}"; else _mc="${MODEL_CONFIG_DECODE:-}"; fi
    [[ -n "$_mc" ]] && eval "model_args=(${_mc})"

    if parallelism_is_wide_ep; then
        # ---- WIDE_EP=1 (MoriEP) ----
        # Per-role all2all: prefill=high_throughput, decode=low_latency. The
        # v1.2.0 image rejects the bare "mori" alias; these names are required.
        local _all2all="${PREFILL_MORI_BACKEND}"
        [[ "$log_prefix" == "decode" ]] && _all2all="${DECODE_MORI_BACKEND}"

        local extra_args=() kv_args=()
        if [[ "$role" == "master" ]]; then
            extra_args+=(--api-server-count=${_GPUS_PER_NODE})
            local kv_config; kv_config=$(_moriio_build_kv_transfer_config "${kv_role}")
            kv_args+=(--kv-transfer-config "${kv_config}")
        else
            extra_args+=(--data-parallel-start-rank "${dp_start_rank}" --headless)
        fi

        # Recipe knobs (overridable via env / models.yaml). DeepSeek-V3 on AITER
        # needs block=16 + MLA off (the block=1 + AITER-MLA fp8 decode kernel
        # GPU-faults), and KV_CACHE_MEMORY_BYTES to SKIP the boot profiling forward
        # (that forward is the only path that calls AITER's eager per-token fp8
        # quant, which has the aiter_tensor_t torch_guard bug -> engine-init crash).
        local _block="${KV_BLOCK_SIZE:-1}"
        local _kvdtype="${KV_CACHE_DTYPE:-fp8}"
        local mem_args=()
        [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]] && mem_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
        # Bound the attention workspace: without --max-model-len vLLM sizes it for the
        # model's full context (DeepSeek-V3 = 163840), producing a ~128 GiB workspace
        # alloc that OOMs alongside the EP-sharded weights. MAX_MODEL_LEN (models.yaml)
        # caps it to what the workload needs.
        local mml_args=()
        [[ -n "${MAX_MODEL_LEN:-}" ]] && mml_args+=(--max-model-len "${MAX_MODEL_LEN}")

        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            _dryrun_emit "moriio" "${log_prefix}" "${role}" \
                vllm serve "${MODEL_PATH}" \
                    -tp 1 \
                    --data-parallel-size "${dp_size}" \
                    --data-parallel-size-local "${DP_PARALLEL_SIZE_LOCAL}" \
                    --data-parallel-address "${dp_addr}" \
                    --data-parallel-rpc-port "${RPC_PORT}" \
                    --enable-expert-parallel \
                    --port "${SERVE_PORT}" \
                    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
                    "${mem_args[@]}" \
                    "${mml_args[@]}" \
                    --kv-cache-dtype "${_kvdtype}" \
                    --block-size "${_block}" \
                    --no-enable-prefix-caching \
                    --all2all-backend "${_all2all}" \
                    --trust-remote-code \
                    --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
                    "${exec_args[@]}" "${extra_args[@]}" "${kv_args[@]}"
            WORKER_PID=0; return 0
        fi

        vllm serve ${MODEL_PATH} \
            -tp 1 \
            --data-parallel-size "${dp_size}" \
            --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
            --data-parallel-address "${dp_addr}" \
            --data-parallel-rpc-port ${RPC_PORT} \
            --enable-expert-parallel \
            --port ${SERVE_PORT} \
            --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.8} \
            "${mem_args[@]}" \
            "${mml_args[@]}" \
            --kv-cache-dtype "${_kvdtype}" \
            --block-size "${_block}" \
            --no-enable-prefix-caching \
            --all2all-backend "${_all2all}" \
            --trust-remote-code \
            --distributed-timeout-seconds ${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
            "${exec_args[@]}" \
            "${extra_args[@]}" \
            "${kv_args[@]}" \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
        WORKER_PID=$!
        return 0
    fi

    # ---- WIDE_EP=0 (moriio + TP) — NEW cell (Stage B) ----
    # MoRIIO KV transfer over a plain tensor-parallel server (no EP). Every node
    # is a full server; the kv-transfer-config is attached on all nodes (no DP
    # master/child split). kv-cache-dtype/block-size come from models.yaml.
    local kv_config; kv_config=$(_moriio_build_kv_transfer_config "${kv_role}")
    local _tp_size="${IO_TP_SIZE:-${GENERIC_TP_SIZE:-8}}"

    # The connector owns the parallelism *degree* (--tensor-parallel-size) for the
    # moriio+TP path, so strip any --tensor-parallel-size N that a models.yaml entry
    # may carry (those entries are shared with the rixl path, which DOES want it in
    # yaml). Prevents a duplicate --tensor-parallel-size on the moriio+TP command.
    local _filtered=() _skip=0 _a
    for _a in "${model_args[@]}"; do
        if [[ "$_skip" == "1" ]]; then _skip=0; continue; fi
        if [[ "$_a" == "--tensor-parallel-size" ]]; then _skip=1; continue; fi
        if [[ "$_a" == "--tensor-parallel-size="* ]]; then continue; fi
        _filtered+=("$_a")
    done
    model_args=("${_filtered[@]}")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        _dryrun_emit "moriio" "${log_prefix}" "${role}" \
            vllm serve "${MODEL_PATH}" \
                --tensor-parallel-size "${_tp_size}" \
                --port "${SERVE_PORT}" \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
                --trust-remote-code \
                --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
                --kv-transfer-config "${kv_config}" \
                "${exec_args[@]}" "${model_args[@]}"
        WORKER_PID=0; return 0
    fi

    vllm serve ${MODEL_PATH} \
        --tensor-parallel-size "${_tp_size}" \
        --port ${SERVE_PORT} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.8} \
        --trust-remote-code \
        --distributed-timeout-seconds ${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
        --kv-transfer-config "${kv_config}" \
        "${exec_args[@]}" \
        "${model_args[@]}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
    WORKER_PID=$!
}

connector_wait_workers_ready() {
    echo "Waiting for prefill & decode servers to be ready..."
    sleep 20
    local TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"
    local SLEEP_SECONDS=10
    local SEARCH_SIGNAL="Application startup complete."
    local PREFILL_LOG=/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log
    local DECODE_LOG=/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log
    _wait_log_signal_or_fail "${PREFILL_LOG}" "prefill master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
    _wait_log_signal_or_fail "${DECODE_LOG}" "decode master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
}

connector_start_proxy() {
    # Ported faithfully from the validated MAD-private PR#324 mori launcher.
    # vllm_router: production router with --kv-connector moriio (needs the binary on
    #   PATH or ROUTER_BINARY set); includes a registration gate so the benchmark
    #   doesn't fire before prefill+decode register (else every request 503s).
    # moriio_toy: in-image toy proxy; resolves the script across the online_serving/
    #   -> disaggregated/ path move.
    # Sets BENCHMARK_PORT (router->ROUTER_PORT, toy->PROXY_PORT) for the driver.
    sleep 10
    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        local PREFILL_URL="http://${PREFILL_MASTER_ADDR}:${SERVE_PORT}"
        local DECODE_URL="http://${DECODE_MASTER_ADDR}:${SERVE_PORT}"
        # Router intra-node DP size = per-node DP rank count. wideEP has
        # DP_PARALLEL_SIZE_LOCAL(=GPUS_PER_NODE) ranks/node; TP (WIDE_EP=0) is a
        # single unit -> 1. Passing 8 on the TP path makes the router round-robin
        # to DP ranks 0..7 while the TP server only has rank 0 -> every non-rank-0
        # request fails "data_parallel_rank N out of range [0,1)" (7/8 -> 500).
        local _router_dp_local="${DP_PARALLEL_SIZE_LOCAL}"
        parallelism_is_wide_ep || _router_dp_local=1
        echo "Starting vllm-router (MoRIIO): HTTP ${ROUTER_PORT}"
        echo "  prefill=${PREFILL_URL}  decode=${DECODE_URL}  dp_local=${_router_dp_local}"
        [ -f /root/.cargo/env ] && source /root/.cargo/env

        local ROUTER_BIN="${ROUTER_BINARY:-$(command -v vllm-router 2>/dev/null || true)}"
        if [ -z "${ROUTER_BIN}" ] || [ ! -x "${ROUTER_BIN}" ]; then
            echo "Error: vllm-router not found. Set ROUTER_BINARY=<path>, or PROXY_TYPE=moriio_toy to use the in-image toy proxy." \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            exit 1
        fi
        echo "Using vllm-router binary: ${ROUTER_BIN}"
        local _PROMETHEUS_PORT="${VLLM_ROUTER_PROMETHEUS_PORT:-29000}"
        "${ROUTER_BIN}" \
            --host 0.0.0.0 \
            --port "${ROUTER_PORT}" \
            --vllm-pd-disaggregation \
            --kv-connector moriio \
            --prefill "${PREFILL_URL}" \
            --decode "${DECODE_URL}" \
            --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
            --intra-node-data-parallel-size "${_router_dp_local}" \
            --policy round_robin \
            --prefill-policy round_robin \
            --decode-policy round_robin \
            --log-level "${VLLM_ROUTER_LOG_LEVEL:-info}" \
            --prometheus-port "${_PROMETHEUS_PORT}" \
            > >(tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
        proxy_pid=$!
        BENCHMARK_PORT=${ROUTER_PORT}
    else
        local _PROXY_SCRIPT=""
        for _candidate in \
            "${MORIIO_TOY_PROXY:-}" \
            "/app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py" \
            "/app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py" \
            "$(python3 -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "..", "examples", "disaggregated", "disaggregated_serving", "moriio_toy_proxy_server.py"))' 2>/dev/null)"; do
            if [ -n "${_candidate}" ] && [ -f "${_candidate}" ]; then
                _PROXY_SCRIPT="${_candidate}"; break
            fi
        done
        if [ -z "${_PROXY_SCRIPT}" ]; then
            echo "Error: moriio_toy_proxy_server.py not found (upstream vLLM path changed?)" \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            exit 1
        fi
        echo "Starting MoRI toy proxy: ${_PROXY_SCRIPT}"
        python "${_PROXY_SCRIPT}" \
            > >(tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
        proxy_pid=$!
        BENCHMARK_PORT=${PROXY_PORT}
    fi
    export BENCHMARK_PORT

    echo "Proxy (${PROXY_TYPE}) ready for benchmarking on ${host_name}:${host_ip}:${BENCHMARK_PORT}"

    # Router-registration gate (vllm_router only): wait for prefill+decode to
    # register via discovery before benchmarking, else requests 503.
    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        local _ROUTER_LOG="/run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log"
        local _REG_TIMEOUT="${ROUTER_REGISTER_TIMEOUT_S:-300}" _waited=0
        echo "Waiting up to ${_REG_TIMEOUT}s for prefill+decode to register with the router..."
        while [ "${_waited}" -lt "${_REG_TIMEOUT}" ]; do
            if grep -qa "Add Prefill" "${_ROUTER_LOG}" 2>/dev/null && \
               grep -qa "Add Decode"  "${_ROUTER_LOG}" 2>/dev/null; then
                echo "Router registration complete after ${_waited}s."; break
            fi
            sleep 5; _waited=$((_waited + 5))
        done
        [ "${_waited}" -ge "${_REG_TIMEOUT}" ] && echo "WARNING: router registration not confirmed in ${_REG_TIMEOUT}s; proceeding."
    else
        sleep 20
    fi

    curl -X POST http://127.0.0.1:${BENCHMARK_PORT}/v1/completions -H "Content-Type: application/json" -d '{
        "prompt": "Who is AMD CEO?",
        "temperature": 0,
        "max_tokens" : 10,
        "top_k": 1
    }'
    sleep 20
}
