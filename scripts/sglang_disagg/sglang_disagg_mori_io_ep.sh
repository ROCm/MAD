#!/bin/bash
# MoRI EP PD entrypoint (used when RUN_MORI=1 in run_xPyD_models.slurm).
# Customize for MoRI expert-parallel + disaggregated launch; until then this
# delegates to the standard Mooncake PD launcher.

_MORI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${_MORI_SCRIPT_DIR}"

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
# The router (proxy): on the first prefill node (MASTER_ADDR) unless the launcher gave
# it a node of its own (ROUTER_NODE=1), whose address it passes as ROUTER_ADDR.
ROUTER_ADDR="${ROUTER_ADDR:-$MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
DP_MODE="${DP_MODE:-0}"

if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "ERROR: MODEL_NAME not set, exiting"
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

# These are expected to be baked into the image
# (docker/di_sglang.ubuntu.amd.Dockerfile, pin set sglang-disagg). Install only what is
# genuinely missing, so a MAD-built image does no work here while an image supplied
# through DOCKER_IMAGE_NAME still self-heals. The unconditional
# "--ignore-installed --force-reinstall flask" this replaces mutated the image on
# every node of every run, downgraded click 8.5.0 -> 8.4.1, and surfaced unrelated
# dependency conflicts in the log.
# py-spy ships a binary rather than an importable module, so check for the command.
# DRY_RUN=1 prints each server command this node would start -- one argument per line,
# between ===DRYRUN and ===END=== lines, as vllm_disagg.sh does -- and starts, installs,
# waits for and writes nothing. For offline checks (scripts/common/tests/snapshot).
_dry_run() { [[ "${DRY_RUN:-0}" == "1" ]]; }
_dry_run_emit() {  # <role> <argv...>
    echo "===DRYRUN role=$1 NODE_RANK=${NODE_RANK}==="
    shift
    printf '%s\n' "$@"
    echo "===END==="
}

if ! _dry_run; then
command -v py-spy >/dev/null 2>&1 || pip install py-spy
python3 -c "import flask" >/dev/null 2>&1 || pip install flask
python3 -c "import yaml"  >/dev/null 2>&1 || pip install pyyaml
fi


host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
# Under DRY_RUN the address is this node's entry in IPADDRS, not the machine running it.
_dry_run && host_ip="$(cut -d, -f$((NODE_RANK + 1)) <<<"${IPADDRS:-}")"
host_name=$(hostname)

# Failing the job so every node, and the CI log, finds out. /run_logs is the job's
# shared log directory, so a marker one node writes there is seen by the others;
# every socket_barrier below watches it. Before, a node that gave up exited alone and
# the nodes waiting on it looped until the job's wall clock: in one run the
# decode server hit "Scheduler hit an exception" at 17:56, the router wait timed out
# 4000s later, and the job still ran to its 6-hour TIMEOUT.
JOB_ABORT_FILE="/run_logs/${SLURM_JOB_ID:-0}/ABORTED"
# Lines SGLang prints only when a server has died.
# A server that rejects its command line exits at once with argparse's
# "<prog>: error: ..." (sglang 0.5.20: "sglang serve: error: unrecognized arguments");
# without it here the wait below sat until its timeout for a server that never ran.
_FATAL_SERVER_LOG_RE='Scheduler hit an exception|Received sigquit from a child process|: error: (unrecognized arguments|ambiguous option|argument |the following arguments are required)'
# First error lines, then the tail: the tail of a dead server is its traceback, and the
# cause is usually earlier.
# Who holds this node's GPU memory, from the kernel (readable inside the container): one
# failure was "free memory on startup is less than desired" on 3 of 8
# GPUs, and nothing in the log could say what held it.
_print_gpu_snapshot() {
    echo "----- GPU memory on $(hostname) -----"
    local d
    for d in /sys/class/drm/card*/device; do
        [ -r "$d/mem_info_vram_used" ] || continue
        echo "$(basename "$(dirname "$d")"): $(( $(cat "$d/mem_info_vram_used") >> 30 )) GiB used of $(( $(cat "$d/mem_info_vram_total") >> 30 )) GiB"
    done
    echo "----- processes holding GPU memory (KFD; host pids) -----"
    ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' '; echo
    command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showpids 2>/dev/null | grep -vE '^=+|^\s*$' | head -n 30 || true
}
_print_server_log() {  # <file>
    echo "----- first error lines of $1 -----"
    grep -nE 'Error|error:|Exception|NCCL WARN|out of memory|hipError|Segmentation fault|core dumped|Memory access fault|died unexpectedly' "$1" 2>/dev/null \
        | grep -vE 'Traceback|raise |Failed to import Triton kernels' \
        | awk '{ k = $0; sub(/^[0-9]+:/, "", k); gsub(/pid=[0-9]+/, "", k);
                 gsub(/[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/, "", k);
                 if (!seen[k]++) print }' | head -n 40 || true
    echo "----- last 80 lines of $1 -----"
    tail -n 80 "$1" 2>/dev/null || true
    _print_gpu_snapshot
}
# A server that dies mid-run of a GPU fault logs only the runtime's line ("Memory access
# fault by GPU node-6 ... Reason: Unknown", "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION")
# and then "Worker proc ... died unexpectedly": no kernel, no address, and the GPU core
# dump fails in the container. The kernel's own record says which engine faulted (a
# compute kernel or a DMA copy), the faulting address and the process, so when the
# server's log shows a fault, save that record next to the job's logs. The container
# runs privileged, so it can read dmesg; if it cannot, the file says so.
#
# The watcher must not keep this script's output open: the container runs
# `vllm_disagg.sh 2>&1 | tee ...`, and tee ends only when every holder of the pipe has
# closed it. A watcher that inherited it kept two finished jobs (their sweeps done)
# running until the 4-hour wall clock. So it runs with its own output redirected, and
# every watcher is killed when this script exits.
_GPU_FAULT_RE='Memory access fault|HSA_STATUS_ERROR|died unexpectedly'
_GPU_FAULT_WATCHERS=""
trap '[ -n "${_GPU_FAULT_WATCHERS}" ] && kill ${_GPU_FAULT_WATCHERS} 2>/dev/null' EXIT
_watch_gpu_faults() {  # <role>: watch this node's <role>_NODE<rank>.log in the background
    local log="${_RUN_LOGS:-/run_logs}/${SLURM_JOB_ID}/${1}_NODE${NODE_RANK}.log"
    local out="${_RUN_LOGS:-/run_logs}/${SLURM_JOB_ID}/gpu_fault_NODE${NODE_RANK}.log"
    (
        until grep -qE "${_GPU_FAULT_RE}" "$log" 2>/dev/null; do sleep 5; done
        sleep 2   # let the runtime finish printing the fault
        {
            echo "=== $(date -u '+%F %T') UTC, ${host_name:-$(hostname)}: GPU fault in ${log}"
            grep -nE "${_GPU_FAULT_RE}" "$log" | head -n 10
            echo "----- kernel log (amdgpu) -----"
            dmesg -T 2>/dev/null | grep -iE 'amdgpu|gmc_v|vm fault|page fault|UTCL2|retry fault|protection fault' | tail -n 80 \
                || echo "(dmesg not readable in this container)"
            _print_gpu_snapshot
        } > "$out" 2>&1
        echo "[gpu-fault] ${1} on ${host_name:-$(hostname)} reported a GPU fault; kernel record in ${out}" \
            >> "${_RUN_LOGS:-/run_logs}/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log"
    ) </dev/null >/dev/null 2>&1 &
    _GPU_FAULT_WATCHERS="${_GPU_FAULT_WATCHERS} $!"
}

# Everything this launcher started, children first. A node that gives up must not leave
# its servers running: they hold the container's output pipe, so the container -- and
# the SLURM job -- stayed up until the wall clock. In one run NODE1's decode
# server had started fine; NODE0 failed, NODE1's barrier gave up and exited, and the
# decode server kept the job alive until it was cancelled.
_descendants() {  # <pid>: every process below it, children first
    local c
    for c in $(pgrep -P "$1" 2>/dev/null); do _descendants "$c"; echo "$c"; done
}
# SIGTERM first, then SIGKILL whatever is left after a grace period. vLLM workers wedged
# in a failed HIP/RCCL call ignore SIGTERM: in one run both nodes gave up and the
# job still ran on, holding its nodes, until it was cancelled by hand.
_kill_own_processes() {
    local pids p i
    pids="$(_descendants $$)"
    [ -n "$pids" ] || return 0
    kill $pids 2>/dev/null || true
    for i in $(seq 1 15); do
        p=""; for p in $pids; do kill -0 "$p" 2>/dev/null && break; p=""; done
        [ -z "$p" ] && return 0
        sleep 1
    done
    kill -9 $pids 2>/dev/null || true
}

_job_fail() {  # <reason>
    echo "ERROR: $1" >&2
    echo "NODE${NODE_RANK} (${host_name}): $1" >> "${JOB_ABORT_FILE}" 2>/dev/null || true
    _kill_own_processes
    exit 1
}

# =============================================================================
# Server and router commands
# =============================================================================

# NIC selection and the servers' NCCL / MoRI environment (connectors/mori.env).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mori_ep_env.sh"

# A loopback RC round trip on every KV NIC before any server starts. MoRI's start-up only
# opens the devices, so an RDMA user space that cannot create QPs (rdma-core < 54 on Thor2
# under the in-kernel bnxt_re driver) would otherwise hang the first KV transfer behind
# --watchdog-timeout.
_rdma_preflight() {
    local dev out rc try spid port=19400 gid="${MORI_IB_GID_INDEX:-3}" failed=""
    if ! command -v ibv_rc_pingpong >/dev/null 2>&1; then
        echo "[rdma-preflight] skipped: ibv_rc_pingpong is not in the image"
        return 0
    fi
    for dev in ${IB_DEVICES//,/ }; do
        dev="${dev%%:*}"
        port=$((port + 1))
        timeout 15 ibv_rc_pingpong -d "$dev" -g "$gid" -n 5 -p "$port" >/dev/null 2>&1 &
        spid=$!
        for try in $(seq 1 10); do
            out="$(timeout 15 ibv_rc_pingpong -d "$dev" -g "$gid" -n 5 -p "$port" 127.0.0.1 2>&1)"
            rc=$?
            grep -q "Couldn't connect" <<<"$out" || break
            sleep 0.3
        done
        kill "$spid" 2>/dev/null
        wait "$spid" 2>/dev/null
        [ "$rc" -eq 0 ] || failed="${failed}${failed:+; }${dev}: $(grep -v '^$' <<<"$out" | tail -n 1)"
    done
    [ -z "$failed" ] || _job_fail "RDMA preflight on ${host_name}: no working QP (GID ${gid}) on ${failed}. The image's RDMA user space does not match this host's NIC driver (docker/pins/README.md); RDMA_PREFLIGHT=0 skips this check"
    echo "[rdma-preflight] loopback RC round trip OK on ${IB_DEVICES} (GID ${gid})"
}

# This node's role and commands, from models.yaml (render.py); the launcher's topology
# (MAD_TOPO_*) decides the role. Sets ROLE, ROLE_INDEX, RUNS_SERVER, RUNS_ROUTER,
# TP_SIZE, SERVER_ARGV, ROUTER_ARGV, IP_FIRST_PREFILL and IP_FIRST_DECODE.
_render_out="$(HOST_IP="${host_ip}" python3 "${SCRIPT_DIR}/render.py" \
    --models "${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}" shell --rank "${NODE_RANK}")" \
    || _job_fail "render.py could not build node ${NODE_RANK}'s commands (error above)"
eval "${_render_out}"
echo "Using model-specific configuration for: $MODEL_NAME (DP_MODE=${DP_MODE}, ${ROLE} ${ROLE_INDEX})"

# The agentic replay reads server-side metrics; render.py adds the server flags.
AGENTIC_METRICS_ENABLED=0
if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_SERVER_METRICS:-0}" == "1" ]]; then
    AGENTIC_METRICS_ENABLED=1
fi

# =============================================================================
# Container Synchronization
# =============================================================================

if ! _dry_run; then
[[ "${RDMA_PREFLIGHT:-1}" == "1" ]] && _rdma_preflight
echo "Waiting at the container creation barrier on $host_name"
python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${BARRIER_PORT} \
    --abort-file "${JOB_ABORT_FILE}" \
    || _job_fail "container creation barrier on port ${BARRIER_PORT} failed on ${host_name}"


# =============================================================================
# Launching the servers and the router
# =============================================================================
# The router's node (NODE_RANK 0, co-located with the first prefill server, or the
# router node) waits for the servers' logs: every server with DP_MODE=0, each role's
# first node with DP_MODE=1. Servers sync on the router's :2322 and stop when it closes.

cd /sgl-workspace/sglang || {
    echo "ERROR: cd /sgl-workspace/sglang failed"
    exit 1
}
fi

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

_ROLE="$ROLE"; _ROLE_INDEX="$ROLE_INDEX"
# Whether this node runs the router and the benchmark client.
_HOSTS_ROUTER="$RUNS_ROUTER"

if [[ "$_HOSTS_ROUTER" == 1 ]]; then
    if [[ "$_ROLE" == prefill ]]; then
        echo "${host_name}:${host_ip} is Prefill Node 0 + Router/proxy (NODE_RANK=0, co-located)"
    else
        # A node of its own: no server here, only the router and the benchmark client.
        echo "${host_name}:${host_ip} is the Router node (proxy + benchmark client, NODE_RANK=${NODE_RANK})"
    fi

    _dry_run || mkdir -p "/run_logs/${SLURM_JOB_ID:-0}"
    _node0_prefill_pid=""

    if [[ "$_ROLE" == prefill ]]; then
        # --- Launch first prefill server on this node (co-located with router) ---
        setup_sglang_worker_env
        _server_cmd=(python3 -m sglang.launch_server "${SERVER_ARGV[@]}")

        if _dry_run; then
            _dry_run_emit prefill "${_server_cmd[@]}"
        else
            PREFILL_LOG="/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log"
            {
                echo "========== PREFILL_CMD (NODE_RANK=0, co-located with router) =========="
                printf '%q ' "${_server_cmd[@]}"; echo
                echo ""
            } | tee "$PREFILL_LOG"
            _dbg "launching prefill server (TP_SIZE=${TP_SIZE})"
            set -x
            "${_server_cmd[@]}" 2>&1 | tee -a "$PREFILL_LOG" >/dev/null &
            set +x
            _node0_prefill_pid=$!
            _dbg "prefill server started pid=${_node0_prefill_pid}"
            _watch_gpu_faults prefill
        fi
    fi

    if ! _dry_run; then

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
                    if grep -Eq "${_FATAL_SERVER_LOG_RE}" "$LOG_FILE" 2>/dev/null; then
                        _print_server_log "$LOG_FILE"
                        _job_fail "prefill NODE${i} died during start-up after ${_elapsed}s (log tail above: $LOG_FILE)"
                    fi
                    [[ -f "${JOB_ABORT_FILE}" ]] && _job_fail "stopped waiting for prefill NODE${i}: $(head -n1 "${JOB_ABORT_FILE}")"
                    if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                        _print_server_log "$LOG_FILE"
                        _job_fail "Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for prefill NODE${i} (${LOG_FILE})"
                    fi
                    sleep "${ROUTER_POLL_SLEEP_SECONDS}"
                done
                echo "Prefill NODE${i} ready."
            done
            for ((i = xP; i <= xP + yD - 1; i++)); do
                LOG_FILE="${_runlog}/decode_NODE${i}.log"
                until [[ -f "$LOG_FILE" ]] && grep -q "${SEARCH_SIGNAL}" "$LOG_FILE" 2>/dev/null; do
                    _elapsed=$(( $(date +%s) - _wait_start_ts ))
                    if grep -Eq "${_FATAL_SERVER_LOG_RE}" "$LOG_FILE" 2>/dev/null; then
                        _print_server_log "$LOG_FILE"
                        _job_fail "decode NODE${i} died during start-up after ${_elapsed}s (log tail above: $LOG_FILE)"
                    fi
                    [[ -f "${JOB_ABORT_FILE}" ]] && _job_fail "stopped waiting for decode NODE${i}: $(head -n1 "${JOB_ABORT_FILE}")"
                    if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                        _print_server_log "$LOG_FILE"
                        _job_fail "Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for decode NODE${i} (${LOG_FILE})"
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
                    _elapsed=$(( $(date +%s) - _wait_start_ts ))
                    if grep -Eq "${_FATAL_SERVER_LOG_RE}" "$LOG_FILE" 2>/dev/null; then
                        _print_server_log "$LOG_FILE"
                        _job_fail "${_log_label} died during start-up after ${_elapsed}s (log tail above: $LOG_FILE)"
                    fi
                    [[ -f "${JOB_ABORT_FILE}" ]] && _job_fail "stopped waiting for ${_log_label}: $(head -n1 "${JOB_ABORT_FILE}")"
                    if (( _elapsed >= ROUTER_READY_TIMEOUT_SECONDS )); then
                        _print_server_log "$LOG_FILE"
                        _job_fail "Timeout (${_elapsed}s >= ${ROUTER_READY_TIMEOUT_SECONDS}s) waiting for ${_log_label} (${LOG_FILE})"
                    fi
                    sleep "${ROUTER_POLL_SLEEP_SECONDS}"
                done
                echo "${_log_label} ready (${LOG_FILE})."
            done
        fi

        echo "Prefill/decode backends report ready per logs; starting sglang_router (proxy)."
    fi

    # Launched only after worker logs confirm servers are up (avoids the proxy probing
    # backends too early). Binds 0.0.0.0 so bench_serving (--host 127.0.0.1 in
    # benchmark_xPyD.sh) can connect.
    _router_cmd=(python3 -m sglang_router.launch_router "${ROUTER_ARGV[@]}")

    if _dry_run; then
        _dry_run_emit router "${_router_cmd[@]}"
        exit 0
    fi
    echo "========== ROUTER_CMD (NODE_RANK=${NODE_RANK}, DP_MODE=${DP_MODE}) ==========" >&2
    printf '%q ' "${_router_cmd[@]}"; echo
    set -x
    "${_router_cmd[@]}" 2>&1 | tee "/run_logs/${SLURM_JOB_ID:-0}/proxy_NODE${NODE_RANK}.log" >/dev/null &
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

    echo "Killing the proxy server (pid=${proxy_pid})"
    kill "${proxy_pid}"

    if [[ -n "${_node0_prefill_pid}" ]]; then
        echo "Killing the co-located prefill server (pid=${_node0_prefill_pid})"
        kill "${_node0_prefill_pid}"
    fi

    if [[ "${benchmark_status}" -ne 0 ]]; then
        echo "ERROR: benchmark failed with status ${benchmark_status}" >&2
        exit "${benchmark_status}"
    fi

elif [[ "$_ROLE" == prefill ]]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-default})"
    setup_sglang_worker_env
    _server_cmd=(python3 -m sglang.launch_server "${SERVER_ARGV[@]}")

    # NOTE: do NOT gate a dp_ep server's later nodes on port 5757 being open.
    # Same rendezvous deadlock as decode: all of a server's nodes must launch concurrently.

    if _dry_run; then
        _dry_run_emit prefill "${_server_cmd[@]}"
        exit 0
    fi
    PREFILL_LOG="/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log"
    mkdir -p "$(dirname "$PREFILL_LOG")"
    {
        echo "========== PREFILL_CMD (NODE_RANK=${NODE_RANK}, prefill ${_ROLE_INDEX}) =========="
        printf '%q ' "${_server_cmd[@]}"; echo
        echo ""
    } | tee "$PREFILL_LOG"
    _dbg "launching prefill server (prefill ${_ROLE_INDEX}, TP_SIZE=${TP_SIZE})"
    set -x
    "${_server_cmd[@]}" 2>&1 | tee -a "$PREFILL_LOG" >/dev/null &
    set +x
    prefill_pid=$!
    _dbg "prefill server started pid=${prefill_pid}"
    _watch_gpu_faults prefill

    _dbg "waiting for proxy server to be up (ROUTER_ADDR=${ROUTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${ROUTER_ADDR}" \
        --node-ports 2322 \
        --abort-file "${JOB_ABORT_FILE}" \
        || _job_fail "the proxy on ${ROUTER_ADDR}:2322 never came up"

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${ROUTER_ADDR}" \
        --remote-port 2322

    echo "Killing the prefill server"
    kill "${prefill_pid}"

elif [[ "$_ROLE" == decode ]]; then
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-default})"
    setup_sglang_worker_env
    _dbg "decode node: decode ${_ROLE_INDEX} TP_SIZE=${TP_SIZE}"

    if [[ "$DP_MODE" == "1" ]]; then
        #export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
        #echo "DP_MODE=0 decode SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"
        export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_DECODE}"
        _dbg "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
        _dbg "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD}"
        _dbg "MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE}"

        # NOTE: do NOT gate a dp_ep server's later nodes on port 5757 being open.
        # torch init_process_group uses a rendezvous: rank 0 opens the port only
        # AFTER all ranks have joined. Waiting for rank 0's port before launching
        # rank 1 creates a permanent deadlock. All decode nodes must launch concurrently.
        _dbg "decode ${_ROLE_INDEX}: launching without gating on dist-init port (rendezvous requires all ranks concurrent)"
    fi

    _server_cmd=(python3 -m sglang.launch_server "${SERVER_ARGV[@]}")

    if _dry_run; then
        _dry_run_emit decode "${_server_cmd[@]}"
        exit 0
    fi
    DECODE_LOG="/run_logs/${SLURM_JOB_ID:-0}/decode_NODE${NODE_RANK}.log"
    mkdir -p "$(dirname "$DECODE_LOG")"
    {
        echo "========== DECODE_CMD (NODE_RANK=${NODE_RANK}, decode ${_ROLE_INDEX}) =========="
        printf '%q ' "${_server_cmd[@]}"; echo
        echo ""
    } | tee "$DECODE_LOG"
    _dbg "launching decode server (decode ${_ROLE_INDEX}, TP_SIZE=${TP_SIZE})"
    set -x
    "${_server_cmd[@]}" 2>&1 | tee -a "$DECODE_LOG" >/dev/null &
    set +x
    decode_pid=$!
    _dbg "decode server started pid=${decode_pid}"
    _watch_gpu_faults decode

    _dbg "waiting for proxy server to be up (ROUTER_ADDR=${ROUTER_ADDR}:2322) ..."
    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${ROUTER_ADDR}" \
        --node-ports 2322 \
        --abort-file "${JOB_ABORT_FILE}" \
        || _job_fail "the proxy on ${ROUTER_ADDR}:2322 never came up"

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${ROUTER_ADDR}" \
        --remote-port 2322

    echo "Killing the decode server"
    kill "${decode_pid}"

else
    echo "ERROR: NODE_RANK=${NODE_RANK} has no role (role '${_ROLE}'; xP=${xP} yD=${yD}, router ${ROUTER_ADDR})" >&2
    exit 1
fi

echo "Script completed successfully"
exit 0
