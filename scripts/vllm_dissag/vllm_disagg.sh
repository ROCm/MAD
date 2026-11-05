#!/bin/bash
# vLLM Disaggregated Server Launcher — unified two-axis driver.
# =============================================================================
# ONE launcher for all KV connectors and parallelism modes. Two axes select the
# behavior; the all-to-all EP backend is validated against the connector:
#
#   CONNECTOR = rixl | moriio          (KV transfer; default rixl via back-compat shim)
#   WIDE_EP   = 0 (TP) | 1 (wideEP)    (parallelism; default 0=TP via back-compat shim)
#   EP_BACKEND= mori | deepep          (only when WIDE_EP=1; default = connector partner)
#
#   | CONNECTOR | WIDE_EP=0 | WIDE_EP=1 EP_BACKEND |
#   | rixl      | TP        | deepep (only)        |
#   | moriio    | TP (new)  | mori   (only)        |
#
# Connector logic lives in connectors/<CONNECTOR>.sh (sourced), providing:
#   connector_init, connector_setup_env, connector_runtime_patch,
#   connector_launch_worker, connector_wait_workers_ready, connector_start_proxy
# Parallelism helpers live in parallelism.sh.
# Per-model flags + env come from models.yaml (parsed below).
#
# Node roles (by NODE_RANK), co-located proxy on rank 0:
#   0           -> Prefill MASTER + Proxy
#   1 .. xP-1   -> Prefill CHILD  (--headless, wideEP only)
#   xP          -> Decode  MASTER
#   xP+1 .. end -> Decode  CHILD   (--headless, wideEP only)
#
# Back-compat shim: RUN_MORI=1 -> moriio/wideEP/mori; RUN_DEEPEP=1 -> rixl/wideEP/deepep.
# DRY_RUN=1 echoes each worker's assembled `vllm serve` argv instead of running.
# =============================================================================

SCRIPT_DIR="${NIXL_COOKBOOK_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"

# =============================================================================
# Initialization — container-side boot setup
# =============================================================================
# Runs first, before any axis or connector work. Everything here must be
# idempotent (a node can re-enter this script) and NON-FATAL: nothing in this
# section is worth losing a multi-hour serving cell over. Add future one-time
# container setup here rather than inline further down.
#
# The image is a moving target -- pins change under us and a package present
# today can be absent in the next nightly -- so these steps ask the environment
# what it actually has instead of encoding a fact about any one tag.

# Python deps this script and its tooling import. Entries are
# `module[:pip-spec]`, space separated, for when the import name differs from
# the pip name. `PY_BOOT_DEPS=off` (or `0`/`none`) skips the step. When nothing
# is missing it prints one line and installs nothing -- the expected path.
#
#   yaml   CRITICAL. The models.yaml parse below does a bare `import yaml`,
#          twice, with no fallback, and it resolves both the per-model env
#          block and the per-role serve flags. Today's images ship PyYAML so
#          this has never fired, but if one stops shipping it the serve path
#          dies at the model catalog -- not somewhere obvious.
#   pandas  Reporting only. `benchmark_parser.py` is the sole tool that reports
#          ITL/TTFT/TPOT, and it falls back to a stdlib table without pandas.
#          `parse_to_csv.py` -- the one the job calls to write CONCURRENCY.csv
#          and perf.csv -- is stdlib-only.
#
# Non-fatal even for yaml: a compute node may have no route to PyPI, and if
# yaml really is absent the existing bare import fails loudly a few lines
# down. This step only gives it a chance to self-heal, and the WARN is the
# diagnosis for that failure.
_ensure_py_deps() {
    local spec="${PY_BOOT_DEPS-yaml:PyYAML pandas}"
    case "${spec}" in ""|off|OFF|0|none) return 0 ;; esac

    local missing=() entry mod pkg
    for entry in ${spec}; do
        mod="${entry%%:*}"
        pkg="${entry#*:}"; [[ "$pkg" == "$entry" ]] && pkg="$mod"
        if ! python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${mod}') else 1)" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[py-deps] present: ${spec}"
        return 0
    fi

    echo "[py-deps] installing missing: ${missing[*]}"
    # --break-system-packages first: newer base images ship a PEP 668 managed
    # interpreter, where a plain install refuses with externally-managed-environment.
    if pip install --quiet --break-system-packages "${missing[@]}" 2>/dev/null \
       || pip install --quiet "${missing[@]}" 2>/dev/null; then
        echo "[py-deps] installed: ${missing[*]}"
    else
        # Say which kind of failure this is. "serving is unaffected" is true for
        # pandas and false for PyYAML, so do not print one blanket reassurance.
        echo "[py-deps] WARN install failed: ${missing[*]}" >&2
        case " ${missing[*]} " in *" PyYAML "*)
            echo "[py-deps] WARN PyYAML absent — the models.yaml parse below will fail, so per-model env and serve flags will NOT be applied" >&2 ;;
        esac
        case " ${missing[*]} " in *" pandas "*)
            echo "[py-deps] pandas absent is harmless — benchmark_parser.py falls back to a stdlib table, and parse_to_csv.py is stdlib-only" >&2 ;;
        esac
    fi
}

_init_container_env() {
    _ensure_py_deps
}
_init_container_env

# =============================================================================
# Axis selection (+ legacy shim) and validation
# =============================================================================
# Legacy flags map to the new axes when CONNECTOR is not explicitly set.
if [[ -z "${CONNECTOR:-}" ]]; then
    if [[ "${RUN_MORI:-0}" == "1" ]]; then
        CONNECTOR=moriio; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-mori}"
    elif [[ "${RUN_DEEPEP:-0}" == "1" ]]; then
        CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-deepep}"
    else
        # No flags -> historical default (== legacy vllm_disagg_server.sh): rixl + TP.
        # Matches the slurm's shim so direct invocation and sbatch agree.
        CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-0}"
    fi
fi
WIDE_EP="${WIDE_EP:-0}"

case "$CONNECTOR" in rixl|moriio) ;; *) echo "Error: invalid CONNECTOR='${CONNECTOR}' (expected rixl|moriio)." >&2; exit 1 ;; esac
case "$WIDE_EP" in 0|1) ;; *) echo "Error: invalid WIDE_EP='${WIDE_EP}' (expected 0|1)." >&2; exit 1 ;; esac

# EP backend defaults to the connector's partner; validate cross-pairings out.
if [[ "$WIDE_EP" == "1" ]]; then
    if [[ "$CONNECTOR" == "moriio" ]]; then
        EP_BACKEND="${EP_BACKEND:-mori}"
        [[ "$EP_BACKEND" == "mori" ]] || { echo "Error: CONNECTOR=moriio supports EP_BACKEND=mori only (got '${EP_BACKEND}')." >&2; exit 1; }
    else
        EP_BACKEND="${EP_BACKEND:-deepep}"
        [[ "$EP_BACKEND" == "deepep" ]] || { echo "Error: CONNECTOR=rixl supports EP_BACKEND=deepep only (got '${EP_BACKEND}')." >&2; exit 1; }
    fi
fi
export CONNECTOR WIDE_EP EP_BACKEND

_CONNECTOR_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.sh"
[[ -f "$_CONNECTOR_FILE" ]] || { echo "Error: connector profile not found: ${_CONNECTOR_FILE}" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/parallelism.sh" ]] || { echo "Error: parallelism.sh not found in ${SCRIPT_DIR}" >&2; exit 1; }

echo "[vllm_disagg] CONNECTOR=${CONNECTOR} WIDE_EP=${WIDE_EP} EP_BACKEND=${EP_BACKEND:-<n/a>}"

# =============================================================================
# Common Environment Configuration
# =============================================================================
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
: "${MODEL_PATH:?MODEL_PATH must be set (path to the model dir/repo)}"
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
echo "[vllm_disagg] topology: xP=${xP} yD=${yD} (total nodes=$((xP + yD)))"
IPADDRS="${IPADDRS:-localhost}"
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"

echo "Listing NIXL_COOKBOOK_PATH: ${NIXL_COOKBOOK_PATH:-<unset>}"
[[ -n "${NIXL_COOKBOOK_PATH:-}" ]] && ls "${NIXL_COOKBOOK_PATH}"

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

# =============================================================================
# Topology math
# =============================================================================
_GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PREFILL_DP_SIZE=$((xP * _GPUS_PER_NODE))
DECODE_DP_SIZE=$((yD * _GPUS_PER_NODE))
DP_PARALLEL_SIZE_LOCAL=${_GPUS_PER_NODE}
PREFILL_DP_START_RANK=$(( NODE_RANK * _GPUS_PER_NODE ))
PREFILL_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' '{print $1}')
DECODE_DP_START_RANK=$(( (NODE_RANK - xP) * _GPUS_PER_NODE ))
DECODE_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+1)}')

# =============================================================================
# Driver helper functions (shared by all connectors)
# =============================================================================
_dryrun_emit() {
    local backend="$1"; shift
    local log_prefix="$1"; shift
    local role="$1"; shift
    echo "===DRYRUN backend=${backend} log_prefix=${log_prefix} role=${role} NODE_RANK=${NODE_RANK}==="
    local a
    for a in "$@"; do printf '%s\n' "$a"; done
    echo "===END==="
}

# A worker that dies at boot never prints the ready signal, so without this the
# wait runs to TIMEOUT_SECONDS (a decode segfault at 07:29, NODE0 waited
# until 08:50). Every node's worker log is scanned, not just the masters'.
_BOOT_FATAL_SIGNALS=(-e "Engine core initialization failed" -e "EngineCore failed to start" -e "Segfault encountered")

# Only this node exits on a boot failure; the others would sit in their barriers
# until the wall (one segfault at 3 min held 8 nodes for 4 h). The host
# srun step sees this marker and exits nonzero, so --kill-on-bad-exit ends the step.
_mark_boot_fatal() {
    echo "$*" > "/run_logs/${SLURM_JOB_ID}/BOOT_FATAL" 2>/dev/null || true
}

_wait_log_signal_or_fail() {
    local LOG_FILE="$1" LABEL="$2" SEARCH_SIGNAL="$3" TIMEOUT_SECONDS="$4" SLEEP_SECONDS="$5"
    local ELAPSED=0 _dead _out
    local _log_dir="/run_logs/${SLURM_JOB_ID}"
    until grep -Fq "${SEARCH_SIGNAL}" "${LOG_FILE}" 2>/dev/null; do
        _dead="$(grep -lF "${_BOOT_FATAL_SIGNALS[@]}" "${_log_dir}"/prefill_NODE*.log "${_log_dir}"/decode_NODE*.log 2>/dev/null | tr '\n' ' ')"
        if [ -n "${_dead}" ]; then
            _out=/run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            {
                echo "Error: worker died during boot while waiting for ${LABEL} (${ELAPSED}s): ${_dead}"
                grep -hnF "${_BOOT_FATAL_SIGNALS[@]}" ${_dead} 2>/dev/null | head -5
            } | tee -a "${_out}"
            _mark_boot_fatal "worker died during boot: ${_dead}"
            exit 1
        fi
        if [ "${ELAPSED}" -ge "${TIMEOUT_SECONDS}" ]; then
            echo "Timeout (${TIMEOUT_SECONDS}s): '${SEARCH_SIGNAL}' not found in ${LABEL}: ${LOG_FILE}" \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            _mark_boot_fatal "timeout waiting for ${LABEL}"
            exit 1
        fi
        sleep "${SLEEP_SECONDS}"; ELAPSED=$((ELAPSED + SLEEP_SECONDS))
    done
    echo "Ready: ${LABEL} (${LOG_FILE})"
}

wait_for_proxy_and_cleanup() {
    local worker_pid="$1" label="$2"
    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py --node-ips ${MASTER_ADDR} --node-ports $PROXY_PORT
    echo "Waiting until proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py --remote-ip ${MASTER_ADDR} --remote-port $PROXY_PORT
    echo "Killing the ${label} server"
    pkill -P "$worker_pid" 2>/dev/null; kill "$worker_pid" 2>/dev/null || true
}

print_node_info() {
    local role_desc="$1"
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "Connector : ${CONNECTOR}  WIDE_EP=${WIDE_EP}  EP_BACKEND=${EP_BACKEND:-<n/a>}"
    echo "${host_name}:${host_ip} is ${role_desc}."
}

# =============================================================================
# Model catalog (models.yaml): export per-model ENV, then resolve per-role FLAGS
# =============================================================================
# PARALLEL_MODE mirrors sglang: WIDE_EP=1 -> dp, WIDE_EP=0 -> tp.
if [[ "$WIDE_EP" == "1" ]]; then PARALLEL_MODE=dp; else PARALLEL_MODE=tp; fi

MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"
MODEL_CONFIG_PREFILL=""
MODEL_CONFIG_DECODE=""
if [[ -n "$MODEL_NAME" && -f "$MODELS_YAML" ]]; then
    export MODELS_YAML MODEL_NAME PARALLEL_MODE
    # 1) Export per-model env: block FIRST (so connector ${VAR:-default} yields to it).
    #    Only set a var that is NOT already in the environment, so a submit-time
    #    `docker -e VAR=...` (already exported) WINS over the yaml value. Precedence:
    #    connector default  <  models.yaml env:  <  submit-time -e.
    _yaml_env="$(python3 - <<'PY'
import os, yaml, shlex
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
cfg = m.get(os.environ["MODEL_NAME"]) or {}
for k, v in (cfg.get("env") or {}).items():
    # skip if already present in the environment (submit-time -e override wins)
    if k in os.environ:
        continue
    print(f'export {k}={shlex.quote(str(v))}')
PY
)"
    [[ -n "$_yaml_env" ]] && eval "$_yaml_env"

    # Fallback: launcher-calculated default if not set by submit-time or yaml
    : ${GPU_MEMORY_UTILIZATION:=$_GPU_MEM_UTIL_LAUNCHER_DEFAULT}
    export GPU_MEMORY_UTILIZATION

    # 2) Resolve per-role flag strings for the active PARALLEL_MODE.
    eval "$(python3 - <<'PY'
import os, shlex, yaml
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
name = os.environ["MODEL_NAME"]; mode = os.environ["PARALLEL_MODE"]
cfg = m.get(name)
if cfg is None:
    import sys
    print(f"WARN: model {name!r} not in models.yaml; using empty flags", file=sys.stderr)
    raise SystemExit(0)
prefill = cfg.get("prefill") or {}; decode = cfg.get("decode") or {}
def compose(role):
    return " ".join(x for x in [
        cfg.get("base_flags",""), cfg.get(f"{mode}_flags",""),
        (role.get(mode,"") if isinstance(role,dict) else ""),
        cfg.get("experimental_flags",""),
    ] if x).strip()
print(f'MODEL_CONFIG_PREFILL={shlex.quote(compose(prefill))}')
print(f'MODEL_CONFIG_DECODE={shlex.quote(compose(decode))}')
PY
)"
    echo "[vllm_disagg] model flags (${PARALLEL_MODE}): prefill='${MODEL_CONFIG_PREFILL}' decode='${MODEL_CONFIG_DECODE}'"
fi
export MODEL_CONFIG_PREFILL MODEL_CONFIG_DECODE

# =============================================================================
# Load parallelism + connector, then initialize
# =============================================================================
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/parallelism.sh"
# shellcheck source=/dev/null
source "${_CONNECTOR_FILE}"
connector_init

echo "-----------------------------Printing node specific details ----------------------"
echo "IPADDRS = ${IPADDRS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}  DECODE_DP_SIZE=${DECODE_DP_SIZE}"
echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}  DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"

# =============================================================================
# Container barrier + connector_runtime_patch (skipped under DRY_RUN)
# =============================================================================
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    _BARRIER_PORT="${CONTAINER_BARRIER_PORT:-2222}"
    # Image has no ss/fuser (tcp_listen.sh). A leftover on this port makes
    # socket_barrier's bind fail in a daemon thread while is_port_open() still
    # sees the old listener as ready -- the barrier would pass without us
    # owning the port. Kill what we can in this PID ns, then bind-or-die.
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/tcp_listen.sh"
    tcp_listen_kill "${_BARRIER_PORT}" || true
    sleep 2
    # Bounded: a node that never launches its container cannot open the port,
    # and an unbounded barrier turns that into a silent hold on every node
    # until walltime. Timing out names the missing peers and fails
    # the job in minutes. Generous by default -- a cold node still has to load
    # a ~50 GiB image before it reaches this line.
    _BARRIER_TIMEOUT="${CONTAINER_BARRIER_TIMEOUT_SECONDS:-1800}"
    echo "Waiting at the container creation barrier on $host_name"
    if ! python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --local-ip ${host_ip} --local-port ${_BARRIER_PORT} --enable-port \
        --node-ips ${IPADDRS} --node-ports ${_BARRIER_PORT} \
        --timeout "${_BARRIER_TIMEOUT}"; then
        echo "Error: container creation barrier failed on $host_name after ${_BARRIER_TIMEOUT}s." >&2
        exit 1
    fi
    connector_runtime_patch
fi

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================
_BENCH_RC=0
if [ "$NODE_RANK" -eq 0 ]; then
    print_node_info "Prefill master + Proxy node (co-located)"
    connector_launch_worker "master" "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" "kv_producer" "prefill"
    local_worker_pid=$WORKER_PID
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[dry-run] rank0 prefill master emitted; skipping proxy/benchmark."; exit 0; }

    connector_wait_workers_ready
    # HOLD_AFTER_READY=1: keep P+D up, skip router/toy proxy and the bench.
    # Children block on MASTER:PROXY_PORT until something listens, then until
    # that port closes — so a later manual proxy on :10001 is the teardown gate.
    if [[ "${HOLD_AFTER_READY:-0}" == "1" ]]; then
        echo "[hold] P+D Application startup complete. Skipping proxy/router and bench."
        echo "[hold] Launch proxy on NODE0 :${PROXY_PORT:-10001} (discovery :${MORI_PROXY_PING_PORT:-31367})."
        echo "[hold] Engines stay until walltime, or until prefill master exits."
        wait "$local_worker_pid" || true
        echo "[hold] prefill master exited; sleeping until slurm walltime"
        sleep infinity
    fi
    connector_start_proxy

    # connector_start_proxy sets BENCHMARK_PORT (router->ROUTER_PORT, toy->PROXY_PORT).
    # Fall back to PROXY_PORT only if the connector didn't set it.
    export BENCHMARK_PORT="${BENCHMARK_PORT:-${PROXY_PORT}}"
    # Agentic replay driver reads the endpoint via AGENTIC_PORT; for the default
    # sweep it is exactly BENCHMARK_PORT (the router/proxy port), so the non-agentic
    # path is unaffected.
    export AGENTIC_PORT="${BENCHMARK_PORT}"

    # Agentic path only: the vLLM PD router serves /v1/chat/completions but 503s on
    # /v1/models under MoRIIO service discovery (empty HTTP worker registry). The
    # shared harness (scripts/common/agentic_lib.sh) gates readiness + served-model
    # resolution on GET /v1/models, so start a tiny side-port shim that answers
    # /v1/models (gated on the router's /health) and stream-proxies everything else
    # to the router, then point AGENTIC_PORT at the shim. Byte-for-byte no-op for the
    # default sweep (guarded by BENCHMARK_SCRIPT_FILE).
    _agentic_shim_pid=""
    if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" ]]; then
        _shim_port="${AGENTIC_SHIM_PORT:-$((BENCHMARK_PORT + 1))}"
        _shim_model="${MODEL:-${MODEL_PATH}}"
        _shim_prefill="${AGENTIC_SERVER_METRICS%% *}"
        AGENTIC_SHIM_PORT="${_shim_port}" \
        AGENTIC_SHIM_UPSTREAM="127.0.0.1:${BENCHMARK_PORT}" \
        AGENTIC_SHIM_MODEL="${_shim_model}" AGENTIC_SHIM_PREFILL="${_shim_prefill}" \
            python3 "$NIXL_COOKBOOK_PATH/agentic_models_shim.py" \
            > >(tee /run_logs/${SLURM_JOB_ID}/agentic_models_shim_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
        _agentic_shim_pid=$!
        export AGENTIC_PORT="${_shim_port}"
        echo "[agentic-shim] models shim on :${_shim_port} -> router :${BENCHMARK_PORT} (model=${_shim_model})"
        sleep 3
    fi

    _BENCH_RC=0
    bash "$NIXL_COOKBOOK_PATH/${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}" || _BENCH_RC=$?

    [[ -n "$_agentic_shim_pid" ]] && { kill "$_agentic_shim_pid" 2>/dev/null || true; }

    echo "Killing the proxy server.."
    pkill -P $proxy_pid 2>/dev/null; kill $proxy_pid 2>/dev/null || true
    echo "Killing the prefill master server.."
    pkill -P $local_worker_pid 2>/dev/null; kill $local_worker_pid 2>/dev/null || true

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    print_node_info "Prefill child node"
    connector_launch_worker "child" "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" "kv_producer" "prefill" "${PREFILL_DP_START_RANK}"
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[dry-run] prefill child emitted."; exit 0; }
    wait_for_proxy_and_cleanup $WORKER_PID "prefill child"

elif [ "$NODE_RANK" -eq "$xP" ]; then
    print_node_info "Decode master node"
    connector_launch_worker "master" "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" "kv_consumer" "decode"
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[dry-run] decode master emitted."; exit 0; }
    wait_for_proxy_and_cleanup $WORKER_PID "decode master"

else
    print_node_info "Decode child node"
    connector_launch_worker "child" "${DECODE_DP_SIZE}" "${DECODE_MASTER_ADDR}" "kv_consumer" "decode" "${DECODE_DP_START_RANK}"
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[dry-run] decode child emitted."; exit 0; }
    wait_for_proxy_and_cleanup $WORKER_PID "decode child"
fi

if [ "${_BENCH_RC:-0}" -eq 0 ]; then
    echo "Script completed successfully."
else
    echo "Script completed with benchmark rc=${_BENCH_RC}."
fi
exit "${_BENCH_RC:-0}"
