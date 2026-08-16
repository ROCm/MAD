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

# Prefer the routable fabric IP (FABRIC_SUBNET, default 10.158.) over hostname -I's
# first entry: nodes with a 10.224.x overlay listed first would bind the socket_barrier
# / advertise host_ip on an unreachable NIC -> prefill<->decode barrier hangs "Waiting
# for nodes". Matches the IPADDRS selection in run_xPyD_models.slurm. Falls back to $1.
FABRIC_SUBNET="${FABRIC_SUBNET:-10.158.}"
host_ip=$(hostname -I | awk -v pfx="$FABRIC_SUBNET" '{f=$1; for(i=1;i<=NF;i++) if(index($i,pfx)==1){f=$i; break} print f}')
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

_wait_log_signal_or_fail() {
    local LOG_FILE="$1" LABEL="$2" SEARCH_SIGNAL="$3" TIMEOUT_SECONDS="$4" SLEEP_SECONDS="$5"
    local ELAPSED=0
    until grep -Fq "${SEARCH_SIGNAL}" "${LOG_FILE}" 2>/dev/null; do
        if [ "${ELAPSED}" -ge "${TIMEOUT_SECONDS}" ]; then
            echo "Timeout (${TIMEOUT_SECONDS}s): '${SEARCH_SIGNAL}' not found in ${LABEL}: ${LOG_FILE}" \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
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
    #    Precedence: image-baked ENV  <  models.yaml env:  <  submit-time -e.
    #    models.yaml MUST override image-baked ENV: a DeepSeek-tuned disagg image
    #    bakes KV_BLOCK_SIZE=16 / VLLM_ROCM_USE_AITER_MLA=0 / VLLM_CUDAGRAPH_MODE=
    #    PIECEWISE etc. as container ENV, which would otherwise shadow a model's own
    #    recipe (GLM-5.1 DSA needs block=1 + AITER sparse MLA on). But a genuine
    #    submit-time `-e VAR=...` must still win. The slurm can tell the two apart
    #    (it runs on the host) and passes MODELS_YAML_PROTECT = the space-separated
    #    list of keys the USER set at submit; the driver protects only those. When
    #    MODELS_YAML_PROTECT is unset (script run directly, no slurm), fall back to
    #    the old "skip if in env" behavior so nothing regresses.
    _yaml_env="$(python3 - <<'PY'
import os, yaml, shlex
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
cfg = m.get(os.environ["MODEL_NAME"]) or {}
protect_raw = os.environ.get("MODELS_YAML_PROTECT")
have_protect = protect_raw is not None
protect = set((protect_raw or "").split())
for k, v in (cfg.get("env") or {}).items():
    if have_protect:
        # 3-tier: yaml overrides baked ENV; only a user submit-time -e (in the
        # protect-list) wins over yaml.
        if k in protect:
            continue
    else:
        # No protect-list (direct run): legacy behavior — any existing env wins.
        if k in os.environ:
            continue
    print(f'export {k}={shlex.quote(str(v))}')
PY
)"
    [[ -n "$_yaml_env" ]] && eval "$_yaml_env"

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

# Allow ${VAR} / ${VAR:-default} inside recipe flag strings, so a recipe can expose
# a per-run knob (e.g. GLM_MAX_MODEL_LEN) that a run overrides by exporting the name
# -- no yaml edit. Deliberately narrow: only those two forms, no command
# substitution, no eval. A recipe with no ${...} is unaffected.
import re
_SUB = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
def expand(s):
    return _SUB.sub(lambda mo: os.environ.get(mo.group(1)) or (mo.group(2) or ""), s)

def compose(role):
    return expand(" ".join(x for x in [
        cfg.get("base_flags",""), cfg.get(f"{mode}_flags",""),
        (role.get(mode,"") if isinstance(role,dict) else ""),
        cfg.get("experimental_flags",""),
    ] if x).strip())
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
# Container barrier + runtime patches (skipped under DRY_RUN)
# =============================================================================
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    _BARRIER_PORT="${CONTAINER_BARRIER_PORT:-2222}"
    for _pid in $(ss -tlnp sport = ${_BARRIER_PORT} 2>/dev/null | grep -oP "pid=\K\d+"); do
        kill -9 "$_pid" 2>/dev/null
    done
    sleep 2
    echo "Waiting at the container creation barrier on $host_name"
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --local-ip ${host_ip} --local-port ${_BARRIER_PORT} --enable-port \
        --node-ips ${IPADDRS} --node-ports ${_BARRIER_PORT}
    connector_runtime_patch
fi

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================
if [ "$NODE_RANK" -eq 0 ]; then
    print_node_info "Prefill master + Proxy node (co-located)"
    connector_launch_worker "master" "${PREFILL_DP_SIZE}" "${PREFILL_MASTER_ADDR}" "kv_producer" "prefill"
    local_worker_pid=$WORKER_PID
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[dry-run] rank0 prefill master emitted; skipping proxy/benchmark."; exit 0; }

    connector_wait_workers_ready
    connector_start_proxy

    # connector_start_proxy sets BENCHMARK_PORT (router->ROUTER_PORT, toy->PROXY_PORT).
    # Fall back to PROXY_PORT only if the connector didn't set it.
    export BENCHMARK_PORT="${BENCHMARK_PORT:-${PROXY_PORT}}"
    bash "$NIXL_COOKBOOK_PATH/${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}"

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

echo "Script completed successfully."
exit 0
