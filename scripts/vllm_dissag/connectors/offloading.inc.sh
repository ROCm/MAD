#!/bin/bash
# Tiered prefix caching — KV offload overlay (orthogonal to CONNECTOR/WIDE_EP).
# Layers a CPU-RAM KV tier (GPU KV -> host) on top of the disagg P/D connector via
# vLLM's MultiConnector. Load reads the first matching sub-connector, save writes to
# all — the offload sub-connector is listed first so a decode worker hits the local
# CPU cache before the P->D fetch. KV_OFFLOAD=none is a no-op.
#
# The CPU tier's backend is selectable (OFFLOAD_BACKEND):
#   native  (default) -> vLLM's built-in OffloadingConnector. Sized by OFFLOAD_CPU_BYTES.
#   lmcache           -> LMCache's LMCacheConnectorV1. Sized by LMCACHE_MAX_LOCAL_CPU_SIZE.
#                        Requires an image with `lmcache` installed (the stock
#                        vllm/vllm-openai-rocm image does NOT ship it); LMCache reads
#                        its config from the process env (see kv_offload_setup_env).
#                        Optionally add a filesystem tier below the CPU tier via
#                        OFFLOAD_DISK_PATH (GPU -> CPU RAM -> local disk). LMCache LRU
#                        spills CPU-evicted chunks to disk instead of dropping them.
#
# Env: KV_OFFLOAD        = none (default) | cpu
#      OFFLOAD_BACKEND   = native (default) | lmcache        (only read when KV_OFFLOAD=cpu)
#      OFFLOAD_CPU_BYTES = pinned host bytes for the native CPU tier (default 100 GB)
#      LMCACHE_MAX_LOCAL_CPU_SIZE  = per-worker CPU tier in GB for lmcache (default 100.0)
#      OFFLOAD_DISK_PATH           = base dir for the lmcache disk tier (unset = no disk tier).
#                                    Must be node-local disk (e.g. /mnt/m2m_nobackup/...), NOT
#                                    tmpfs. A per-host subdir is appended to avoid P/D collisions.
#      LMCACHE_MAX_LOCAL_DISK_SIZE = per-worker disk tier in GB (default 0.0 = unbounded-ish)

KV_OFFLOAD="${KV_OFFLOAD:-none}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-native}"
OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES:-107374182400}"

kv_offload_enabled() {
    [[ "${KV_OFFLOAD:-none}" != "none" ]]
}

# Validate the KV_OFFLOAD tier and (when active) its backend. Exits on bad input.
_kv_offload_validate() {
    case "${KV_OFFLOAD}" in
        none|cpu) ;;
        *)
            echo "Error: unsupported KV_OFFLOAD='${KV_OFFLOAD}' (expected none|cpu)." >&2
            exit 1
            ;;
    esac
    kv_offload_enabled || return 0
    case "${OFFLOAD_BACKEND}" in
        native|lmcache) ;;
        *)
            echo "Error: unsupported OFFLOAD_BACKEND='${OFFLOAD_BACKEND}' (expected native|lmcache)." >&2
            exit 1
            ;;
    esac
}

# Echo the kv-transfer-config for `vllm serve`: base JSON unchanged when none, else a
# MultiConnector wrapping [<offload sub-connector>, base]. The offload sub-connector is
# OffloadingConnector (native) or LMCacheConnectorV1 (lmcache).
kv_offload_wrap() {
    local base_json="$1"
    if ! kv_offload_enabled; then
        printf '%s' "$base_json"
        return 0
    fi
    _kv_offload_validate

    OFFLOAD_BACKEND="${OFFLOAD_BACKEND}" OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES}" \
    _BASE_JSON="${base_json}" python3 - <<'PY'
import json, os
base = json.loads(os.environ["_BASE_JSON"])
# vLLM's MultiConnector rebuilds each sub-connector as
# KVTransferConfig(**sub_dict, engine_id=engine_id) (multi_connector.py
# _get_connector_classes_and_configs). It reads engine_id via dict.get() WITHOUT
# popping it, so a sub-connector dict that still carries an "engine_id" key raises
# "got multiple values for keyword argument 'engine_id'". Lift engine_id off the
# base dict to the outer MultiConnector; vLLM's fallback (ktc.get("engine_id",
# outer.engine_id)) then re-applies it to the base (and offload) sub-connector.
engine_id = base.pop("engine_id", None)
backend = os.environ["OFFLOAD_BACKEND"]
if backend == "lmcache":
    # LMCache is configured entirely via the process env (LMCACHE_*), so the
    # sub-connector dict is just the connector name + role. See kv_offload_setup_env.
    offload = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
    }
else:
    offload = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(os.environ["OFFLOAD_CPU_BYTES"]),
        },
    }
multi = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "connectors": [offload, base],
    },
}
if engine_id is not None:
    multi["engine_id"] = engine_id
print(json.dumps(multi))
PY
}

# Export the process env the active offload backend needs before `vllm serve`.
# none/native: no-op (OffloadingConnector is fully configured via its JSON dict).
# lmcache: LMCache reads its tier config from the environment, so export the CPU-tier
#   size + the determinism/metrics vars the llm-d guide validated. Passthrough: a
#   submit-time export of any of these wins (`${VAR:-default}`).
kv_offload_setup_env() {
    kv_offload_enabled || return 0
    [[ "${OFFLOAD_BACKEND}" == "lmcache" ]] || return 0
    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-100.0}"
    # Stable hashing across workers so prefix keys match; LMCache prometheus multiproc dir.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/lmcache_prometheus}"
    mkdir -p "${PROMETHEUS_MULTIPROC_DIR}" 2>/dev/null || true
    echo "[kv_offload] lmcache CPU tier: LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE} GB/worker" \
         "PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR}"

    # Optional filesystem tier below the CPU tier. LMCache (LRU) spills CPU-evicted
    # chunks here instead of dropping them; retrieval checks CPU then disk. Sharded
    # per-GPU by LMCache (local_disk_path_sharding=by_gpu); we append per-host so the
    # prefill and decode nodes don't collide on a shared mount.
    [[ -n "${OFFLOAD_DISK_PATH:-}" ]] || return 0
    local disk_dir="${OFFLOAD_DISK_PATH%/}/$(hostname)"
    mkdir -p "${disk_dir}" 2>/dev/null || true
    export LMCACHE_LOCAL_DISK="${disk_dir}"
    export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-0.0}"
    echo "[kv_offload] lmcache disk tier: LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK}" \
         "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE} GB/worker"
}
