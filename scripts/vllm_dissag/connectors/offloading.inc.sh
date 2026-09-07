#!/bin/bash
# Tiered prefix caching — KV offload overlay (orthogonal to CONNECTOR/WIDE_EP).
# Layers a CPU-RAM KV tier (GPU KV -> host) on top of the disagg P/D connector via
# vLLM's MultiConnector. Load reads the first matching sub-connector, save writes to
# all — the offload sub-connector is listed first so a decode worker hits the local
# CPU cache before the P->D fetch. KV_OFFLOAD=none is a no-op.
#
# The CPU tier's backend is selectable (OFFLOAD_BACKEND):
#   native  (default) -> vLLM's built-in OffloadingConnector. CPU tier sized by
#                        OFFLOAD_CPU_BYTES; fully configured via its JSON dict.
#   lmcache           -> LMCache's LMCacheConnectorV1. CPU tier sized by
#                        LMCACHE_MAX_LOCAL_CPU_SIZE. Requires an image with `lmcache`
#                        installed (the stock vllm/vllm-openai-rocm image does NOT ship
#                        it); LMCache reads its config from the process env.
#
# Either backend can add a filesystem tier below the CPU tier via OFFLOAD_DISK_PATH
# (GPU -> CPU RAM -> disk):
#   native  -> an OffloadingConnector TieringOffloadingSpec with an `fs` secondary tier.
#   lmcache -> LMCache (LRU) spills CPU-evicted chunks to its local_disk.
# A per-host subdir is appended so the prefill and decode nodes don't collide on a
# shared mount.
#
# Env: KV_OFFLOAD        = none (default) | cpu
#      OFFLOAD_BACKEND   = native (default) | lmcache        (only read when KV_OFFLOAD=cpu)
#      OFFLOAD_CPU_BYTES = pinned host bytes for the native CPU tier (default 100 GB)
#      LMCACHE_MAX_LOCAL_CPU_SIZE  = per-worker CPU tier in GB for lmcache (default 100.0)
#      OFFLOAD_DISK_PATH           = base dir for a filesystem tier (unset = no disk tier).
#                                    Node-local disk (e.g. /mnt/m2m_nobackup/...), NOT tmpfs.
#      LMCACHE_MAX_LOCAL_DISK_SIZE = lmcache only: per-worker disk tier in GB (default 0.0)

KV_OFFLOAD="${KV_OFFLOAD:-none}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-native}"
export OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES:-107374182400}"

kv_offload_enabled() {
    [[ "${KV_OFFLOAD:-none}" != "none" ]]
}

# Per-host dir for a filesystem tier; empty when OFFLOAD_DISK_PATH is unset. The
# per-host subdir keeps the prefill and decode nodes from colliding on a shared mount.
_kv_offload_fs_dir() {
    [[ -n "${OFFLOAD_DISK_PATH:-}" ]] || return 0
    printf '%s/%s' "${OFFLOAD_DISK_PATH%/}" "$(hostname)"
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

    _OFFLOAD_FS_DIR="$(_kv_offload_fs_dir)" \
    _BASE_JSON="${base_json}" python3 - <<'PY'
import json, os
base = json.loads(os.environ["_BASE_JSON"])
# MultiConnector passes engine_id to each sub-connector as a kwarg; a sub-dict
# still carrying "engine_id" raises "got multiple values". Pop it here and set it
# on the outer dict; vLLM re-applies it to sub-connectors via its fallback.
engine_id = base.pop("engine_id", None)
backend = os.environ["OFFLOAD_BACKEND"]
if backend == "lmcache":
    # LMCache reads its config from the env (LMCACHE_*); see kv_offload_setup_env.
    offload = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
    }
else:
    extra = {"cpu_bytes_to_use": int(os.environ["OFFLOAD_CPU_BYTES"])}
    fs_dir = os.environ.get("_OFFLOAD_FS_DIR", "")
    if fs_dir:
        # Filesystem tier: GPU -> CPU RAM -> fs, via TieringOffloadingSpec.
        extra["spec_name"] = "TieringOffloadingSpec"
        extra["block_size"] = 256
        extra["secondary_tiers"] = [{
            "type": "fs",
            "root_dir": fs_dir,
            "n_read_threads": 16,
            "n_write_threads": 16,
        }]
    offload = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
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

# Export env vars the active offload backend reads before `vllm serve`, and create
# the filesystem-tier dir when OFFLOAD_DISK_PATH is set. Submit-time exports win.
kv_offload_setup_env() {
    kv_offload_enabled || return 0

    local disk_dir=""
    if [[ -n "${OFFLOAD_DISK_PATH:-}" ]]; then
        disk_dir="$(_kv_offload_fs_dir)"
        mkdir -p "${disk_dir}" || echo "[kv_offload] WARNING: failed to create ${disk_dir}" >&2
    fi

    if [[ "${OFFLOAD_BACKEND}" != "lmcache" ]]; then
        [[ -n "${disk_dir}" ]] && echo "[kv_offload] native fs tier: root_dir=${disk_dir}"
        return 0
    fi

    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-100.0}"
    # Stable hashing across workers so prefix keys match; LMCache prometheus multiproc dir.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/lmcache_prometheus}"
    mkdir -p "${PROMETHEUS_MULTIPROC_DIR}" || echo "[kv_offload] WARNING: failed to create ${PROMETHEUS_MULTIPROC_DIR}" >&2
    echo "[kv_offload] lmcache CPU tier: LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE} GB/worker" \
         "PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR}"

    # Optional disk tier: LMCache spills CPU-evicted chunks here (LRU, per-GPU sharded).
    [[ -n "${disk_dir}" ]] || return 0
    export LMCACHE_LOCAL_DISK="${disk_dir}"
    export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-0.0}"
    echo "[kv_offload] lmcache disk tier: LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK}" \
         "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE} GB/worker"
}
