#!/bin/bash
# KV offload overlay (orthogonal to CONNECTOR/WIDE_EP): layers a CPU-RAM (optionally + fs)
# KV tier on top of the disagg P/D connector via vLLM's MultiConnector. The offload
# sub-connector is listed first so a decode worker hits the local cache before the P->D
# fetch. KV_OFFLOAD=none is a no-op. See README "Tiered prefix caching" for usage.
#
# Env: KV_OFFLOAD       none (default) | cpu | fs   (fs needs OFFLOAD_DISK_PATH; not for SimpleCPU)
#      OFFLOAD_BACKEND  OffloadingConnector|native (default) | SimpleCPUOffloadConnector | LMCacheConnectorV1|lmcache
#      OFFLOAD_CPU_BYTES_PER_NODE   single per-node host-RAM budget for every backend (default 100 GiB)
#      OFFLOAD_DISK_PATH            fs-tier base dir (node-accessible, not tmpfs); a per-host subdir is appended
#      LMCACHE_MAX_LOCAL_DISK_SIZE  lmcache only: per-worker disk tier in GB (default 0.0)
#      LMCACHE_MAX_LOCAL_CPU_SIZE   lmcache only: explicit per-worker GB, overrides the derived value
#
# Sizing: OffloadingConnector pins cpu_bytes_to_use / world_size per rank (world_size = intra-engine
# TP degree; DP replicas don't reduce it). With ranks_per_node = DP_PARALLEL_SIZE_LOCAL and the
# connector-set _OFFLOAD_TP_WORLD (1 for wideEP, TP size for TP):
#   native/SimpleCPU  cpu_bytes_to_use = PER_NODE * _OFFLOAD_TP_WORLD / ranks_per_node
#   lmcache           LMCACHE_MAX_LOCAL_CPU_SIZE = (PER_NODE / 1 GiB) / ranks_per_node   (GB per worker)

KV_OFFLOAD="${KV_OFFLOAD:-none}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-OffloadingConnector}"
export OFFLOAD_CPU_BYTES_PER_NODE="${OFFLOAD_CPU_BYTES_PER_NODE:-107374182400}"

# Worker ranks on this node (one vLLM worker process per GPU).
_kv_offload_ranks_per_node() {
    local n="${DP_PARALLEL_SIZE_LOCAL:-1}"
    (( n > 0 )) || n=1
    echo "$n"
}

# native/SimpleCPU per-rank cpu_bytes_to_use from the per-node budget + TP world.
_kv_offload_bytes_per_rank() {
    local ranks tpw
    ranks="$(_kv_offload_ranks_per_node)"
    tpw="${_OFFLOAD_TP_WORLD:-1}"
    (( tpw > 0 )) || tpw=1
    echo $(( OFFLOAD_CPU_BYTES_PER_NODE * tpw / ranks ))
}

# lmcache per-worker CPU tier (GB) from the per-node budget (topology: /ranks).
_kv_offload_lmcache_gb_per_worker() {
    local ranks; ranks="$(_kv_offload_ranks_per_node)"
    awk -v b="$OFFLOAD_CPU_BYTES_PER_NODE" -v n="$ranks" 'BEGIN{printf "%.1f", b/1073741824/n}'
}

kv_offload_enabled() {
    [[ "${KV_OFFLOAD:-none}" != "none" ]]
}

kv_offload_fs_enabled() {
    [[ "${KV_OFFLOAD:-none}" == "fs" ]]
}

# Per-host dir for the filesystem tier. Only meaningful when KV_OFFLOAD=fs.
# The per-host subdir keeps prefill and decode nodes from colliding on a shared mount.
_kv_offload_fs_dir() {
    kv_offload_fs_enabled || return 0
    printf '%s/%s' "${OFFLOAD_DISK_PATH%/}" "$(hostname)"
}

# Validate the KV_OFFLOAD tier and (when active) its backend. Exits on bad input.
_kv_offload_validate() {
    case "${KV_OFFLOAD}" in
        none|cpu|fs) ;;
        *)
            echo "Error: unsupported KV_OFFLOAD='${KV_OFFLOAD}' (expected none|cpu|fs)." >&2
            exit 1
            ;;
    esac
    kv_offload_enabled || return 0
    case "${OFFLOAD_BACKEND}" in
        OffloadingConnector|native|SimpleCPUOffloadConnector|LMCacheConnectorV1|lmcache) ;;
        *)
            echo "Error: unsupported OFFLOAD_BACKEND='${OFFLOAD_BACKEND}' (expected OffloadingConnector|SimpleCPUOffloadConnector|LMCacheConnectorV1)." >&2
            exit 1
            ;;
    esac
    if kv_offload_fs_enabled; then
        if [[ -z "${OFFLOAD_DISK_PATH:-}" ]]; then
            echo "Error: KV_OFFLOAD=fs requires OFFLOAD_DISK_PATH to be set." >&2
            exit 1
        fi
        if [[ "${OFFLOAD_BACKEND}" == "SimpleCPUOffloadConnector" ]]; then
            echo "Error: OFFLOAD_BACKEND=SimpleCPUOffloadConnector has no filesystem tier; use KV_OFFLOAD=cpu, or OffloadingConnector/LMCacheConnectorV1 for fs." >&2
            exit 1
        fi
    fi
}

# Echo the kv-transfer-config for `vllm serve`: base JSON unchanged when none, else a
# MultiConnector wrapping [<offload sub-connector>, base]. The offload sub-connector is
# OffloadingConnector (native), SimpleCPUOffloadConnector, or LMCacheConnectorV1 (lmcache).
kv_offload_wrap() {
    local base_json="$1"
    if ! kv_offload_enabled; then
        printf '%s' "$base_json"
        return 0
    fi

    OFFLOAD_CPU_BYTES="$(_kv_offload_bytes_per_rank)" OFFLOAD_BACKEND="${OFFLOAD_BACKEND}" \
    _OFFLOAD_FS_DIR="$(_kv_offload_fs_dir)" _BASE_JSON="${base_json}" python3 - <<'PY'
import json, os
base = json.loads(os.environ["_BASE_JSON"])
# MultiConnector rebuilds each sub-connector as KVTransferConfig(**sub_dict, engine_id=...),
# so a sub-dict that still carries "engine_id" raises "multiple values for 'engine_id'".
# Lift it to the outer MultiConnector; vLLM re-applies it to each sub-connector.
engine_id = base.pop("engine_id", None)
backend = os.environ["OFFLOAD_BACKEND"]
if backend in ("LMCacheConnectorV1", "lmcache"):
    # LMCache reads its config from the env (LMCACHE_*); see kv_offload_setup_env.
    offload = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
    }
elif backend == "SimpleCPUOffloadConnector":
    offload = {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(os.environ["OFFLOAD_CPU_BYTES"]),
        },
    }
else:  # OffloadingConnector / native
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
# the filesystem-tier dir when KV_OFFLOAD=fs. Submit-time exports win.
kv_offload_setup_env() {
    kv_offload_enabled || return 0

    local disk_dir=""
    if kv_offload_fs_enabled; then
        disk_dir="$(_kv_offload_fs_dir)"
        mkdir -p "${disk_dir}" || echo "[kv_offload] WARNING: failed to create ${disk_dir}" >&2
    fi

    if [[ "${OFFLOAD_BACKEND}" != "LMCacheConnectorV1" && "${OFFLOAD_BACKEND}" != "lmcache" ]]; then
        # per-rank cpu_bytes_to_use is derived in kv_offload_wrap at launch (needs _OFFLOAD_TP_WORLD)
        echo "[kv_offload] ${OFFLOAD_BACKEND}: OFFLOAD_CPU_BYTES_PER_NODE=${OFFLOAD_CPU_BYTES_PER_NODE}B (ranks/node=$(_kv_offload_ranks_per_node))"
        [[ -n "${disk_dir}" ]] && echo "[kv_offload] OffloadingConnector fs tier: root_dir=${disk_dir}"
        return 0
    fi

    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-$(_kv_offload_lmcache_gb_per_worker)}"
    # Stable hashing across workers so prefix keys match; LMCache prometheus multiproc dir.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/lmcache_prometheus}"
    mkdir -p "${PROMETHEUS_MULTIPROC_DIR}" || echo "[kv_offload] WARNING: failed to create ${PROMETHEUS_MULTIPROC_DIR}" >&2
    echo "[kv_offload] lmcache CPU tier: LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE} GB/worker" \
         "PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR}"

    # Disk tier: LMCache spills CPU-evicted chunks here (LRU, per-GPU sharded).
    [[ -n "${disk_dir}" ]] || return 0
    export LMCACHE_LOCAL_DISK="${disk_dir}"
    export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-0.0}"
    echo "[kv_offload] lmcache disk tier: LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK}" \
         "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE} GB/worker"
}

# Run in the parent shell at source time: validate (so bad input aborts the launcher, not a
# $(...) subshell) and export tier env before `vllm serve`. No-op when KV_OFFLOAD=none.
_kv_offload_validate
kv_offload_setup_env

# --enable-prefix-caching when the run needs it (agentic, KV_OFFLOAD, or explicit), else the
# default sweep's --no-enable-prefix-caching.
kv_prefix_cache_arg() {
    if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" || "${ENABLE_PREFIX_CACHE:-0}" == "1" \
          || "${BENCHMARK_SCRIPT:-}" == "agentic" ]] || kv_offload_enabled; then
        echo "--enable-prefix-caching"
    else
        echo "--no-enable-prefix-caching"
    fi
}
