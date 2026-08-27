#!/bin/bash
# Tiered prefix caching — KV offload overlay (orthogonal to CONNECTOR/WIDE_EP).
# Layers vLLM's OffloadingConnector (GPU KV -> CPU-RAM tier) on top of the disagg
# P/D connector via MultiConnector. Load reads the first matching sub-connector,
# save writes to all — OffloadingConnector is listed first so a decode worker hits
# the local CPU cache before the P->D fetch. KV_OFFLOAD=none is a no-op.
#
# Env: KV_OFFLOAD        = none (default) | cpu
#      OFFLOAD_CPU_BYTES = pinned host bytes for the CPU tier (default 100 GB)

KV_OFFLOAD="${KV_OFFLOAD:-none}"
OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES:-107374182400}"

kv_offload_enabled() {
    [[ "${KV_OFFLOAD:-none}" != "none" ]]
}

# Echo the kv-transfer-config for `vllm serve`: base JSON unchanged when none,
# else a MultiConnector wrapping [OffloadingConnector, base].
kv_offload_wrap() {
    local base_json="$1"
    if ! kv_offload_enabled; then
        printf '%s' "$base_json"
        return 0
    fi

    case "${KV_OFFLOAD}" in
        cpu) ;;
        *)
            echo "Error: unsupported KV_OFFLOAD='${KV_OFFLOAD}' (expected none|cpu)." >&2
            exit 1
            ;;
    esac

    OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES}" _BASE_JSON="${base_json}" python3 - <<'PY'
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
