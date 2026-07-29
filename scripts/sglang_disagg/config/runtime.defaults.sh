#!/bin/bash
# Entrypoint runtime defaults - aggregates Cluster/Launcher/Model inline ${VAR:-default}s
# lifted from sglang_disagg_mori_io_ep.sh. See ../CONFIG.md for the bucket taxonomy.
# TODO(stage2): split into per-bucket defaults (defaults-cluster/launcher/model).
#
# Sourced by the entrypoint after SCRIPT_DIR is set (line 7) and before first use (line 37).
# SCRIPT_DIR must be set by the caller (used by MODELS_YAML below).

# Cluster: rendezvous + topology
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
IPADDRS="${IPADDRS:-localhost}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Launcher: barrier + router readiness knobs
BARRIER_PORT="${BARRIER_PORT:-4342}"
SEARCH_SIGNAL="${SEARCH_SIGNAL:-The server is fired up and ready to roll!}"
ROUTER_READY_TIMEOUT_SECONDS="${ROUTER_READY_TIMEOUT_SECONDS:-4000}"
ROUTER_POLL_SLEEP_SECONDS="${ROUTER_POLL_SLEEP_SECONDS:-10}"

# Model: per-worker TP size + models.yaml path
GENERIC_TP_SIZE="${GENERIC_TP_SIZE:-8}"
MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"
