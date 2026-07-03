#!/bin/bash
# Shared parallelism helpers for the unified vLLM disagg launcher.
# =============================================================================
# The WIDE_EP axis (0=TP, 1=wideEP) decides the *parallelism* shape of a worker.
# The exact `vllm serve` template still lives in each connector (the argv order
# and a few flags genuinely differ by connector/EP backend, and we preserve them
# byte-for-byte for parity), but the pieces that are IDENTICAL across connectors
# live here so there is one source of truth:
#
#   - master/child role args for wideEP (DP) workers
#   - the DP "degree" math is computed in the driver (PREFILL_DP_SIZE etc.)
#
# Sourced by vllm_disagg.sh before the connector profile.
# =============================================================================

# parallelism_role_args <role> <dp_start_rank> <gpus_per_node>
# Echoes the wideEP (DP) master/child distinguishing flags, space-separated:
#   master -> --api-server-count=<gpus_per_node>
#   child  -> --data-parallel-start-rank <r> --headless
# (TP mode has no master/child distinction; callers don't invoke this for TP.)
parallelism_role_args() {
    local role="$1" dp_start_rank="$2" gpus_per_node="${3:-8}"
    if [[ "$role" == "master" ]]; then
        printf -- '--api-server-count=%s' "${gpus_per_node}"
    else
        printf -- '--data-parallel-start-rank %s --headless' "${dp_start_rank}"
    fi
}

# parallelism_is_wide_ep -> return 0 if WIDE_EP=1, else 1.
parallelism_is_wide_ep() {
    [[ "${WIDE_EP:-0}" == "1" ]]
}
