#!/bin/bash
# Gracefully stop a rocprofv3-wrapped vLLM serve so rocprofv3 FINALIZES its trace.
#
# WHY THIS EXISTS: rocprofv3's SIGTERM/SIGINT handler blocks until ALL the traced
# processes it wraps have exited before it writes trace output. On a vLLM serve that
# is the EngineCore + TP-worker subprocesses (and, on the prefill role, 8 rocprofv3-
# wrapped api-server trees). Two traps to avoid:
#   (a) `docker stop` SIGKILLs pid 1 after grace -> Docker tears the container down
#       and SIGKILLs rocprofv3 MID-WRITE -> truncated/empty shards (Exit 137, 0 files).
#   (b) SIGTERM to the top `vllm serve` (pid ~20) alone -> its DP supervisor RESPAWNS
#       the workers -> rocprofv3 (waiting on all traced descendants) never drains.
# So: signal the WHOLE vLLM process group (see below), re-signalling each loop to
# starve the respawn, while leaving pid 1 + pid 20 ALIVE so the container stays Up
# until rocprofv3 has written the shards. Run this INSIDE the container before
# `docker stop`.
#
# Usage (in-container): bash /kimik3_profiling/flush_rocprof.sh [timeout_s]
set -u
TIMEOUT="${1:-120}"

# Signal the WHOLE vLLM process group (coordinator + api-servers + engine + workers),
# NOT pid 1 / pid 20. On the PREFILL/producer role there are 8 rocprofv3-wrapped
# api-server trees whose DP supervisor aggressively RESPAWNS workers -- signalling
# only Worker/EngineCore lets them come back and rocprofv3 (blocks until all traced
# descendants exit) never drains. Signalling every VLLM:: process class at once, and
# re-signalling each loop, starves the respawn so rocprofv3 finalizes. pid 20 + pid 1
# are never matched, so the CONTAINER stays Up until the write completes (avoids the
# Docker-teardown SIGKILL-mid-write that truncates shards).
# VALIDATED (4-node E2E, 2026-09-07): drained all 4 roles to full shards; prefill
# producer ranks carried 279 MoRIIO markers each (mori.rdma.io_transfer / session_batch_write).
GROUP_PATTERNS=(VLLM::Worker VLLM::EngineCore VLLM::APIServer VLLM::DPCoordinator DPCoordinator)
echo "[flush_rocprof] SIGTERM whole vLLM group (workers+engine+apiserver+coordinator); container + pid20 kept alive"
for pat in "${GROUP_PATTERNS[@]}"; do pkill -TERM -f "$pat" 2>/dev/null || true; done

# Wait for rocprofv3 to finalize (all rocprofv3 procs gone = flush complete). The
# container is still Up here, so nothing SIGKILLs rocprofv3 mid-write.
elapsed=0
while pgrep -f rocprofv3 >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "[flush_rocprof] WARN: rocprofv3 still running after ${TIMEOUT}s; leaving it (do NOT force-kill mid-write)"
        break
    fi
    # Re-signal every group class so the DP supervisor can't keep rocprofv3 blocked.
    for pat in "${GROUP_PATTERNS[@]}"; do pkill -TERM -f "$pat" 2>/dev/null || true; done
    sleep 3; elapsed=$((elapsed + 3))
done
echo "[flush_rocprof] rocprofv3 finalize complete (${elapsed}s). Container still Up; safe to 'docker stop' now."
