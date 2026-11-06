#!/bin/bash
# =============================================================================
# GLM-5.2-MXFP4 — 1P1D prefill/decode disaggregation on 2x MI355X (8x gfx950)
#   ATOM native engine (atom.entrypoints.openai_server) + atomesh router
#   KV cache moves prefill -> decode over the mooncake TransferEngine (TCP).
#
# Per-NODE launcher. Run once per role:
#   PROXY_IP=<prefillIP> DECODE_IP=<decodeIP> ROLE=prefill ./serve_atom.sh   # node A
#   PROXY_IP=<prefillIP> DECODE_IP=<decodeIP> ROLE=decode  ./serve_atom.sh   # node B
#   PROXY_IP=<prefillIP> DECODE_IP=<decodeIP> ROLE=router  ./serve_atom.sh   # node A, after A+B are up
#
# Two non-obvious settings make this work on this class of node (see README):
#   1) MXFP4 online_quant_config excludes "*expert*" -> experts stay MXFP4 and
#      are NOT re-quantized. The FP8 per-block re-quant deadlocks the decode.
#   2) MC_FORCE_TCP=1 -> mooncake moves KV over TCP. Without it mooncake attempts
#      GPU-direct RDMA writes and every transfer returns "chunk error -1" (these
#      nodes have no amdgpu_peermem, so GPU-direct RDMA registration is unavailable).
# =============================================================================
set -uo pipefail
ROLE="${ROLE:?prefill|decode|router}"
IMG="${IMG:-rocm/atom-dev:latest}"
MODEL="${MODEL:-/models/GLM-5.2-MXFP4}"          # weights staged here on every node
MOUNT="${MOUNT:-/models}"                          # host dir mounted into the container (must contain MODEL)
TP="${TP:-8}"
CONTAINER="${CONTAINER:-atom_${ROLE}}"
LOG="${LOG:-$PWD/logs}"; mkdir -p "$LOG"; LOG="$(cd "$LOG" && pwd)"   # absolute (docker -v needs it)
CACHE_ROOT="${CACHE_ROOT:-$PWD/cache}"; mkdir -p "$CACHE_ROOT" 2>/dev/null || true
CACHE_ROOT="$(cd "$CACHE_ROOT" && pwd)"

PROXY_IP="${PROXY_IP:?prefill/proxy node IP}"
PING_PORT="${PING_PORT:-36367}"; ROUTER_PORT="${ROUTER_PORT:-30000}"
PF_PORT="${PF_PORT:-8010}"; DC_PORT="${DC_PORT:-8020}"; HS_PORT="${HS_PORT:-6301}"

# --- ionic RDMA provider libs (mooncake initializes an RDMA context even in TCP
#     mode, so the host provider libs must be visible inside the container) ------
IONIC_VER_SO=$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.0.* 2>/dev/null | head -1)
NICM=()
[ -f /usr/lib/x86_64-linux-gnu/libionic.so ] && NICM+=(-v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro)
[ -f /usr/lib/x86_64-linux-gnu/libionic.so.1 ] && NICM+=(-v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro)
[ -n "$IONIC_VER_SO" ] && NICM+=(-v "${IONIC_VER_SO}:${IONIC_VER_SO}:ro")
[ -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ] && NICM+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
[ -f /etc/libibverbs.d/ionic.driver ] && NICM+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)

# --- engine + mooncake-TCP env -----------------------------------------------
MENV=(
  # ATOM: QuickReduce custom all-reduce (TP8) + GLM DSA QK-norm quant fusion.
  -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 -e AITER_USE_FLYDSL_MOE_SORTING=1
  -e ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
  # mooncake KV over TCP (the transfer fix). MC_FORCE_TCP is load-bearing:
  # without it mooncake tries GPU-direct RDMA writes -> "chunk error -1".
  -e MC_FORCE_TCP="${MC_FORCE_TCP:-1}" -e MC_TCP_ENABLE_CONNECTION_POOL=true
  -e MC_SOCKET_IFNAME="${SOCKET_IFNAME:-ens3}" -e MC_LOG_LEVEL="${MC_LOG_LEVEL:-INFO}"
  -e MORI_SOCKET_IFNAME="${SOCKET_IFNAME:-ens3}"
  -e RDMAV_FORK_SAFE=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e PYTHONHASHSEED=0
  -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME:-ens3}" -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME:-ens3}"
  -e NCCL_IB_RETRY_CNT=15 -e NCCL_IB_TIMEOUT=22
  # First-run AITER JIT compiles (tens of seconds each, serialized) can exceed
  # NCCL's watchdog/heartbeat -> rank killed. Disable the heartbeat kill + extend
  # timeouts so the cold compiles survive.
  -e TORCH_NCCL_ENABLE_MONITORING=0 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0 -e NCCL_TIMEOUT=3600
  -e ATOM_ENGINE_READY_TIMEOUT_S=3600
)
DEV=(--device /dev/kfd --device /dev/dri --device /dev/infiniband)
COMMON=(--network host --ipc host --privileged --group-add video
  --cap-add IPC_LOCK --cap-add NET_ADMIN --ulimit memlock=-1 --ulimit stack=67108864
  --ulimit nofile=1048576:1048576 --shm-size 128G
  -v "$MOUNT":"$MOUNT":ro -v /sys:/sys
  -v /sys/class/infiniband:/sys/class/infiniband:ro -v "$LOG":/out)

# per-worker isolated JIT caches (host-persisted so recompiles survive restarts)
CACHE="$CACHE_ROOT/atomcache_${ROLE}"; mkdir -p "$CACHE" 2>/dev/null || true
CENV=(-e HOME=/out -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton
  -e AITER_JIT_DIR=/cache/aiter -v "$CACHE":/cache)

# GLM-5.2 MXFP4 online-quant config. exclude_layer "*expert*" keeps the MoE
# experts in MXFP4 (not re-quantized) -> avoids the decode online-quant deadlock.
QCFG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'

serve(){ local port=$1 kvrole=$2
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --entrypoint bash "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${MENV[@]}" "${CENV[@]}" \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$IMG" -c "
      # mooncake native libs live in the venv site-packages; put them + rocm on the loader path.
      export LD_LIBRARY_PATH=\$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
      python -m atom.entrypoints.openai_server --model $MODEL --host 0.0.0.0 --server-port $port \
        -tp $TP --kv_cache_dtype fp8 --no-enable_prefix_caching \
        --online_quant_config '$QCFG' \
        --kv-transfer-config '{\"kv_role\":\"$kvrole\",\"kv_connector\":\"mooncake\",\"protocol\":\"tcp\",\"proxy_ip\":\"$PROXY_IP\",\"proxy_ping_port\":$PING_PORT,\"http_port\":$port,\"handshake_port\":$HS_PORT}' \
        2>&1 | tee /out/atom_$ROLE.log"
  echo "[atom-$ROLE] launched on :$port (mooncake-tcp $kvrole, proxy=$PROXY_IP)" | tee "$LOG/serve_$ROLE.log"
}

case "$ROLE" in
  prefill) serve "$PF_PORT" kv_producer ;;
  decode)  serve "$DC_PORT" kv_consumer ;;
  router)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER" --network host -v "$LOG":/out \
      -v "$MOUNT":"$MOUNT":ro "$IMG" bash -lc "
      exec atomesh launch --host 0.0.0.0 --port $ROUTER_PORT --pd-disaggregation \
        --prefill http://$PROXY_IP:$PF_PORT --decode http://${DECODE_IP:?}:$DC_PORT \
        --policy random --backend atom --disable-health-check --disable-circuit-breaker \
        --log-level info 2>&1 | tee /out/atom_router.log"
    echo "[atom-router] launched :$ROUTER_PORT (prefill=$PROXY_IP:$PF_PORT decode=$DECODE_IP:$DC_PORT)" | tee "$LOG/serve_router.log"
    ;;
  *) echo "ROLE must be prefill|decode|router"; exit 1 ;;
esac
