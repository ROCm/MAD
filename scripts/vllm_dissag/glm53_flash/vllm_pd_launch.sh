#!/bin/bash
# =============================================================================
# GLM-5.3-Flash vLLM + MoRIIO disaggregated P/D launcher (per node, per role).
#
# CURRENT PROVEN RECIPE = base image `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix`
# + the 11 Python overlays in ./patches/ (the verified connector / model / aiter
# fixes; see patches/README.md). This launcher bind-mounts them (OVERLAYS=1,
# default) so disagg recalls correctly to 871K tokens. Other host bind-mounts:
# ionic RDMA userspace libs, an optional glibc-swap closure for the ionic driver,
# an optional patched-mori .so set (atomic-MR strip), and a persistent JIT cache.
# (A fully self-contained image with the fixes baked in-source is a follow-up;
# set OVERLAYS=0 to serve the base image bare — it will NOT recall at depth.)
#
# Roles:
#   ROLE=prefill  -> vllm serve (kv_producer) on PF_PORT
#   ROLE=decode   -> vllm serve (kv_consumer) on DC_PORT
#   ROLE=proxy    -> vllm-router (pd-disaggregation, service discovery)
#
# Required env: ROLE, HOST_IP (this node ip), PROXY_IP (prefill/proxy node ip),
#   DECODE_IP, IMG, MODEL.
# Common optional env (defaults in []):
#   MODE=tp4|tp8|ep [tp4]          MoRIIO KV only (tp) or +expert-parallel (ep)
#   GPUUTIL [0.5]  MAXLEN [940000]  KV_DTYPE [auto]  BLOCK_SIZE [4]
#   SPARSE_IDX_MB [512]  A2A [mori_high_throughput]  DP [8]
#   EXTRA_ARGS                      extra `vllm serve` args. RECOMMENDED long-ctx
#                                   recipe = CHUNKED prefill on the prefill leg:
#                                   --enable-chunked-prefill --max-num-batched-tokens 16384
#                                   (keeps per-chunk KV via the connector's cross-
#                                   chunk accumulation + dodges the single-chunk
#                                   Triton compile wall <~780K -> exact recall to
#                                   max_model_len). Single-chunk (--max-num-batched-
#                                   tokens >= MAXLEN) also works but caps ~100K.
#   ROUTER_DP_LOCAL [8]             1 for TP, 8 for EP/DP8 (proxy role)
#   WORKDIR [/tmp/glm53_disagg]     writable dir for logs + compilation configs
#   JITCACHE_OVERRIDE              node-local dir for the /cache JIT cache
#   GLIBC_SWAP [0] + HOSTLIBS       fresh-node glibc-2.39 closure for ionic driver
#   MORI_PATCHED [0] + MORI_SO_DIR  patched mori .so set (ionic atomic-MR strip)
#   ROUTER_BIN                      path to the vllm-router binary (proxy role)
# =============================================================================
set -uo pipefail
ROLE="${ROLE:?prefill|decode|proxy}"
IMG="${IMG:?container image (built from the glmv53flash Dockerfile)}"
MODEL="${MODEL:?path to GLM-5.3-Flash weights (mounted into the container)}"
DP="${DP:-8}"
HOST_IP="${HOST_IP:?this node ip}"
PROXY_IP="${PROXY_IP:?prefill/proxy node ip}"
DECODE_IP="${DECODE_IP:?decode node ip}"
WORKDIR="${WORKDIR:-/tmp/glm53_disagg}"
LOG="${LOG:-$WORKDIR/logs}"
CONTAINER="${CONTAINER:-vllm_${ROLE}}"
mkdir -p "$WORKDIR" "$LOG" 2>/dev/null || true

PF_PORT=20005; DC_PORT=40005; PROXY_PORT=10001
PROXY_PING=${PROXY_PING:-36382}; NOTIFY=${NOTIFY:-61005}

# ---- cudagraph / compilation ----
PREFILL_CUDAGRAPH_MODE="${PREFILL_CUDAGRAPH_MODE:-NONE}"
DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE:-NONE}"
CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8 16 32 64 128 256}"
_compcfg_write() {  # $1=role $2=mode -> writes file, echoes path
  local role="$1"; local mode="$2"; local f="${WORKDIR}/compcfg_${role}.json"
  mkdir -p "${WORKDIR}" 2>/dev/null || true   # ensure dir exists before write (else file silently missing)
  # COMPACT JSON (no spaces) so an unquoted $(cat file) is a single argv token
  # when expanded inside the container shell (see DC_CC below).
  if [ "${EAGER:-1}" = "1" ]; then
    printf '{"mode":0,"cudagraph_mode":"NONE","custom_ops":["+quant_fp8"]}' > "$f" 2>/dev/null || true
  else
    printf '{"cudagraph_mode":"%s","custom_ops":["+quant_fp8"]}' "$mode" > "$f" 2>/dev/null || true
  fi
  echo "$f"
}
PF_CFG_FILE="$(_compcfg_write prefill "$PREFILL_CUDAGRAPH_MODE")"
DC_CFG_FILE="$(_compcfg_write decode "$DECODE_CUDAGRAPH_MODE")"
# vLLM's --compilation-config needs INLINE JSON (a file path is rejected as
# json_invalid). To pass JSON through the docker `bash -lc "..."` heredoc without the
# host shell mangling its quotes, defer the substitution to the CONTAINER's shell:
# the compcfg_*.json file is mounted (via $WORKDIR) at the same path, so we emit a
# literal $(cat <file>) that the container's bash expands at runtime. The \$ keeps it
# unexpanded on the host side; single quotes wrap it as one argv token in-container.
if [ "${EAGER:-1}" = "1" ]; then PF_CC=""; DC_CC=""
else PF_CC="--compilation-config \$(cat $PF_CFG_FILE)"; DC_CC="--compilation-config \$(cat $DC_CFG_FILE)"; fi

# MODE: tp4 / tp8 = tensor-parallel, MoRIIO KV only. ep = DP + expert-parallel + a2a.
MODE="${MODE:-tp4}"
if [ "$MODE" = "tp8" ]; then PARALLEL="-tp 8"; EP_FLAGS=""
elif [ "$MODE" = "tp4" ]; then PARALLEL="-tp 4"; EP_FLAGS=""
else
  PARALLEL="-tp 1 --data-parallel-size ${DP} --data-parallel-size-local ${DP} --data-parallel-address ${HOST_IP} --data-parallel-rpc-port 13345 --api-server-count=${DP}"
  EP_FLAGS="--enable-expert-parallel --all2all-backend ${A2A:-allgather_reducescatter}"
fi

IBDEV="${IBDEV:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"

# ---- ionic RDMA userspace bind-mounts ----
NICM=()
for f in /usr/lib/x86_64-linux-gnu/libionic.so /usr/lib/x86_64-linux-gnu/libionic.so.1 \
         /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so /etc/libibverbs.d/ionic.driver; do
  [ -e "$f" ] && NICM+=(-v "$f:$f:ro")
done
for f in $(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.* 2>/dev/null); do
  case "$f" in *:*) continue;; esac
  NICM+=(-v "$f:$f:ro")
done

# ---- optional glibc-2.39 + RDMA closure swap (fresh nodes whose host ionic
# driver needs GLIBC_2.38 while the image ships 2.35). Provide HOSTLIBS pointing
# at a dir with lib/ (glibc runtime) + etc/libibverbs.d/. Off by default. ----
GLIBC_SWAP="${GLIBC_SWAP:-0}"
HOSTLIBS="${HOSTLIBS:-}"
if [ "$GLIBC_SWAP" = "1" ] && [ -n "$HOSTLIBS" ] && [ -d "$HOSTLIBS/lib" ]; then
  for f in "$HOSTLIBS"/lib/*; do
    b=$(basename "$f")
    case "$b" in libionic*|libionic-rdmav34.so) continue;; esac
    NICM+=(-v "$f:/usr/lib/x86_64-linux-gnu/$b:ro")
  done
  NICM+=(-v "$HOSTLIBS/lib/ld-linux-x86-64.so.2:/lib64/ld-linux-x86-64.so.2:ro")
  NICM+=(-v "$HOSTLIBS/lib/libc.so.6:/lib/x86_64-linux-gnu/libc.so.6:ro")
  [ -e "$HOSTLIBS/etc/libibverbs.d/mlx5.driver" ] && \
    NICM+=(-v "$HOSTLIBS/etc/libibverbs.d/mlx5.driver:/etc/libibverbs.d/mlx5.driver:ro")
fi

# ---- persistent JIT cache (per role) ----
JITCACHE="${JITCACHE_OVERRIDE:-$WORKDIR/jitcache_${ROLE}}"
mkdir -p "$JITCACHE" 2>/dev/null || true
NICM+=(-v "${JITCACHE}:/cache")
CENV=(
  -e HOME=/cache -e TRITON_CACHE_DIR=/tmp/triton_${ROLE}_$$
  -e AITER_JIT_DIR=/cache/aiter -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e MORI_KERNEL_DIR=/cache/mori
)

# ---- vLLM + MoRI(IO/EP) env ----
ENVS=(
  -e MORI_EP_OVER_RDMA="${MORI_EP_OVER_RDMA:-0}"
  -e MORI_NO_ATOMIC_MR="${MORI_NO_ATOMIC_MR:-1}"
  -e MORIIO_DEFER_WRITES="${MORIIO_DEFER_WRITES:-1}"
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=${AITER_MOE:-1}
  -e AITER_KSPLIT=${AITER_KSPLIT:-0}
  -e VLLM_ROCM_USE_AITER_MLA="${AITER_MLA:-1}"
  -e VLLM_ROCM_USE_AITER_RMSNORM=1
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
  -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0 -e VLLM_USE_AITER_TRITON_SILU_MUL=0
  # Cross-node DP16 (EP16 2P/2D) only: the aiter fp8-BMM precompile loop runs a GPU
  # alloc/compile storm that corrupts the cross-node gloo epoll fd -> gloo nfds=-1
  # segfault. Default off avoids it (runtime falls back to torch.bmm, BF16, correct).
  # Harmless for TP/EP8 (intra-node XGMI DP, no cross-node gloo). Set =1 to restore.
  -e VLLM_ROCM_USE_AITER_FP8BMM="${AITER_FP8BMM:-0}"
  -e VLLM_USE_V1=1 -e VLLM_LOGGING_LEVEL=INFO
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${SPARSE_IDX_MB:-512}"
  -e VLLM_ALL2ALL_BACKEND="${A2A:-allgather_reducescatter}"
  -e VLLM_ENGINE_READY_TIMEOUT_S=10800 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 -e VLLM_RPC_TIMEOUT=300000
  -e PYTORCH_ALLOC_CONF=expandable_segments:False -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_ENABLE_SDMA=0
  # ---- MoRI fabric (ionic, PFC pri-3 lossless TC=96) ----
  -e MORI_ENABLE_DMABUF_REG="${DMABUF:-1}" -e HSA_USE_UDMABUF="${DMABUF:-1}"
  -e MORI_IO_DISABLE_ATOMIC_MR=1
  -e MORI_IB_HCA=${MORI_HCA:-ionic} -e MORI_IB_GID_INDEX=${MORI_GID:-1} -e MORI_SOCKET_IFNAME=${SOCKET_IFNAME:-ens3}
  -e MORI_RDMA_TC=96 -e MORI_IO_TC=96 -e MORI_RDMA_SL=3 -e MORI_IO_SL=3
  -e MORI_IO_RAIL_AFFINITY=1 -e MORI_IO_ENABLE_CHUNKING=1 -e MORI_IO_CHUNK_BYTES=262144
  -e MORI_NUM_QP_PER_PE=${MORI_QP:-4} -e VLLM_MORIIO_QP_PER_TRANSFER=${MORIIO_QP:-4} -e VLLM_MORIIO_NUM_WORKERS=${MORIIO_NW:-4}
  -e VLLM_MORIIO_TRANSFER_TIMEOUT_S=600 -e VLLM_MORIIO_DEFERRED_TIMEOUT_S=1800 -e VLLM_HANDSHAKE_TIMEOUT_MINS=30
  -e MORI_RDMA_DEVICES="$IBDEV" -e RDMAV_FORK_SAFE=1 -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
  -e MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"
  # ---- NCCL/GLOO control plane on mlx5 (ionic stays the MoRI data plane) ----
  -e GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:-ens3} -e NCCL_SOCKET_IFNAME=${SOCKET_IFNAME:-ens3}
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} -e NCCL_IB_GID_INDEX=3 -e NCCL_NET_GDR_LEVEL=3 -e NCCL_CROSS_NIC=1
  -e NCCL_IB_RETRY_CNT=15 -e NCCL_IB_TIMEOUT=22 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e TORCH_NCCL_ENABLE_MONITORING=0 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  -e HIP_VISIBLE_DEVICES=${GPU_IDS:-0,1,2,3,4,5,6,7}
)
DEV=(--device /dev/kfd --device /dev/dri --device /dev/infiniband)
COMMON=(--network host --ipc host --privileged --group-add video
  --cap-add IPC_LOCK --cap-add NET_ADMIN --ulimit memlock=-1:-1 --ulimit stack=67108864
  --ulimit nofile=1048576:1048576 --shm-size 128G
  -v "$MODEL:$MODEL:ro" -v "$WORKDIR:$WORKDIR" -v /sys/class/infiniband:/sys/class/infiniband:ro)

# ---- optional extra bind-mounts (iteration hook). EXTRA_MOUNTS is a
# space-separated list of host:container[:ro] specs, e.g. to overlay a single
# patched .py onto the in-image path without rebuilding. ----
if [ -n "${EXTRA_MOUNTS:-}" ]; then
  for _m in ${EXTRA_MOUNTS}; do COMMON+=(-v "$_m"); done
fi

# ---- optional patched-mori .so set (ionic atomic-MR strip). Ideally baked into
# the image; provide MORI_SO_DIR to overlay a host-built set. ----
if [ "${MORI_PATCHED:-0}" = "1" ] && [ -n "${MORI_SO_DIR:-}" ]; then
  for _so in ${MORI_SO_DIR}/libmori_*.so; do
    [ -e "$_so" ] || continue
    COMMON+=(-v "${_so}:/usr/local/lib/python3.12/dist-packages/mori/$(basename "$_so"):ro")
  done
fi

# ---- Python overlays (OVERLAYS=1, default on): the verified connector / model /
# aiter fixes that make GLM-5.3-Flash disagg recall correct (to 871K tokens) on
# top of the base image. Bind-mounted from ./patches/ (see patches/README.md for
# the file->destination manifest). These ARE the fix for the currently-proven
# recipe; the fully in-source image is a follow-up. Set OVERLAYS=0 to serve the
# base image bare (will NOT recall correctly at depth). ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-$SCRIPT_DIR/patches}"
_SP="/usr/local/lib/python3.12/dist-packages"
if [ "${OVERLAYS:-1}" = "1" ] && [ -d "$PATCH_DIR" ]; then
  declare -A _OVL=(
    [moriio_connector_hma.py]="$_SP/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    [moriio_engine.py]="$_SP/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
    [moriio_common.py]="$_SP/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
    [moriio_layout.py]="$_SP/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
    [attn_utils.py]="$_SP/vllm/v1/worker/gpu/attn_utils.py"
    [indexer.py]="$_SP/vllm/v1/attention/backends/mla/indexer.py"
    [glm5next_attention.py]="$_SP/vllm/models/glm5next/nvidia/attention.py"
    [usercustomize.py]="$_SP/usercustomize.py"
    [gemm_op_a8w8_tip.py]="$_SP/aiter/ops/gemm_op_a8w8.py"
    [fused_moe_tip.py]="$_SP/aiter/fused_moe.py"
    [batched_gemm_a16wfp4.py]="$_SP/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py"
  )
  for _f in "${!_OVL[@]}"; do
    [ -e "$PATCH_DIR/$_f" ] && COMMON+=(-v "$PATCH_DIR/$_f:${_OVL[$_f]}:ro")
  done
fi

KV_EXTRA_PF='{"proxy_ip":"'$PROXY_IP'","proxy_port":"'$PROXY_PORT'","proxy_ping_port":"'$PROXY_PING'","http_port":"'$PF_PORT'","local_ping_port":"61555","handshake_port":"8405","notify_port":"'$NOTIFY'"}'
KV_EXTRA_DC='{"proxy_ip":"'$PROXY_IP'","proxy_port":"'$PROXY_PORT'","proxy_ping_port":"'$PROXY_PING'","http_port":"'$DC_PORT'","local_ping_port":"4583","handshake_port":"7305","notify_port":"'$NOTIFY'"}'

case "$ROLE" in
  prefill)
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
      -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
      vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
        --port $PF_PORT --gpu_memory_utilization ${GPUUTIL:-0.5} \
        --kv-cache-dtype ${KV_DTYPE:-auto} --block-size ${BLOCK_SIZE:-4} --no-enable-prefix-caching \
        --trust-remote-code $PF_CC \
        --max-model-len ${MAXLEN:-940000} ${EXTRA_ARGS:-} \
        --kv-transfer-config '{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"9711\",\"kv_connector_extra_config\":$KV_EXTRA_PF}' \
        2>&1 | tee $LOG/vllm_prefill.log"
    echo "[vllm-prefill] launched :$PF_PORT host=$HOST_IP proxy=$PROXY_IP"
    ;;
  decode)
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
      -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
      vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
        --port $DC_PORT --gpu_memory_utilization ${GPUUTIL:-0.5} \
        --kv-cache-dtype ${KV_DTYPE:-auto} --block-size ${BLOCK_SIZE:-4} --no-enable-prefix-caching \
        --trust-remote-code $DC_CC \
        --cudagraph-capture-sizes $CAPTURE_SIZES \
        --max-model-len ${MAXLEN:-940000} ${EXTRA_ARGS:-} \
        --kv-transfer-config '{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"6301\",\"kv_connector_extra_config\":$KV_EXTRA_DC}' \
        2>&1 | tee $LOG/vllm_decode.log"
    echo "[vllm-decode] launched :$DC_PORT host=$HOST_IP proxy=$PROXY_IP"
    ;;
  proxy)
    # vLLM router (pd-disaggregation, service discovery). The self-contained image
    # ships vllm-router at /usr/local/bin/vllm-router (default). Set ROUTER_BIN to a
    # HOST path to override with an external build (then it is bind-mounted in).
    # ROUTER_DP_LOCAL: 1 for TP, 8 for EP.
    ROUTER_BIN="${ROUTER_BIN:-/usr/local/bin/vllm-router}"
    RDP="${ROUTER_DP_LOCAL:-8}"
    ROUTER_MNT=(); case "$ROUTER_BIN" in /usr/local/bin/*) ;; *) ROUTER_MNT=(-v "$ROUTER_BIN:$ROUTER_BIN:ro") ;; esac
    docker rm -f vllm_proxy >/dev/null 2>&1
    docker run -d --name vllm_proxy --network host "${ROUTER_MNT[@]}" -v "$WORKDIR:$WORKDIR" \
      --entrypoint bash "$IMG" -lc "
      $ROUTER_BIN --host 0.0.0.0 --port $PROXY_PORT \
        --vllm-pd-disaggregation --kv-connector moriio \
        --vllm-discovery-address 0.0.0.0:$PROXY_PING \
        --intra-node-data-parallel-size $RDP \
        --policy round_robin --prefill-policy round_robin --decode-policy round_robin \
        --worker-startup-timeout-secs 3600 --max-concurrent-requests ${ROUTER_MAX_CONC:-1024} \
        --request-timeout-secs ${ROUTER_REQ_TIMEOUT:-3600} --retry-max-retries 1 \
        --log-level info 2>&1 | tee $LOG/vllm_router.log"
    echo "[vllm-router] launched :$PROXY_PORT (discovery :$PROXY_PING, dp_local=$RDP)"
    ;;
esac
