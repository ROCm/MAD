#!/bin/bash
# =============================================================================
# PR#205 vLLM + MoRIIO disaggregated P/D launcher for GLM-5.2-FP8 on AINIC a77.
# ===== EP16 VARIANT: expert-parallel across 2 NODES per role (DP16). =====
# Each role (prefill, decode) is a 2-node DP16 group. The MoE all-to-all now
# crosses the ionic RDMA fabric, so we use Tej's PR#558 host-CPU-proxy path
# (MORI_EP_OVER_RDMA=1): the host rings the NIC doorbell via ibv_post_send
# because GPU-initiated IBGDA doorbell MMIO fails under KVM/VFIO passthrough.
#
# 1P1D-EP16 topology (4 nodes):
#   prefill: master(rank0,api) + headless(rank8)   -> DP16, kv_producer
#   decode : master(rank0,api) + headless(rank8)   -> DP16, kv_consumer
#   proxy  : vllm-router (on prefill master, or any control node)
#
# Per-node launcher. Env:
#   ROLE=prefill|decode|proxy
#   NODE_ROLE=master|headless   (EP16: which of the 2 nodes in this role's DP group)
#   HOST_IP     = this node ens3 ip
#   DP_MASTER_IP= the role's rank-0 (master) ens3 ip (headless dials this for DP rendezvous)
#   PROXY_IP    = router/discovery node ens3 ip
#   DECODE_IP   = decode master ens3 ip
#   DP (default 16 total), DP_LOCAL (default 8 GPUs/node), START_RANK (master 0 / headless 8)
# =============================================================================
set -uo pipefail
ROLE="${ROLE:?prefill|decode|proxy}"
IMG="${IMG:-vllm-mori-pr558:ionic}"
MODEL="${MODEL:-/shared_nfs/ravgupta_disagg205/models/GLM-5.2-FP8}"
NODE_ROLE="${NODE_ROLE:-master}"       # master | headless  (EP16 two-node DP group)
DP="${DP:-16}"                         # TOTAL data-parallel size across both nodes
DP_LOCAL="${DP_LOCAL:-8}"              # GPUs on this node
START_RANK="${START_RANK:-0}"          # master=0, headless=8
HOST_IP="${HOST_IP:?this node ens3 ip}"
DP_MASTER_IP="${DP_MASTER_IP:-$HOST_IP}"  # rank-0 node for this role's DP rendezvous
PROXY_IP="${PROXY_IP:?router/discovery node ens3 ip}"
DECODE_IP="${DECODE_IP:?decode master ens3 ip}"
LOG="${LOG:-/shared_nfs/ravgupta_disagg205/logs}"
CONTAINER="${CONTAINER:-vllm_${ROLE}_${NODE_ROLE}}"

PF_PORT=20005; DC_PORT=40005; PROXY_PORT=10001
PROXY_PING=36367; NOTIFY=61005

# ---- cudagraph / compilation (PR#206 pattern, but keeping our newer vLLM defaults) ----
# NEVER --enforce-eager on AITER images (eager worker w/o +quant_fp8 routes fp8 quant through an
# AITER op whose signature mismatches -> init crash). Express no-cudagraph as cudagraph_mode:NONE
# WITH custom_ops:[+quant_fp8]. GLM(MLA+DSA): prefill=NONE, decode=FULL_AND_PIECEWISE (better TPOT
# than plain PIECEWISE) + use_inductor_graph_partition (splits at cudagraph-unsafe MLA KV-update).
PREFILL_CUDAGRAPH_MODE="${PREFILL_CUDAGRAPH_MODE:-NONE}"
DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8 16 32 64 128 256}"
IGP="${USE_INDUCTOR_GRAPH_PARTITION:-1}"   # GLM opts in; splits unsafe ops to eager
# KV_CACHE_MEMORY_BYTES: pass --kv-cache-memory-bytes to SKIP the boot profiling forward (the only
# path that calls AITER eager per-token fp8 quant with the aiter_tensor_t torch_guard bug -> crash).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"

# Write plain JSON to a per-role file in the NFS workdir (mounted at same path in the container).
# The serve command reads it INSIDE the container via "\$(cat FILE)" so the container's shell
# substitutes the raw (unescaped) JSON as a single argv — no host/docker quote-escaping involved.
WORKDIR_NFS="/shared_nfs/ravgupta_disagg205"
_compcfg_write() {  # $1=role $2=mode -> writes file, echoes its path
  local role="$1" mode="$2" igp=""
  [ "$IGP" = "1" ] && igp=', "use_inductor_graph_partition": true'
  local f="${WORKDIR_NFS}/compcfg_${role}.json"
  printf '{"cudagraph_mode": "%s", "custom_ops": ["+quant_fp8"]%s}\n' "$mode" "$igp" > "$f" 2>/dev/null || true
  echo "$f"
}
PF_CFG_FILE="$(_compcfg_write prefill "$PREFILL_CUDAGRAPH_MODE")"
DC_CFG_FILE="$(_compcfg_write decode "$DECODE_CUDAGRAPH_MODE")"
MEM_ARG=""; [ -n "$KV_CACHE_MEMORY_BYTES" ] && MEM_ARG="--kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES"

# ---- MTP / speculative decoding (optimization). SPEC=mtp enables the model's built-in MTP layer
# (GLM-5.2 config: num_nextn_predict_layers=1). Written to a file to dodge docker -lc quote-escaping.
# On the decode side this is where the TPOT win lands; prefill can also carry it (harmless).
SPEC="${SPEC:-off}"; SPEC_TOK="${SPEC_TOK:-1}"
SPEC_ARG=""
if [ "$SPEC" = "mtp" ]; then
  SPEC_FILE="${WORKDIR_NFS}/speccfg_${ROLE}.json"
  printf '{"method": "mtp", "num_speculative_tokens": %s}\n' "$SPEC_TOK" > "$SPEC_FILE" 2>/dev/null || true
  SPEC_ARG="--speculative-config \"\$(cat $SPEC_FILE)\""
fi

# EP16: DP16 expert-parallel spanning 2 nodes per role, using vLLM INTERNAL DP LB:
# rank-0 node (master) runs a SINGLE API server + the DP coordinator at
# --data-parallel-address; rank-8 node (headless) joins with --headless
# --data-parallel-start-rank 8. NOTE: --api-server-count>1 flips vLLM into
# hybrid/external LB which FORBIDS --headless ("Remote engine N must not use
# --headless in external or hybrid dp lb mode") -> master must NOT set it here.
# -tp 1 (pure DP/EP). all2all crosses ionic -> MORI_EP_OVER_RDMA=1 host-CPU proxy.
# CRITICAL: the master must NOT pass --data-parallel-start-rank. arg_utils.py:2189
# flips data_parallel_hybrid_lb=True whenever start_rank is set on a non-headless
# node, which then FORBIDS the headless secondary ("must not use --headless in
# hybrid/external dp lb mode"). Master => rank 0 implicit, internal LB, single API
# server + DP coordinator. Only the HEADLESS node carries --data-parallel-start-rank.
DP_RPC_PORT=13345
if [ "$NODE_ROLE" = "headless" ]; then
  PARALLEL="-tp 1 --data-parallel-size ${DP} --data-parallel-size-local ${DP_LOCAL} \
    --data-parallel-start-rank ${START_RANK} --data-parallel-address ${DP_MASTER_IP} \
    --data-parallel-rpc-port ${DP_RPC_PORT} --headless"
else
  PARALLEL="-tp 1 --data-parallel-size ${DP} --data-parallel-size-local ${DP_LOCAL} \
    --data-parallel-address ${DP_MASTER_IP} --data-parallel-rpc-port ${DP_RPC_PORT}"
fi
EP_FLAGS="--enable-expert-parallel --all2all-backend ${A2A:-mori_high_throughput}"

IBDEV="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"

# ---- ionic RDMA userspace bind-mounts (from ClusterSphere cluster_rdma_env_recommender) ----
# All 5 host provider libs the recommender lists, incl the versioned .so I was missing before.
NICM=()
for f in /usr/lib/x86_64-linux-gnu/libionic.so /usr/lib/x86_64-linux-gnu/libionic.so.1 \
         /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so /etc/libibverbs.d/ionic.driver; do
  [ -e "$f" ] && NICM+=(-v "$f:$f:ro")
done
# versioned libionic (glob catches 1.1.54.0-187 and any future rev)
for f in $(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.* 2>/dev/null); do
  NICM+=(-v "$f:$f:ro")
done

# ---- Persistent JIT cache (per-role, on NFS): stops mori/aiter/triton/inductor recompiling
# every launch (the collective_rpc mq-timeout that killed decode). First run compiles, later
# runs load instantly. HOME=/cache so MORI's $HOME/.mori/jit persists too.
JITCACHE="/shared_nfs/ravgupta_disagg205/jitcache_${ROLE}"
mkdir -p "$JITCACHE" 2>/dev/null || true
NICM+=(-v "${JITCACHE}:/cache")
CENV=(
  -e HOME=/cache
  -e TRITON_CACHE_DIR=/cache/triton
  -e AITER_JIT_DIR=/cache/aiter
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  -e MORI_KERNEL_DIR=/cache/mori
)

# ---- vLLM + MoRI(IO/EP) env ----
ENVS=(
  # ---- AITER env (ported from ROCm/MAD PR#206 connectors/moriio.sh — validated GLM recipe) ----
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1
  -e VLLM_ROCM_USE_AITER_MLA="${AITER_MLA:-1}"     # GLM keeps MLA on (DeepSeek uses 0)
  -e VLLM_ROCM_USE_AITER_RMSNORM=1
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
  -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0 -e VLLM_USE_AITER_TRITON_SILU_MUL=0
  -e VLLM_USE_V1=1 -e VLLM_LOGGING_LEVEL=INFO
  -e VLLM_ALL2ALL_BACKEND="${A2A:-mori_high_throughput}"
  # PR#206 timeouts (bigger ready timeout; RPC 300000ms)
  -e VLLM_ENGINE_READY_TIMEOUT_S=10800 -e VLLM_RINGBUFFER_WARNING_INTERVAL=3600
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
  -e VLLM_RPC_TIMEOUT=300000
  # ---- ROCm-7.2.x platform env (PR#206): MUST reach PID1 via docker -e; pytorch reads at import ----
  -e PYTORCH_ALLOC_CONF=expandable_segments:False -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
  -e HSA_ENABLE_IPC_MODE_LEGACY=0
  -e GPU_MAX_HW_QUEUES=2 -e HIP_FORCE_DEV_KERNARG=1 -e HSA_ENABLE_SDMA=0
  # ---- MoRI fabric (ionic a77, PFC pri-3 lossless TC=96) ----
  # EP16 INTERNODE: Tej PR#558 host-CPU-proxy. GPU-initiated IBGDA doorbell MMIO fails
  # across KVM/VFIO passthrough on ionic, so the host posts the send (ibv_post_send) via
  # TransportType::PROXY. Required whenever the EP all2all crosses nodes (EP>=16).
  -e MORI_EP_OVER_RDMA="${MORI_EP_OVER_RDMA:-1}"
  -e MORI_ENABLE_DMABUF_REG="${DMABUF:-1}" -e HSA_USE_UDMABUF="${DMABUF:-1}"
  -e MORI_IO_DISABLE_ATOMIC_MR=1
  -e MORI_IB_HCA=ionic -e MORI_IB_GID_INDEX=1 -e MORI_SOCKET_IFNAME=ens3
  -e MORI_RDMA_TC=96 -e MORI_IO_TC=96 -e MORI_RDMA_SL=3 -e MORI_IO_SL=3
  -e MORI_IO_RAIL_AFFINITY=1 -e MORI_IO_ENABLE_CHUNKING=1 -e MORI_IO_CHUNK_BYTES=262144
  # PR#206 MoRIIO QP/worker tuning (was 1/8 -> 4/4)
  -e MORI_NUM_QP_PER_PE=4 -e VLLM_MORIIO_QP_PER_TRANSFER=4 -e VLLM_MORIIO_NUM_WORKERS=4
  -e VLLM_MORIIO_TRANSFER_TIMEOUT_S=600 -e VLLM_MORIIO_DEFERRED_TIMEOUT_S=1800
  -e VLLM_HANDSHAKE_TIMEOUT_MINS=30
  -e MORI_RDMA_DEVICES="$IBDEV"
  -e RDMAV_FORK_SAFE=1 -e HSA_NO_SCRATCH_RECLAIM=1
  -e AITER_USE_FLYDSL_MOE_SORTING=1
  -e AITER_ONLINE_TUNE=0 -e AITER_TUNE_GEMM=0
  # ---- shmem sizing. The MoRI symmetric heap is allocated ON-GPU at connector init,
  #      BEFORE vLLM's profiling mem-check -> it directly eats into the util budget.
  #      32GB heap consumed ~65GB/GPU (double-buffered + proxy) and blew the 0.80 check
  #      (only 222/288 free). 16GB (EP8's proven value) is plenty for 1P1D-EP16. ----
  -e ROCSHMEM_HEAP_SIZE=8589934592 -e ROCSHMEM_MAX_NUM_CONTEXTS=256
  -e MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"
  # ---- NCCL/GLOO CONTROL plane. EP16 is 2 nodes/role, so the DP GroupCoordinator
  #      now needs a CROSS-NODE NCCL comm (EP8/TP8 never did — DP was intra-node XGMI).
  #      mlx5_0 IB is not routable between these ionic nodes -> ncclCommInitRank fails
  #      "unhandled system error". The DP-coordinator traffic is tiny (per-step token
  #      allreduce); the heavy expert-all2all + KV go over MoRI/ionic, NOT NCCL. So
  #      force NCCL onto TCP/ens3 (NCCL_IB_DISABLE=1) — demonstrably routes (DP
  #      rendezvous worked). Set NCCL_P2P for intra-node XGMI to stay enabled. ----
  -e GLOO_SOCKET_IFNAME=ens3 -e NCCL_SOCKET_IFNAME=ens3
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" -e NCCL_NET_GDR_LEVEL=3 -e NCCL_CROSS_NIC=1
  -e NCCL_IB_RETRY_CNT=15 -e NCCL_IB_TIMEOUT=22 -e NCCL_IGNORE_CPU_AFFINITY=1
  # long JIT compiles survive
  -e TORCH_NCCL_ENABLE_MONITORING=0 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
)
DEV=(--device /dev/kfd --device /dev/dri --device /dev/infiniband)
COMMON=(--network host --ipc host --privileged --group-add video
  --cap-add IPC_LOCK --cap-add NET_ADMIN --ulimit memlock=-1:-1 --ulimit stack=67108864
  --ulimit nofile=1048576:1048576 --shm-size 128G
  -v /shared_nfs:/shared_nfs -v /sys:/sys
  -v /sys/class/infiniband:/sys/class/infiniband:ro)
mkdir -p "$LOG" 2>/dev/null || true

KV_EXTRA_PF='{"proxy_ip":"'$PROXY_IP'","proxy_port":"'$PROXY_PORT'","proxy_ping_port":"'$PROXY_PING'","http_port":"'$PF_PORT'","local_ping_port":"61555","handshake_port":"8405","notify_port":"'$NOTIFY'"}'
KV_EXTRA_DC='{"proxy_ip":"'$PROXY_IP'","proxy_port":"'$PROXY_PORT'","proxy_ping_port":"'$PROXY_PING'","http_port":"'$DC_PORT'","local_ping_port":"4583","handshake_port":"7305","notify_port":"'$NOTIFY'"}'

case "$ROLE" in
  prefill)
    # PREFILL DP16: master(rank0)=api+kv-transfer, headless(rank8)=DP worker only
    # (no http port, no kv-transfer-config — it joins the master's DP group). all2all
    # = mori_high_throughput, crossing ionic via MORI_EP_OVER_RDMA=1 host proxy.
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    if [ "$NODE_ROLE" = "headless" ]; then
      docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
        -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
        vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
          --gpu_memory_utilization ${GPUUTIL:-0.85} $MEM_ARG \
          --kv-cache-dtype fp8 --block-size 1 --no-enable-prefix-caching \
          --trust-remote-code \
          --compilation-config \"\$(cat $PF_CFG_FILE)\" \
          --max-model-len ${MAXLEN:-262144} \
          2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_prefill_headless.log"
      echo "[vllm-prefill:headless] rank$START_RANK dp=$DP host=$HOST_IP master=$DP_MASTER_IP"
    else
      docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
        -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
        vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
          --port $PF_PORT --gpu_memory_utilization ${GPUUTIL:-0.85} $MEM_ARG $SPEC_ARG \
          --kv-cache-dtype fp8 --block-size 1 --no-enable-prefix-caching \
          --trust-remote-code \
          --compilation-config \"\$(cat $PF_CFG_FILE)\" \
          --max-model-len ${MAXLEN:-262144} \
          --kv-transfer-config '{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"9711\",\"kv_connector_extra_config\":$KV_EXTRA_PF}' \
          2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_prefill.log"
      echo "[vllm-prefill:master] :$PF_PORT dp=$DP host=$HOST_IP proxy=$PROXY_IP"
    fi
    ;;
  decode)
    # DECODE DP16: master(rank0)=api+kv-transfer, headless(rank8)=DP worker only.
    # cudagraph FULL_AND_PIECEWISE, mori_low_latency all2all over ionic.
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    if [ "$NODE_ROLE" = "headless" ]; then
      docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
        -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
        vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
          --gpu_memory_utilization ${GPUUTIL:-0.90} $MEM_ARG \
          --kv-cache-dtype fp8 --block-size 1 --no-enable-prefix-caching \
          --trust-remote-code \
          --compilation-config \"\$(cat $DC_CFG_FILE)\" \
          --cudagraph-capture-sizes $CAPTURE_SIZES \
          --max-model-len ${MAXLEN:-262144} \
          2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_decode_headless.log"
      echo "[vllm-decode:headless] rank$START_RANK dp=$DP host=$HOST_IP master=$DP_MASTER_IP"
    else
      docker run -d --name "$CONTAINER" "${DEV[@]}" "${COMMON[@]}" "${NICM[@]}" "${ENVS[@]}" "${CENV[@]}" \
        -e HOST_IP=$HOST_IP --entrypoint bash "$IMG" -lc "
        vllm serve --model $MODEL $PARALLEL $EP_FLAGS \
          --port $DC_PORT --gpu_memory_utilization ${GPUUTIL:-0.90} $MEM_ARG $SPEC_ARG \
          --kv-cache-dtype fp8 --block-size 1 --no-enable-prefix-caching \
          --trust-remote-code \
          --compilation-config \"\$(cat $DC_CFG_FILE)\" \
          --cudagraph-capture-sizes $CAPTURE_SIZES \
          --max-model-len ${MAXLEN:-262144} \
          --kv-transfer-config '{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"6301\",\"kv_connector_extra_config\":$KV_EXTRA_DC}' \
          2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_decode.log"
      echo "[vllm-decode:master] :$DC_PORT dp=$DP host=$HOST_IP proxy=$PROXY_IP"
    fi
    ;;
  proxy)
    # PROXY_TYPE=vllm_router (default, production — raviguptaamd/router@82dc981, PR#206 pin:
    #   DP-rank round-robin + MoRIIO KV-notify dpfix) OR moriio_toy (fallback, low-con only).
    # Both use pure service discovery: workers self-register to :36367 (their kv-transfer-config
    # proxy_ip=this node, proxy_ping_port=36367). Router listens HTTP on $PROXY_PORT.
    PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
    ROUTER_BIN=/shared_nfs/ravgupta_disagg205/router_build/vllm-router
    RDP="${ROUTER_DP_LOCAL:-8}"   # EP: DP replicas/node = 8; TP: 1
    docker rm -f vllm_proxy >/dev/null 2>&1
    if [ "$PROXY_TYPE" = "vllm_router" ] && [ -x "$ROUTER_BIN" ]; then
      docker run -d --name vllm_proxy --network host -v /shared_nfs:/shared_nfs \
        --entrypoint bash "$IMG" -lc "
        $ROUTER_BIN --host 0.0.0.0 --port $PROXY_PORT \
          --vllm-pd-disaggregation --kv-connector moriio \
          --vllm-discovery-address 0.0.0.0:$PROXY_PING \
          --intra-node-data-parallel-size $RDP \
          --policy round_robin --prefill-policy round_robin --decode-policy round_robin \
          --worker-startup-timeout-secs 3600 \
          --max-concurrent-requests ${ROUTER_MAX_CONC:-1024} \
          --request-timeout-secs ${ROUTER_REQ_TIMEOUT:-3600} \
          --retry-max-retries 1 \
          --log-level info 2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_router.log"
      echo "[vllm-router] launched :$PROXY_PORT (discovery :$PROXY_PING, dp_local=$RDP, round_robin)"
    else
      docker run -d --name vllm_proxy --network host -v /shared_nfs:/shared_nfs \
        --entrypoint bash "$IMG" -lc "
        python /app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py \
          --port $PROXY_PORT 2>&1 | tee /shared_nfs/ravgupta_disagg205/$(basename $LOG)/vllm_proxy.log"
      echo "[vllm-toy-proxy] launched :$PROXY_PORT (registration on :$PROXY_PING)"
    fi
    ;;
esac
