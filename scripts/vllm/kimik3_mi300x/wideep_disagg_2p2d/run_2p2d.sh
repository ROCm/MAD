#!/bin/bash
# Kimi-K3 MXFP4 2P/2D wide-EP DISAGGREGATED serve: DP/EP16 per role, MoRI-EP
# (mori all2all) + MoRIIO connector (prefill->decode KV + KDA state transfer).
#
# Topology (4 nodes, 8 GPU each = EP16 per pool):
#   Prefill pool: P-master (rank0, proxy+kv_producer) + P-worker (rank8, headless)
#   Decode  pool: D-master (kv_consumer)             + D-worker (headless)
# Run this per node with ROLE + the shared *_ADDR env set (see run_2p2d_launch.sh).
#
# Applies connector fixes baked into the disagg image (VLLM_REF=v3).
set -euo pipefail

IMAGE="${IMAGE:-kimik3-wideep-disagg:latest}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your Kimi-K3-MXFP4 weights path}"
ROLE="${ROLE:?ROLE=prefill_master|prefill_worker|decode_master|decode_worker}"
PMASTER="${PMASTER:?prefill master eth0 IP}"
DMASTER="${DMASTER:?decode master eth0 IP}"
PROXY_IP="${PROXY_IP:-$PMASTER}"
LOGHOST="${LOGHOST:-$HOME/k3disagg/logs}"; mkdir -p "$LOGHOST"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
# Bound the K3 MoE profiling M. On gfx942 the tuned FlyDSL a8w4 configs are
# gfx950-only + unsharded (896 experts), so our EP16 (56-expert) shard ALWAYS
# falls to the heuristic FlyDSL kernel. That heuristic crashes LLVM codegen
# ("Do not know how to expand this operator's operand") at the giant profiling
# shape sorted-M=131072 (= max_num_batched_tokens 16384 x topk 8). Shrinking
# max-num-batched-tokens shrinks the profiling M so the heuristic kernel compiles.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
# K3 MoE requant path. gfx942 has NO scaled-MXFP4 MFMA and the a16w4 SiTUv2
# heuristic FlyDSL kernel CANNOT codegen on gfx942 (LLVM ExpandIntegerOperand on
# a 128-bit buffer->LDS async load; all tuned a8w4 configs are gfx950-only). The
# PROVEN-coherent colocated path (logbook: NIAH 3/3 @9600) requants MoE to
# packed-int4 and runs it through Situv2 (dtype torch.int4, per_1x32) with
# AITER_SITUV2_A8W4=1 + AITER PR#4471 (SiTUv2 in the int4 stage1 epilogue).
QUANT_CONFIG="${QUANT_CONFIG:-{\"moe\":{\"weight\":\"int4_per_group_32\"}}}"
GPU_UTIL="${GPU_UTIL:-0.88}"
# --- RDMA fabric (OVERRIDABLE) -----------------------------------------------
# Defaults are validated Broadcom Thor2 (bnxt RoCE) values:
# ibv device names rdma0..rdma7, host NIC eno0, GID index 3. On a DIFFERENT
# fabric (e.g. Mellanox mlx5) override these, e.g.:
#   NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
#   RDMA_DEVICES=mlx5_0,mlx5_2,...  SOCKET_IFNAME=eth0  IB_GID_INDEX=3 THOR2_BNXT_FIX=0
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
NCCL_IB_HCA_VAL="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
RDMA_DEVICES="${RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"
# Thor2 bnxt libibverbs ABI fix (host v34 driver vs image v59). Set THOR2_BNXT_FIX=1
# on Broadcom Thor2; default OFF for Mellanox mlx5 (OCI).
THOR2_BNXT_FIX="${THOR2_BNXT_FIX:-0}"
THOR2_LIBIBVERBS_HOST="${THOR2_LIBIBVERBS_HOST:-/usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0}"
THOR2_LIBIBVERBS_IMG="${THOR2_LIBIBVERBS_IMG:-/usr/lib/x86_64-linux-gnu/libibverbs.so.1.16.62.0}"
THOR2_BNXT_HOST="${THOR2_BNXT_HOST:-/usr/local/lib/libbnxt_re-rdmav34.so}"
THOR2_BNXT_IMG="${THOR2_BNXT_IMG:-/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so}"
if [ "${THOR2_BNXT_FIX}" = "1" ]; then
  BNXT_MOUNTS="-v ${THOR2_LIBIBVERBS_HOST}:${THOR2_LIBIBVERBS_IMG}:ro -v ${THOR2_BNXT_HOST}:${THOR2_BNXT_IMG}:ro"
else
  BNXT_MOUNTS=""
fi
# -----------------------------------------------------------------------------
# Skip the boot memory-profiling forward (profile_run) entirely by pinning KV
# cache size — mirrors DeepSeek #181 (which used it for an AITER fp8 profiling
# bug). profile_run's dummy forward hangs under tp8xDP2 + mori all2all (sampler
# gather / _sync_device all2all deadlocks); pinning KV bytes bypasses all of it.
# At TP2/DP8 weights are ~137.5 GiB/GPU + 16 GiB MoRI heap, so KV room is tighter
# (~10 GiB) than at TP8 (was 14e9). Pin 8 GiB — ample for K3's tiny MLA KV
# (kv_lora_rank=512, only 24/93 full-attn layers; ~13.5 KiB/tok fp8 => ~600k tok).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-8000000000}"
# Wide-EP shape per pool: TP2 x DP8 -> EP16, no PP (per user: "No PP — disagg via
# MoRIIO, DP/EP via MoRI-EP"). 16 GPUs/pool (2 nodes x 8). world=16 => EP16 (896/16
# =56 experts/GPU via MoRI-EP all2all). DP8 = 8 independent attention streams (the
# 69/93 KDA recurrent layers stay DP-local => best decode throughput); TP2 shards
# the 106.5 GiB replicated attn+shared-expert weight to 53.3 GiB/GPU so it FITS
# (pure TP1/DP16 = 190.7 GiB weights > 192 GiB HBM; see Confluence 1830010189).
# Per-GPU: experts 84.2 + repl 53.3 = 137.5 GiB weights + 16 GiB MoRI heap. DP_SIZE
# is the GLOBAL dp count (=world/TP=16/2=8); DP_LOCAL=4 dp ranks/node (8 GPUs/2).
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-8}"
DP_LOCAL="${DP_LOCAL:-4}"
SERVE_PORT=20005; RPC_PORT=13345
# JIT cache persistence (best-practice): host dir keyed per image AND per role
# CLASS. Prefill (mori_high_throughput + cudagraph NONE) and decode
# (mori_low_latency + PIECEWISE) compile DIFFERENT kernel variants under the SAME
# aiter .so filenames (module_moe_asm/moe_sorting_opus/...). Node-local nvme keeps
# prefill (025/043) and decode (047/048) caches physically separate; the per-CLASS
# subdir also prevents collision if a node ever hosts both roles. Do NOT put this
# on a SHARED FS (would let prefill/decode clobber each other's same-named .so).
ROLE_CLASS="${ROLE%%_*}"   # prefill | decode
JIT_HOST="${JIT_HOST:-/tmp/$USER/vllm_jit_cache/k3disagg_${ROLE_CLASS}}"; mkdir -p "$JIT_HOST"

VLLM_SP=/usr/local/lib/python3.12/dist-packages/vllm

# Per-pool topology: TP2 x DP8 -> EP16 (see the TP_SIZE/DP_SIZE block above).
# Master node hosts DP ranks 0..DP_LOCAL-1; worker node hosts DP_LOCAL..2*DP_LOCAL-1
# (=> --data-parallel-start-rank ${DP_LOCAL}). gpu-util 0.88 for the 16 GiB MoRI
# shmem heap reserved before the vLLM snapshot.
case "$ROLE" in
  prefill_master) DP_ADDR=$PMASTER; KV_ROLE=kv_producer; BACKEND=mori_high_throughput; CG=NONE; HEADLESS=""; START="" ;;
  prefill_worker) DP_ADDR=$PMASTER; KV_ROLE=kv_producer; BACKEND=mori_high_throughput; CG=NONE; HEADLESS="--headless"; START="--data-parallel-start-rank ${DP_LOCAL}" ;;
  decode_master)  DP_ADDR=$DMASTER; KV_ROLE=kv_consumer; BACKEND=mori_low_latency;  CG=PIECEWISE; HEADLESS=""; START="" ;;
  decode_worker)  DP_ADDR=$DMASTER; KV_ROLE=kv_consumer; BACKEND=mori_low_latency;  CG=PIECEWISE; HEADLESS="--headless"; START="--data-parallel-start-rank ${DP_LOCAL}" ;;
  *) echo "bad ROLE=$ROLE"; exit 1 ;;
esac
# Optional cudagraph override (DECODE_CG=NONE bypasses PIECEWISE capture to isolate
# a capture-time GPU fault; PREFILL_CG likewise). Decode capture of the KDA conv
# path can fault on some builds; NONE trades decode-graph perf for stability.
case "$ROLE" in
  prefill_*) [ -n "${PREFILL_CG:-}" ] && CG="$PREFILL_CG" ;;
  decode_*)  [ -n "${DECODE_CG:-}"  ] && CG="$DECODE_CG"  ;;
esac
# Optional all2all-backend override (test: mori_low_latency on both pools to rule
# out mori_high_throughput/InterNodeV1 as the profile-forward all2all deadlock).
if [[ "$ROLE" == prefill_* && -n "${PREFILL_BACKEND:-}" ]]; then BACKEND="$PREFILL_BACKEND"; fi
if [[ "$ROLE" == decode_*  && -n "${DECODE_BACKEND:-}"  ]]; then BACKEND="$DECODE_BACKEND"; fi

IS_MASTER=0; [[ "$ROLE" == *_master ]] && IS_MASTER=1
CONTAINER="k3disagg_${ROLE}"
docker rm -f "$CONTAINER" 2>/dev/null || true

# kv-transfer-config on ALL ranks (masters AND headless workers). A headless
# worker hosts real DP ranks (e.g. decode rank1 on the worker node); WITHOUT
# --kv-transfer-config its engine never instantiates the MoRIIO connector, so it
# binds NO handshake listener -> prefill can't transfer KV to those ranks ->
# "Timed out waiting for write_ready_flags" -> EngineDead. (Was gated on
# IS_MASTER, which silently made every worker-node DP rank a transfer black hole.)
# To survive host-bash -> ssh -> docker -c quoting, pass JSON as base64 via env.
KVCFG_B64=""
if true; then
  # Peer pool's per-DP-pod node IPs (ordered by pod index = global_dp_rank//dp_local).
  # A prefill (kv_producer) handshakes the DECODE pool -> needs decode hosts; a decode
  # (kv_consumer) notifies the PREFILL pool -> needs prefill hosts. Without this the
  # connector falls back to a single peer host (the master), so KV writes/notifies to
  # ranks on the peer's WORKER node silently miss -> that node's decode ranks generate
  # context-free (the 50%/DP2, ~88%/DP8 wrong-answer alternation). Consumed by the
  # apply_kimik3_moriio_pod_hosts patcher as multi_pod_hosts in the handshake.
  if [[ "$ROLE" == prefill_* ]]; then PEER_POD_HOSTS="${DECODE_POD_HOSTS:-}"; else PEER_POD_HOSTS="${PREFILL_POD_HOSTS:-}"; fi
  # MORIIO_READ_MODE=1 selects the connector's READ path (decode pulls KV, sync,
  # returns N-1) instead of WRITE (prefill pushes, async, returns N). READ is the
  # more-tested vLLM disagg path; toggled for A/B against the WRITE decode-consume bug.
  READ_MODE_JSON=""; if [ "${MORIIO_READ_MODE:-0}" = "1" ]; then READ_MODE_JSON=",\"read_mode\":\"true\""; fi
  KVCFG_JSON="{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_port\":\"9711\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_port\":\"30000\",\"proxy_ping_port\":\"36367\",\"http_port\":\"${SERVE_PORT}\",\"local_ping_port\":\"61555\",\"handshake_port\":\"8405\",\"notify_port\":\"61005\",\"moriio_pod_hosts\":\"${PEER_POD_HOSTS}\",\"post_batch_size\":${MORIIO_POST_BATCH_SIZE:--1},\"qp_per_transfer\":${MORIIO_QP_PER_TRANSFER:-1},\"num_workers\":${MORIIO_NUM_WORKERS:-1}${READ_MODE_JSON}}}"
  KVCFG_B64=$(printf '%s' "$KVCFG_JSON" | base64 -w0)
fi
# api-server ONLY on masters (headless workers must NOT bind an api-server).
# api-server-count MUST be <= data-parallel-size: the frontend DP load-balancer
# round-robins requests across data_parallel_rank [0, api_server_count). Match it
# to DP_SIZE so every request lands on a live DP rank.
if [[ $IS_MASTER -eq 1 ]]; then
  APISERVERS="--api-server-count ${DP_SIZE} --port ${SERVE_PORT}"
else
  APISERVERS=""
fi

echo "[disagg] node=$(hostname -s) role=$ROLE dp_addr=$DP_ADDR kv_role=${KV_ROLE:-none} backend=$BACKEND"

docker run -d --name "$CONTAINER" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
  --shm-size 128g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=524288:524288 \
  -e VLLM_ROCM_USE_AITER_MLA=0 \
  -e AITER_SITUV2_A8W4=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 \
  -e VLLM_USE_AITER_TRITON_SILU_MUL=0 -e VLLM_ROCM_USE_AITER_RMSNORM=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} -e GLOO_SOCKET_IFNAME=${SOCKET_IFNAME} \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=${NCCL_IB_HCA_VAL} -e MORI_RDMA_DEVICES=${RDMA_DEVICES} -e MORI_SOCKET_IFNAME=${SOCKET_IFNAME} \
  -e NCCL_IB_GID_INDEX=${IB_GID_INDEX} -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  -e MORIIO_SKIP_MAMBA="${MORIIO_SKIP_MAMBA:-0}" \
  -e VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}" \
  -e AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-0}" -e AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}" \
  -e MORI_GPU_ARCHS=gfx942 -e MORI_IB_GID_INDEX=${IB_GID_INDEX} -e MORI_IB_ENABLE_RELAXED_ORDERING=1 \
  -e MORI_NUM_QP_PER_PE=8 -e MORI_SHMEM_HEAP_SIZE=17179869184 \
  -e MORI_RDMA_TC=41 -e MORI_RDMA_SL=0 -e MORI_IO_SL=1 \
  -e VLLM_MORIIO_QP_PER_TRANSFER="${VLLM_MORIIO_QP_PER_TRANSFER:-2}" -e VLLM_MORIIO_NUM_WORKERS="${VLLM_MORIIO_NUM_WORKERS:-4}" \
  -e AITER_JIT_DIR=/opt/vllm_cache/aiter -e TRITON_CACHE_DIR=/opt/vllm_cache/triton \
  -e VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
  -e KVCFG_B64="$KVCFG_B64" \
  -e QUANT_CONFIG="$QUANT_CONFIG" \
  -v "$MODEL_DIR":/model:ro -v "$LOGHOST":/logs \
  -v "$JIT_HOST":/opt/vllm_cache \
  ${BNXT_MOUNTS} \
  --entrypoint bash \
  "$IMAGE" -c "
    set -e
    mkdir -p /opt/vllm_cache/aiter /opt/vllm_cache/triton /opt/vllm_cache/vllm
    # The NFS/DockerHub image tar is a STALE build shipping flydsl 0.2.2, but the
    # K3 int4 SiTUv2 MoE path (_setup_kernel_k3_situ_gfx942 -> compile_moe_gemm1)
    # hard-requires flydsl>=0.2.4 (ImportError otherwise -> WorkerProc init fails ->
    # pool never starts). Dockerfile.kimik3_disagg installs 0.2.4 (line 148) but this
    # tar predates that. Bump at container start (once per node, pure-python, fast).
    FLYDSL_VER=\$(python3 -c 'import flydsl,importlib.metadata as m; print(m.version(\"flydsl\"))' 2>/dev/null || echo 0)
    if [ \"\$FLYDSL_VER\" != \"0.2.4\" ]; then
      echo \"[disagg] flydsl \$FLYDSL_VER -> upgrading to 0.2.4 (K3 int4 SiTUv2 requires >=0.2.4)\"
      pip install --no-cache-dir 'flydsl==0.2.4' 2>&1 | tail -1
    fi
    QUANTARG=()
    if [ -n \"\$QUANT_CONFIG\" ]; then
      QUANTARG=(--quantization-config \"\$QUANT_CONFIG\")
    fi
    KVARG=()
    if [ -n \"\$KVCFG_B64\" ]; then
      KVJSON=\$(printf '%s' \"\$KVCFG_B64\" | base64 -d)
      KVARG=(--kv-transfer-config \"\$KVJSON\")
    fi
    echo '[disagg] launching vllm serve...'
    vllm serve /model --served-model-name kimi-k3 --tensor-parallel-size ${TP_SIZE} \
      --data-parallel-size ${DP_SIZE} --data-parallel-size-local ${DP_LOCAL} \
      --data-parallel-address ${DP_ADDR} --data-parallel-rpc-port ${RPC_PORT} ${START} ${HEADLESS} \
      --enable-expert-parallel --all2all-backend ${BACKEND} \
      --trust-remote-code --reasoning-parser kimi_k3 --mm-encoder-tp-mode data --safetensors-load-strategy ${LOAD_STRATEGY:-prefetch} \
      --no-enable-prefix-caching --kv-cache-dtype ${KV_CACHE_DTYPE:-fp8} --block-size ${BLOCK_SIZE:-16} \
      --kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
      --max-model-len ${MAX_MODEL_LEN} --max-num-seqs 8 --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --gpu-memory-utilization ${GPU_UTIL} \
      --distributed-timeout-seconds 7200 \
      --compilation-config '{\"cudagraph_mode\":\"${CG}\",\"custom_ops\":[\"+quant_fp8\"]}' \
      ${APISERVERS} \"\${QUANTARG[@]}\" \"\${KVARG[@]}\" 2>&1 | tee /logs/vllm_${ROLE}.log
  "
echo "[disagg] $ROLE started -> $LOGHOST/vllm_${ROLE}.log"
