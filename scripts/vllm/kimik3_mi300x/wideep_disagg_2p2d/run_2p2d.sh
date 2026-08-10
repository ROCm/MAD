#!/bin/bash
# Kimi-K3 MXFP4 2P/2D wide-EP DISAGGREGATED serve: DP/EP16 per role, MoRI-EP
# (mori all2all) + MoRIIO connector (prefill->decode KV + KDA state transfer).
#
# Topology (4 nodes, 8 GPU each = EP16 per pool):
#   Prefill pool: P-master (rank0, proxy+kv_producer) + P-worker (rank8, headless)
#   Decode  pool: D-master (kv_consumer)             + D-worker (headless)
# Run this per node with ROLE + the shared *_ADDR env set (see run_2p2d_launch.sh).
#
# Applies the KDA MoRIIO patchers at container start (idempotent) so the MoRIIO
# connector carries K3's ~69 KDA (GDN) recurrent+conv state, not just MLA KV.
set -euo pipefail

IMAGE="${IMAGE:-kimik3-wideep-disagg:latest}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your Kimi-K3-MXFP4 weights path}"
ROLE="${ROLE:?ROLE=prefill_master|prefill_worker|decode_master|decode_worker}"
PMASTER="${PMASTER:?prefill master eth0 IP}"
DMASTER="${DMASTER:?decode master eth0 IP}"
PROXY_IP="${PROXY_IP:-$PMASTER}"
PATCHER_DIR="${PATCHER_DIR:-$HOME/k3disagg/patchers}"
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
# Defaults are the validated Broadcom Thor2 (bnxt RoCE) values used on tus1-p3:
# ibv device names rdma0..rdma7, host NIC eno0, GID index 3. On a DIFFERENT
# fabric (e.g. Mellanox mlx5) override these, e.g.:
#   NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
#   RDMA_DEVICES=mlx5_0,mlx5_2,...  SOCKET_IFNAME=eth0  IB_GID_INDEX=3 THOR2_BNXT_FIX=0
SOCKET_IFNAME="${SOCKET_IFNAME:-eno0}"
NCCL_IB_HCA_VAL="${NCCL_IB_HCA:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
RDMA_DEVICES="${RDMA_DEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"
# Thor2 bnxt libibverbs ABI fix (host v34 driver vs image v59). Needed on Thor2;
# harmless elsewhere but set THOR2_BNXT_FIX=0 to skip the two -v mounts.
THOR2_BNXT_FIX="${THOR2_BNXT_FIX:-1}"
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
  -e K3_MAMBA_N1_FORCE="${K3_MAMBA_N1_FORCE:-}" \
  -e K3_WRITE_FENCE="${K3_WRITE_FENCE:-}" \
  -e VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}" \
  -e K3_MLA_SINGLE_SPLIT="${K3_MLA_SINGLE_SPLIT:-1}" \
  -e K3_GROUP_ROUTING="${K3_GROUP_ROUTING:-1}" \
  -e K3_EXTRA_FIXES="${K3_EXTRA_FIXES:-0}" \
  -e K3_CHUNK_GATE_SLACK="${K3_CHUNK_GATE_SLACK:-2}" \
  -e K3_CHUNK_GATE_DEBUG="${K3_CHUNK_GATE_DEBUG:-0}" \
  -e K3_XFER_PROBE="${K3_XFER_PROBE:-0}" \
  -e K3_MLA_FULL_PREFILL="${K3_MLA_FULL_PREFILL:-1}" \
  -e K3_FORCE_PREFILL_KDA="${K3_FORCE_PREFILL_KDA:-0}" \
  -e K3_WRITE_FENCE_MS="${K3_WRITE_FENCE_MS:-20}" \
  -e K3_WRITE_DEVSYNC="${K3_WRITE_DEVSYNC:-0}" \
  -e K3_KDA_CONV_DEBUG="${K3_KDA_CONV_DEBUG:-0}" \
  -e K3_FWD_BREADCRUMB="${K3_FWD_BREADCRUMB:-0}" \
  -e K3_WRITE_BC="${K3_WRITE_BC:-0}" \
  -e K3_KDA_STATE_PROBE="${K3_KDA_STATE_PROBE:-0}" \
  -e K3_MAMBA_BC="${K3_MAMBA_BC:-0}" \
  -e K3_DECODE_RECV_PROBE="${K3_DECODE_RECV_PROBE:-0}" \
  -e K3_HS_BC="${K3_HS_BC:-0}" \
  -e K3_INPUTS_PROBE="${K3_INPUTS_PROBE:-0}" \
  -e AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-0}" -e AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}" \
  -e MORI_GPU_ARCHS=gfx942 -e MORI_IB_GID_INDEX=${IB_GID_INDEX} -e MORI_IB_ENABLE_RELAXED_ORDERING=1 \
  -e MORI_NUM_QP_PER_PE=8 -e MORI_SHMEM_HEAP_SIZE=17179869184 \
  -e MORI_RDMA_TC=41 -e MORI_RDMA_SL=0 -e MORI_IO_SL=1 \
  -e VLLM_MORIIO_QP_PER_TRANSFER="${VLLM_MORIIO_QP_PER_TRANSFER:-2}" -e VLLM_MORIIO_NUM_WORKERS="${VLLM_MORIIO_NUM_WORKERS:-4}" \
  -e AITER_JIT_DIR=/opt/vllm_cache/aiter -e TRITON_CACHE_DIR=/opt/vllm_cache/triton \
  -e VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
  -e KVCFG_B64="$KVCFG_B64" \
  -e QUANT_CONFIG="$QUANT_CONFIG" \
  -v "$MODEL_DIR":/model:ro -v "$LOGHOST":/logs -v "$PATCHER_DIR":/patchers:ro \
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
    echo '[disagg] applying KDA MoRIIO patchers...'
    # KDA/HMA/sampler patchers baked into image source (kimik3-wideep-disagg).
    # Runtime-relax the over-strict aiter#4471 packed-int4 guard (the grafted
    # K3-aware AITER lacks compile_moe_gemm1(act=); the proven image runs the
    # same aiter fine -> guard is a false positive on this stack).
    VLLM_SP=\$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null)
    if [ -f /patchers/apply_kimik3_aiter_situv2_int4.py ]; then
      python3 /patchers/apply_kimik3_aiter_situv2_int4.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_situ_aiter_gfx942.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_situ_aiter_gfx942.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_mxfp4_int4_guard_relax.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_mxfp4_int4_guard_relax.py \"\$VLLM_SP\" || true
    fi
    # Reconcile MoRIIO self.block_size from an attention (non-mamba) layer so the
    # MLA block_size (1536) doesn't trip the guard against KDA's block_size=1.
    if [ -f /patchers/apply_kimik3_moriio_block_size_fix.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_block_size_fix.py \"\$VLLM_SP\" || true
    fi
    # Use the PADDED physical mamba page for KDA block stride/geometry (fixes the
    # producer-side GPU memory fault: unpadded conv+ssm page drifts RDMA offsets OOB).
    if [ -f /patchers/apply_kimik3_moriio_mamba_page_pad_fix.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_mamba_page_pad_fix.py \"\$VLLM_SP\" || true
    fi
    # Thread real tp_size into get_port_offset so per-rank ports don't collide when
    # DP-local>1 AND TP>1 (wide-EP TP2xDP-local-4: dp0/tp1 and dp1/tp0 both bound
    # handshake port 8406 -> ZMQError Address already in use -> listener dies -> pool
    # hangs on 'No available shared memory broadcast block'). offset = dp*tp_size+tp.
    if [ -f /patchers/apply_kimik3_moriio_port_offset_tpsize.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_port_offset_tpsize.py \"\$VLLM_SP\" || true
    fi
    # Multi-NODE disagg: advertise the peer pool's per-pod node IPs so prefill can
    # reach decode ranks on the HEADLESS worker node (else KV writes to that node's
    # ranks miss -> context-free generation -> 50%/DP2 88%/DP8 wrong-answer alternation).
    if [ -f /patchers/apply_kimik3_moriio_pod_hosts.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_pod_hosts.py \"\$VLLM_SP\" || true
    fi
    # ROOT-CAUSE FIX: route KDA/mamba state transfer by the MAMBA KV-cache group's
    # block ids (group [1]), not attention's (group [0]). Without this, decode reads
    # zero KDA state -> fluent but context-free output. Load-bearing; always on.
    if [ -f /patchers/apply_kimik3_moriio_mamba_blockids.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_mamba_blockids.py \"\$VLLM_SP\" || true
    fi
    # ROOT-CAUSE FIX: remote_tp_size=1 (un-advertising router) collapses ALL prefill
    # ranks to decode tp0 -> only 1/8 decode shards get KV -> context-free. Normalize
    # degenerate remote TP to local world_size (symmetric TP). Load-bearing; always on.
    if [ -f /patchers/apply_kimik3_moriio_remote_tp_fix.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_remote_tp_fix.py \"\$VLLM_SP\" || true
    fi
    # ROOT-CAUSE FIX: mamba/KDA N-vs-N-1 boundary. Prefill computes h(N-1) (drop last
    # prompt token), decode recomputes token N from h(N-1). Without this, decode
    # double-counts the last token in the recurrent state -> echoes it -> wrong output.
    # Ports vLLM's own nixl/mooncake hybrid-PD handling. Load-bearing; always on.
    if [ -f /patchers/apply_kimik3_moriio_mamba_n1.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_mamba_n1.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_moriio_group_routing.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_group_routing.py \"\$VLLM_SP\" || true
    fi
    if [ \"\${K3_EXTRA_FIXES:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_chunked_allgrp.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_chunked_allgrp.py \"\$VLLM_SP\" || true
    fi
    if [ \"\${K3_EXTRA_FIXES:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_chunk_gate_fix.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_chunk_gate_fix.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_xfer_probe.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_xfer_probe.py \"\$VLLM_SP\" || true
    fi
    # k3-mla-boundary: clamp final MLA block RDMA copy to valid slots (fix decode recall)
    if [ \"\${K3_ENABLE_CLAMP:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_moriio_mla_boundary_clamp.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_mla_boundary_clamp.py \"\$VLLM_SP\" || true
    fi
    # k3-mla-full: the real fix (prefill computes N MLA / KDA stays N-1). Clamp above OFF by default.
    if [ -f /patchers/apply_kimik3_mla_full_prefill.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_mla_full_prefill.py \"\$VLLM_SP\" || true
    fi
    # k3-force-prefill-kda: route disagg boundary token through prefill KDA kernel
    if [ -f /patchers/apply_kimik3_force_prefill_kda.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_force_prefill_kda.py \"\$VLLM_SP\" || true
    fi
    # Force single-split TRITON_MLA decode (deterministic; avoids uninit-tail merge)
    if [ -f /patchers/apply_kimik3_mla_single_split.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_mla_single_split.py \"\$VLLM_SP\" || true
    fi
    # RDMA write-then-notify ordering fence (K3_WRITE_FENCE=delay): settle before
    # write_done so decode does not read stale HBM (non-deterministic recall).
    if [ -f /patchers/apply_kimik3_moriio_write_fence.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_write_fence.py \"\$VLLM_SP\" || true
    fi
    # Diagnostic breadcrumbs for the WRITE KV delivery (K3_WRITE_BC=1).
    if [ \"\${K3_WRITE_BC:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_moriio_write_bc.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_write_bc.py \"\$VLLM_SP\" || true
    fi
    # Ground-truth KDA state probe (K3_KDA_STATE_PROBE=1): norm of recurrent/conv
    # state at the slot decode reads -- ~0 means transferred state didn't land.
    if [ \"\${K3_KDA_STATE_PROBE:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_kda_state_probe.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kda_state_probe.py \"\$VLLM_SP\" || true
    fi
    # Decode-side receive probe (K3_DECODE_RECV_PROBE=1): on write completion,
    # read decode's OWN KV slot norm -- ~0 means RDMA bytes never landed on decode.
    if [ \"\${K3_DECODE_RECV_PROBE:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_decode_recv_probe.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_decode_recv_probe.py \"\$VLLM_SP\" || true
    fi
    # Handshake dial breadcrumb (K3_HS_BC=1): logs self_tp/dial_tp/port/path so we
    # can see if all prefill ranks wrongly dial the same decode rank.
    if [ \"\${K3_HS_BC:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_handshake_dial_bc.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_handshake_dial_bc.py \"\$VLLM_SP\" || true
    fi
    # Decode inputs probe (K3_INPUTS_PROBE=1): logs num_computed_tokens/positions/
    # block_table in _prepare_inputs -- to find the WRITE-mode decode-consume bug.
    if [ \"\${K3_INPUTS_PROBE:-0}\" = \"1\" ] && [ -f /patchers/apply_kimik3_decode_inputs_probe.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_decode_inputs_probe.py \"\$VLLM_SP\" || true
    fi
    # Diagnostics: MORIIO_SKIP_MAMBA=1 isolates the KDA transfer; KDA OOB bounds log.
    if [ -f /patchers/apply_kimik3_moriio_mamba_diag.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_mamba_diag.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_moriio_save_skip_mamba.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_moriio_save_skip_mamba.py \"\$VLLM_SP\" || true
    fi
    # Guard gather_initial_states against OOB KDA state idx (the disagg producer
    # prefill GPU memory fault: mis-flagged has_initial_state -> reads a stale/OOB block).
    if [ -f /patchers/apply_kimik3_kda_gather_guard.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kda_gather_guard.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_kda_conv_debug.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kda_conv_debug.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_fwd_breadcrumb.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_fwd_breadcrumb.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_layer_breadcrumb.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_layer_breadcrumb.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_kda_fa_contiguous.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kda_fa_contiguous.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_kda_internal_bc.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kda_internal_bc.py \"\$VLLM_SP\" || true
    fi
    if [ -f /patchers/apply_kimik3_kvzero_bounds.py ] && [ -n \"\$VLLM_SP\" ]; then
      python3 /patchers/apply_kimik3_kvzero_bounds.py \"\$VLLM_SP\" || true
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
