#!/bin/bash
# =============================================================================
# MiniMax-M3 MXFP4 — single-node vLLM + MoRIIO P/D disaggregation over XGMI
# =============================================================================
# Simulates prefill/decode disaggregation on ONE MI355X node:
#   prefill (kv_producer) on GPU0-3 TP4  +  decode (kv_consumer) on GPU4-7 TP4
#   fronted by vllm-router; KV moves prefill->decode via MoRIIO.
#
# BACKEND=xgmi  -> GPU-to-GPU XGMI KV transfer (intra-node, the goal)
# BACKEND=rdma  -> RDMA-over-NIC KV transfer (default upstream mode)
#
# Usage:
#   BACKEND=xgmi ./run_pd_singlenode.sh          # launch P+D+router, run smoke inference
#   BACKEND=rdma ./run_pd_singlenode.sh
# Override MODEL / LOG / UCXDEV via env as needed.
# Run under `nohup ... &` or via ssh so containers persist (they are `docker -d`).
# =============================================================================
set -uo pipefail

# ---- config (override via env) ----------------------------------------------
IMG="${IMG:-rocm/vllm-dev:vllm-0.23.1-rocm723-mi35x-mori-0625}"
ROUTER_IMG="${ROUTER_IMG:-vllm/vllm-router:nightly-20260721-1fbcde7}"
MODEL="${MODEL:-/models/MiniMax-M3-MXFP4}"     # flat dir w/ weights + remote-code .py (see README)
LOG="${LOG:-$PWD/logs}"; mkdir -p "$LOG"; LOG="$(cd "$LOG" && pwd)"   # absolute (docker -v needs it)
BACKEND="${BACKEND:-xgmi}"                       # xgmi | rdma
UCXDEV="${UCXDEV:-ionic_0:1}"                    # RoCE NIC for UCX (rdma backend); harmless for xgmi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"            # lower to ~0.30 on shared/occupied nodes
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MOUNT="${MOUNT:-/models}"                        # host dir mounted into containers (must contain MODEL)

HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{print $7}')
PING=36367; ROUTER_PORT=30000; PF_PORT=2584; DC_PORT=2585

# Per-worker MoRIIO ZMQ ports MUST differ between the two co-located engines
# (both share host network) or they collide on bind (Address already in use).
PF_HS=6301; PF_NT=61005      # prefill handshake / notify
DC_HS=6311; DC_NT=61015      # decode  handshake / notify

MENV="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_ENGINE_READY_TIMEOUT_S=3600"
# NOTE: --disable-custom-all-reduce is REQUIRED for two co-located TP groups on one node
# (otherwise AITER custom all-reduce IPC handles collide -> hipIpcGetMemHandle EINVAL).
SERVE_FLAGS="--tensor-parallel-size 4 --max-num-batched-tokens 32768 --max-num-seqs 512 \
--block-size 128 --language-model-only --attention-backend TRITON_ATTN --moe-backend aiter \
--no-enable-prefix-caching --gpu-memory-utilization $GPU_MEM_UTIL --tool-call-parser minimax_m3 \
--reasoning-parser minimax_m3 --enable-auto-tool-choice --disable-custom-all-reduce \
--max-model-len $MAX_MODEL_LEN"

echo "[run] $(date) host=$HOST_IP backend=$BACKEND model=$MODEL mem_util=$GPU_MEM_UTIL" | tee "$LOG/run.log"
docker rm -f vm_prefill vm_decode vm_router >/dev/null 2>&1 || true

serve(){ local name=$1 gpus=$2 role=$3 port=$4 hs=$5 nt=$6
  local envargs=""; for kv in $MENV; do envargs+=" -e $kv"; done
  docker run -d --name "$name" --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --ulimit memlock=-1 --ulimit stack=67108864 --init --group-add video --ipc host \
    --shm-size 128G --network host --privileged --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined -v "$MOUNT":"$MOUNT" -v "$LOG":/out \
    -e HIP_VISIBLE_DEVICES=$gpus -e CUDA_VISIBLE_DEVICES=$gpus \
    -e UCX_NET_DEVICES=$UCXDEV -e VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST_IP \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600+port)) $envargs \
    "$IMG" -c "vllm serve $MODEL --served-model-name minimaxm3 --host 0.0.0.0 --port $port --trust-remote-code \
      --kv-transfer-config '{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"$role\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$HOST_IP\",\"proxy_ping_port\":\"$PING\",\"http_port\":\"$port\",\"handshake_port\":\"$hs\",\"notify_port\":\"$nt\",\"read_mode\":true,\"backend\":\"$BACKEND\"}}' \
      $SERVE_FLAGS 2>&1 | tee /out/$name.log"; }

# 1) router first (workers self-register to it via ZMQ discovery on $PING)
docker run -d --name vm_router --network host --ulimit nofile=1048576:1048576 -v "$LOG":/out "$ROUTER_IMG" \
  bash -lc "exec vllm-router --vllm-pd-disaggregation --kv-connector moriio --vllm-discovery-address 0.0.0.0:$PING \
    --port $ROUTER_PORT --host 0.0.0.0 --policy consistent_hash --prefill-policy consistent_hash \
    --decode-policy consistent_hash --log-level info 2>&1 | tee /out/vm_router.log"
echo "[run] router :$ROUTER_PORT (discovery :$PING)" | tee -a "$LOG/run.log"; sleep 8

# 2) both workers (distinct handshake/notify ports)
serve vm_prefill 0,1,2,3 kv_producer $PF_PORT $PF_HS $PF_NT
serve vm_decode  4,5,6,7 kv_consumer $DC_PORT $DC_HS $DC_NT
echo "[run] prefill($PF_PORT,GPU0-3)+decode($DC_PORT,GPU4-7) launched" | tee -a "$LOG/run.log"

# 3) wait until BOTH register to the router (discovery mode: /v1/models stays empty;
#    check the router log for the "Add Prefill"/"Add Decode" registration lines).
for i in $(seq 1 120); do
  P=$(docker logs vm_router 2>&1 | grep -c "Add Prefill" || true)
  D=$(docker logs vm_router 2>&1 | grep -c "Add Decode"  || true)
  ps=$(docker inspect -f '{{.State.Status}}' vm_prefill 2>/dev/null)
  ds=$(docker inspect -f '{{.State.Status}}' vm_decode  2>/dev/null)
  echo "[run] t=$((i*10))s router_prefill=$P router_decode=$D pf=$ps dc=$ds" | tee -a "$LOG/run.log"
  [ "$P" -ge 1 ] && [ "$D" -ge 1 ] && { echo "[run] both workers registered" | tee -a "$LOG/run.log"; break; }
  { [ "$ps" = exited ] || [ "$ds" = exited ]; } && { echo "[run] a worker exited early — see $LOG/vm_*.log" | tee -a "$LOG/run.log"; exit 1; }
  sleep 10
done
sleep 10

# 4) smoke inference through the router (exercises P->D KV transfer over $BACKEND).
# Retry: right after registration the first request can fail while the engine warms up.
echo "[run] === INFERENCE via router ($BACKEND) ===" | tee -a "$LOG/run.log"
REQ='{"model":"minimaxm3","messages":[{"role":"user","content":"What is 17 times 23? Reply with only the final number."}],"max_tokens":200,"temperature":0}'
for attempt in $(seq 1 12); do
  resp=$(curl -s --max-time 120 http://$HOST_IP:$ROUTER_PORT/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$REQ" 2>&1)
  if echo "$resp" | grep -q '"choices"'; then
    echo "$resp" | tee "$LOG/infer.json" >/dev/null
    echo "[run] inference OK (attempt $attempt)" | tee -a "$LOG/run.log"
    echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print('[run] answer:',d['choices'][0]['message'].get('content'))" 2>/dev/null | tee -a "$LOG/run.log"
    break
  fi
  echo "[run] inference attempt $attempt not ready, retrying in 15s..." | tee -a "$LOG/run.log"; sleep 15
done
echo "[run] confirm backend: $(docker logs vm_decode 2>&1 | grep -oE 'Using MoRIIO backend: [A-Z]+' | tail -1)" | tee -a "$LOG/run.log"
echo "[run] DONE $(date)" | tee -a "$LOG/run.log"
