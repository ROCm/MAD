#!/bin/bash
# =============================================================================
# GLM-5.3-Flash EP8 1P/1D disaggregated orchestrator (DP8 + expert-parallel,
# MoRIIO KV transfer + allgather/reducescatter MoE dispatch).
#
# Drives the two legs over `spur exec <job-id>` (this cluster's per-node exec);
# on another cluster replace drive() with your own remote exec. The fix is
# in-image (VLLM_REF); this orchestrator only sets serving env + bring-up order.
#
# Usage:
#   PF_JOB=<a> DC_JOB=<b> PF_IP=<a_ip> DC_IP=<b_ip> \
#   IMG=<image> MODEL=<weights> ROUTER_BIN=<router> \
#   INFRA_ENV="GLIBC_SWAP=1 HOSTLIBS=<dir> MORI_PATCHED=1 MORI_SO_DIR=<dir>" \
#     bash orch_ep8.sh
#
# EP8 specifics vs TP4:
#   MODE=ep, DP=8, A2A=allgather_reducescatter (the patched mori shmem crashes in
#   the MoRI-EP all2all path; AgRs keeps DP8+EP compute + MoRIIO KV disagg).
#   GPUUTIL 0.40 + SPARSE_IDX_MB 4096 (more VRAM headroom than TP4).
#   ROUTER_DP_LOCAL=8 (NOT 1). MAXLEN 32768 with single-chunk prefill to match.
# =============================================================================
set -uo pipefail
PF_JOB="${PF_JOB:?prefill node handle}"; DC_JOB="${DC_JOB:?decode node handle}"
PF_IP="${PF_IP:?prefill node ip}"; DC_IP="${DC_IP:?decode node ip}"
REMOTE_DIR="${REMOTE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMG="${IMG:-rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix}"  # base image; launcher mounts ./patches/ overlays
MODEL="${MODEL:?GLM-5.3-Flash weights on the nodes}"
ROUTER_BIN="${ROUTER_BIN:?vllm-router binary on the nodes}"
PROXY_PING="${PROXY_PING:-36390}"
WORKDIR="${WORKDIR:-/tmp/glm53_ep8}"
MAXLEN="${MAXLEN:-32768}"                 # raise (+ the prefill batch below) for longer ctx
INFRA_ENV="${INFRA_ENV:-}"                # GLIBC_SWAP/HOSTLIBS/MORI_PATCHED/MORI_SO_DIR passthrough
ROUTER="http://${PF_IP}:10001"

COMMON="MODE=ep DP=8 A2A=allgather_reducescatter GPUUTIL=0.40 SPARSE_IDX_MB=4096 \
EAGER=1 MORIIO_DEFER_WRITES=1 MORI_EP_OVER_RDMA=1 MORI_NO_ATOMIC_MR=1 \
KV_DTYPE=auto BLOCK_SIZE=4 MAXLEN=$MAXLEN \
IMG=$IMG MODEL=$MODEL WORKDIR=$WORKDIR PROXY_IP=$PF_IP DECODE_IP=$DC_IP PROXY_PING=$PROXY_PING $INFRA_ENV"

drive(){ local j="$1"; shift; timeout "${TO:-90}" spur exec "$j" -- bash -lc "$*" </dev/null 2>&1 | tail -3; }
wait_ready(){ for i in $(seq 1 60); do
    local r; r=$(TO=25 drive "$1" "docker logs vllm_$2 2>&1 | grep -c 'Application startup complete'")
    [ "$(echo "$r"|tail -1)" != "0" ] && { echo "  $2 READY"; return 0; }; sleep 15
  done; echo "  $2 TIMEOUT"; return 1; }

echo "=== [0] PRECLEAN ==="
drive "$PF_JOB" "docker rm -f vllm_prefill vllm_proxy >/dev/null 2>&1; echo cleaned-pf"
drive "$DC_JOB" "docker rm -f vllm_decode >/dev/null 2>&1; echo cleaned-dc"

echo "=== [1] ROUTER on prefill node (ROUTER_DP_LOCAL=8 for EP) ==="
drive "$PF_JOB" "cd $REMOTE_DIR && ROLE=proxy MODE=ep ROUTER_DP_LOCAL=8 ROUTER_BIN=$ROUTER_BIN \
  IMG=$IMG MODEL=$MODEL WORKDIR=$WORKDIR $INFRA_ENV \
  HOST_IP=$PF_IP PROXY_IP=$PF_IP DECODE_IP=$DC_IP PROXY_PING=$PROXY_PING bash vllm_pd_launch.sh"

echo "=== [2] DECODE first ==="
# NOTE: decode kept EAGER here. TP4 decode CUDA graphs give ~6x lower TPOT (see
# RESULTS.md + run_flash_disagg_tp4.sh) but graphs were NOT yet verified for EP8's
# DP8+expert-parallel decode path -- flip to EAGER=0 DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE
# (drop --enforce-eager) once EP8 graphs are validated (recall + no crash).
drive "$DC_JOB" "cd $REMOTE_DIR && ROLE=decode $COMMON HOST_IP=$DC_IP DECODE_CUDAGRAPH_MODE=NONE \
  EXTRA_ARGS='--max-num-seqs 128 --enforce-eager' bash vllm_pd_launch.sh"

echo "=== [3] PREFILL second (single-chunk prefill: --max-num-batched-tokens = MAXLEN) ==="
drive "$PF_JOB" "cd $REMOTE_DIR && ROLE=prefill $COMMON HOST_IP=$PF_IP \
  EXTRA_ARGS='--max-num-seqs 512 --max-num-batched-tokens $MAXLEN' bash vllm_pd_launch.sh"

echo "=== [4] WAIT for both legs ==="
wait_ready "$DC_JOB" decode  || exit 1
wait_ready "$PF_JOB" prefill || exit 1

echo "=== [5] WARM handshake (retry; mori CreateSession cold-start) ==="
for k in 1 2 3 4; do
  R=$(TO=70 drive "$PF_JOB" "curl -s -m 60 $ROUTER/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":6,\"temperature\":0}'")
  echo "$R" | grep -q text_completion && { echo "  SERVED"; break; }; sleep 6
done
echo "=== DONE. Router: $ROUTER ==="
