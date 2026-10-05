#!/bin/bash
# =============================================================================
# GLM-5.3-Flash DISAGGREGATED 1P1D (TP4) over MoRIIO on ionic MI355X (gfx950)
# ONE-COMMAND reproducible recipe. Brings up prefill+decode+router in the
# correct order, warms the handshake, and runs a recall smoke test.
#
# WHAT THIS PROVES: disagg KV-transfer with the MLA kernel-block-scale fix
# (moriio_layout.py) + per-group cross-chunk KV accumulation (moriio_connector.py)
# -> correct needle retrieval to 871K tokens (near the model's max_model_len),
# all needle depths. See RESULTS.md for the full grid.
#
# RECIPE NOTE: prefill uses CHUNKED prefill (--enable-chunked-prefill
# --max-num-batched-tokens 16384). Chunked prefill both (a) keeps every chunk's
# per-group KV (the connector accumulates blocks across chunks) and (b) avoids
# the single-chunk Triton compile wall (<~780K), so it recalls correctly all the
# way to max_model_len. (Single-chunk --max-num-batched-tokens>=max-model-len
# also works but caps ~100K on VRAM/compile; chunked supersedes it.)
#
# USAGE (from a host that can `spur exec` the jobs):
#   PF_JOB=6409 DC_JOB=6410 PF_IP=10.245.155.8 DC_IP=10.245.156.221 \
#     bash run_flash_disagg_tp4.sh
# Re-run any time; it tears down and relaunches cleanly.
# =============================================================================
set -uo pipefail

# ---- CONFIG (override via env) ----------------------------------------------
# This orchestrator drives the two legs over `spur exec <job-id>` (this cluster's
# per-node exec). On a different cluster, replace `drive()` with your own remote
# exec (ssh, srun, ...). The per-leg env below is the portable part -- it is what
# vllm_pd_launch.sh consumes; the fix set is the ./patches/ overlays the launcher
# mounts onto the base image (OVERLAYS=1, default).
PF_JOB="${PF_JOB:?prefill node handle (spur job id, e.g. 6409)}"
DC_JOB="${DC_JOB:?decode  node handle (spur job id, e.g. 6410)}"
PF_IP="${PF_IP:?prefill node ip}"
DC_IP="${DC_IP:?decode  node ip}"
# Repo path of the launcher, INSIDE the node's filesystem (default: this dir on a
# shared mount). Override REMOTE_DIR to wherever this recipe dir is on the nodes.
REMOTE_DIR="${REMOTE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMG="${IMG:-rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix}"  # base image; launcher mounts ./patches/ overlays
MODEL="${MODEL:?path to GLM-5.3-Flash weights on the nodes}"
ROUTER_BIN="${ROUTER_BIN:-/usr/local/bin/vllm-router}"   # in-image by default; set to a host path to override
PROXY_PING="${PROXY_PING:-36382}"
WORKDIR="${WORKDIR:-/tmp/glm53_disagg}"                  # writable logs/config dir on the nodes
# Optional node infra (leave unset if not needed): GLIBC_SWAP + HOSTLIBS for the
# ionic-driver glibc closure; MORI_PATCHED + MORI_SO_DIR for a host-built patched
# mori .so set. Passed through verbatim.
INFRA_ENV="${INFRA_ENV:-}"
ROUTER="http://${PF_IP}:10001"

# Portable per-leg env. GPUUTIL 0.5 (VRAM headroom), MoRIIO KV disagg, TP4.
COMMON_ENV="GPUUTIL=0.5 SPARSE_IDX_MB=512 EAGER=1 MORIIO_DEFER_WRITES=1 \
MORI_NO_ATOMIC_MR=1 KV_DTYPE=auto BLOCK_SIZE=4 MAXLEN=940000 MODE=tp4 \
IMG=$IMG MODEL=$MODEL WORKDIR=$WORKDIR PROXY_IP=$PF_IP DECODE_IP=$DC_IP PROXY_PING=$PROXY_PING $INFRA_ENV"

drive(){ local j="$1"; shift; timeout "${TO:-90}" spur exec "$j" -- bash -lc "$*" </dev/null 2>&1 | tail -3; }
wait_ready(){ # $1=job $2=name
  for i in $(seq 1 40); do
    local r; r=$(TO=25 drive "$1" "docker logs vllm_$2 2>&1 | grep -c 'Application startup complete'")
    [ "$(echo "$r"|tail -1)" = "1" ] && { echo "  $2 READY"; return 0; }
    sleep 15
  done; echo "  $2 TIMEOUT"; return 1; }

echo "=== [0] PRECLEAN ==="
drive "$PF_JOB" "docker rm -f vllm_prefill vllm_proxy >/dev/null 2>&1; echo cleaned-pf"
drive "$DC_JOB" "docker rm -f vllm_decode >/dev/null 2>&1; echo cleaned-dc"
# warm page cache (optional, speeds reloads)
drive "$PF_JOB" "(cat $MODEL/*.safetensors >/dev/null 2>&1 &); echo warm-pf" >/dev/null
drive "$DC_JOB" "(cat $MODEL/*.safetensors >/dev/null 2>&1 &); echo warm-dc" >/dev/null

echo "=== [1] ROUTER on prefill node (ROUTER_DP_LOCAL=1 for TP — else HTTP 400) ==="
drive "$PF_JOB" "cd $REMOTE_DIR && ROLE=proxy MODE=tp4 ROUTER_DP_LOCAL=1 ROUTER_BIN=$ROUTER_BIN \
  IMG=$IMG MODEL=$MODEL WORKDIR=$WORKDIR $INFRA_ENV \
  HOST_IP=$PF_IP PROXY_IP=$PF_IP DECODE_IP=$DC_IP PROXY_PING=$PROXY_PING \
  bash vllm_pd_launch.sh"

echo "=== [2] DECODE first (order matters: prefill handshakes to decode) ==="
# Decode runs with CUDA graphs (FULL_AND_PIECEWISE): ~6x lower TPOT / ~3.4x higher
# decode throughput vs --enforce-eager, recall unchanged (verified 8K/60K, see
# RESULTS.md). EAGER=0 here overrides the COMMON_ENV EAGER=1; the launcher adds
# --cudagraph-capture-sizes, so do NOT repeat it in EXTRA_ARGS (would duplicate).
# Prefill stays eager (graphs don't help the single big prefill pass).
drive "$DC_JOB" "cd $REMOTE_DIR && ROLE=decode $COMMON_ENV EAGER=0 DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE HOST_IP=$DC_IP \
  EXTRA_ARGS='--max-num-seqs 256' \
  bash vllm_pd_launch.sh"

echo "=== [3] PREFILL second (chunked prefill: keeps per-chunk KV + dodges compile wall) ==="
drive "$PF_JOB" "cd $REMOTE_DIR && ROLE=prefill $COMMON_ENV HOST_IP=$PF_IP \
  EXTRA_ARGS='--max-num-seqs 256 --enforce-eager --enable-chunked-prefill --max-num-batched-tokens 16384' bash vllm_pd_launch.sh"

echo "=== [4] WAIT for both legs (weights ~5-6 min each) ==="
wait_ready "$DC_JOB" decode  || exit 1
wait_ready "$PF_JOB" prefill || exit 1

echo "=== [5] WARM handshake (mori CreateSession cold-start; retry a few) ==="
for k in 1 2 3 4; do
  R=$(TO=70 drive "$PF_JOB" "curl -s -m 60 $ROUTER/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":6,\"temperature\":0}'")
  echo "$R" | grep -q text_completion && { echo "  SERVED: $(echo "$R"|grep -oE '\"text\":\"[^\"]*'|head -1)"; break; }
  sleep 6
done

echo "=== [6] RECALL SMOKE (exact needle retrieval) ==="
TO=90 drive "$PF_JOB" "python3 - <<'PY'
import json,urllib.request
ROUTER='$ROUTER'; MODEL='$MODEL'
for W in (500, 8000):
    f='The quiet river wound through the green valley beneath a pale sky. '
    n=max(1, W//len(f.split())); P=[]
    for i in range(n):
        if i==int(n*0.9): P.append('The special access code is DELTA-9931. ')
        P.append(f)
    body=''.join(P)+' Recall the special access code stated above. The special access code is'
    d=json.dumps({'model':MODEL,'prompt':body,'max_tokens':8,'temperature':0}).encode()
    r=urllib.request.Request(ROUTER+'/v1/completions',data=d,headers={'Content-Type':'application/json'})
    j=json.loads(urllib.request.urlopen(r,timeout=120).read()); a=j['choices'][0]['text']
    print('  ~%dw tok=%s OK=%s ANS=%r'%(W,j.get('usage',{}).get('prompt_tokens'),'9931' in a,a))
PY"
echo "=== DONE. Router: $ROUTER ==="
