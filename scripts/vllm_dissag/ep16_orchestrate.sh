#!/bin/bash
# EP16 1P1D disagg orchestrator (4 nodes). Runs on the LOGIN host; drives the 4
# node containers via `spur exec <jobid>`. Each role is a 2-node DP16 group.
#
#   prefill: 007(master,rank0,+router) + 044(headless,rank8)
#   decode : 043(master,rank0)         + 042(headless,rank8)
#
# Order: load image (parallel) -> router -> prefill master+headless & decode
# master+headless (all EP workers register/rendezvous) -> wait ready -> bench.
set -uo pipefail
LAUNCH=/shared_nfs/ravgupta_disagg205/vllm_pd_ep16_launch.sh
TAR=/shared_nfs/ravgupta_disagg205/vllm-mori-pr558-ionic.tar
LOGROOT=/shared_nfs/ravgupta_disagg205/logs_ep16
IMG=vllm-mori-pr558:ionic

# node jobid ip
PF_M_J=1941; PF_M_IP=10.245.156.83    # 008 prefill master + router (RW-NFS; 007 was RO)
PF_H_J=1937; PF_H_IP=10.245.154.29    # 044 prefill headless
DC_M_J=1939; DC_M_IP=10.245.155.134   # 043 decode master
DC_H_J=1936; DC_H_IP=10.245.149.97    # 042 decode headless
ROUTER_IP=$PF_M_IP                     # router lives on prefill master

# tunables (1P1D-EP16, GLM-5.2-FP8)
MAXLEN=${MAXLEN:-131072}
PF_KVBYTES=${PF_KVBYTES:-20000000000}
DC_KVBYTES=${DC_KVBYTES:-20000000000}
A2A_PF=${A2A_PF:-mori_high_throughput}
A2A_DC=${A2A_DC:-mori_low_latency}

drive(){ local j=$1; shift; timeout ${TO:-1200} spur exec "$j" -- bash -lc "$*" </dev/null 2>/dev/null | grep -vE "Triton|triton"; }

echo "===== [1] load image on all 4 nodes (parallel) ====="
for J in $PF_M_J $PF_H_J $DC_M_J $DC_H_J; do
  ( r=$(TO=1800 drive $J "docker images $IMG -q|grep -q . && echo HAVE || (docker load -i $TAR >/dev/null 2>&1 && echo LOADED || echo FAIL)"); echo "[j$J] img=$r" ) &
done
wait
echo "===== [2] router on prefill-master (007) ====="
TO=120 drive $PF_M_J "ROLE=proxy HOST_IP=$PF_M_IP PROXY_IP=$ROUTER_IP DECODE_IP=$DC_M_IP \
  LOG=$LOGROOT ROUTER_DP_LOCAL=8 bash $LAUNCH"
sleep 8

echo "===== [3] prefill master(007,rank0) + headless(044,rank8) ====="
TO=120 drive $PF_M_J "ROLE=prefill NODE_ROLE=master START_RANK=0 DP=16 DP_LOCAL=8 \
  HOST_IP=$PF_M_IP DP_MASTER_IP=$PF_M_IP PROXY_IP=$ROUTER_IP DECODE_IP=$DC_M_IP \
  A2A=$A2A_PF MAXLEN=$MAXLEN KV_CACHE_MEMORY_BYTES=$PF_KVBYTES GPUUTIL=0.85 \
  LOG=$LOGROOT bash $LAUNCH"
TO=120 drive $PF_H_J "ROLE=prefill NODE_ROLE=headless START_RANK=8 DP=16 DP_LOCAL=8 \
  HOST_IP=$PF_H_IP DP_MASTER_IP=$PF_M_IP PROXY_IP=$ROUTER_IP DECODE_IP=$DC_M_IP \
  A2A=$A2A_PF MAXLEN=$MAXLEN KV_CACHE_MEMORY_BYTES=$PF_KVBYTES GPUUTIL=0.85 \
  LOG=$LOGROOT bash $LAUNCH"

echo "===== [4] decode master(043,rank0) + headless(042,rank8) ====="
TO=120 drive $DC_M_J "ROLE=decode NODE_ROLE=master START_RANK=0 DP=16 DP_LOCAL=8 \
  HOST_IP=$DC_M_IP DP_MASTER_IP=$DC_M_IP PROXY_IP=$ROUTER_IP DECODE_IP=$DC_M_IP \
  A2A=$A2A_DC MAXLEN=$MAXLEN KV_CACHE_MEMORY_BYTES=$DC_KVBYTES GPUUTIL=0.90 \
  LOG=$LOGROOT bash $LAUNCH"
TO=120 drive $DC_H_J "ROLE=decode NODE_ROLE=headless START_RANK=8 DP=16 DP_LOCAL=8 \
  HOST_IP=$DC_H_IP DP_MASTER_IP=$DC_M_IP PROXY_IP=$ROUTER_IP DECODE_IP=$DC_M_IP \
  A2A=$A2A_DC MAXLEN=$MAXLEN KV_CACHE_MEMORY_BYTES=$DC_KVBYTES GPUUTIL=0.90 \
  LOG=$LOGROOT bash $LAUNCH"

echo "===== launched. Tail logs in $LOGROOT (vllm_prefill.log/_headless, vllm_decode.log/_headless, vllm_router.log)."
echo "Wait for router 'Add Prefill'+'Add Decode' before benchmarking."
