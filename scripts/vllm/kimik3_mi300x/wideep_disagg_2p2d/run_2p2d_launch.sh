#!/bin/bash
# Orchestrate the K3 2P/2D disagg bring-up from the head/control host.
# Prefill pool: PM (master) + PW (worker). Decode pool: DM (master) + DW (worker).
# Router runs on PM (proxy_ip). Launch order: workers first, then masters, then router.
set -euo pipefail

# --- node eth0 IPs (EDIT for your 4-node allocation) ---
# Prefill pool = 2 nodes (master+worker); decode pool = 2 nodes (master+worker).
# PM_NODE/DM_NODE are the ssh hostnames; *_IP are the eth0 IPs the peers dial.
PM_NODE=${PM_NODE:-<prefill-master-host>}; PM_IP=${PM_IP:-<prefill-master-ip>}   # prefill master + proxy/router
PW_NODE=${PW_NODE:-<prefill-worker-host>}; PW_IP=${PW_IP:-<prefill-worker-ip>}   # prefill worker
DM_NODE=${DM_NODE:-<decode-master-host>};  DM_IP=${DM_IP:-<decode-master-ip>}    # decode master
DW_NODE=${DW_NODE:-<decode-worker-host>};  DW_IP=${DW_IP:-<decode-worker-ip>}    # decode worker

# Path to THIS recipe folder on the control host (scp'd to each node). Override REPO=.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
# Kimi-K3 MXFP4 weights path (must exist on every node; local nvme recommended).
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to your Kimi-K3-MXFP4 path}
# Optional per-node override: MODEL_DIR_144=/path/on/144 (suffix = last hostname segment).
model_dir_for() {
  local host="$1"
  local suf="${host##*-}"
  local v="MODEL_DIR_${suf}"
  if [[ -n "${!v+x}" ]]; then echo "${!v}"; else echo "$MODEL_DIR"; fi
}
SSH="ssh -o StrictHostKeyChecking=no"
# Load-bearing env only. TP2xDP8 -> EP16 per pool (no PP). See README for knobs.
COMMON="IMAGE=${IMAGE:-kimik3-wideep-disagg:latest} TP_SIZE=${TP_SIZE:-2} DP_SIZE=${DP_SIZE:-8} DP_LOCAL=${DP_LOCAL:-4} KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-8000000000} PREFILL_BACKEND=mori_low_latency DECODE_CG=${DECODE_CG:-NONE} MODEL_DIR=${MODEL_DIR} PMASTER=$PM_IP DMASTER=$DM_IP PROXY_IP=$PM_IP DECODE_POD_HOSTS=$DM_IP,$DW_IP PREFILL_POD_HOSTS=$PM_IP,$PW_IP THOR2_BNXT_FIX=${THOR2_BNXT_FIX:-0} SOCKET_IFNAME=${SOCKET_IFNAME:-eth0} NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9} RDMA_DEVICES=${RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9} IB_GID_INDEX=${IB_GID_INDEX:-3} MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048} GPU_UTIL=${GPU_UTIL:-0.88} LOAD_STRATEGY=${LOAD_STRATEGY:-lazy} AUTO_ROUTER=${AUTO_ROUTER:-0}"

deploy() {  # $1=node
  $SSH "$1" 'mkdir -p ~/k3disagg/logs' 2>/dev/null
  scp -o StrictHostKeyChecking=no "$REPO/run_2p2d.sh" "$REPO/load_image.sh" "$1:~/k3disagg/" >/dev/null
  # Self-restoring: ensure the disagg image is present (pull from DockerHub if not).
  $SSH "$1" "cd ~/k3disagg && TAG='${IMAGE:-kimik3-wideep-disagg:latest}' HUB_IMAGE='${HUB_IMAGE:-}' DOCKER_USER='${DOCKER_USER:-}' DOCKER_PAT='${DOCKER_PAT:-}' bash load_image.sh" 2>&1 | tail -1
}

echo "=== deploy scripts+image to 4 nodes ==="
for n in $PM_NODE $PW_NODE $DM_NODE $DW_NODE; do deploy "$n"; echo "  $n ok"; done

run_role() {  # $1=node $2=role
  local md; md=$(model_dir_for "$1")
  $SSH "$1" "cd ~/k3disagg && $COMMON MODEL_DIR=$md ROLE=$2 bash run_2p2d.sh" 2>&1 | tail -1
}

echo "=== start WORKERS first ==="
run_role "$PW_NODE" prefill_worker
run_role "$DW_NODE" decode_worker
sleep 5
echo "=== start MASTERS ==="
run_role "$PM_NODE" prefill_master
run_role "$DM_NODE" decode_master

# Auto-start the router in the discovery window so the engines' _ping threads
# connect before exhausting MAX_PING_RETRIES (a late router misses discovery ->
# "0 prefill 0 decode"). AUTO_ROUTER=1 waits for both masters' /v1/models, then
# launches the router on PM.
if [ "${AUTO_ROUTER:-0}" = "1" ]; then
  echo "=== AUTO_ROUTER: waiting for both masters, then starting router ==="
  ( for t in $(seq 1 180); do
      pm=$($SSH $PM_NODE "curl -s -m 5 http://$PM_IP:20005/v1/models 2>/dev/null | grep -c kimi-k3" 2>/dev/null)
      dm=$($SSH $DM_NODE "curl -s -m 5 http://$DM_IP:20005/v1/models 2>/dev/null | grep -c kimi-k3" 2>/dev/null)
      if [ "${pm:-0}" -ge 1 ] && [ "${dm:-0}" -ge 1 ]; then
        echo "[auto-router] both masters ready; starting router"
        $SSH $PM_NODE "docker exec k3disagg_prefill_master bash -c 'setsid nohup vllm-router --host 0.0.0.0 --port 30000 --vllm-pd-disaggregation --kv-connector moriio --prefill http://$PM_IP:20005 --decode http://$DM_IP:20005 --vllm-discovery-address 0.0.0.0:36367 --intra-node-data-parallel-size ${DP_LOCAL:-4} --moriio-dp-size ${DP_SIZE:-8} --policy round_robin --prefill-policy round_robin --decode-policy round_robin --log-level info > /logs/router.log 2>&1 < /dev/null &'"
        break
      fi
      sleep 20
    done ) &
  echo "[auto-router] watcher started (pid $!)"
fi

echo ""
echo "=== bring-up started. Watch for 'Application startup complete' in: ==="
echo "  PM: $SSH $PM_NODE 'docker logs -f k3disagg_prefill_master'"
echo "  DM: $SSH $DM_NODE 'docker logs -f k3disagg_decode_master'"
echo ""
echo "=== once BOTH masters are up, start the router on $PM_NODE: ==="
cat <<EOF
  $SSH $PM_NODE 'docker exec -d k3disagg_prefill_master bash -c "vllm-router --host 0.0.0.0 --port 30000 --vllm-pd-disaggregation --kv-connector moriio --prefill http://$PM_IP:20005 --decode http://$DM_IP:20005 --vllm-discovery-address 0.0.0.0:36367 --intra-node-data-parallel-size ${DP_LOCAL:-4} --moriio-dp-size ${DP_SIZE:-8} --policy round_robin --prefill-policy round_robin --decode-policy round_robin --log-level info > /logs/router.log 2>&1"'
EOF
echo ""
echo "=== test (after router shows 'Add Prefill'+'Add Decode'): ==="
echo "  curl http://$PM_IP:30000/v1/models"
echo "  python3 niah_probe.py --url http://$PM_IP:30000 --model kimi-k3 --ctx 6000 --depths 0.1,0.5,0.9"
