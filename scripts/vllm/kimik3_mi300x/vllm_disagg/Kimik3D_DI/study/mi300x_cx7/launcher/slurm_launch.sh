#!/bin/bash
# MI300X + CX-7 SLURM launch wrapper for 2P/2D disagg (TP2/DP8/EP16).
# Differences from the MI325X ssh path:
#  - node roles come from a SLURM nodelist (not hardcoded IPs)
#  - scripts live on NFS (/shared_inference/ravgupta/k3_study or home); model is node-local NVMe
#  - fabric = mlx5/RoCE (FABRIC=mlx5, THOR2_BNXT_FIX=0), env from env/{v4_env,fabric_env}.sh
#
# Usage (inside a SLURM allocation, or via --jobid):
#   JOBID=231589 NODES="a b c d" bash slurm_launch.sh
# Roles: node1=prefill_master(+router), node2=prefill_worker, node3=decode_master, node4=decode_worker.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVDIR="$(cd "$HERE/../env" && pwd)"
JOBID="${JOBID:?set JOBID to the running SLURM job}"
# resolve nodelist -> array
if [ -z "${NODES:-}" ]; then
  NODES=$(squeue -j "$JOBID" -h -o "%N" | xargs -I{} scontrol show hostnames {} | tr '\n' ' ')
fi
read -r PM PW DM DW _ <<< "$NODES"
[ -z "${DW:-}" ] && { echo "need 4 nodes, got: $NODES"; exit 1; }
echo "roles: PM=$PM PW=$PW DM=$DM DW=$DW"

# per-node IPs on the mgmt/control plane (eth0) — used for DP addr + router prefill/decode URLs
ip_of(){ srun --jobid="$JOBID" -N1 -n1 -w "$1" bash -c "ip -4 -o addr show eth0 2>/dev/null | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null; }
PM_IP=$(ip_of "$PM"); DM_IP=$(ip_of "$DM"); PW_IP=$(ip_of "$PW"); DW_IP=$(ip_of "$DW")
echo "ips: PM=$PM_IP PW=$PW_IP DM=$DM_IP DW=$DW_IP"

# common env exported to every role launch
COMMON="MODEL_DIR=/mnt/m2m_nobackup/models_blog/Kimi-K3-MXFP4 \
IMAGE=${IMAGE:-kimik3-wideep-disagg:v4-mi300x-cx7} \
TP_SIZE=2 DP_SIZE=8 DP_LOCAL=4 KV_CACHE_MEMORY_BYTES=20000000000 K3_WRITE_READBACK=1 \
MAX_MODEL_LEN=${MAX_MODEL_LEN:-320000} \
FABRIC=mlx5 THOR2_BNXT_FIX=0 \
NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
RDMA_DEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
SOCKET_IFNAME=eth0 IB_GID_INDEX=3 \
PREFILL_BACKEND=mori_high_throughput DECODE_BACKEND=mori_low_latency DECODE_CG=FULL_AND_PIECEWISE \
PMASTER=$PM_IP DMASTER=$DM_IP PROXY_IP=$PM_IP \
PREFILL_POD_HOSTS=$PM_IP,$PW_IP DECODE_POD_HOSTS=$DM_IP,$DW_IP"

# launcher lives on NFS so all nodes see the same copy; JIT cache on NFS too (per-role subdir)
LAUNCHER="${LAUNCHER:-/shared_inference/ravgupta/k3_study/launcher/run_2p2d.sh}"
run_role(){ local node="$1" role="$2"
  srun --jobid="$JOBID" -N1 -n1 -w "$node" bash -c \
    "cd \$(dirname $LAUNCHER) && $COMMON ROLE=$role JIT_HOST=/shared_inference/ravgupta/k3_study/jit/\${role%%_*} bash $LAUNCHER" &
}
echo "== launch workers =="; run_role "$PW" prefill_worker; run_role "$DW" decode_worker; sleep 8
echo "== launch masters =="; run_role "$PM" prefill_master; run_role "$DM" decode_master
echo "== workers+masters launched; start router with AUTO_ROUTER once both masters serve /v1/models =="
echo "router: docker exec -d k3disagg_prefill_master vllm-router --host 0.0.0.0 --port 30000 --vllm-pd-disaggregation --kv-connector moriio --prefill http://$PM_IP:20005 --decode http://$DM_IP:20005 --vllm-discovery-address 0.0.0.0:36367 --intra-node-data-parallel-size 4 --moriio-dp-size 8 --policy round_robin --prefill-policy round_robin --decode-policy round_robin"
