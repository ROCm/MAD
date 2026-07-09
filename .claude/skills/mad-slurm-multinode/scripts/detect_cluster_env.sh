#!/usr/bin/env bash
# Propose cluster-specific values for mad.env / manifests by inspecting the node.
# Read-only. The agent confirms these with the user before writing configs;
# they are best-effort guesses, not authoritative.
#
# This reflects the node it runs on. The login/jump node and the compute nodes
# can differ (HCAs, GPU arch, interfaces), so for compute-node values run it on
# an allocated node, e.g.:
#   srun -p <partition> [--reservation <res>] [--nodelist <node>] -N1 \
#     bash detect_cluster_env.sh
set -u

echo "== mad-slurm-multinode cluster env detection (proposals only) =="

# --- GPU architecture ---
arch=""
if command -v rocminfo >/dev/null 2>&1; then
  arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
elif command -v rocm-smi >/dev/null 2>&1; then
  arch=$(rocm-smi --showhw 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
fi
echo "MAD_SYSTEM_GPU_ARCHITECTURE : ${arch:-<unknown; check rocminfo>}"

# --- RDMA / IB HCAs ---
hcas=""
if command -v ibv_devices >/dev/null 2>&1; then
  hcas=$(ibv_devices 2>/dev/null | awk 'NR>2 {print $1}' | grep -E '^(mlx5_|rdma|bnxt_re)' | sort)
fi
if [ -z "$hcas" ] && [ -d /sys/class/infiniband ]; then
  hcas=$(ls /sys/class/infiniband 2>/dev/null | grep -E '^(mlx5_|rdma|bnxt_re)' | sort)
fi
if [ -n "$hcas" ]; then
  list=$(echo "$hcas" | sed 's/$/:1/' | paste -sd, -)
  echo "NCCL_IB_HCA (all ports :1)   : $list"
  if echo "$hcas" | grep -q '^rdma'; then
    fam="AINIC (ionic) -> RDMAV_DRIVERS=ionic, IBV_DRIVERS=ionic, RCCL_AINIC_ROCE=1, GID likely 1"
  elif echo "$hcas" | grep -q '^mlx5'; then
    fam="CX7/Mellanox (mlx5) -> RDMAV_DRIVERS=mlx5, IBV_DRIVERS=mlx5, GID likely 3"
  elif echo "$hcas" | grep -q '^bnxt_re'; then
    fam="Broadcom Thor2 (bnxt_re) -> RDMAV_DRIVERS=bnxt_re, IBV_DRIVERS=bnxt_re, GID: check show_gids"
  else
    fam="unknown vendor -> archetype not auto-classified; confirm with the user (see references/cluster-types.md)"
  fi
  echo "archetype guess              : $fam"
  echo "  (note: NCCL_IB_HCA usually lists only the GPU-attached HCAs; trim as needed)"
else
  echo "NCCL_IB_HCA                  : <no mlx5_/rdma devices found; check ibv_devices>"
fi

# --- HCA <-> netdev / NUMA affinity (management vs data-plane) ---
# Some HCAs back a routed management iface (e.g. mlx5_1->eth0, mlx5_6->eth1)
# rather than a GPU rail. NCCL_IB_HCA should list only the data-plane HCAs, so
# split them by their backing netdev: eth*/eno*/bond* = management (skip),
# rdma*/ib*/none = data-plane (use). Confirm against GPU<->NIC affinity below.
echo "HCA -> netdev / NUMA affinity:"
if command -v ibdev2netdev >/dev/null 2>&1; then
  ibdev2netdev 2>/dev/null | sed 's/^/    /'
fi
mgmt_hcas=""; data_hcas=""
if [ -d /sys/class/infiniband ]; then
  for d in /sys/class/infiniband/*; do
    [ -e "$d" ] || continue
    dev=$(basename "$d")
    numa=$(cat "$d/device/numa_node" 2>/dev/null)
    nd=$(ls "$d/device/net" 2>/dev/null | paste -sd, -)
    printf '    %-10s numa=%-3s netdev=%s\n' "$dev" "${numa:-?}" "${nd:-none}"
    if echo "$nd" | grep -qE '^(eth|eno|ens|enp|em|bond)'; then
      mgmt_hcas="$mgmt_hcas $dev"
    else
      data_hcas="$data_hcas $dev"
    fi
  done
fi
if [ -n "$data_hcas" ]; then
  dlist=$(echo $data_hcas | tr ' ' '\n' | sed 's/$/:1/' | paste -sd, -)
  echo "  management HCAs (skip)      :${mgmt_hcas:- none}"
  echo "  data-plane HCAs (use)       :$data_hcas"
  echo "  NCCL_IB_HCA (data-plane :1) : $dlist"
  echo "  (heuristic: a HCA backing a routed eth*/bond iface is management;"
  echo "   rdma*/ib* rails attached to the GPUs are data-plane — confirm by affinity)"
fi

# --- GPU PCIe bus / NUMA (to match data-plane rails to GPUs) ---
if command -v rocm-smi >/dev/null 2>&1; then
  echo "GPU PCIe bus (rocm-smi --showbus):"
  rocm-smi --showbus 2>/dev/null | grep -iE 'GPU|bus' | sed 's/^/    /' | head -20
fi

# --- management interface ---
echo "candidate NCCL_SOCKET_IFNAME :"
if command -v ip >/dev/null 2>&1; then
  ip -br addr 2>/dev/null | awk '$2=="UP" && $1!="lo" {print "    "$1"  "$3}'
else
  echo "    <ip not available>"
fi
echo "  (pick the routable management iface, e.g. eth0 on CX7, eno0 on AINIC)"

# --- RoCEv2 GID index ---
if command -v show_gids >/dev/null 2>&1; then
  echo "RoCEv2 GID candidates (show_gids, look for v2):"
  show_gids 2>/dev/null | grep -iE 'v2|RoCE' | head -10 | sed 's/^/    /'
else
  echo "NCCL_IB_GID_INDEX            : <show_gids not found; CX7 often 3, AINIC often 1 — verify>"
fi

echo "== confirm all of the above with the user before writing mad.env/manifest =="
