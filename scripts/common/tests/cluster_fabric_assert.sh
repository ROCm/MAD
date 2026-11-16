#!/bin/bash
# Offline checks for cluster.sh's fabric FALLBACK (what a run gets when it passes
# no fabric settings), on fake sysfs trees. No cluster, no GPUs.
#
# The first three layouts are clusters that already ran on the fixed lists; they
# must resolve exactly as they did. The fourth names its AINIC rails ionic_N and also
# carries a management mlx5_0. Explicit values always win.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

# _layout NAME DEFAULT_ROUTE_IFACE "IFACES" DRIVER:DEV...
_layout() {
    local root="$TMP/$1" iface="$2" ifaces="$3" spec d drv; shift 3
    mkdir -p "$root/ib" "$root/net" "$root/drivers" "$root/bin"
    for spec in "$@"; do
        drv="${spec%%:*}"; d="${spec#*:}"
        mkdir -p "$root/ib/$d/device" "$root/drivers/$drv"
        ln -s "$root/drivers/$drv" "$root/ib/$d/device/driver"
    done
    for d in $ifaces; do mkdir -p "$root/net/$d"; done
    printf '#!/bin/sh\necho "default via 10.0.0.1 dev %s proto dhcp"\n' "$iface" > "$root/bin/ip"
    chmod +x "$root/bin/ip"
}
# _resolve NAME [VAR=VALUE...] -> the fabric cluster.sh settles on
cat > "$TMP/resolve.sh" <<'RESOLVE'
. "$_CLUSTER_SH" >/dev/null 2>&1
echo "arch=$CLUSTER_ARCHETYPE gid=$NCCL_IB_GID_INDEX if=$NCCL_SOCKET_IFNAME kv=$KV_IB_DEVICE hca=${NCCL_IB_HCA:-} mori=${MORI_RDMA_DEVICES:-}"
RESOLVE
_resolve() {
    local root="$TMP/$1"; shift
    # The python3 that runs the launchers (cluster.sh reads the fabric catalog with PyYAML).
    env -i HOME="$TMP" USER=t PATH="$root/bin:$(dirname "$(command -v python3)"):/usr/bin:/bin" \
        CLUSTER_SYSFS_IB="$root/ib" CLUSTER_SYSFS_NET="$root/net" _CLUSTER_SH="$DIR/cluster.sh" "$@" \
        bash "$TMP/resolve.sh"
}
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }

_layout cx7 eth0 "eth0 lo" $(for i in 0 1 2 3 4 5 6 7 8 9; do echo mlx5_core:mlx5_$i; done)
_layout ainic eno0 "eno0 lo" $(for i in 0 1 2 3 4 5 6 7; do echo ionic:rdma$i; done)
_layout thor2 fenic0 "fenic0 lo" $(for i in 0 1 2 3 4 5 6 7; do echo bnxt_en:bnxt_re$i; done)
_layout ionic ens3 "ens3 enP2p0s9 lo" mlx5_core:mlx5_0 $(for i in 0 1 2 3 4 5 6 7; do echo ionic:ionic_$i; done)
_layout bare eth0 "eth0 lo"
R8="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
B8="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7"
I8="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"

echo "=== existing clusters resolve as before ==="
_is "$(_resolve cx7)"   "arch=cx7 gid=3 if=eth0 kv=mlx5_1 hca= mori="                     "CX7 (mlx5_0..9): no rails set, connectors keep their own"
_is "$(_resolve ainic)" "arch=ainic gid=1 if=eno0 kv=rdma0 hca=$R8 mori=$R8"             "AINIC named rdma0..7"
_is "$(_resolve thor2)" "arch=thor2 gid=3 if=fenic0 kv=bnxt_re0 hca=$B8 mori=$B8"        "Thor2 bnxt_re0..7"
_is "$(_resolve bare)"  "arch=unknown gid=3 if=eth0 kv=mlx5_1 hca= mori="                "no RDMA devices: the old defaults"

echo "=== AINIC named ionic_N, with a management mlx5 NIC ==="
_is "$(_resolve ionic)"  "arch=ainic gid=1 if=ens3 kv=ionic_0 hca=$I8 mori=$I8"           "ainic, rails ionic_0..7, default-route interface"

echo "=== an explicit value always wins over the fallback ==="
_is "$(_resolve ionic NCCL_IB_HCA=ionic_0 NCCL_SOCKET_IFNAME=enX KV_IB_DEVICE=ionic_3 NCCL_IB_GID_INDEX=0)" \
    "arch=ainic gid=0 if=enX kv=ionic_3 hca=ionic_0 mori=$I8"                            "passed HCA, interface, KV NIC and GID are kept"
_is "$(_resolve ionic CLUSTER_ARCHETYPE=cx7)" "arch=cx7 gid=3 if=ens3 kv=mlx5_1 hca= mori=" \
    "a passed archetype is not re-detected"

echo "=== a family's own settings ==="
cat > "$TMP/env.sh" <<'ENVSH'
. "$_CLUSTER_SH" >/dev/null 2>&1
echo "roce=${RCCL_AINIC_ROCE:-}"
ENVSH
_env() { local root="$TMP/$1"; shift
    env -i HOME="$TMP" USER=t PATH="$root/bin:$(dirname "$(command -v python3)"):/usr/bin:/bin" \
        CLUSTER_SYSFS_IB="$root/ib" CLUSTER_SYSFS_NET="$root/net" _CLUSTER_SH="$DIR/cluster.sh" "$@" bash "$TMP/env.sh"; }
_is "$(_env ainic)" "roce=1" "ainic sets RCCL_AINIC_ROCE"
_is "$(_env ainic RCCL_AINIC_ROCE=0)" "roce=0" "...unless the run set it"
_is "$(_env cx7)" "roce=" "other families do not"

echo "=== a new NIC family is a catalog entry ==="
python3 - "$DIR/../../compat/rocm.yaml" "$TMP/catalog.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["fabrics"]["newnic"] = {"match": ["vnic"], "gid_index": 5, "socket_ifname": "enx0", "rdma_drivers": "vnic",
                          "kv_nic": "vnic0", "rails": {"driver": "vnic_drv", "default": ["vnic0", "vnic1"]}}
c["fabric_detection"]["order"].insert(0, "newnic")
yaml.safe_dump(c, open(sys.argv[2], "w"))
PY
_layout newnic enx0 "enx0 lo" $(for i in 0 1 2; do echo vnic_drv:vnic$i; done)
_is "$(_resolve newnic MAD_CATALOG="$TMP/catalog.yaml")" "arch=newnic gid=5 if=enx0 kv=vnic0 hca=vnic0,vnic1,vnic2 mori=vnic0,vnic1,vnic2" \
    "detected, rails found by driver, values from the catalog -- no code change"
_is "$(_resolve cx7 MAD_CATALOG="$TMP/none.yaml" 2>/dev/null)" "arch=unknown gid=3 if=eth0 kv=mlx5_1 hca= mori=" \
    "no catalog: the old cx7 values, with a warning"

echo "======================================================"
echo "  cluster_fabric_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
