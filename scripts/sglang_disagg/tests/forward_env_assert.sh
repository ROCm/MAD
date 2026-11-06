#!/bin/bash
# Offline check: which settings run_xPyD_models.slurm forwards into the sglang container.
# A setting the caller passed must reach the container; one only cluster.sh defaulted
# must not, on a cx7/unknown fabric, so those runs start exactly as before.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
# The launcher's forwarding block, exactly as it runs, with cluster.sh sourced from here.
awk '/^_FABRIC_VARS=/{f=1} f{print} /^echo "\[fabric\]/{exit}' "$DIR/sglang_disagg/run_xPyD_models.slurm" \
  | sed "s#\. \"\${MAD_SCRIPTS_DIR}/common/cluster.sh\"#. \"$DIR/common/cluster.sh\" >/dev/null 2>\&1#" > "$TMP/block.sh"
_fwd() { env -i PATH="$PATH" HOME="$TMP" USER=t CLUSTER_SYSFS_IB="$TMP/none" "$@" bash "$TMP/block.sh" 2>&1 | tail -1 | sed 's/^.*forwarded to the container://; s/^ *//'; }
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }

_is "$(_fwd CLUSTER_ARCHETYPE=cx7)" "none (container defaults)" "cx7, nothing passed: container defaults, as before"
_is "$(_fwd CLUSTER_ARCHETYPE=cx7 GPUS_PER_NODE=4 GENERIC_TP_SIZE=4)" "-e GPUS_PER_NODE=4 -e GENERIC_TP_SIZE=4" \
    "a passed node shape reaches the container"
_is "$(_fwd CLUSTER_ARCHETYPE=cx7 NCCL_IB_HCA=mlx5_0)" "-e NCCL_IB_HCA=mlx5_0" "cx7: a passed fabric value is forwarded, cluster.sh's are not"
_is "$(_fwd CLUSTER_ARCHETYPE=ainic NCCL_IB_HCA=ionic_0 NCCL_SOCKET_IFNAME=ens3)" \
    "-e NCCL_IB_HCA=ionic_0 -e NCCL_IB_GID_INDEX=1 -e NCCL_SOCKET_IFNAME=ens3 -e GLOO_SOCKET_IFNAME=ens3 -e MORI_RDMA_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 -e MORI_IB_GID_INDEX=1 -e IB_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 -e RCCL_AINIC_ROCE=1" \
    "ainic: passed values plus what cluster.sh derived"

echo "======================================================"
echo "  forward_env_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
