#!/bin/bash
# Offline check of scripts/common/launcher/launcher_lib.sh: node selection, addresses,
# running the node program and the verdict, as every launcher uses them.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$DIR/common/launcher/launcher_lib.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

# Stand-ins for what the launchers source before the library.
cat > "$TMP/stubs.sh" <<'EOF'
cluster_expand_nodelist() { tr ',' '\n' <<<"$1"; }
dispatch_node_addresses() { case "$1" in node1) echo "192.168.0.1 10.158.0.1" ;; *) echo "192.168.0.${1#node}" ;; esac; }
dispatch_is_leader() { [ "${LEADER:-1}" = 1 ]; }
dispatch_run() { echo "dispatch_run $*" >> "$CALLS"; [ "$1" = node ] && return "${NODE_RC:-0}"; return 0; }
config_write() { echo "config_write $1" >> "$CALLS"; }
cluster_job_verdict() { echo "verdict $*" >> "$CALLS"; return "$1"; }
EOF
_run() { env -i PATH="$PATH" CALLS="$TMP/calls" SLURM_JOB_ID=7 LOG_PATH="$TMP/logs" MODEL_NAME=M "$@" \
             bash -c 'set -euo pipefail; . "$0"; . "$1"; '"$BODY" "$TMP/stubs.sh" "$LIB" 2>&1; }

echo "=== launcher_select_nodes ==="
BODY='launcher_select_nodes 3 jn; echo "SEL=$SELECTED_NODELIST NN=$SLURM_NNODES NL=$SLURM_JOB_NODELIST TPN=$SLURM_TASKS_PER_NODE JN=$SLURM_JOB_NAME"'
out="$(_run SLURM_JOB_NODELIST=node2,node10,node3,node4)"
_has "$out" "SEL=node2,node10,node3" "the first N nodes, in the scheduler's order (node10 stays after node2)"
_has "$out" "NN=3 NL=node2,node10,node3 TPN=1(x3)" "the SLURM_* view is narrowed to them"
_has "$out" "JN=jn" "a default job name"
out="$(_run SLURM_JOB_NODELIST=node1,node2,node3,node4 SLURM_JOB_NAME=mine)"
_has "$out" "JN=mine" "a job name already set is kept"
out="$(_run SLURM_JOB_NODELIST=node1,node2)"; rc=$?
_is "$rc" "1" "a short allocation is refused"
_has "$out" "topology needs 3 nodes but the allocation has 2" "with the reason"

echo ""
echo "=== launcher_node_ips ==="
BODY='SELECTED_NODES=$(printf "node1\nnode2\nnode3"); launcher_node_ips ${PREFIX:-}; echo "IPS=$IPADDRS M=$MASTER_NODE/$MASTER_ADDR R=${ROUTER_ADDR:-none}"'
out="$(_run)"
_has "$out" "IPS=192.168.0.1,192.168.0.2,192.168.0.3 M=node1/192.168.0.1 R=none" "first address of each node; master is the first node"
out="$(_run PREFIX=10.158.)"
_has "$out" "IPS=10.158.0.1,192.168.0.2,192.168.0.3" "the address in the subnet, else the first (also under set -e -o pipefail)"
out="$(_run ROUTER_NODE=1)"
_has "$out" "R=192.168.0.3" "ROUTER_NODE=1: the last node is the router"

echo ""
echo "=== launcher_run_nodes / launcher_verdict ==="
BODY='SELECTED_NODELIST=node1,node2; launcher_run_nodes /x/node.sh "docker rm c"; echo "RC=$LAUNCHER_RC"; launcher_verdict 1; echo "after verdict"'
rm -f "$TMP/calls"; out="$(_run)"; rc=$?
_is "$(cat "$TMP/calls")" "config_write $TMP/logs/7/config.resolved.json
dispatch_run node node1,node2 bash /x/node.sh
dispatch_run cleanup node1,node2 bash -c docker rm c
verdict 0 $TMP/logs/7/perf.csv $(pwd)/perf_M.csv 1" "record, run, clean up, verdict on the leader"
_is "$rc" "0" "a passing run ends 0"
rm -f "$TMP/calls"; out="$(_run NODE_RC=3)"; rc=$?
_has "$out" "RC=3" "the nodes' status is kept through the cleanup"
_is "$rc" "3" "and the leader's verdict fails the job with it"
rm -f "$TMP/calls"; out="$(_run NODE_RC=4 LEADER=0)"; rc=$?
_is "$rc" "4" "a non-leader ends with its own node's status"
_is "$(grep -c '^verdict\|^config_write' "$TMP/calls")" "0" "and neither records the config nor gives a verdict"

echo ""
echo "======================================================"
echo "  launcher_lib_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
