#!/bin/bash
# Offline check of scripts/common/dispatch/dispatch.sh beyond the passing runs the
# snapshot harness compares (snapshot.py --equivalence spur-array): which dispatch is
# chosen, a failing node's status reaching the verdict, gathers seen alike by every
# task, and a task that never starts failing the others instead of hanging them.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$DIR/common/dispatch/dispatch.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

B="$TMP/bin"; mkdir -p "$B"
printf '#!/bin/bash\nn=${SLURMD_NODENAME:-node1}; case "$1" in -I) echo "10.0.0.${n#node} ";; *) echo "$n";; esac\n' > "$B/hostname"
printf '#!/bin/bash\necho "srun $*" >> "$TMP_LOG"\n' > "$B/srun"
chmod +x "$B"/*

echo "=== which dispatch ==="
_mode() { env -i PATH="$B:$PATH" "$@" bash -c '. "$0"; dispatch_mode' "$LIB"; }
_is "$(_mode)" "slurm-srun" "default: srun under one allocation"
_is "$(_mode SLURM_ARRAY_TASK_ID=0)" "spur-array" "a job-array task: spur-array"
_is "$(_mode SLURM_ARRAY_TASK_ID=0 MAD_DISPATCH=slurm-srun)" "slurm-srun" "MAD_DISPATCH wins over detection"

echo ""
echo "=== slurm-srun: the srun the launchers always ran ==="
_srun() { rm -f "$TMP/srun.log"; env -i PATH="$B:$PATH" TMP_LOG="$TMP/srun.log" SLURM_NNODES=2 bash -c ". \"$LIB\"; $1" >/dev/null; cat "$TMP/srun.log"; }
_is "$(_srun 'dispatch_node_addresses node2')" "srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist=node2 bash -c hostname -I" "node addresses"
_is "$(_srun 'dispatch_gather probe /bin/bash -c true')" "srun --nodes=2 --ntasks=2 /bin/bash -c true" "gather"
_is "$(_srun 'dispatch_run node node1,node2 bash x.sh')" "srun --nodelist=node1,node2 bash x.sh" "run"

echo ""
echo "=== spur-array: tasks of one run ==="
# Task k of N: runs BODY after sourcing the library, as node<k+1>.
_task() { # k N rendezvous_dir body
    env -i PATH="$B:$PATH" SLURMD_NODENAME="node$(( $1 + 1 ))" SLURM_ARRAY_JOB_ID=77 SLURM_ARRAY_TASK_ID="$1" \
        MAD_NNODES="$2" MAD_RENDEZVOUS_DIR="$3" MAD_RENDEZVOUS_TIMEOUT=5 \
        bash -c '. "$0"; _dispatch_array_init >/dev/null || exit 9; '"$4" "$LIB"
}
R="$TMP/r1"
for k in 0 1 2; do _task $k 3 "$R" 'echo "$SLURM_JOB_ID $SLURM_NNODES $SLURM_JOB_NODELIST $SLURM_PROCID"; dispatch_gather probe bash -c "echo from \$(hostname)"' > "$TMP/t$k.out" & done; wait
_is "$(head -1 "$TMP/t1.out")" "77 3 node1,node2,node3 1" "every task sees the whole run (job id, size, nodes, own rank)"
_is "$(tail -n +2 "$TMP/t0.out" | tr '\n' ' ')" "from node1 from node2 from node3 " "a gather prints every node's output in rank order"
_is "$(tail -n +2 "$TMP/t2.out" | tr '\n' ' ')" "from node1 from node2 from node3 " "...the same on every task"

R="$TMP/r2"
for k in 0 1 2; do _task $k 3 "$R" 'dispatch_run node node1,node2,node3 bash -c "exit \$(( SLURM_PROCID == 1 ? 7 : 0 ))"; echo "rc=$?"' > "$TMP/u$k.out" & done; wait
_is "$(cat "$TMP/u0.out")" "rc=7" "rank 0 gets the worst status of the run (node 2 exited 7)"
_is "$(cat "$TMP/u1.out")" "rc=7" "the failing node's task ends with its own status"
_is "$(cat "$TMP/u2.out")" "rc=0" "a passing node's task ends with its own status"

# Run size from a wrapper's SLURM_NNODES (madengine), with no MAD_NNODES or array size.
R="$TMP/r4"
for k in 0 1; do env -i PATH="$B:$PATH" SLURMD_NODENAME="node$((k + 1))" SLURM_ARRAY_JOB_ID=78 SLURM_ARRAY_TASK_ID=$k \
    SLURM_NNODES=2 MAD_RENDEZVOUS_DIR="$R" MAD_RENDEZVOUS_TIMEOUT=5 \
    bash -c '. "$0"; _dispatch_array_init >/dev/null || exit 9; echo "$SLURM_NNODES $SLURM_JOB_NODELIST"' "$LIB" > "$TMP/w$k.out" & done; wait
_is "$(cat "$TMP/w0.out")" "2 node1,node2" "run size from a wrapper's SLURM_NNODES (madengine)"

R="$TMP/r3"
_t0=$SECONDS
_task 0 2 "$R" 'echo joined' > "$TMP/v0.out" 2>&1; _rc=$?
_is "$_rc" "9" "a task whose peer never starts fails..."
_has "$(cat "$TMP/v0.out")" "rank(s) 1 did not report host_rank* within 5s" "...saying which rank is missing"
_is "$([ $((SECONDS - _t0)) -lt 15 ] && echo bounded || echo hung)" "bounded" "...within MAD_RENDEZVOUS_TIMEOUT"

echo "======================================================"
echo "  dispatch_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
