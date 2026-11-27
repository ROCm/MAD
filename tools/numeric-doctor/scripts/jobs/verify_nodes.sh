#!/bin/bash
#SBATCH --job-name=nd-verify-nodes
#SBATCH --time=01:00:00
# debug-numeric step 6: does the reproducer reproduce on OTHER nodes? (`step6-finish.md`)  The package's own
# script runs as a vendor would (package_run.sh: empty environment, from the package's dir) on each of the
# job's K nodes at once -- distinct hosts by construction -- each on its own copy of the package (a run writes
# into its dir). A reproducer that reproduced only on the node 5b ran on looked complete and was not
# (2026-11-23: a stream-K GEMM band present on one node only).
#
#   PACKAGE=<extracted package dir> [SCRIPT=run.sh] [PACKAGE_ENV="K=V ..."] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name nodes --gpus 8 --nodes 2 --exclusive \
#       --exclude <the 5b host> --step 'step6[G1]' -- verify_nodes.sh
#
# Per node: <ART>/verify_nodes/<job>/<host>.log (the package's output) and its verdict from the package's
# LADDER_RESULT= line (a native package's NATIVE_TOTAL=<hits>/<runs>: reproduced when hits > 0). Last line:
#   NODES_REPRODUCED=<k>/<K> hosts=<host>:<REPRODUCED|NOT_REPRODUCED|ERROR>,... [NODE_DEPENDENT]
# k < K with no ERROR = the node is a variable of the bug: ledger it, say it in the README, step 7 reads it.
set -u
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
P=${PACKAGE:?PACKAGE=<extracted package dir>}
P=$(cd -- "$P" && pwd -P) || nd_die "no package dir $PACKAGE"
[ -f "$P/${SCRIPT:-run.sh}" ] || nd_die "no ${SCRIPT:-run.sh} in $P"
K=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
# independent runs, not one multi-rank job: no node may be ended because another finished first (nd_multinode's
# srun --wait grace, 60 s, killed the second node's replay mid-run -- 2026-11-24); 0 = srun waits for all
export ND_NODE_GRACE=0
nd_multinode "$K"
OUT=${OUT:-${ND_ART_HOST:-$PWD}/verify_nodes/${SLURM_JOB_ID:-$$}}
mkdir -p "$OUT" && chmod 777 "$OUT"
echo "verify on $K node(s): $ND_HOSTS  package=$P  out=$OUT"
# one task per node: copy the package, run it through package_run.sh, log per Slurm node name (a node's hostname can differ:
# Slurm node-a reported hostname node-b, 2026-11-24)
"${ND_LAUNCH[@]}" bash -c '
  h=${SLURMD_NODENAME:-$(hostname -s)}; d='"$OUT"'/pkg.$h
  rm -rf "$d" && cp -a '"$P"' "$d" && chmod -R a+rwX "$d"
  PACKAGE=$d SCRIPT='"${SCRIPT:-run.sh}"' bash '"$ND_REPO_HOST"'/scripts/jobs/package_run.sh > '"$OUT"'/$h.log 2>&1
  echo "NODE_DONE $h rc=$?"'
k=0 n=0 errs=0 res=()
for h in ${ND_HOSTS//,/ }; do
  n=$((n + 1)); f=$OUT/$h.log
  line=$(grep -hoE '^(LADDER_RESULT|REPRO_RESULT|NATIVE_TOTAL)=.*' "$f" 2>/dev/null | tail -n 1)
  case $line in
    NATIVE_TOTAL=*) h_=${line#NATIVE_TOTAL=}; h_=${h_%%/*}                # a native package: <hits>/<runs>
                    if [ "${h_:-0}" -gt 0 ] 2>/dev/null; then v=REPRODUCED; else v=NOT_REPRODUCED; fi ;;
    *) v=$(sed -E 's/^[A-Z_]+=([A-Z_]+).*/\1/' <<<"$line") ;;
  esac
  case $v in
    REPRODUCED) k=$((k + 1)) ;;
    NOT_REPRODUCED|CLEAN|NOT_SET_UP) v=NOT_REPRODUCED ;;   # NOT_SET_UP: the futility stop, ran clean -- not a package error
    *) v=ERROR; errs=$((errs + 1)) ;;
  esac
  res+=("$h:$v")
  echo "  $h: ${line:-no result line -- read $f}"
done
tag=""; [ "$errs" = 0 ] && [ "$k" -lt "$n" ] && [ "$k" -gt 0 ] && tag=" NODE_DEPENDENT"
[ "$errs" = 0 ] && [ "$k" = 0 ] && tag=" REPRODUCES_NOWHERE_ELSE"
echo "NODES_REPRODUCED=$k/$n hosts=$(IFS=,; echo "${res[*]}")$tag"
