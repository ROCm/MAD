#!/bin/bash
#SBATCH --job-name=nd-exec
#SBATCH --nodes=1
#SBATCH --time=00:30:00
# One command on a GPU node, in the model's environment -- 5a grouping, a 5N candidate
# (native_run.sh / native_variants.sh), a step-7 experiment, a step-8 validation.  
#
#   ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name <name> [--gpus 8] -- exec.sh \
#       [--gpus all|<id,...>] [--env K=V]... -- '<command with $ND_REPO / $ND_ART>'
#
# Everything after exec.sh is passed to `nd run` (default --gpus all here: a job owns its
# GPUs).  Last line: EXEC_RC=<rc>.  The command runs under `set -o pipefail`: `python3 ... | grep`
# reports the python's failure, not grep's 0 (2026-11-14: EXEC_RC=0 on a command whose every
# python call failed).  141 = a stage cut off by a reader that stopped early (`| head`).
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
# The options (before --) go to nd run as they are; the command (after --) becomes ONE string
# with pipefail in front (nd_join_cmd).  Prefixing the last word instead broke a command given as
# words: `-- bash $ND_ART/t.sh` ran `bash set -o pipefail; t.sh` ("bash: set: No such file or
# directory" in every such log), and a quoted argument with spaces was re-split (2026-11-20).
opts=() cmd=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; cmd=("$@"); break; }; opts+=("$1"); shift; done
[ ${#cmd[@]} -gt 0 ] || nd_die "exec.sh: no command after -- (exec.sh [--gpus all|<id,...>] [--env K=V]... -- '<command>')"
case " ${opts[*]-} " in *" --gpus "*) ;; *) opts=(--gpus all "${opts[@]+"${opts[@]}"}") ;; esac
args=("${opts[@]}" -- "set -o pipefail; $(nd_join_cmd "${cmd[@]}")")
echo "[$(date)] host=$(hostname -s) job=${SLURM_JOB_ID:-local} runtime=$RUNTIME"
# In the background with a TERM trap: `nd stop` and the walltime warning (--signal=B:TERM)
# signal this batch shell only; an untrapped foreground run would die with it, and Slurm
# would then KILL the command before it could stop gracefully [L2].
"$ND" run "${args[@]}" &
c=$!
trap 'echo "[$(date)] TERM -- stopping"; kill -TERM $c 2>/dev/null' TERM INT
nd_wait_pid $c; rc=$?
echo "EXEC_RC=$rc"
exit $rc
