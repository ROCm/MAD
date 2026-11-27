#!/bin/bash
#SBATCH --job-name=nd-package
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Run a shipped package's own script -- a reproducer's run.sh, a fix's validate.sh -- as a vendor
# would on a GPU node: from the package's directory, in an EMPTY environment (no ND_ENV, no
# profile, none of the investigation's variables), so what is verified is what ships.
#
#   PACKAGE=<extracted package dir> [SCRIPT=run.sh] [PACKAGE_ENV="K=V ..."] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name verify --gpus 8 --exclusive --step '<tag>' -- package_run.sh
#
# Why a job script and not `sbatch run.sh` + `nd adopt`: the step-6/8 verification is the same either
# way, but a plain sbatch can be refused by an unattended session's permissions (2026-11-23: three
# validation runs could not verify as documented and improvised `env -u ...` wrappers), and this
# path is recorded by nd submit (--step) like every other job.  ND_ENV only gets the job queued; the
# package's script sees none of it.
#
# Kept: HOME, USER, LOGNAME, PATH, LANG, TERM, TMPDIR and PACKAGE_ENV (an override the package's
# README documents, e.g. IMAGE=<another release> for a differential replay).  SLURM_* is not kept, so
# a package script that resubmits itself from a login node (no /dev/kfd) is not fooled into thinking
# it is the job; here it runs on the allocated GPU node and starts its runtime directly.
# Last line: PACKAGE_RC=<rc>.
set -uo pipefail
P=${PACKAGE:?PACKAGE=<extracted package dir>}
S=${SCRIPT:-run.sh}
P=$(cd -- "$P" && pwd -P) || { echo "PACKAGE_RC=2 (no dir $PACKAGE)"; exit 2; }
[ -f "$P/$S" ] || { echo "PACKAGE_RC=2 (no $S in $P)"; exit 2; }
[ -e /dev/kfd ] || echo "!! no /dev/kfd on $(hostname -s): the package's script may try to submit itself"
keep=()
for v in HOME USER LOGNAME PATH LANG TERM TMPDIR; do [ -n "${!v:-}" ] && keep+=("$v=${!v}"); done
# shellcheck disable=SC2206
extra=(${PACKAGE_ENV:-})
echo "[$(date)] host=$(hostname -s) job=${SLURM_JOB_ID:-local} package=$P script=$S env=${keep[*]%%=*} ${extra[*]:-}"
cd -- "$P" || exit 2
env -i "${keep[@]}" "${extra[@]}" bash "./$S" &
pid=$!
trap 'echo "[$(date)] TERM -- stopping the package script"; kill -TERM $pid 2>/dev/null' TERM INT
wait $pid; rc=$?
echo "PACKAGE_RC=$rc"
exit $rc
