#!/bin/bash
# Entry point of every pod of a slurm_multi card on Kubernetes (see mad_k8s.py).
#
# madengine's pod spec provides:
#   MAD_K8S_NNODES      nodes (pods) in this run
#   MAD_K8S_NODE_PREFIX pod hostname prefix: pod i is "<prefix>-<i>" (Indexed Job)
#   MAD_K8S_JOB_ID      numeric id shared by every pod (stands in for SLURM_JOB_ID)
#   MAD_K8S_IMAGE       the image every pod runs
#   MAD_K8S_SCRIPT      the card's script, relative to /workspace
#   MAD_K8S_ARGS        the card's args
#   MAD_K8S_MULTIPLE_RESULTS  the card's multiple_results file name, if any
#   JOB_COMPLETION_INDEX      set by Kubernetes for an Indexed Job
#
# Every pod runs the card's script as its own node (MAD_DISPATCH=k8s-indexed): its
# scripts reach their nodes through scripts/common/dispatch/dispatch.sh, the pods
# meet in a rendezvous directory on the shared results volume, each starts its own
# node's part, and pod 0 makes the verdict. Nothing fans out from one pod to the
# others; `docker run` runs in the pod and `scontrol show hostnames` lists the pods
# (mad_k8s.py). A card whose scripts predate dispatch.sh (an older MAD checkout)
# is refused: update MAD.
set -u

WORKSPACE="${MAD_K8S_WORKSPACE:-/workspace}"
RESULTS="${MAD_K8S_RESULTS:-/results}"   # the shared results volume
STATE="${MAD_K8S_STATE:-/tmp/mad-k8s}"
SHIM="${WORKSPACE}/.mad-k8s"
mkdir -p "$STATE/containers" "$SHIM/bin"

# What `docker run` starts a container from: the image's environment and working
# directory, before anything below adds to them.
export MAD_K8S_IMAGE_WORKDIR="$PWD"
python3 -c 'import json, os, sys; json.dump(dict(os.environ), open(sys.argv[1], "w"))' "$STATE/base_env.json"

script="${WORKSPACE}/${MAD_K8S_SCRIPT}"
if [ ! -f "$(dirname "$script")/../common/dispatch/dispatch.sh" ]; then
    echo "[mad-k8s] ${MAD_K8S_SCRIPT} does not reach its nodes through common/dispatch/dispatch.sh" \
         "(a MAD checkout from before per-pod dispatch); update MAD to run it on Kubernetes" >&2
    mkdir -p "${RESULTS}/${MAD_K8S_NODE_PREFIX}-${JOB_COMPLETION_INDEX:-0}" && \
        echo 2 > "${RESULTS}/${MAD_K8S_NODE_PREFIX}-${JOB_COMPLETION_INDEX:-0}/exit_code"
    exit 2
fi
for tool in docker scontrol; do
    printf '#!/bin/bash\nexec python3 %s/mad_k8s.py %s "$@"\n' "$SHIM" "$tool" > "$SHIM/bin/$tool"
    chmod +x "$SHIM/bin/$tool"
done
export PATH="$SHIM/bin:$PATH"

idx="${JOB_COMPLETION_INDEX:-0}"
# This pod's node name, as the other pods address it. Not `hostname`: with host
# networking that is the Kubernetes node's name.
export MAD_K8S_SELF="${MAD_K8S_NODE_PREFIX}-${idx}"
nodes=""
for i in $(seq 0 $(( MAD_K8S_NNODES - 1 ))); do
    nodes="${nodes:+$nodes,}${MAD_K8S_NODE_PREFIX}-${i}"
done
# The allocation, as SLURM would describe it to the batch script.
export SLURM_JOB_ID="$MAD_K8S_JOB_ID" SLURM_JOBID="$MAD_K8S_JOB_ID"
export SLURM_JOB_NAME="${MAD_K8S_NODE_PREFIX}"
export SLURM_JOB_NODELIST="$nodes" SLURM_NODELIST="$nodes"
export SLURM_NNODES="$MAD_K8S_NNODES" SLURM_JOB_NUM_NODES="$MAD_K8S_NNODES"
export SLURM_NTASKS="$MAD_K8S_NNODES" SLURM_NPROCS="$MAD_K8S_NNODES"
export SLURM_NODEID="$idx" SLURM_PROCID="$idx" SLURM_LOCALID=0
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$WORKSPACE}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-kubernetes}"
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
export MAD_DEPLOYMENT_TYPE=kubernetes

# Publish the run's results where madengine's collector reads them: what the card
# declares (multiple_results), else the launcher's per-job perf.csv under LOG_PATH
# -- the same places the SLURM path looks.
publish_results() {
    local out="${RESULTS}/${MAD_K8S_SELF}" published="" cand
    mkdir -p "$out"
    if [ -n "${MAD_K8S_MULTIPLE_RESULTS:-}" ]; then
        published="$(find . "${LOG_PATH:-/nonexistent}" -maxdepth 4 -name "$MAD_K8S_MULTIPLE_RESULTS" -type f 2>/dev/null | head -1)"
    fi
    for cand in "${LOG_PATH:-}/${SLURM_JOB_ID}/perf.csv" "./perf.csv"; do
        [ -z "$published" ] && [ -s "$cand" ] && published="$cand"
    done
    if [ -n "$published" ]; then
        cp "$published" "$out/perf.csv" && echo "[mad-k8s] results: $published -> $out/perf.csv"
    else
        echo "[mad-k8s] no perf CSV found (multiple_results='${MAD_K8S_MULTIPLE_RESULTS:-}', LOG_PATH='${LOG_PATH:-}')" >&2
    fi
}

export MAD_DISPATCH=k8s-indexed
cd "$(dirname "$script")"
echo "[mad-k8s] node ${idx} (${MAD_K8S_SELF}): running $(basename "$script") ${MAD_K8S_ARGS:-} as its own node of ${SLURM_JOB_NODELIST}"
# Kubernetes stops a pod with TERM to this process. Bash runs a trap only once a
# foreground child has returned, so the launcher runs in the background and the TERM
# is passed on to it at once: its own teardown (stopping its containers, the
# verdict) runs as it does on scancel.
# shellcheck disable=SC2086
bash "$(basename "$script")" ${MAD_K8S_ARGS:-} &
launcher=$!
trap 'kill -TERM "$launcher" 2>/dev/null' TERM INT
while :; do
    wait "$launcher"; rc=$?
    kill -0 "$launcher" 2>/dev/null || break   # a trap ended the wait, not the launcher
done
trap - TERM INT
echo "[mad-k8s] $(basename "$script") exited ${rc} on node ${idx}"
[ "$idx" = "0" ] && publish_results
mkdir -p "${RESULTS}/${MAD_K8S_SELF}" && echo "$rc" > "${RESULTS}/${MAD_K8S_SELF}/exit_code"
[ "$idx" = "0" ] && exit "$rc"
# Pod 0's status is the run's: dispatch.sh's rank 0 waits for every rank's status and
# makes the verdict, so this node's failure is already in it (and in exit_code above).
# A nonleader exiting nonzero would fail the Job itself (backoffLimit 0), and
# Kubernetes would then stop pod 0 before it published the results and its verdict.
[ "$rc" = 0 ] || echo "[mad-k8s] node ${idx}: status ${rc} recorded for pod 0's verdict; this pod exits 0"
exit 0
