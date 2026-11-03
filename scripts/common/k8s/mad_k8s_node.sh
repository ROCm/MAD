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
set -u

WORKSPACE="${MAD_K8S_WORKSPACE:-/workspace}"
STATE="${MAD_K8S_STATE:-/tmp/mad-k8s}"
SHIM="${WORKSPACE}/.mad-k8s"
mkdir -p "$STATE/containers" "$SHIM/bin"

# What `docker run` starts a container from: the image's environment and working
# directory, before anything below adds to them.
export MAD_K8S_IMAGE_WORKDIR="$PWD"
python3 -c 'import json, os, sys; json.dump(dict(os.environ), open(sys.argv[1], "w"))' "$STATE/base_env.json"

for tool in srun docker scontrol; do
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

if [ "$idx" != "0" ]; then
    echo "[mad-k8s] node ${idx} (${MAD_K8S_SELF} on $(hostname)): serving tasks for ${MAD_K8S_NODE_PREFIX}-0"
    exec python3 "$SHIM/mad_k8s.py" agent
fi

# Node 0: the batch host. It serves tasks too -- srun on SLURM runs a step's task on
# the batch node like any other.
python3 "$SHIM/mad_k8s.py" agent &
agent_pid=$!
release() {
    python3 "$SHIM/mad_k8s.py" shutdown-peers
    kill "$agent_pid" 2>/dev/null
}
trap 'release; exit 143' TERM INT

python3 "$SHIM/mad_k8s.py" wait-peers || { echo "[mad-k8s] not every node came up" >&2; release; exit 1; }

script="${WORKSPACE}/${MAD_K8S_SCRIPT}"
cd "$(dirname "$script")"
echo "[mad-k8s] running $(basename "$script") ${MAD_K8S_ARGS:-} on ${SLURM_JOB_NODELIST}"
# shellcheck disable=SC2086
bash "$(basename "$script")" ${MAD_K8S_ARGS:-}
rc=$?
echo "[mad-k8s] $(basename "$script") exited ${rc}"

# Publish the run's results where madengine's collector reads them: what the card
# declares (multiple_results), else the launcher's per-job perf.csv under LOG_PATH
# -- the same places the SLURM path looks.
out="/results/${MAD_K8S_SELF}"
mkdir -p "$out"
published=""
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
echo "$rc" > "$out/exit_code"

release
exit "$rc"
