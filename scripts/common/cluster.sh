#!/bin/bash
# =============================================================================
# cluster.sh -- single, generic site config for the MAD multinode launchers.
# -----------------------------------------------------------------------------
# This file is *sourced* by:
#   scripts/sglang_disagg/run_xPyD_models.slurm
#   scripts/vllm_dissag/run_xPyD_models.slurm
#   scripts/vllm_multinode/run_multinode.slurm
#
# Every value is `${VAR:-default}`, so the layering is always:
#
#       environment  >  the defaults below
#
# That one property is what lets the same model card run identically under both
# CI paths, because both of them speak only environment variables:
#
#   STANDALONE  the Jenkins pipeline writes standalone_env.sh, sources it, and
#               submits with `sbatch --export=ALL`.
#   MADENGINE   madengine merges the card's env_vars into the sbatch script it
#               generates (madengine deployment/slurm.py, _build_env_vars).
#
# Neither path has to know about this file. Whatever either one sets wins;
# whatever neither sets lands on the same site default here. Adding a knob here
# is therefore automatically available to both.
#
# WHAT BELONGS HERE: facts about the cluster -- filesystem roots, the partition,
# fabric device names, port numbers, timeouts.
# WHAT DOES NOT: model-specific performance flags. Those live in models.yaml,
# keyed by model name, and are the launcher's business rather than the site's.
#
# Sourced under two different shells: run_multinode.slurm sets
# `set -euo pipefail`, the two disagg launchers set no flags at all. So nothing
# here may rely on `set -e` to catch an error, and every expansion must carry a
# default -- an unset name under `set -u` is a fatal error, not an empty string.
# =============================================================================

# --------------------------------------------------------------- weights
# Local NVMe first, shared NFS second. /mnt/m2m_nobackup is per-node and NOT
# uniform across this cluster; /shared_inference is one NFS export that every
# node reads at once. Order matters: a checkpoint too large to sit in page cache
# is read at NFS speed on every node simultaneously, which is the single largest
# component of bring-up for the big models.
#
# MODEL_DIR, when set, is appended as a final candidate rather than replacing the
# list, which is the behaviour the disagg launchers already had.
export NVME_ROOT="${NVME_ROOT:-/mnt/m2m_nobackup}"
export SHARED_MOUNT="${SHARED_MOUNT:-/shared_inference}"
export MODEL_DIR_CANDIDATES="${MODEL_DIR_CANDIDATES:-${NVME_ROOT}/models_blog ${SHARED_MOUNT}/models_blog}"
export MODEL_NAME="${MODEL_NAME:-None}"
export MODEL_DIR="${MODEL_DIR:-}"
export MODEL_PATH="${MODEL_PATH:-}"

# --------------------------------------------------------------- paths
export LOG_PATH="${LOG_PATH:-${SHARED_MOUNT}/${USER:-$(id -un)}/model_blog_logs}"

# --------------------------------------------------------------- slurm
# These mirror what the scheduler already tells a running job; they matter when
# a launcher is run by hand outside sbatch, where the SLURM_* names are unset.
export SBATCH_PARTITION="${SBATCH_PARTITION:-amd-rccl}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-${SBATCH_PARTITION}}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-m2m}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
# Deliberately the invoking user's cwd, not a person's home directory. A
# hardcoded home only ever works for the one account it names.
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# --------------------------------------------------------------- topology
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export xP="${xP:-1}"
export yD="${yD:-1}"
export TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"
export PP_SIZE="${PP_SIZE:-1}"

# --------------------------------------------------------------- fabric
# USE_CX7_NICS=1 selects the 8 CX7 rail NICs for KV transfer; 0 keeps KV on the
# management NIC, which is cross-rail safe but is a fraction of the bandwidth and
# only shows up as a bottleneck once the input sequence is long enough to move
# real KV. Rail NICs require the allocated nodes to share a rail.
export USE_CX7_NICS="${USE_CX7_NICS:-0}"
export KV_IB_DEVICE="${KV_IB_DEVICE:-mlx5_1}"
export FABRIC_SUBNET_PREFIX="${FABRIC_SUBNET_PREFIX:-10.158.}"

# --------------------------------------------------------------- ports
export SERVE_PORT="${SERVE_PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PROXY_PORT="${PROXY_PORT:-10001}"
export ROUTER_PORT="${ROUTER_PORT:-30000}"

# --------------------------------------------------------------- timeouts
export LOG_WAIT_TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"
export DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
export CPU_DISTRIBUTED_TIMEOUT_SECONDS="${CPU_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"

# --------------------------------------------------------------- container
export DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-256G}"

# =============================================================================
# Functions
# =============================================================================

# cluster_check_model_path <path> <label>
#
# Is <path> present, and non-empty, on EVERY allocated node? Ranks share one
# MODEL_PATH, so a partial hit is useless: one node without the weights fails the
# whole run, deep into bring-up, with a much worse message than this one.
#
# The srun body always exits 0. A missing path is the expected outcome of walking
# a candidate list, not a step failure, and a non-zero srun leaves a
# "<jobid>.N bash FAILED 1:0" row in sacct that reads as a fault in a healthy job.
#
# Non-empty matters as much as present: a staged directory can exist while being
# empty or a broken partial copy, and `[ -d ]` alone would happily select it over
# a working NFS copy.
cluster_check_model_path() {
    local path="$1"
    local label="${2:-$1}"
    local nodes="${SLURM_NNODES:-1}"
    local model_label="${path##*/}"
    local out rc n_found

    # Four outcomes, not two. "Missing" alone cannot distinguish a node with no
    # local NVMe mount at all from a node that has the mount but was never given
    # this model -- and those need completely different fixes, one from the
    # cluster team and one from whoever stages weights. Reporting them the same
    # way is why that question keeps getting argued from memory.
    #
    # EMPTY is its own case because a staged directory can exist while holding
    # nothing: a `[ -d ]` test alone would pick it over a working NFS copy.
    echo "Checking ${label}: ${path}"
    out="$(srun --nodes="${nodes}" --ntasks="${nodes}" /bin/bash -c "
        _p='${path}'
        _d=\"\$(dirname \"\$_p\")\"
        if [ ! -d \"\$_d\" ]; then
            echo \"\$(hostname): - no \$_d on this node (not mounted)\"
        elif [ ! -d \"\$_p\" ]; then
            echo \"\$(hostname): - \$_d present, ${model_label} NOT staged\"
        elif [ -z \"\$(ls -A \"\$_p\" 2>/dev/null)\" ]; then
            echo \"\$(hostname): - \$_p exists but is EMPTY\"
        else
            echo \"\$(hostname): + Found \$_p\"
        fi
        exit 0
    " 2>/dev/null)"
    rc=$?
    [ -n "${out}" ] && echo "${out}"

    # srun failing, or reporting from fewer nodes than were allocated, must not
    # read as "available" -- that would send ranks at a path some cannot see.
    if [ "${rc}" -ne 0 ] || [ -z "${out//[[:space:]]/}" ]; then
        echo "x ${label} could not be checked (srun exit ${rc})"
        return 1
    fi
    n_found="$(printf '%s\n' "${out}" | grep -c 'Found')"
    if [ "${n_found}" -eq "${nodes}" ]; then
        echo "+ ${label} available on ALL nodes"
        return 0
    fi
    echo "x ${label} NOT available on all nodes (${n_found}/${nodes})"
    return 1
}

# cluster_resolve_model_path
#
# Sets MODEL_PATH to the first candidate that holds MODEL_NAME on every node.
# An explicit MODEL_PATH short-circuits the probe entirely; an explicit MODEL_DIR
# is appended as a final candidate, which is the order the disagg launchers used.
# Returns 1 when nothing matched, so the caller decides whether that is fatal.
cluster_resolve_model_path() {
    if [ -n "${MODEL_PATH:-}" ]; then
        echo "MODEL_PATH set explicitly, skipping the location probe: ${MODEL_PATH}"
        return 0
    fi

    local candidates="${MODEL_DIR_CANDIDATES}"
    # Append an explicit MODEL_DIR unless the list already covers it.
    if [ -n "${MODEL_DIR:-}" ]; then
        case " ${candidates} " in
            *" ${MODEL_DIR%/} "*) : ;;
            *) candidates="${candidates} ${MODEL_DIR%/}" ;;
        esac
    fi

    local dir
    for dir in ${candidates}; do
        if cluster_check_model_path "${dir%/}/${MODEL_NAME}" "${dir%/}"; then
            MODEL_PATH="${dir%/}/${MODEL_NAME}"
            MODEL_DIR="${dir%/}"
            export MODEL_PATH MODEL_DIR
            echo ""
            echo "+ Selected MODEL_PATH: ${MODEL_PATH} (available on all nodes)"
            return 0
        fi
    done

    echo ""
    echo "x FATAL: model '${MODEL_NAME}' is not on ALL allocated nodes in any of:"
    for dir in ${candidates}; do
        echo "  - ${dir%/}/${MODEL_NAME}"
    done
    echo ""
    echo "Every rank loads from the same path, so it must exist on every node."
    return 1
}

# cluster_mark_dir <dir> <one-line description>
#
# Leave a note saying what wrote here and whether it is safe to delete.
#
# Everything CI creates on shared storage accumulates: per-job log directories,
# per-image JIT caches, staged image tarballs. None of it is ever read by a human
# who was not already looking for it, and none of it says so. Someone else
# eventually finds a directory holding tens of gigabytes under an account that is
# not theirs and has no way to judge whether deleting it breaks a running job --
# so it is left alone, forever, by everyone.
#
# CACHEDIR.TAG is the existing convention on this filesystem (there is one in
# /shared_inference/models_blog) and backup tools skip directories carrying it.
# The README is for the person with the du output and a full disk.
#
# Written once and never rewritten, so this costs nothing on the runs after the
# first and never clobbers a note someone has edited.
cluster_mark_dir() {
    local dir="$1"
    local what="${2:-CI working data}"
    [ -d "$dir" ] || return 0
    [ -e "$dir/README.ci" ] && return 0

    # The tag body is specified by the Cache Directory Tagging Standard; tools
    # match on this exact first line.
    printf 'Signature: 8a477f597d28d172789f06886806bc55\n# Cache directory created by MAD CI.\n' \
        > "$dir/CACHEDIR.TAG" 2>/dev/null || true

    {
        echo "What: ${what}"
        echo "Written by: MAD CI (${USER:-unknown}) via ${0##*/}"
        echo "First created: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo ""
        echo "Safe to delete when no job is running. Nothing here is an input to a"
        echo "future run: logs are archived by the build that produced them, and"
        echo "caches are rebuilt on demand (slower first run, then back to normal)."
        echo ""
        echo "Contents accumulate and are not pruned automatically."
    } > "$dir/README.ci" 2>/dev/null || true
    return 0
}

# cluster_nvme_mount
#
# Echoes the docker -v flag for local NVMe, or nothing when this node has no such
# mount. Guarded on -d because docker AUTO-CREATES a directory at a bind-mount
# source that does not exist: an unconditional -v would shadow the model with an
# empty directory on a node without the mount, turning a clean fallback into a
# confusing empty-model-dir failure. Evaluated per node, inside the srun body.
cluster_nvme_mount() {
    local root="${NVME_ROOT:-/mnt/m2m_nobackup}"
    [ -d "${root}" ] && printf -- '-v %s:%s' "${root}" "${root}"
    return 0
}
