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
# ways of launching it, because both of them speak only environment variables:
#
#   sbatch      the caller exports the card's env_vars and submits the batch
#               script with `sbatch --export=ALL`.
#   madengine   madengine merges the card's env_vars into the sbatch script it
#               generates (madengine deployment/slurm.py, _build_env_vars).
#
# Neither path has to know about this file. Whatever either one sets wins;
# whatever neither sets lands on the same site default here. Adding a knob here
# is therefore automatically available to both.
#
# WHAT BELONGS HERE: facts about the cluster -- filesystem roots, the partition,
# fabric device names, port numbers, timeouts.
#
# What has to be known BEFORE a job is submitted -- the partition to ask for, GPUs per
# node, exclusivity -- is in clusters/<name>.json instead: pass it to madengine as
# --additional-context-file, or read it for the sbatch options (clusters/README.md). The partition appears in both on purpose:
# there it picks the allocation, here it is only a default for running a launcher by
# hand outside sbatch (see the slurm section below). Keep them in step.
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
# The directory the weights actually live in, which is not always MODEL_NAME.
# A card can be a CONFIGURATION alias: Kimi-K2-Instruct-MoRI-AB and
# Kimi-K2-Instruct-DeepEP-AB differ only in their all-to-all backend and share
# one checkpoint, so the catalog key and the weights directory diverge. A
# launcher that sets MODEL_WEIGHTS_NAME (see scripts/sglang_disagg) gets that
# directory resolved; everything else keeps resolving MODEL_NAME as before.
export MODEL_WEIGHTS_NAME="${MODEL_WEIGHTS_NAME:-${MODEL_NAME}}"
export MODEL_DIR="${MODEL_DIR:-}"
export MODEL_PATH="${MODEL_PATH:-}"

# Whether shared NFS is an acceptable place to load these weights from.
#
# For most models it is: a 63 GB checkpoint is served from page cache and the
# difference does not show. For the very large ones it is not. A checkpoint too
# big to cache is read at NFS speed by every node at once, and at that size the
# read alone can outlast the time limit the partition allows -- so falling back
# to NFS does not degrade the run, it produces a job that cannot finish.
#
# Cards for those models set REQUIRE_LOCAL_WEIGHTS=1. Local NVMe then becomes the
# only candidate, and a node missing it -- or holding a different variant -- is a
# hard failure that says what to fix, rather than a silent hour-long fallback.
#
# Left at 0 by default: it is a property of the MODEL, not of the site, and the
# cards that need it declare it. Smaller models keep the NFS fallback that has
# always worked for them.
export REQUIRE_LOCAL_WEIGHTS="${REQUIRE_LOCAL_WEIGHTS:-0}"

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
# The values below were one cluster's, hardcoded. They are not portable: the
# RDMA driver, the GID index and the control-plane interface all change with the
# adapter vendor, and getting one wrong does not fail loudly -- RCCL initializes
# zero NICs and falls back to TCP, so the run still completes and still reports a
# number, measured over the wrong transport.
#
# CLUSTER_ARCHETYPE names the fabric family. It is DETECTED from the adapters
# present, because the node knows and a human guessing does not. Set it
# explicitly to override, or set any individual variable below -- everything here
# is ${VAR:-default}, so an explicit value always wins over the archetype.
#
#   cx7      Mellanox CX7 / RoCE          mlx5_*      GID 3   iface eth0
#   ainic    AMD AINIC / Pollara          rdma0..7    GID 1   iface eno0
#   thor2    Broadcom Thor2 / RoCE        bnxt_re0..7 GID 3   iface fenic0
#
# Archetype facts and the confirm-on-node procedure:
# .claude/skills/mad-slurm-multinode/references/cluster-types.md
cluster_detect_archetype() {
    local devs=""
    if [ -d /sys/class/infiniband ]; then
        devs="$(ls /sys/class/infiniband 2>/dev/null)"
    elif command -v ibv_devices >/dev/null 2>&1; then
        devs="$(ibv_devices 2>/dev/null | awk 'NR>2 {print $1}')"
    fi
    case "${devs}" in
        *bnxt_re*) echo thor2 ;;
        *rdma*)    echo ainic ;;
        *mlx5*)    echo cx7 ;;
        *)         echo unknown ;;
    esac
}

export CLUSTER_ARCHETYPE="${CLUSTER_ARCHETYPE:-$(cluster_detect_archetype)}"

case "${CLUSTER_ARCHETYPE}" in
    ainic)
        _arch_gid=1;  _arch_iface=eno0;   _arch_drivers=ionic;   _arch_kv_nic=rdma0
        # Without this the AINIC path falls back to verbs/sockets silently.
        export RCCL_AINIC_ROCE="${RCCL_AINIC_ROCE:-1}"
        ;;
    thor2)
        _arch_gid=3;  _arch_iface=fenic0; _arch_drivers=bnxt_re; _arch_kv_nic=bnxt_re0
        ;;
    *)
        # cx7 and unknown share these: the previous hardcoded values, so a node
        # this cannot classify behaves exactly as before rather than differently.
        _arch_gid=3;  _arch_iface=eth0;   _arch_drivers=mlx5;    _arch_kv_nic=mlx5_1
        ;;
esac

export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${_arch_gid}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_arch_iface}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export RDMAV_DRIVERS="${RDMAV_DRIVERS:-${_arch_drivers}}"
export IBV_DRIVERS="${IBV_DRIVERS:-${_arch_drivers}}"

# USE_CX7_NICS=1 selects the 8 CX7 rail NICs for KV transfer; 0 keeps KV on the
# management NIC, which is cross-rail safe but is a fraction of the bandwidth and
# only shows up as a bottleneck once the input sequence is long enough to move
# real KV. Rail NICs require the allocated nodes to share a rail. Meaningful on
# cx7 only; the other archetypes name their KV device directly.
export USE_CX7_NICS="${USE_CX7_NICS:-0}"
export KV_IB_DEVICE="${KV_IB_DEVICE:-${_arch_kv_nic}}"

# RDMA rails: the GPU-attached NICs that NCCL and MoRI move data over. The
# connectors default to one CX7 cluster's list (mlx5_0,2,3,4,5,7,8,9), which names
# devices that do not exist on an AINIC or Thor2 node -- NCCL then skips them and
# falls back to TCP, and the run still reports a number. So on those fabrics the
# rails come from the archetype. On cx7 nothing is set here: each connector keeps
# the choice it already makes (rixl.sh even probes ibstat, which a preset
# NCCL_IB_HCA would switch off), so the fabric this was validated on is unchanged.
case "${CLUSTER_ARCHETYPE}" in
    ainic) _arch_rails="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7" ;;
    thor2) _arch_rails="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7" ;;
    *)     _arch_rails="" ;;
esac
if [ -n "${_arch_rails}" ]; then
    export NCCL_IB_HCA="${NCCL_IB_HCA:-${_arch_rails}}"
    export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_arch_rails}}"
fi
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
            # Fingerprint the weights, do not just confirm a directory exists.
            # This path can resolve to DIFFERENT models on different nodes: on this
            # cluster Kimi-K3 is a real directory on some nodes and a symlink to
            # Kimi-K3-MXFP4 on others. Every rank loads from one MODEL_PATH, so a
            # split allocation would run two quantizations in one job and produce
            # wrong numbers rather than an error.
            #
            # config.json is the discriminator: it is small, always present in a
            # HuggingFace layout, and carries quantization_config, so variants of
            # one model differ in it. readlink is reported too, purely so the
            # message can say WHY they differ.
            _fp=nocfg
            [ -f \"\$_p/config.json\" ] && _fp=\"\$(md5sum \"\$_p/config.json\" 2>/dev/null | cut -c1-12)\"
            _rl=\"\$(readlink -f \"\$_p\" 2>/dev/null || echo \"\$_p\")\"
            echo \"\$(hostname): + Found \$_p fp=\$_fp real=\$_rl\"
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
        # Present everywhere is not sufficient: it must be the SAME weights
        # everywhere. Differing fingerprints reject this candidate rather than
        # failing the run, so the walk falls through to shared NFS -- one copy
        # every node reads, therefore uniform by construction. Slower, correct.
        local n_fp
        n_fp="$(printf '%s\n' "${out}" | grep -o 'fp=[^ ]*' | sort -u | grep -vc '^fp=nocfg$')"
        if [ "${n_fp:-0}" -gt 1 ] 2>/dev/null; then
            echo "x ${label} holds DIFFERENT weights on different nodes:"
            printf '%s\n' "${out}" | sed 's/^/      /'
            echo "  Every rank loads one MODEL_PATH, so this would run two variants in"
            echo "  one job and report numbers rather than an error. Rejecting this"
            echo "  candidate; falling through to the next."
            return 1
        fi
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
    # NVMe only: drop every candidate under the shared mount. Done by filtering
    # rather than by hardcoding one path, so an overridden MODEL_DIR_CANDIDATES
    # with several local roots still works.
    if [ "${REQUIRE_LOCAL_WEIGHTS}" = "1" ]; then
        local _keep="" _c
        for _c in ${candidates}; do
            case "${_c}" in
                "${SHARED_MOUNT%/}"/*) : ;;
                *) _keep="${_keep} ${_c}" ;;
            esac
        done
        candidates="${_keep# }"
        echo "REQUIRE_LOCAL_WEIGHTS=1: ${MODEL_NAME} is too large to load over shared"
        echo "storage in the time this partition allows; considering local roots only."
    fi
    # Append an explicit MODEL_DIR unless the list already covers it.
    if [ -n "${MODEL_DIR:-}" ]; then
        case " ${candidates} " in
            *" ${MODEL_DIR%/} "*) : ;;
            *) candidates="${candidates} ${MODEL_DIR%/}" ;;
        esac
    fi

    local dir
    for dir in ${candidates}; do
        if cluster_check_model_path "${dir%/}/${MODEL_WEIGHTS_NAME}" "${dir%/}"; then
            MODEL_PATH="${dir%/}/${MODEL_WEIGHTS_NAME}"
            MODEL_DIR="${dir%/}"
            export MODEL_PATH MODEL_DIR
            echo ""
            echo "+ Selected MODEL_PATH: ${MODEL_PATH} (available on all nodes)"
            return 0
        fi
    done

    echo ""
    echo "x FATAL: model '${MODEL_NAME}' is not usable on ALL allocated nodes in any of:"
    for dir in ${candidates}; do
        echo "  - ${dir%/}/${MODEL_WEIGHTS_NAME}"
    done
    echo ""
    echo "Every rank loads from the same path, so it must exist, be non-empty, and be"
    echo "the SAME weights on every node."
    if [ "${REQUIRE_LOCAL_WEIGHTS}" = "1" ]; then
        echo ""
        echo "This card sets REQUIRE_LOCAL_WEIGHTS=1, so shared storage was deliberately"
        echo "not considered: at this model's size an NFS read would not finish inside the"
        echo "partition time limit. Failing now rather than burning the allocation."
        echo ""
        echo "To fix: stage ${MODEL_NAME} on local NVMe on the nodes this job can draw,"
        echo "with the path resolving to the same weights on each. The per-node lines"
        echo "above say which nodes are missing it and which hold a different variant."
    fi
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

# --------------------------------------------------------------- GPU architecture
#
# A recipe is built for one GPU architecture. The Kimi-K3 disagg recipes on MI300X
# requantize the MoE to int4 because gfx942 has no scaled-MXFP4 MFMA and turn off
# AITER MLA because that kernel is gfx950-only; the MI355X recipes do the opposite.
# Run on the other GPU, neither fails cleanly: they take the allocation, load
# ~1.5 TB of weights and die in kernel codegen or return wrong numbers.
#
# The model card's skip_gpu_arch says the same thing to an orchestrator, but only
# the orchestrator reads it, and on SLURM it runs on a login node with no GPU to
# compare against. So the launcher checks for itself, on the nodes it was given.
# Because it happens here, it is identical whether madengine or plain sbatch submitted the job.

# _cluster_local_gpu_arch
#
# Echoes this node's gfx name, or nothing. Read from the KFD topology the kernel
# driver publishes, so it works on a host with no ROCm userspace installed; the
# container has rocminfo, the bare node often does not. gfx_target_version encodes
# major*10000 + minor*100 + stepping, and the gfx name spells minor and stepping in
# hex: 90402 -> gfx942, 90500 -> gfx950, 90010 -> gfx90a. CPU agents report 0.
_cluster_local_gpu_arch() {
    local _p _v _arch=""
    for _p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$_p" ] || continue
        _v="$(awk '$1=="gfx_target_version"{print $2}' "$_p" 2>/dev/null)"
        [ -n "$_v" ] && [ "$_v" != "0" ] || continue
        _arch="$(printf 'gfx%d%x%x' $((_v / 10000)) $(((_v / 100) % 100)) $((_v % 100)))"
        break
    done
    if [ -z "$_arch" ] && command -v rocminfo >/dev/null 2>&1; then
        _arch="$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]\+' || true)"
    fi
    echo "$_arch"
}

# cluster_detect_gpu_arch
#
# Sets and exports MAD_GPU_ARCH to the architecture of EVERY allocated node, or
# leaves it empty when that cannot be established. An explicit MAD_GPU_ARCH wins
# and skips the probe. Returns 1 only for a MIXED allocation: ranks of one job on
# two architectures cannot run one recipe, and that is never the scheduler's
# intent, so it is reported rather than resolved.
cluster_detect_gpu_arch() {
    if [ -n "${MAD_GPU_ARCH:-}" ]; then
        echo "GPU arch: ${MAD_GPU_ARCH} (set explicitly, not probed)"
        export MAD_GPU_ARCH
        return 0
    fi
    local out archs n
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
        n="${SLURM_NNODES:-1}"
        # Same shape as cluster_check_model_path: the body always exits 0, so a node
        # that cannot answer shows up as a missing line, not as a failed step in sacct.
        out="$(srun --nodes="${n}" --ntasks="${n}" /bin/bash -c \
            "$(declare -f _cluster_local_gpu_arch); echo \"\$(hostname) \$(_cluster_local_gpu_arch)\"; exit 0" \
            2>/dev/null || true)"
    else
        out="$(hostname) $(_cluster_local_gpu_arch)"
    fi
    [ -n "${out}" ] && printf '%s\n' "${out}" | sed 's/^/  gpu arch: /'
    # Sourced under `set -euo pipefail` by run_multinode.slurm: a probe that finds
    # nothing must fall through to the warning below, not end the job.
    archs="$(printf '%s\n' "${out}" | awk 'NF>=2{print $2}' | sort -u || true)"
    MAD_GPU_ARCH=""
    case "$(printf '%s\n' "${archs}" | grep -c . || true)" in
        0)  echo "GPU arch: could not be determined on these nodes" ;;
        1)  MAD_GPU_ARCH="${archs}"
            echo "GPU arch: ${MAD_GPU_ARCH} on all allocated nodes" ;;
        *)  echo "x The allocation mixes GPU architectures: $(echo ${archs})." >&2
            echo "  One job runs one recipe; constrain the allocation to one GPU type." >&2
            return 1 ;;
    esac
    export MAD_GPU_ARCH
    return 0
}

# cluster_require_gpu_arch <model> <allowed archs>
#
# Fails when the recipe for <model> declares the architectures it supports and the
# allocation is none of them. <allowed archs> is comma- or space-separated; empty
# means the recipe does not restrict, and nothing is checked. When the architecture
# cannot be determined the run continues with a warning: refusing a job on missing
# information would be worse than the failure this check exists to prevent.
# GPU_ARCH_CHECK=0 bypasses the check, for bringing up a recipe on a new GPU.
cluster_require_gpu_arch() {
    local model="$1" allowed="${2:-}"
    cluster_detect_gpu_arch || return 1
    # Label the results with the GPU the run actually used rather than a default.
    export PERF_GPU_ARCH="${PERF_GPU_ARCH:-${MAD_GPU_ARCH}}"
    [ -n "${allowed//[, ]/}" ] || return 0
    if [ "${GPU_ARCH_CHECK:-1}" = "0" ]; then
        echo "GPU_ARCH_CHECK=0: not enforcing ${model}'s supported archs (${allowed})"
        return 0
    fi
    if [ -z "${MAD_GPU_ARCH}" ]; then
        echo "WARN: ${model} supports only ${allowed}, and this allocation's GPU arch could" >&2
        echo "      not be determined, so that is NOT being enforced." >&2
        return 0
    fi
    local a
    for a in ${allowed//,/ }; do
        [ "${a}" = "${MAD_GPU_ARCH}" ] && { echo "+ ${model} supports ${MAD_GPU_ARCH}"; return 0; }
    done
    echo "x ${model} supports only: ${allowed}. These nodes are ${MAD_GPU_ARCH}." >&2
    echo "  The recipe is specific to its GPU (quantization path, attention kernels," >&2
    echo "  image build arch), so it would fail late or report wrong numbers here." >&2
    echo "  Run it on a partition with ${allowed}, or use the model's ${MAD_GPU_ARCH} recipe." >&2
    return 1
}

# cluster_report_rdma_rails
#
# Prints the rails this job will use and warns about any that is absent or not
# ACTIVE on this node. Evidence only -- it changes nothing. A rail list that
# names the wrong devices does not fail: NCCL drops them and runs over TCP. This
# line is what makes that visible in the job log instead of only in the numbers.
cluster_report_rdma_rails() {
    local rails="${NCCL_IB_HCA:-${MORI_RDMA_DEVICES:-}}" d st bad=""
    if [ -z "${rails}" ]; then
        echo "RDMA rails: connector default (fabric ${CLUSTER_ARCHETYPE:-unknown})"
        return 0
    fi
    for d in ${rails//,/ }; do
        d="${d%%:*}"
        st="$(cat "/sys/class/infiniband/${d}/ports/1/state" 2>/dev/null || true)"
        case "${st}" in *ACTIVE*) : ;; *) bad="${bad} ${d}(${st:-absent})" ;; esac
    done
    echo "RDMA rails: ${rails} (fabric ${CLUSTER_ARCHETYPE:-unknown})"
    [ -z "${bad}" ] || echo "WARN: rails not ACTIVE on $(hostname):${bad} -- traffic on them will fall back or fail" >&2
    return 0
}
