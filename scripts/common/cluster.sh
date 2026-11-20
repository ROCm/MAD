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
# node, exclusivity -- is the submitter's: madengine's SLURM presets (amd-rccl, 8 GPUs per
# node, exclusive) or the sbatch options you pass. The partition here is only a default for
# running a launcher by hand outside sbatch (see the slurm section below).
# WHAT DOES NOT: model-specific performance flags. Those live in models.yaml,
# keyed by model name, and are the launcher's business rather than the site's.
#
# Sourced under two different shells: run_multinode.slurm sets
# `set -euo pipefail`, the two disagg launchers set no flags at all. So nothing
# here may rely on `set -e` to catch an error, and every expansion must carry a
# default -- an unset name under `set -u` is a fatal error, not an empty string.
# =============================================================================

# --------------------------------------------------------------- site profile
# MAD_CLUSTER (compat/sites) or MAD_CLUSTER_FILE (a site's own file of that shape):
# the site's platform, fabric family and defaults, set here only where the run passed nothing.
if [ -n "${MAD_CLUSTER:-}${MAD_CLUSTER_FILE:-}" ]; then
    if _cluster_out="$(python3 "$(dirname "${BASH_SOURCE[0]}")/platform/platform.py" cluster 2>&1)"; then
        eval "${_cluster_out}"
    else
        echo "ERROR: site profile: ${_cluster_out}" >&2
        _CLUSTER_PLATFORM_ERROR="site profile: ${_cluster_out}"
    fi
fi

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

# How the launchers reach their nodes: srun under one allocation, or one job-array
# task per node (dispatch/dispatch.sh).
# shellcheck source=dispatch/dispatch.sh
. "$(dirname "${BASH_SOURCE[0]}")/dispatch/dispatch.sh"

# One hostname per line from a SLURM nodelist, compressed or not.
# Stock SLURM gives the compressed form (node[01-03,05]), which only `scontrol
# show hostnames` expands. Spur, a SLURM-compatible scheduler, gives the list
# already expanded (nodeA,nodeB) and has no `scontrol show hostnames` at all, so
# calling it there returns nothing and every node list built from it is empty.
# A list with no brackets needs no expanding; only the compressed form goes to
# scontrol.
cluster_expand_nodelist() {
    local nl="${1:-}"
    [ -n "$nl" ] || return 0
    if [[ "$nl" != *"["* ]]; then
        tr ',' '\n' <<<"$nl" | sed '/^$/d'
    else
        scontrol show hostnames "$nl"
    fi
}
# A container-creation barrier port for this job: 21000 + job id mod 8000. One per job,
# so jobs sharing a node do not meet on one port, and no launcher has to free a fixed
# port first (`fuser -k` killed whatever else listened there). The range avoids every
# fixed port the launchers and engines use (2222 .. 20005, 29000 and up).
cluster_barrier_port() {
    echo $(( 21000 + ${SLURM_JOB_ID:-0} % 8000 ))
}
# How many rows of a perf.csv are FAILURE (a stalled cell, lost requests, zero
# throughput, an errored NIAH length -- see the parsers). The launchers fail the
# job on any, so a standalone sbatch run reports what madengine reports for the
# same rows: a run whose server died mid-sweep wrote seven FAILURE rows, its
# containers exited 0, and the job ended COMPLETED.
cluster_perf_failures() {
    python3 - "$1" 2>/dev/null <<'PY' || grep -c ',FAILURE,' "$1" 2>/dev/null
import csv, sys
rows = csv.DictReader(open(sys.argv[1], newline=""))
print(sum(1 for r in rows if (r.get("status") or "").strip().upper() == "FAILURE"))
PY
}
# The job's verdict, the same in every launcher:
#   cluster_job_verdict <srun_rc> <perf_src> <perf_dst> [require_csv=1]
# Publishes perf_src at perf_dst first -- madengine resolves a card's
# multiple_results against the directory it launched the job from, and it collects
# the cells that passed even from a job that failed -- then fails (returns
# non-zero) when a node's container exited non-zero, when there are no results
# (unless require_csv=0, a run that serves without benchmarking), or when any row
# is FAILURE. A launcher ends with `cluster_job_verdict ... || exit $?`.
cluster_job_verdict() {
    local rc="$1" src="$2" dst="$3" require="${4:-1}" logs failed
    logs="$(dirname "$src")"
    if [ -f "$src" ]; then
        cp "$src" "$dst" && echo "perf CSV -> ${dst} ($(( $(wc -l < "$dst") - 1 )) rows)"
    fi
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: a node's container exited with status ${rc}; see the per-node logs under ${logs}." >&2
        return "$rc"
    fi
    if [ ! -f "$src" ]; then
        [ "$require" = 1 ] || return 0
        echo "ERROR: no perf CSV at ${src}; the benchmark produced no parseable results." >&2
        return 1
    fi
    # A header and no rows is no result either: a collector that found no metrics
    # still writes the header, and madengine would record "0 successful, 0 failed".
    if [ "$require" = 1 ] && [ "$(awk 'NR > 1 && NF' "$src" | wc -l)" -eq 0 ]; then
        echo "ERROR: ${src} has no result rows; the benchmark produced no parseable results." >&2
        return 1
    fi
    failed="$(cluster_perf_failures "$src" || true)"
    if [ "${failed:-0}" -gt 0 ] 2>/dev/null; then
        echo "ERROR: ${failed} row(s) of ${src} are FAILURE; see the per-node logs under ${logs}." >&2
        return 1
    fi
    return 0
}
# --------------------------------------------------------------- topology
# GPUs per node: what the run passed, else its platform's (MAD_PLATFORM, else the catalog's
# default_platform -- compat/hardware.yaml `platforms`). A named platform is also verified against
# the nodes by cluster_require_gpu_arch (accelerator, GPU count).
if _platform_out="$(python3 "$(dirname "${BASH_SOURCE[0]}")/platform/platform.py" shell 2>&1)"; then
    eval "${_platform_out}"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-${_platform_gpus}}"
else
    echo "ERROR: platform: ${_platform_out}" >&2
    _CLUSTER_PLATFORM_ERROR="${_platform_out}"
fi
[ -n "${GPUS_PER_NODE:-}" ] || { echo "ERROR: GPUS_PER_NODE is unknown (no platform catalog); set it" >&2; _CLUSTER_PLATFORM_ERROR="GPUS_PER_NODE unknown"; }
export GPUS_PER_NODE
export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export xP="${xP:-1}"
export yD="${yD:-1}"
export TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"
# No PP_SIZE default here. Only the colocated launcher reads it, and its default is
# the node count; exporting 1 from here made that default unreachable, so a plain
# 2-node sbatch failed the TP*PP == GPUs check (8x1 != 16).

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
# The families (cx7, ainic, thor2 today) and their values are in compat/hardware.yaml `fabrics`.
#
# Where the node lists its RDMA devices and network interfaces. Parameters only so
# the offline tests can point detection at a fake tree.
export CLUSTER_SYSFS_IB="${CLUSTER_SYSFS_IB:-/sys/class/infiniband}"
export CLUSTER_SYSFS_NET="${CLUSTER_SYSFS_NET:-/sys/class/net}"

# The archetype's usual control interface if this node has it, else the
# interface of the IPv4 default route, else the usual name unchanged.
cluster_socket_ifname() {
    local want="$1" dflt
    if [ -e "${CLUSTER_SYSFS_NET}/${want}" ]; then echo "$want"; return; fi
    dflt="$(ip -4 route show default 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit }}')"
    echo "${dflt:-$want}"
}

# The family and its fallback values are data: compat/hardware.yaml `fabrics`, settled for this node
# by scripts/common/platform/fabric.py (detected from the RDMA devices and drivers unless
# CLUSTER_ARCHETYPE names the family). A new NIC family is a catalog entry, not a code branch.
_arch_gid=3; _arch_iface=eth0; _arch_drivers=mlx5; _arch_kv_nic=mlx5_1; _arch_rails=""; _arch_env=()
_arch_rails_export=""; _arch_tc=""; _arch_sl=""
if _fabric_out="$(python3 "$(dirname "${BASH_SOURCE[0]}")/platform/fabric.py" shell 2>&1)"; then
    eval "${_fabric_out}"
else
    echo "WARN: the fabric catalog could not be read (${_fabric_out}); using the cx7 fallback values" >&2
    CLUSTER_ARCHETYPE="${CLUSTER_ARCHETYPE:-unknown}"
fi
export CLUSTER_ARCHETYPE
for _kv in "${_arch_env[@]}"; do                 # the family's own settings, unless the run set them
    _kn="${_kv%%=*}"; [ -n "${!_kn:-}" ] || export "${_kv}"
done
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${_arch_gid}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(cluster_socket_ifname "${_arch_iface}")}"
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
# CLUSTER_FABRIC_DERIVED=1: this family's rails and settings are derived here (the catalog names
# its rails), so a launcher forwards them into the containers, whose defaults are one CX7 site's.
_fabric_derived_given="${CLUSTER_FABRIC_DERIVED:-}"
export CLUSTER_FABRIC_DERIVED=0
if [ -n "${_arch_rails_export}" ]; then
    CLUSTER_FABRIC_DERIVED=1
    export NCCL_IB_HCA="${NCCL_IB_HCA:-${_arch_rails_export}}"
    export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_arch_rails_export}}"
fi
[ -n "${_fabric_derived_given}" ] && CLUSTER_FABRIC_DERIVED="${_fabric_derived_given}"   # config wins
# The family's values for the containers, whose own defaults they replace (the connectors fall
# back to these, not to one site's names): forwarded by name (node_lib.sh node_mad_env_args).
# A value passed through config (submit env, mad-config, a site profile) wins over the family's.
export MAD_FABRIC_RAILS="${MAD_FABRIC_RAILS:-${_arch_rails}}" MAD_FABRIC_GID="${MAD_FABRIC_GID:-${_arch_gid}}" \
       MAD_FABRIC_IFACE="${MAD_FABRIC_IFACE:-${_arch_iface}}" MAD_FABRIC_KV_NIC="${MAD_FABRIC_KV_NIC:-${_arch_kv_nic}}" \
       MAD_FABRIC_RDMA_TC="${MAD_FABRIC_RDMA_TC:-${_arch_tc}}" MAD_FABRIC_RDMA_SL="${MAD_FABRIC_RDMA_SL:-${_arch_sl}}"
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
    out="$(dispatch_gather "model-${model_label}-${label//[^A-Za-z0-9]/_}" /bin/bash -c "
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

# GPUs on this node: the KFD topology nodes with a GPU target (each logical GPU in a
# partition mode counts). Empty when it cannot tell.
_cluster_local_gpu_count() {
    local _p _v _n=0
    for _p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$_p" ] || continue
        _v="$(awk '$1=="gfx_target_version"{print $2}' "$_p" 2>/dev/null)"
        [ -n "$_v" ] && [ "$_v" != "0" ] && _n=$((_n + 1))
    done
    [ "$_n" -gt 0 ] && echo "$_n"
    return 0
}

# cluster_detect_gpu_arch
#
# Sets and exports MAD_GPU_ARCH to the architecture of EVERY allocated node, or
# leaves it empty when that cannot be established. An explicit MAD_GPU_ARCH wins
# and skips the probe. Returns 1 only for a MIXED allocation: ranks of one job on
# two architectures cannot run one recipe, and that is never the scheduler's
# intent, so it is reported rather than resolved.
cluster_detect_gpu_arch() {
    [ "${_CLUSTER_ARCH_PROBED:-0}" = 1 ] && return 0     # probed already (a recipe variant needed it)
    if [ -n "${MAD_GPU_ARCH:-}" ]; then
        echo "GPU arch: ${MAD_GPU_ARCH} (set explicitly, not probed)"
        export MAD_GPU_ARCH
        return 0
    fi
    local out archs n
    if [ -n "${SLURM_JOB_ID:-}" ] && { _dispatch_per_task || command -v srun >/dev/null 2>&1; }; then
        n="${SLURM_NNODES:-1}"
        # Same shape as cluster_check_model_path: the body always exits 0, so a node
        # that cannot answer shows up as a missing line, not as a failed step in sacct.
        out="$(dispatch_gather gpu-arch /bin/bash -c \
            "$(declare -f _cluster_local_gpu_arch _cluster_local_gpu_count); echo \"\$(hostname) \$(_cluster_local_gpu_arch) \$(_cluster_local_gpu_count)\"; exit 0" \
            2>/dev/null || true)"
    else
        out="$(hostname) $(_cluster_local_gpu_arch) $(_cluster_local_gpu_count)"
    fi
    # The fewest GPUs any node has (third column), for cluster_require_gpu_arch.
    [ -n "${MAD_GPU_COUNT:-}" ] || \
        MAD_GPU_COUNT="$(printf '%s\n' "${out}" | awk 'NF>=3 && $3>0 {print $3}' | sort -n | head -1 || true)"
    export MAD_GPU_COUNT
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
    _CLUSTER_ARCH_PROBED=1
    return 0
}

# cluster_gpu_arch_for_recipe MODELS_YAML MODEL: a recipe written as per-accelerator variants
# (models.yaml `variants:`, not an alias that pins one) runs the variant of the GPU it lands on,
# so the GPU is found before the recipe is applied. Any other recipe: nothing is probed here.
cluster_gpu_arch_for_recipe() {
    local yaml="$1" model="$2" lib
    lib="${MAD_SCRIPTS_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}/common/recipe/models.py"
    [ -f "$lib" ] && [ -f "$yaml" ] || return 0
    python3 "$lib" "$yaml" "$model" --picks-variant 2>/dev/null || return 0
    cluster_detect_gpu_arch
}

# cluster_require_gpu_arch <model> <allowed archs>
#
# Fails when the recipe for <model> declares the architectures it supports and the
# allocation is none of them. <allowed archs> is comma- or space-separated; empty
# means the recipe does not restrict, and nothing is checked. When the architecture
# cannot be determined the run continues with a warning: refusing a job on missing
# information would be worse than the failure this check exists to prevent.
# GPU_ARCH_CHECK=0 bypasses the check, for bringing up a recipe on a new GPU.
# cluster_check_recipe MODELS_YAML MODEL ENGINE [ROLES [FLAGS]]: resolve the recipe's declared serving roles
# (models.yaml `roles:`) against this allocation's GPU and the image's engine build -- named by
# MAD_ENGINE_BUILD (ENGINE@IMAGE), else by MAD_IMAGE_COMPONENTS (the image's builds, which madengine
# and the pipeline pass for an image built from a spec) -- using scripts/common/recipe/recipe.py with the
# capability catalog in compat/ (scripts/common/catalog.py). A recipe that declares no roles passes. A refusal
# names what the hardware or build lacks. FLAGS: the engine flags of roles the recipe holds no
# flags for (colocated serving: the card's COLOCATED_EXTRA_ARGS). RECIPE_CHECK=0 skips it. The resolution is recorded
# in config.resolved.json as MAD_RECIPE_ROLES.
cluster_check_recipe() {
    local yaml="$1" model="$2" engine="$3" roles="${4:-}" flags="${5-}" out rc exports
    local tool="${MAD_SCRIPTS_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}/common/recipe/recipe.py"
    local catalog="${MAD_CATALOG:-$(dirname "$tool")/../../../compat}"   # compat/ (or one catalog file)
    if [ "${RECIPE_CHECK:-1}" = "0" ]; then
        echo "RECIPE_CHECK=0: not resolving ${model}'s roles (nor adding the engine arguments or"
        echo "  per-role artifacts they would give)"; return 0
    fi
    if [ -z "${MAD_GPU_ARCH:-}" ]; then
        echo "WARN: GPU arch unknown: ${model}'s roles are NOT resolved (no added engine arguments)" >&2
        return 0
    fi
    if [ ! -f "$tool" ] || [ ! -e "$catalog" ]; then
        echo "WARN: no capability catalog (${catalog}); ${model}'s roles are NOT being checked" >&2
        return 0
    fi
    exports="$(mktemp)"
    out="$(python3 "$tool" check --catalog "$catalog" --recipes "$yaml" --model "$model" --exports "$exports" \
           --arch "$MAD_GPU_ARCH" ${MAD_ENGINE_BUILD:+--build "$MAD_ENGINE_BUILD"} \
           ${MAD_IMAGE_COMPONENTS:+--image-components "$MAD_IMAGE_COMPONENTS"} ${roles:+--roles "$roles"} \
           ${5+--flags "$flags"} --engine "$engine" 2>&1)"; rc=$?
    [ -n "$out" ] && printf '%s\n' "$out"
    # 2: no such recipe -- whether the model needs one is the launcher's check, not this one.
    [ "$rc" -eq 2 ] && { rm -f "$exports"; return 0; }
    if [ "$rc" -eq 0 ]; then
        # Each role's artifact (MAD_ROLE_ARTIFACT_<ROLE>, found on the nodes by
        # cluster_resolve_role_artifacts) and the resolved engine arguments its flags lack
        # (MAD_ROLE_ENGINE_ARGS_<ROLE>, which the role scripts append).
        # shellcheck source=/dev/null
        . "$exports"
        local v
        while read -r v; do
            [ -n "$v" ] && declare -F config_note >/dev/null && config_note "$v" "${!v}" recipe-check
        done < <(sed -n 's/.*export \([A-Z0-9_]*\)=.*/\1/p' "$exports")
        rm -f "$exports"
        out="$(sed -n 's/^\[recipe\] .* resolves on /resolves on /p' <<<"$out")"
        if [ -n "$out" ] && declare -F config_note >/dev/null; then config_note MAD_RECIPE_ROLES "$out" recipe-check; fi
        return 0
    fi
    rm -f "$exports"
    echo "x ${model}'s roles do not resolve on ${MAD_GPU_ARCH}${MAD_ENGINE_BUILD:+ with ${MAD_ENGINE_BUILD}}${MAD_IMAGE_COMPONENTS:+ with the builds of its image} (above)." >&2
    echo "  Use the model's recipe for this GPU, or RECIPE_CHECK=0 to run it anyway." >&2
    return 1
}

# cluster_resolve_role_artifacts: for every role whose recipe names its own artifact
# (MAD_ROLE_ARTIFACT_<ROLE>, from cluster_check_recipe), the path the role loads, found the way
# cluster_resolve_model_path finds the model (an absolute path is checked as given): exported
# as MAD_ROLE_MODEL_PATH_<ROLE> for the role scripts. A role without one serves MODEL_PATH.
cluster_resolve_role_artifacts() {
    local var role name path
    for var in $(compgen -v MAD_ROLE_ARTIFACT_); do
        role="${var#MAD_ROLE_ARTIFACT_}"; name="${!var}"
        [ -n "$name" ] || continue
        local given="MAD_ROLE_MODEL_PATH_${role}"
        if [ -n "${!given:-}" ]; then                  # a path passed through config wins
            echo "Role ${role,,}: ${!given} (set explicitly)"; continue
        fi
        echo "Role ${role,,}: the recipe names its own artifact, ${name}"
        if [[ "$name" == /* ]]; then
            cluster_check_model_path "$name" "$(dirname "$name")" || { echo "x FATAL: ${name} (${role,,}) is not usable on all nodes"; return 1; }
            path="$name"
        else
            # In a subshell: the probe sets MODEL_PATH / MODEL_DIR, which stay the model's.
            path="$(MODEL_PATH=""; MODEL_WEIGHTS_NAME="$name"; MODEL_NAME="$name"
                    cluster_resolve_model_path >&2 && printf %s "$MODEL_PATH")" || return 1
        fi
        export "MAD_ROLE_MODEL_PATH_${role}=${path}"
        declare -F config_note >/dev/null && config_note "MAD_ROLE_MODEL_PATH_${role}" "$path" recipe-check
    done
    return 0
}

# One GPU's memory in bytes on this node: the largest memory bank of the first GPU in the KFD
# topology (CLUSTER_SYSFS_KFD overrides the root, for tests). Empty when it cannot tell.
_cluster_local_gpu_memory() {
    local root="${CLUSTER_SYSFS_KFD:-/sys/class/kfd/kfd/topology/nodes}" n v b sz best=0
    for n in "$root"/*; do
        [ -r "$n/properties" ] || continue
        v="$(awk '$1=="gfx_target_version"{print $2}' "$n/properties" 2>/dev/null)"
        [ -n "$v" ] && [ "$v" != "0" ] || continue
        for b in "$n"/mem_banks/*/properties; do
            sz="$(awk '$1=="size_in_bytes"{print $2}' "$b" 2>/dev/null)"
            [ -n "$sz" ] && [ "$sz" -gt "$best" ] && best="$sz"
        done
        break
    done
    [ "$best" -gt 0 ] && echo "$best"
    return 0
}

# cluster_gpu_memory_for_recipe MODELS_YAML MODEL: when the recipe sizes memory from the GPU
# (models.yaml `memory:` fractions, scripts/common/recipe/models.py), export
# MAD_GPU_MEMORY_BYTES: as set, else one GPU's memory probed on every node, the smallest. A
# recipe that does not: nothing is probed. Refuses when the size cannot be found.
cluster_gpu_memory_for_recipe() {
    local yaml="$1" model="$2" lib out sizes
    lib="${MAD_SCRIPTS_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}/common/recipe/models.py"
    [ -f "$lib" ] && [ -f "$yaml" ] || return 0
    python3 "$lib" "$yaml" "$model" --needs-memory 2>/dev/null || return 0
    if [ -n "${MAD_GPU_MEMORY_BYTES:-}" ]; then
        echo "GPU memory: ${MAD_GPU_MEMORY_BYTES} bytes (set explicitly, not probed)"
        export MAD_GPU_MEMORY_BYTES
        return 0
    fi
    if [ -n "${SLURM_JOB_ID:-}" ] && { _dispatch_per_task || command -v srun >/dev/null 2>&1; }; then
        out="$(dispatch_gather gpu-memory /bin/bash -c \
            "CLUSTER_SYSFS_KFD='${CLUSTER_SYSFS_KFD:-}'; $(declare -f _cluster_local_gpu_memory); echo \"\$(hostname) \$(_cluster_local_gpu_memory)\"; exit 0" \
            2>/dev/null || true)"
    else
        out="$(hostname) $(_cluster_local_gpu_memory)"
    fi
    sizes="$(printf '%s\n' "${out}" | awk 'NF>=2 && $2>0 {print $2}' | sort -n || true)"
    if [ -z "$sizes" ]; then
        echo "x ${model}'s recipe sizes memory from the GPU, and this allocation's GPU memory could" >&2
        echo "  not be read. Set MAD_GPU_MEMORY_BYTES (one GPU, in bytes)." >&2
        return 1
    fi
    MAD_GPU_MEMORY_BYTES="$(printf '%s\n' "$sizes" | head -1)"
    export MAD_GPU_MEMORY_BYTES
    echo "GPU memory: ${MAD_GPU_MEMORY_BYTES} bytes per GPU (the smallest on the allocated nodes)"
    return 0
}

# The probes verify the platform rather than decide it: a platform the run named
# (MAD_PLATFORM) must have the accelerator the nodes report, and no node may have fewer GPUs
# than the run will use per node (GPUS_PER_NODE). Unknown facts are not refused.
cluster_verify_platform() {
    if [ -n "${_CLUSTER_PLATFORM_ERROR:-}" ]; then
        echo "x platform: ${_CLUSTER_PLATFORM_ERROR}" >&2
        return 1
    fi
    if [ -n "${MAD_PLATFORM_ACCELERATOR:-}" ] && [ -n "${MAD_GPU_ARCH:-}" ] \
       && [ "${MAD_PLATFORM_ACCELERATOR}" != "${MAD_GPU_ARCH}" ]; then
        echo "x MAD_PLATFORM=${MAD_PLATFORM} is ${MAD_PLATFORM_ACCELERATOR}, but these nodes are ${MAD_GPU_ARCH}." >&2
        return 1
    fi
    if [ -n "${MAD_GPU_COUNT:-}" ] && [ "${MAD_GPU_COUNT}" -lt "${GPUS_PER_NODE}" ]; then
        echo "x The run uses ${GPUS_PER_NODE} GPUs per node, but a node has ${MAD_GPU_COUNT}." >&2
        echo "  Set GPUS_PER_NODE, or MAD_PLATFORM to this node type (compat/hardware.yaml platforms)." >&2
        return 1
    fi
    [ -n "${MAD_PLATFORM:-}" ] && echo "+ platform ${MAD_PLATFORM}: ${MAD_GPU_ARCH:-arch unknown}, ${GPUS_PER_NODE} GPUs per node"
    return 0
}

cluster_require_gpu_arch() {
    local model="$1" allowed="${2:-}"
    cluster_detect_gpu_arch || return 1
    # Label the results with the GPU the run actually used rather than a default.
    export PERF_GPU_ARCH="${PERF_GPU_ARCH:-${MAD_GPU_ARCH}}"
    cluster_verify_platform || return 1
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

# A per-node task (spur-array, k8s-indexed) joins the others here, before any launcher reads the
# run's nodes: from now on SLURM_JOB_ID / SLURM_NNODES / SLURM_JOB_NODELIST describe
# the whole run. Nothing to do under one allocation.
if _dispatch_per_task; then
    _dispatch_array_init || { echo "ERROR: could not join the other tasks of this run" >&2; exit 1; }
fi
