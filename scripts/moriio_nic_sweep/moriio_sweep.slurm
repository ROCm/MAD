#!/bin/bash
#SBATCH --job-name=moriio-sweep
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output="%x-%j.out"
#SBATCH --error="%x-%j.err"
# ---------------------------------------------------------------------------
# Two-node MoRI-IO RDMA block-size sweep across all 8 AINIC rails.
#
# This is the batch wrapper for run_moriio_sweep.sh. It does one thing the
# script cannot do for itself: put the same workload-testing container on both
# nodes, with the same device passthrough and the same host RDMA vendor
# libraries that the vLLM disagg launcher uses, then start rank 0 and rank 1.
#
# WHY THE WORKLOAD IMAGE AND NOT rocm/pytorch
# -------------------------------------------
# The number we want is the bandwidth MoRI-IO actually delivers *in the software
# stack we ship*. The deploy-mori skill suggests a stock rocm/pytorch base, which
# would measure a different MoRI build against a different ROCm/libibverbs
# combination -- a real number about a stack nobody runs. The image below is the
# one models.yaml pins for GLM-5.2 disagg serving, so its libmori, its ROCm, and
# its torch are the ones the KV transfer will use in production. Same reason the
# env defaults in run_moriio_sweep.sh mirror the serving recipe
# (../vllm_dissag/connectors/moriio.env.aac) rather than MoRI's own defaults.
#
# CONTAINER FLAGS
# ---------------
# Mirrored from run_xPyD_models.slurm's proven docker invocation, because each
# one there was added to fix a specific failure:
#   --device /dev/infiniband   ibv_open_device() needs the uverbs char devices
#   --network host             RoCE OOB bootstrap uses host IPs; also lets the
#                              container read the real enp193s0f1np1 address
#   --ipc host                 large pinned host buffers
#   --privileged / memlock=-1  RDMA memory registration pins pages; the default
#                              64 KB memlock fails MR registration outright
#   $_RDMA_MOUNTS              the HOST's RDMA stack: sysfs, provider plugins,
#                              vendor + core libs. Built from the cookbook's
#                              cluster-sphere ionic recipe -- see the block
#                              itself for what each group fixes.
#   --group-add keep-groups    rootless podman drops supplementary groups
#
# WHAT IS DELIBERATELY NOT MIRRORED
# ---------------------------------
# No JIT cache mount, no cookbook mount, no model mounts, none of the ~90 vLLM
# `-e` vars. This job never imports vLLM. Adding them would only widen the
# surface on which the measurement could fail for reasons unrelated to RDMA.
#
# USAGE
#   export DOCKER_IMAGE_NAME=rocmshared/vllm-disagg:glm52-gfx950-ionic-v1
#   export MORI_SRC=$HOME/test_vllm_40549/mori-src            # FULL checkout
#   sbatch -w mi355-gpu-45,mi355-gpu-46 moriio_sweep.slurm
#
# Or, against an existing allocation, skip sbatch entirely and drive the two
# ranks with `srun --jobid=<id> --overlap` -- see run_on_job5642.sh.
#
# Optional:
#   FABRIC_SUBNET=10.2.80.   control-plane subnet (default; AAC mgmt NIC)
#   CHUNK_SWEEP=1            also run the fixed-block / varying-chunk experiment
#   BLOCK_MIN / BLOCK_MAX / ITERS / BATCH   forwarded to the script
#
# Results land in ./results/ next to this file (the submit dir, which is on
# $HOME and therefore shared): one log per rank. Rank 0 carries the table;
# rank 1's table is all zeros by construction -- see the note in the script.
# ---------------------------------------------------------------------------
set -u

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE" >&2; exit 1; }

if [[ ! -f "$HERE/run_moriio_sweep.sh" ]]; then
  echo "Error: run_moriio_sweep.sh not found next to this file ($HERE)." >&2
  exit 1
fi

# ---- image -----------------------------------------------------------------
# No default on purpose. The workload image tag encodes the mori/aiter versions
# under test; silently falling back to some other image would produce a number
# that is not about the stack we ship. Same contract as run_xPyD_models.slurm.
if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
  echo "Error: DOCKER_IMAGE_NAME is not set." >&2
  echo "  Use the image models.yaml pins for the workload under test, e.g." >&2
  echo "    export DOCKER_IMAGE_NAME=\$(grep -m1 -oE 'rocmshared/[^ ]+' \\" >&2
  echo "        \$HOME/TW_aac_mi355X_DSV4/MAD/scripts/vllm_dissag/models.yaml)" >&2
  exit 1
fi
export DOCKER_IMAGE_NAME

# ---- mori source -----------------------------------------------------------
# benchmark.py lives under tests/, which a SPARSE checkout will not materialise
# even though `git ls-files` lists it. Check for the file, not the directory.
# MORI_SRC=SKIP means the image already carries the harness at /opt/mori (built
# by the Dockerfile in this directory, pinned to the image's own MORI_REF). In
# that case there is nothing to mount and nothing to check on the host.
MORI_SRC="${MORI_SRC:-$HOME/tencent_mi308_wideEP/code/mori}"
if [[ "$MORI_SRC" == "SKIP" ]]; then
  echo "MORI_SRC=SKIP: harness expected inside the image at /opt/mori."
elif [[ ! -f "$MORI_SRC/tests/python/io/benchmark.py" ]]; then
  echo "Error: $MORI_SRC/tests/python/io/benchmark.py not found." >&2
  echo "  MORI_SRC must be a FULL mori checkout. If you cloned with" >&2
  echo "  core.sparseCheckout=true, tests/ is tracked but not on disk:" >&2
  echo "    git -C \$MORI_SRC config core.sparseCheckout   # -> true means this" >&2
  exit 1
fi
export MORI_SRC
MORI_MNT="/opt/mori"           # where it appears inside the container
export MORI_MNT
# Mounting SKIP over /opt/mori would hide the baked-in harness, so emit no -v.
if [[ "$MORI_SRC" == "SKIP" ]]; then
  _MORI_SRC_MOUNT=""
else
  _MORI_SRC_MOUNT="-v $MORI_SRC:$MORI_MNT"
fi

# ---- control plane ---------------------------------------------------------
# Same picker as run_xPyD_models.slurm: `hostname -I` lists several addresses and
# the first is not the routable one. On AAC the PUBLIC address answers ICMP but
# has TCP firewalled node-to-node, so "the one that pings" is the wrong test --
# pick by subnet. 10.2.80. is the mgmt NIC that actually carries TCP here.
FABRIC_SUBNET="${FABRIC_SUBNET:-10.2.80.}"
_pick_ip() { awk -v pfx="$FABRIC_SUBNET" '{f=$1; for(i=1;i<=NF;i++) if(index($i,pfx)==1){f=$i; break} print f}'; }

# Attach mode. With MORIIO_JOBID set this file runs as a plain script on the
# login node and borrows an allocation that already exists, instead of queueing
# a second one behind it. --overlap is required: without it srun waits for the
# job step resources the parent already holds and simply blocks forever.
SRUN_OPTS=""
if [[ -n "${MORIIO_JOBID:-}" ]]; then
  SRUN_OPTS="--jobid=${MORIIO_JOBID} --overlap"
  : "${SLURM_JOB_ID:=${MORIIO_JOBID}}"
  if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
    SLURM_JOB_NODELIST=$(scontrol show job "$MORIIO_JOBID" -o 2>/dev/null \
                         | tr ' ' '\n' | awk -F= '/^NodeList=/{print $2; exit}')
    [[ -z "$SLURM_JOB_NODELIST" || "$SLURM_JOB_NODELIST" == "(null)" ]] && {
      echo "Error: job $MORIIO_JOBID has no NodeList (not running?)." >&2; exit 1; }
  fi
  echo "[attach] borrowing job ${MORIIO_JOBID}: ${SLURM_JOB_NODELIST}"
fi
export SRUN_OPTS SLURM_JOB_ID

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=$(echo "$NODES" | wc -l)
if [[ "$NNODES" != "2" ]]; then
  echo "Error: this sweep is exactly 2 nodes (initiator + target); got $NNODES." >&2
  exit 1
fi
MASTER_NODE=$(echo "$NODES" | head -n1)
MASTER_ADDR=$(srun ${SRUN_OPTS} --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" bash -c 'hostname -I' | _pick_ip)
MASTER_PORT="${MASTER_PORT:-29511}"

if [[ -z "$MASTER_ADDR" ]]; then
  echo "Error: could not resolve a $FABRIC_SUBNET address on $MASTER_NODE." >&2
  exit 1
fi
if [[ "$MASTER_ADDR" != ${FABRIC_SUBNET}* ]]; then
  # Loud, not fatal: a site with a different mgmt subnet can still be right.
  echo "[warn] MASTER_ADDR=$MASTER_ADDR is not on FABRIC_SUBNET=$FABRIC_SUBNET."
  echo "[warn] If the torchrun rendezvous hangs, this is why -- set FABRIC_SUBNET."
fi
export MASTER_ADDR MASTER_PORT FABRIC_SUBNET

OUTDIR="${OUTDIR:-$HERE/results}"
mkdir -p "$OUTDIR"
export OUTDIR

echo "=============================================================="
echo " MoRI-IO 2-node block-size sweep"
echo "   job        : ${SLURM_JOB_ID:-local}"
echo "   nodes      : $(echo "$NODES" | paste -sd,)"
echo "   master     : ${MASTER_ADDR}:${MASTER_PORT}  (subnet ${FABRIC_SUBNET})"
echo "   image      : ${DOCKER_IMAGE_NAME}"
echo "   mori src   : ${MORI_SRC} -> ${MORI_MNT}"
echo "   results    : ${OUTDIR}"
echo "   chunk swp  : ${CHUNK_SWEEP:-0}"
echo "=============================================================="

# ---- container runtime -----------------------------------------------------
# AAC compute nodes deny the docker socket to normal users but ship podman,
# which is CLI-compatible for every flag used here. Decided on the SUBMIT host;
# if that disagrees with the compute nodes, override CONTAINER_CLI at submit.
if [ -z "${CONTAINER_CLI:-}" ]; then
  if docker info >/dev/null 2>&1; then CONTAINER_CLI=docker
  elif command -v podman >/dev/null 2>&1; then CONTAINER_CLI=podman
  else echo "Error: neither a usable docker daemon nor podman found." >&2; exit 1; fi
fi
export CONTAINER_CLI
echo "[runtime] CONTAINER_CLI=$CONTAINER_CLI"

export CONT_NAME="moriio_sweep_${SLURM_JOB_ID:-local}"
export HERE

# Knobs forwarded into the container. Only forward what is actually set, so the
# script's own documented defaults stay in charge of anything not overridden.
#
# Forwarded BY NAME (`-e VAR`, no `=value`) rather than `-e VAR=$VAR`. Both
# runtimes read the value from the client environment in that form, and it is
# the only quoting-safe option here: CHUNK_SWEEP_LIST is a space-separated list,
# and `-e CHUNK_SWEEP_LIST=16384 32768 ...` would word-split inside the
# unquoted ${SWEEP_ENV} expansion -- the tail of the list would be parsed as the
# image name. Each var is re-exported so it survives sbatch -> srun -> runtime.
export SWEEP_ENV=""
for _v in BLOCK_MIN BLOCK_MAX ITERS BATCH OP_TYPE MEM_TYPE NUM_DEV \
          QP_PER_TRANSFER WORKER_THREADS CHUNK_BYTES MAX_CHUNKS NUMA_NODE \
          IFNAME MORI_RDMA_DEVICES MORI_IB_GID_INDEX MORI_DEVICE_NIC \
          CHUNK_SWEEP CHUNK_SWEEP_BLOCK CHUNK_SWEEP_LIST BENCH_TIMEOUT; do
  if [ -n "${!_v:-}" ]; then export "${_v}"; SWEEP_ENV="$SWEEP_ENV -e ${_v}"; fi
done
[ -n "$SWEEP_ENV" ] && echo "[env] forwarding:${SWEEP_ENV//-e /}"

srun ${SRUN_OPTS} --nodelist="$(echo "$NODES" | paste -sd,)" --ntasks="$NNODES" bash -c '
set -u
RANK=$SLURM_PROCID
echo "[node] rank $RANK on $(hostname)"

$CONTAINER_CLI rm -f "$CONT_NAME" 2>/dev/null || true

# Pull is best-effort: on a node with the image already cached but no registry
# route, a hard failure here would kill a runnable job.
$CONTAINER_CLI pull "$DOCKER_IMAGE_NAME" 2>/dev/null || true

# --- host RDMA exposure: cluster-sphere ionic pattern ---------------------
# Follows dist-inf-cookbook/cluster-sphere/cluster-rdma-env-recommender
# (_docker_cmd_ionic) rather than the literal paths in that suites own
# rdma_perf_tests.slurm. The recommender DISCOVERS the vendor libraries by
# glob and emits a mount per hit; the slurm hard-codes
# libionic.so.1.0.54.0-149.g3304be71, and these nodes carry
# libionic.so.1.1.54.0-184. A hard-coded version string is a bind mount that
# fails container creation the day the driver is updated, so the glob form is
# the one worth copying.
#
# Four groups, each for a distinct failure:
#
#  1. SYSFS. /sys/class/infiniband, /sys/class/net, /sys/bus/pci. This is the
#     group the earlier hand-mirrored block was MISSING and it matters most
#     here. MoRIs NIC auto-detect matches device NAME prefixes (^mlx5,
#     ^bnxt_re, ^ionic); AAC names its devices rocep*s0, so the name match
#     fails and it falls through to readlink /sys/class/infiniband/<dev>/
#     device/driver. With sysfs masked that readlink returns nothing and the
#     detect silently DEFAULTS TO MLX5 -- a wrong backend, not an error.
#     MORI_DEVICE_NIC=ionic pins it anyway; this makes the pin agree with what
#     the process can actually see, and is what lets ibv_devinfo inside the
#     container report the same 8 rails the preflight checked outside it.
#  2. PROVIDER PLUGIN. libibverbs dlopens the vendor .so named by
#     /etc/libibverbs.d/<x>.driver. Both must be present: ionic.driver says
#     "driver ionic", and libibverbs/libionic-rdmav34.so is what that resolves
#     to. The image ships the mlx5 plugin; without these the container
#     enumerates ZERO devices.
#  3. VENDOR + CORE LIBS. libionic.so*, libibverbs, librdmacm, and the libnl
#     pair the recommender also checks for -- the providers link against
#     libnl-3/libnl-route-3 and a version skew there is a dlopen failure that
#     surfaces as "no devices" rather than as a link error.
#  4. /etc/rdma, for the same reason as /etc/libibverbs.d.
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu

# (1) sysfs -- read-only; the container must not be able to write these
for _d in /sys/class/infiniband /sys/class/net /sys/bus/pci; do
    [ -d "$_d" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_d:$_d:ro"
done

# (2) provider plugins + driver descriptors. Mount the whole libibverbs/ dir:
# a single -v per plugin would silently drop a provider added by a later
# driver package, and the dir has nothing in it that is not a provider.
[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"
[ -d /etc/rdma ]             && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/rdma:/etc/rdma:ro"

# (3) core + vendor + libnl, discovered not hard-coded.
# -f, not -e: a DANGLING symlink fails the bind mount and the container never
# starts (create exit 125), so resolve-then-test. .a archives are skipped --
# libionic.a is present here and is useless to a running process.
for _pattern in libibverbs.so libibverbs.so.1 libibverbs.so.1.* \
                librdmacm.so librdmacm.so.1 librdmacm.so.1.* \
                libionic.so libionic.so.[0-9]* \
                libmlx5.so* libbnxt_re*.so* libefa.so* libhns.so* \
                libnl-3.so.200* libnl-route-3.so.200*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        case "$_vlib" in *.a) continue ;; esac
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done
echo "[host-rdma] rank $RANK mounts:"
printf "%s\n" $_RDMA_MOUNTS | paste -d" " - - | sed "s/^/    /"

# Two independent things must be true, so check them independently. The user
# space library alone is not enough (nothing to open), and the plugin alone is
# not enough (nothing to open it with). Neither is a warning: either one
# missing means zero ibv devices, and a sweep over zero devices is not a
# slower number, it is no number at all.
case "$_RDMA_MOUNTS" in
  *"$_LIBDIR/libionic.so"*) : ;;
  *) echo "[host-rdma] FATAL: no libionic.so* on $(hostname)."
     echo "[host-rdma] Expected $_LIBDIR/libionic.so{,.1,.1.x.y}; this cluster is"
     echo "[host-rdma] Pensando AINIC and the image ships mlx5 only."
     exit 2 ;;
esac
if [ ! -f "$_LIBDIR/libibverbs/libionic-rdmav34.so" ] || [ ! -f /etc/libibverbs.d/ionic.driver ]; then
     echo "[host-rdma] FATAL: ionic PROVIDER PLUGIN missing on $(hostname)."
     echo "[host-rdma] Need both $_LIBDIR/libibverbs/libionic-rdmav34.so and"
     echo "[host-rdma] /etc/libibverbs.d/ionic.driver -- libibverbs dlopens the"
     echo "[host-rdma] former only because the latter names it. Without the pair"
     echo "[host-rdma] the container enumerates ZERO devices. Not running a fake sweep."
     exit 2
fi

if [ "${CONTAINER_CLI}" = "podman" ]; then
    _GROUP_ADD_ARGS="--group-add keep-groups"; _SHM_SIZE_ARG=""
else
    _GROUP_ADD_ARGS="--group-add video";       _SHM_SIZE_ARG="--shm-size ${DOCKER_SHM_SIZE:-64G}"
fi

# Site mounts: bind only what EXISTS. podman fails container creation on a
# missing bind source (docker would auto-create a dir).
_OPT_MOUNTS=""
for _d in /shared_inference /mnt/m2m_nobackup /shared/data /shared/apps /media/NVME /it-share; do
    [ -d "$_d" ] && _OPT_MOUNTS="$_OPT_MOUNTS -v $_d:$_d"
done

$CONTAINER_CLI run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    ${_GROUP_ADD_ARGS} \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    ${_SHM_SIZE_ARG} \
    --ulimit nofile=524288:524288 \
    --ulimit memlock=-1:-1 \
    -v "$HOME":"$HOME" \
    ${_OPT_MOUNTS} \
    ${_MORI_SRC_MOUNT} \
    -v "$OUTDIR":"$OUTDIR" \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e MORI_REPO="$MORI_MNT" \
    -e OUTDIR="$OUTDIR" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-local}" \
    ${SWEEP_ENV} \
    --name "${CONT_NAME}_r${RANK}" \
    "$DOCKER_IMAGE_NAME" -c "
        bash \"$HERE/run_moriio_sweep.sh\" \
            --rank $RANK \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT
    "
_rc=$?
echo "[node] rank $RANK container exit=$_rc"
exit $_rc
'
_srun_rc=$?

srun ${SRUN_OPTS} --nodelist="$(echo "$NODES" | paste -sd,)" --ntasks="$NNODES" bash -c \
  '$CONTAINER_CLI rm -f "${CONT_NAME}_r${SLURM_PROCID}" 2>/dev/null || true' >/dev/null 2>&1

echo "=============================================================="
echo " sweep exit=${_srun_rc}"
echo " logs: ${OUTDIR}"
ls -1 "$OUTDIR" 2>/dev/null | tail -10
echo "=============================================================="
exit "$_srun_rc"
