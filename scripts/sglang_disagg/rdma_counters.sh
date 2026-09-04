#!/usr/bin/env bash
# Sample the RDMA adapters' operation counters into a CSV, one row per counter per sample.
#
# The expert all-to-all reaches no RCCL log and a trace names kernels without saying what they put
# on the wire, so the adapter is the only place that says how much crossed the fabric. Reading
# sysfs perturbs nothing, so this runs on a tuned job as well as a profiled one.
#
# Counts are per adapter and per node, never per rank or kernel, and include every other user of
# the NIC (here the mooncake KV transfer), so an absolute count is a ceiling; in an A/B that
# traffic is common-mode and largely cancels in the difference.
#
# Usage:
#   ./rdma_counters.sh --out <file.csv> [--interval 30]  # sample until killed
#   ./rdma_counters.sh --out <file.csv> --once           # one sample and exit
#   ./rdma_counters.sh --out <file.csv> --devices mlx5_0,mlx5_2   # only these adapters
#
# --devices: a node here has ten adapters and the run is given eight (IB_DEVICES); the other two
# carry other jobs' traffic. Default is every adapter, so no device is dropped silently.
#
# RDMA_SYSFS_ROOT overrides the sysfs root, so this can be exercised on a machine with no adapter.
#
# Size, measured here: 10 mlx5 adapters x 53 counters is ~530 rows and 28 kB a sample, so the
# default 30 s interval costs about 30 MB per node over an eight-hour job.
#
# Output columns: epoch_ns,device,port,counter,value
set -uo pipefail

OUT=""
INTERVAL=30
ONCE=0
DEVICES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        --devices) DEVICES="$2"; shift 2 ;;
        *) echo "rdma_counters.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT" ]]; then
    echo "rdma_counters.sh: --out is required" >&2
    exit 2
fi

# A zero interval makes the inner countdown run zero times, so the outer loop samples as fast as
# the shared filesystem allows; anything that is not a positive integer is refused here.
if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 1 )); then
    echo "rdma_counters.sh: --interval must be a positive whole number of seconds, got '${INTERVAL}'" >&2
    exit 2
fi

# A node with no RDMA device is not an error; the same launcher runs on such hosts. A header-only
# file records that the sampler ran and found nothing, which is not the same fact as not running.
mkdir -p "$(dirname "$OUT")"
if [[ "$ONCE" == "1" ]]; then
    # A single snapshot appends: two `--once` calls are how a caller makes a window.
    [[ -s "$OUT" ]] || echo "epoch_ns,device,port,counter,value" > "$OUT"
else
    # The sampling loop owns its file and truncates it: on a SLURM requeue the same path comes
    # back, and appending would splice two attempts into one window.
    echo "epoch_ns,device,port,counter,value" > "$OUT"
fi

SYSFS_ROOT="${RDMA_SYSFS_ROOT:-/sys/class/infiniband}"
if [[ ! -d "$SYSFS_ROOT" ]] || [[ -z "$(ls -A "$SYSFS_ROOT" 2>/dev/null)" ]]; then
    echo "rdma_counters.sh: no RDMA devices under ${SYSFS_ROOT}; nothing to sample" >&2
    exit 0
fi

# Devices present but no counters is the container case: `/sys/class/infiniband/<dev>` is a
# symlink into `/sys/devices/...`, and docker gives the class directory with the targets absent.
# Measured here: 10 devices visible, 0 counter files, until `/sys/devices` is bind-mounted
# read-only.
#
# Checked per *requested* device, not node-wide and not over the directories that happen to exist:
# an unused adapter exposing counters would mask a requested one that does not, and the totals
# would come out short while looking complete.
_missing=""
_present=""
if [[ -n "$DEVICES" ]]; then
    IFS=',' read -ra _wanted <<< "$DEVICES"
else
    _wanted=()
    for _dev in "$SYSFS_ROOT"/*; do
        [[ -d "$_dev" ]] && _wanted+=("$(basename "$_dev")")
    done
fi

for _name in "${_wanted[@]}"; do
    [[ -n "$_name" ]] || continue
    _dev="${SYSFS_ROOT}/${_name}"
    if [[ -d "$_dev" ]] \
       && { compgen -G "${_dev}/ports/*/counters" >/dev/null \
            || compgen -G "${_dev}/ports/*/hw_counters" >/dev/null; }; then
        _present="${_present}${_name} "
    else
        _missing="${_missing}${_name} "
    fi
done

if [[ -n "$_missing" ]]; then
    echo "rdma_counters.sh: no counters under: ${_missing% }" >&2
    echo "rdma_counters.sh: in a container the class entries are symlinks into /sys/devices;" >&2
    echo "rdma_counters.sh: add '-v /sys/devices:/sys/devices:ro' to the docker run options." >&2
fi

if [[ -z "$_present" ]]; then
    # Two outcomes, two exit codes. A host with no RDMA at all is supported (the empty DEVICES
    # default) and exits 0 with a header-only file; adapters that were *asked for* and cannot be
    # sampled are a failed measurement, since exit 0 there reads as a node that sent nothing.
    echo "rdma_counters.sh: none of the requested adapters exposes counters; nothing to sample" >&2
    [[ -n "$DEVICES" ]] && exit 3
    exit 0
fi

# A partial set is refused rather than sampled: totals over some of a run's adapters are a wrong
# number a reader cannot tell from a right one.
if [[ -n "$_missing" ]] && [[ -n "$DEVICES" ]]; then
    echo "rdma_counters.sh: refusing to sample a partial set of the requested adapters" >&2
    exit 3
fi

_sample() {
    local now dev port group file name value
    now="$(date +%s%N)"
    for dev in "$SYSFS_ROOT"/*; do
        [[ -d "$dev" ]] || continue
        if [[ -n "$DEVICES" ]] && [[ ",${DEVICES}," != *",$(basename "$dev"),"* ]]; then
            continue
        fi
        for port in "$dev"/ports/*; do
            [[ -d "$port" ]] || continue
            # `counters` are the portable port totals, `hw_counters` the vendor's own, where the
            # per-operation-type ones live under names mlx5 and bnxt_re spell differently -- so
            # nothing is hardcoded and the reader decides what is interesting.
            for group in counters hw_counters; do
                [[ -d "$port/$group" ]] || continue
                for file in "$port/$group"/*; do
                    [[ -f "$file" ]] || continue
                    # One unreadable counter (permissions, EINVAL for an unsupported one) must not
                    # end the sample for every other counter on the node.
                    value="$(cat "$file" 2>/dev/null)" || continue
                    [[ "$value" =~ ^[0-9]+$ ]] || continue
                    name="$(basename "$file")"
                    echo "${now},$(basename "$dev"),$(basename "$port"),${name},${value}"
                done
            done
        done
    done
}

if [[ "$ONCE" == "1" ]]; then
    _sample >> "$OUT"
    exit 0
fi

# SIGTERM is how the launcher stops this, and the last sample has to land before it does: the
# window ends when the servers stop, not one interval earlier.
_stop=0
trap '_stop=1' TERM INT
while [[ "$_stop" == "0" ]]; do
    _sample >> "$OUT"
    for ((_i = 0; _i < INTERVAL && _stop == 0; _i++)); do sleep 1; done
done
_sample >> "$OUT"
