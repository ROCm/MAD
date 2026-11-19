#!/bin/bash
# Which nodes of a SLURM cluster can actually run docker?
#
# madengine reaches docker through passwordless sudo, and two independent things
# go wrong per node: sssd/PAM breaks so sudo refuses ("PAM account management
# error"), or dockerd is simply not running ("dial unix /var/run/docker.sock").
# Both fail long before the model loads, and both look like a bug in the harness
# rather than a sick node, so probe before spending an allocation.
#
# A revoked allocation is NOT a verdict: it means the node was taken between the
# submit and the step, which says nothing about its health. Reported separately
# so a busy cluster does not quietly grow the exclude list.
#
# A scheduler refusal (wrong partition, unknown node name, accounting) is not a
# verdict either: the probe never reached the node. Reported as UNPROBED.
#
# Usage: probe_sudo.sh [-p PARTITION] <node> [node...]
#   Ruby (default partition meta64):
#        probe_sudo.sh $(sinfo -p meta64 -h -t idle,mixed -N -o "%N")
#   DCCS (gfx942):
#        probe_sudo.sh -p compute $(sinfo -p compute -h -t idle,mixed -N -o "%N")
set -u

PARTITION="${PROBE_PARTITION:-meta64}"
if [ "${1:-}" = "-p" ]; then PARTITION="$2"; shift 2; fi

ok=() bad=() unknown=() unprobed=()
for n in "$@"; do
    out=$(timeout 120 srun -p "$PARTITION" -w "$n" -N1 -t 00:02:00 --overlap \
            sudo -n /usr/bin/docker version --format '{{.Server.Version}}' 2>&1)
    rc=$?
    ver=$(printf '%s' "$out" | tail -1)
    # 124 is timeout(1): the node was busy and srun never got past the queue, so
    # docker was never reached. Calling that unhealthy would quietly grow the
    # exclude list every time the cluster is full.
    if [ "$rc" -eq 124 ]; then
        unknown+=("$n"); echo "BUSY $n  (probe timed out in the queue; not a health verdict)"
        continue
    fi
    case "$out" in
        *"has been revoked"*|*"Required node not available"*)
            unknown+=("$n"); echo "BUSY $n  (allocation revoked; not a health verdict)" ;;
        *"Unable to allocate"*|*"invalid partition"*|*"Invalid node name"*)
            unprobed+=("$n"); echo "UNPROBED $n  (scheduler refused, node never reached: ${ver:0:60})" ;;
        *)
            case "$ver" in
                [0-9]*) ok+=("$n");  echo "OK   $n  docker $ver" ;;
                *)      bad+=("$n"); echo "BAD  $n  ${ver:0:70}" ;;
            esac ;;
    esac
done

echo
echo "healthy  : ${#ok[@]}"
echo "unhealthy: ${#bad[@]}"
if [ ${#bad[@]} -gt 0 ]; then
    printf 'exclude_nodes: '
    (IFS=,; printf '%s\n' "${bad[*]}")
fi
[ ${#unknown[@]} -gt 0 ] && echo "inconclusive (re-probe when free): ${unknown[*]}"
[ ${#unprobed[@]} -gt 0 ] && echo "not probed (fix partition/node names; -p $PARTITION was used): ${unprobed[*]}"
exit 0
