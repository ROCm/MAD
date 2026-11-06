#!/bin/bash
# Which processes take this node's GPU memory while the servers start, sampled from the
# host. Started by run_xPyD_models.slurm on each node next to the container.
#
# Why: jobs failed minutes in with "Free memory on device cuda:6 (147.57/191.98 GiB) on
# startup is less than desired" although _check_gpus_clean had just seen every GPU at
# 0 GiB, so the memory was taken after the check, by a process of this job or one that
# arrived meanwhile. The container cannot name it: the kernel's per-process GPU records
# (/sys/class/kfd/kfd/proc/<pid>) use host pids, which the container's pid namespace
# does not show. From the host, each holder's pid resolves to its command line and its
# container, and the per-GPU amounts show which GPU it holds memory on.
#
# Usage: gpu_mem_trace.sh <output file>
#   GPU_MEM_TRACE_SECS      how long to sample (default 900: past every server's startup)
#   GPU_MEM_TRACE_INTERVAL  seconds between samples (default 5)
# A sample is written only when it differs from the previous one, so the file stays small.
# Test roots: _KFD_ROOT (default /sys/class/kfd/kfd), _PROC_ROOT (default /proc),
# _GPU_SYSFS_ROOT (default /sys/class/drm).
out="$1"
kfd="${_KFD_ROOT:-/sys/class/kfd/kfd}"
proc="${_PROC_ROOT:-/proc}"
drm="${_GPU_SYSFS_ROOT:-/sys/class/drm}"
secs="${GPU_MEM_TRACE_SECS:-900}"
step="${GPU_MEM_TRACE_INTERVAL:-5}"

# KFD gpu_id -> "GPU<n> (<PCI address>)", n in the order the runtime numbers the GPUs
# (topology nodes with a gpu_id, CPUs have 0), which is the cuda:<n> of vLLM's messages.
declare -A gpu_name
gpu_ids=""
n=0
for node in $(ls "$kfd/topology/nodes" 2>/dev/null | sort -n); do
    id=$(cat "$kfd/topology/nodes/$node/gpu_id" 2>/dev/null)
    [ -n "$id" ] && [ "$id" != "0" ] || continue
    dom=$(sed -n 's/^domain //p' "$kfd/topology/nodes/$node/properties" 2>/dev/null)
    loc=$(sed -n 's/^location_id //p' "$kfd/topology/nodes/$node/properties" 2>/dev/null)
    pci=""
    [ -n "$loc" ] && pci=$(printf "%04x:%02x:%02x.%x" "${dom:-0}" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7)))
    gpu_name[$id]="GPU${n}${pci:+ ($pci)}"; gpu_ids="$gpu_ids $id"
    n=$((n + 1))
done

_sample() {
    local p f id gib cmd cont line d used=""
    # Each GPU's total, which also counts memory no KFD process record shows.
    for d in "$drm"/card*/device; do
        [ -r "$d/mem_info_vram_used" ] || continue
        used="${used} $(basename "$(readlink -f "$d")")=$(( $(cat "$d/mem_info_vram_used") >> 30 ))GiB"
    done
    echo "  used per GPU:${used:- (not readable)}"
    for p in $(ls "$kfd/proc" 2>/dev/null | sort -n); do
        line=""
        for f in "$kfd/proc/$p"/vram_*; do
            [ -r "$f" ] || continue
            gib=$(( $(cat "$f" 2>/dev/null || echo 0) >> 30 ))
            [ "$gib" -gt 0 ] || continue
            id=${f##*/vram_}
            line="${line} ${gpu_name[$id]:-gpu_id $id}=${gib}GiB"
        done
        [ -n "$line" ] || continue
        # Renamed processes (vLLM's "VLLM::Worker_TP3") pad their command line with spaces.
        cmd=$(tr '\0' ' ' < "$proc/$p/cmdline" 2>/dev/null | cut -c1-160 | sed 's/ *$//')
        cont=$(grep -oE '[0-9a-f]{64}' "$proc/$p/cgroup" 2>/dev/null | head -n 1 | cut -c1-12)
        echo "  pid $p [$([ -n "$cont" ] && echo "container $cont" || echo host)] ${cmd:-(exited)}:${line}"
    done
}

{
    echo "# GPU memory holders on $(hostname), sampled every ${step}s for ${secs}s; a sample is printed when it changes."
    for id in $gpu_ids; do echo "# kfd gpu_id $id = ${gpu_name[$id]}"; done
} > "$out"
prev=""
end=$(( $(date +%s) + secs ))
while [ "$(date +%s)" -lt "$end" ]; do
    cur="$(_sample)"
    if [ "$cur" != "$prev" ]; then
        { echo "=== $(date -u '+%F %T') UTC"; echo "$cur"; } >> "$out"
        prev="$cur"
    fi
    sleep "$step"
done
