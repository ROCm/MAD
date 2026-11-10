#!/bin/bash
# Offline check of vllm_disagg.sh's pre-start GPU check (_check_gpus_clean) against a
# fake /sys/class/drm: a node whose GPUs are idle starts, a node with one GPU holding
# memory fails before the barrier and names that GPU, and GPU_CLEAN_CHECK=0 skips it.
# Runs the real functions, cut out of the script. No GPUs needed.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has()    { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot() { grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }

_fake_node() { # <root> <used GiB per card...>: cards 1.. with PCI-looking targets; card0 is a display (no VRAM files)
    local root=$1 i=1 g; shift
    mkdir -p "$root/pci/0000:03:00.0" && ln -s "$root/pci/0000:03:00.0" "$root/card0" 2>/dev/null; mkdir -p "$root/card0/device"
    for g in "$@"; do
        local pci="$root/pci/0000:$(printf '%02x' $((i+16))):00.0"; mkdir -p "$pci"
        echo $(( g << 30 )) > "$pci/mem_info_vram_used"; echo $(( 192 << 30 )) > "$pci/mem_info_vram_total"
        mkdir -p "$root/card$i"; ln -s "$pci" "$root/card$i/device"; i=$((i+1))
    done
}
FUNCS="$(sed -n '/^_GPU_SYSFS_ROOT=/,/^_print_log_tail() {/p' "$DIR/vllm_disagg.sh" | sed '$d')"
_run() { # <root> [VAR=value...]: what the check prints, then "STARTED" if it let the node go on
    local root=$1; shift
    env -i PATH="$PATH" _GPU_SYSFS_ROOT="$root" host_name=node-x "$@" bash -c "
        _job_fail() { echo \"JOB_FAIL: \$1\"; exit 1; }
        $FUNCS
        _check_gpus_clean
        echo STARTED" 2>&1
}

echo "=== pre-start GPU check ==="
_fake_node "$TMP/clean" 0 1 0 0 1 0 0 0
O="$(_run "$TMP/clean")"
_has    "$O" "STARTED" "idle GPUs (driver reservations under the allowance): the node starts"
_hasnot "$O" "JOB_FAIL" "idle GPUs: no failure"
_has    "$O" "[gpu-check] node-x: 8 GPUs idle before start (most used: 1 GiB, allowance 4 GiB)" "a clean node says the check ran and what it saw"

_fake_node "$TMP/dirty" 0 0 0 0 0 40 0 0
O="$(_run "$TMP/dirty")"
_hasnot "$O" "STARTED" "a GPU holding 40 GiB: the node does not start"
_has    "$O" "JOB_FAIL: GPUs on node-x already hold memory before start (card6 (0000:16:00.0) 40 GiB; allowance 4 GiB)" \
        "the failure names the node, the GPU, its PCI address and the amount"
_has    "$O" "card6 (0000:16:00.0): 40 GiB used of 192 GiB" "the snapshot lists every GPU's memory"
_hasnot "$O" "card0" "a display device without VRAM files is ignored"

O="$(_run "$TMP/dirty" GPU_CLEAN_CHECK=0)"
_has "$O" "STARTED" "GPU_CLEAN_CHECK=0 skips the check"
O="$(_run "$TMP/dirty" GPU_CLEAN_MAX_USED_GIB=64)"
_has "$O" "STARTED" "GPU_CLEAN_MAX_USED_GIB raises the allowance"

_has "$(grep -n '_check_gpus_clean$' "$DIR/vllm_disagg.sh" | head -1)" "_check_gpus_clean" "the launcher calls the check"
_has "$(awk '/_check_gpus_clean$/{c=NR} /socket_barrier.py \\/{if(!b)b=NR} END{print (c && b && c<b) ? "before" : "after"}' "$DIR/vllm_disagg.sh")" "before" \
     "the check runs before the container barrier (so peers see the abort file)"

echo "======================================================"
echo "  gpu_clean_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
