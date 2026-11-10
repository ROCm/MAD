#!/bin/bash
# Offline check of the GPU-fault capture in sglang_disagg_mori_io_ep.sh: the same watcher
# as vllm_dissag/vllm_disagg.sh (kept identical), started after every server launch. Runs
# the launcher's own functions with a stub dmesg: a faulted server's log produces
# gpu_fault_NODE<n>.log with the kernel's amdgpu lines, and the launcher's output pipe
# still closes when it exits (the watcher must not hold it). No GPUs.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has()    { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
L="$DIR/sglang_disagg_mori_io_ep.sh"; V="$DIR/../vllm_dissag/vllm_disagg.sh"
_watcher() { sed -n '/^# A server that dies mid-run of a GPU fault/,/^}$/{/_watch_gpu_faults() {/,/^}$/!{/^_GPU_FAULT_WATCHERS=\|^_GPU_FAULT_RE=\|^trap /p};/_watch_gpu_faults() {/,/^}$/p}' "$1"; }

echo "=== GPU fault capture (sglang) ==="
_has "$(diff <(_watcher "$L") <(_watcher "$V") >/dev/null && echo same || echo differs)" "same" "the watcher is the vLLM launcher's, unchanged"
_has "$(grep -c '^    _watch_gpu_faults prefill$' "$L")/$(grep -c '^    _watch_gpu_faults decode$' "$L")" "2/1" "every server launch (2 prefill paths, decode) starts a watcher"

mkdir -p "$TMP/bin" "$TMP/logs/7"
printf '#!/bin/sh\necho "[t] amdgpu 0000:75:00.0: amdgpu: [mmhub0] no-retry page fault (src_id:0 ring:128 vmid:3)"\necho "[t] usb 1-1: new device"\n' > "$TMP/bin/dmesg"; chmod +x "$TMP/bin/dmesg"
FUNCS="$(sed -n '/^_print_gpu_snapshot() {/,/^}$/p' "$L"; _watcher "$L")"
printf '%s\n' 'Memory access fault by GPU node-3 (Agent handle: 0x5c) on address 0x7134. Reason: Unknown.' 'Fatal Python error: Aborted' \
    > "$TMP/logs/7/prefill_NODE0.log"
O="$(env -i PATH="$TMP/bin:$PATH" _RUN_LOGS="$TMP/logs" SLURM_JOB_ID=7 NODE_RANK=0 bash -c "$FUNCS
_watch_gpu_faults prefill; sleep 9; echo script-done" 2>&1 | timeout 30 cat)"
C="$(cat "$TMP/logs/7/gpu_fault_NODE0.log" 2>/dev/null)"
_has "$C" "Memory access fault by GPU node-3" "the capture quotes the runtime's fault line"
_has "$C" "no-retry page fault" "the capture has the kernel's amdgpu record"
_has "$O" "script-done" "the launcher's output arrives and its pipe closes"
_has "$(grep -c 'GPU fault' "$TMP/logs/7/proxy_NODE0.log" 2>/dev/null)" "1" "the notice goes to the node's proxy log, not the output pipe"

echo "======================================================"
echo "  fault_capture_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
