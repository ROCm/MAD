#!/bin/bash
# Offline checks of two diagnostics, using the real code cut out of the scripts:
#   - vllm_disagg.sh's _watch_gpu_faults: when a server log shows a GPU fault, the
#     kernel's amdgpu lines (dmesg) and the GPU snapshot land in gpu_fault_NODE<n>.log;
#     a healthy server writes nothing; every server role starts a watcher.
#   - run_xPyD_models.slurm's CONTAINER_ENV: each listed name becomes a bare `-e NAME`
#     on docker run, comma- or space-separated, values never inlined.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has()    { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot() { grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }

echo "=== GPU fault capture ==="
mkdir -p "$TMP/bin" "$TMP/logs/42" "$TMP/sys/card1/device"
printf '#!/bin/sh\necho "[Wed Oct  8 17:27:57 2026] amdgpu 0000:16:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:3 pasid:32772)"\necho "[Wed Oct  8 17:27:57 2026] amdgpu 0000:16:00.0: amdgpu:   Faulty UTCL2 client ID: TCP (0x8)"\necho "[Wed Oct  8 17:27:57 2026] eth0: link up"\n' > "$TMP/bin/dmesg"; chmod +x "$TMP/bin/dmesg"
echo $((150 << 30)) > "$TMP/sys/card1/device/mem_info_vram_used"; echo $((192 << 30)) > "$TMP/sys/card1/device/mem_info_vram_total"
FUNCS="$(sed -n '/^_GPU_SYSFS_ROOT=/,/^_print_log_tail() {/p' "$DIR/vllm_disagg.sh" | sed '$d')"
_watch() { # <role> <log content>: start the watcher, write the log, give it time; print the capture file
    printf '%s\n' "$2" > "$TMP/logs/42/${1}_NODE1.log"
    env -i PATH="$TMP/bin:$PATH" _RUN_LOGS="$TMP/logs" SLURM_JOB_ID=42 NODE_RANK=1 host_name=node-x \
        _GPU_SYSFS_ROOT="$TMP/sys" bash -c "$FUNCS
        _watch_gpu_faults $1; sleep 9; kill %1 2>/dev/null; true" >/dev/null 2>&1
    cat "$TMP/logs/42/gpu_fault_NODE1.log" 2>/dev/null; rm -f "$TMP/logs/42/gpu_fault_NODE1.log"
}
O="$(_watch decode 'INFO serving
:0:rocdevice.cpp :3586: Callback: Queue 0x7f aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: code: 0x29
(EngineCore_DP5 pid=1211) ERROR [multiproc_executor.py:284] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.')"
_has    "$O" "GPU fault in $TMP/logs/42/decode_NODE1.log" "a faulted server's capture names the log"
_has    "$O" "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" "the capture quotes the runtime's fault line"
_has    "$O" "Faulty UTCL2 client ID: TCP" "the capture has the kernel's amdgpu fault record (which engine faulted)"
_hasnot "$O" "eth0: link up" "unrelated kernel lines are left out"
_has    "$O" "card1" "the capture has the GPU memory snapshot"
O="$(_watch prefill 'INFO serving
INFO Application startup complete.')"
_has "${O:-<none>}" "<none>" "a healthy server writes no capture"
_has "$(grep -c '^    _watch_gpu_faults \(prefill\|decode\)$' "$DIR/vllm_disagg.sh")" "4" "all four server roles start a watcher"
# The script's output goes through `| tee` in the container: a watcher that still holds it
# when the script ends keeps the container, and the job, up until the wall clock.
printf 'INFO serving\n' > "$TMP/logs/42/decode_NODE1.log"
_t0=$(date +%s)
O="$(env -i PATH="$PATH" _RUN_LOGS="$TMP/logs" SLURM_JOB_ID=42 NODE_RANK=1 bash -c "$FUNCS
    _watch_gpu_faults decode; echo \"watchers:\${_GPU_FAULT_WATCHERS}\"; echo script-done" 2>&1 | timeout 20 cat)"
_rc=$?; _t=$(( $(date +%s) - _t0 ))
_has "$O" "script-done" "the script's output arrives"
_has "rc=$_rc took<10s=$([ "$_t" -lt 10 ] && echo yes || echo "no(${_t}s)")" "rc=0 took<10s=yes" \
     "the output pipe closes as soon as the script exits (no watcher holds it)"
_wpids="$(sed -n 's/^watchers://p' <<<"$O")"
sleep 1; _alive=""; for p in $_wpids; do kill -0 "$p" 2>/dev/null && _alive="$_alive $p"; done
_has "started=[${_wpids# }] alive=[${_alive# }]" "alive=[]" "the watcher is killed when the script exits"
_has "${_wpids:-none}" "${_wpids:-x}" "the script recorded its watcher's pid"

echo ""
echo "=== CONTAINER_ENV ==="
LINES="$(sed -n '/^_EXTRA_ENV=""$/,/^for _v in \${CONTAINER_ENV/p' "$DIR/run_xPyD_models.slurm")"
_env() { env -i CONTAINER_ENV="$1" bash -c "$LINES"'
echo "[$_EXTRA_ENV]"'; }
_has "$(_env 'HSA_NO_SCRATCH_RECLAIM,ROCM_AITER_FA')" "[ -e HSA_NO_SCRATCH_RECLAIM -e ROCM_AITER_FA]" "comma-separated names become bare -e NAME"
_has "$(_env 'A B,C')" "[ -e A -e B -e C]" "spaces and commas both separate"
_has "$(_env '')" "[]" "unset: nothing added"
_has "$(grep -c '^    \${_EXTRA_ENV} \\$' "$DIR/run_xPyD_models.slurm")" "1" "docker run carries the extra -e flags"

echo "======================================================"
echo "  fault_capture_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
