#!/bin/bash
# Offline checks of gpu_mem_trace.sh against a fake kernel tree (KFD topology and
# per-process records, /proc, the GPUs' sysfs), and of how run_xPyD_models.slurm runs it:
#   - each holder is named by command line and container, with GiB per GPU, the GPU
#     numbered as vLLM's cuda:<n>;
#   - a sample is written only when it changes;
#   - the batch script starts it next to the container, off the job's output, and stops it.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has()    { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot() { grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }

echo "=== gpu_mem_trace.sh ==="
K="$TMP/kfd"; P="$TMP/proc"; D="$TMP/drm"
# Topology: node 0 is the CPU (gpu_id 0), nodes 1 and 2 are GPUs 0 and 1.
for n in 0 1 2; do mkdir -p "$K/topology/nodes/$n"; done
echo 0 > "$K/topology/nodes/0/gpu_id"
echo 51234 > "$K/topology/nodes/1/gpu_id"; printf 'domain 0\nlocation_id %d\n' $((0x11 << 8)) > "$K/topology/nodes/1/properties"
echo 60001 > "$K/topology/nodes/2/gpu_id"; printf 'domain 0\nlocation_id %d\n' $((0x2f << 8)) > "$K/topology/nodes/2/properties"
# GPUs' sysfs: card1 -> 0000:11:00.0, card2 -> 0000:2f:00.0
for c in "1 0000:11:00.0 40" "2 0000:2f:00.0 0"; do
    set -- $c; mkdir -p "$D/pci/$2" "$D/card$1"; ln -s "$D/pci/$2" "$D/card$1/device"
    echo $(($3 << 30)) > "$D/pci/$2/mem_info_vram_used"
done
# Processes: 4100 in a container holds 40 GiB on GPU1 only; 4200 on the host holds under 1 GiB.
mkdir -p "$K/proc/4100" "$K/proc/4200" "$P/4100" "$P/4200"
echo $((40 << 30)) > "$K/proc/4100/vram_60001"; echo 0 > "$K/proc/4100/vram_51234"
echo $((1 << 20)) > "$K/proc/4200/vram_51234"
printf 'python3\0-m\0vllm.entrypoints.openai.api_server\0--port\0' > "$P/4100/cmdline"
echo "0::/system.slice/docker-$(printf 'ab%.0s' {1..32}).scope" > "$P/4100/cgroup"
printf 'VLLM::Worker_TP0%40s' "" > "$P/4200/cmdline"; echo "0::/user.slice" > "$P/4200/cgroup"
_trace() { env -i PATH="$PATH" _KFD_ROOT="$K" _PROC_ROOT="$P" _GPU_SYSFS_ROOT="$D" \
    GPU_MEM_TRACE_SECS="$1" GPU_MEM_TRACE_INTERVAL=1 bash "$DIR/gpu_mem_trace.sh" "$TMP/out.log"; cat "$TMP/out.log"; }
O="$(_trace 2)"
_has    "$O" "# kfd gpu_id 51234 = GPU0 (0000:11:00.0)" "GPUs are numbered in topology order with their PCI address"
_has    "$O" "# kfd gpu_id 60001 = GPU1 (0000:2f:00.0)" "the CPU node is skipped in the numbering"
_has    "$O" "pid 4100 [container abababababab] python3 -m vllm.entrypoints.openai.api_server --port: GPU1 (0000:2f:00.0)=40GiB" \
        "a holder is named by command line and container, with GiB on the GPU it holds"
_hasnot "$O" "GPU0 (0000:11:00.0)=0GiB" "GPUs where it holds nothing are left out"
_hasnot "$O" "pid 4200" "a process holding under 1 GiB is left out"
_has    "$O" "used per GPU: 0000:11:00.0=40GiB 0000:2f:00.0=0GiB" "each sample has every GPU's total"
_has    "$(grep -c '^=== ' <<<"$O")" "1" "an unchanged sample is not written again"
# A holder that appears while sampling gets its own sample.
rm -f "$K/proc/4200/vram_51234"
( sleep 0.5; echo $((7 << 30)) > "$K/proc/4200/vram_51234" ) &
O="$(_trace 5)"; wait   # margin: each sample forks several tools, slow under load
_has "$(grep -c '^=== ' <<<"$O")" "2" "a change writes a new sample"
_has "$O" "pid 4200 [host] VLLM::Worker_TP0: GPU0 (0000:11:00.0)=7GiB" "a host process is marked as such; a renamed process loses its space padding"
# Without the kernel tree (no KFD), it still writes its header and exits on time.
_t0=$(date +%s)
O="$(env -i PATH="$PATH" _KFD_ROOT="$TMP/none" _PROC_ROOT="$P" _GPU_SYSFS_ROOT="$TMP/none" \
    GPU_MEM_TRACE_SECS=1 GPU_MEM_TRACE_INTERVAL=1 bash "$DIR/gpu_mem_trace.sh" "$TMP/out2.log"; echo "rc=$?"; cat "$TMP/out2.log")"
_has "$O" "rc=0" "no GPUs readable: exits cleanly"
_has "took<5=$([ $(( $(date +%s) - _t0 )) -lt 5 ] && echo yes || echo no)" "took<5=yes" "it stops after GPU_MEM_TRACE_SECS"

echo ""
echo "=== run_xPyD_models.slurm ==="
S="$(cat "$DIR/run_xPyD_models.slurm")"
_has "$S" 'bash "$NIXL_REPO_DIR/gpu_mem_trace.sh" "$_trace_out" </dev/null >/dev/null 2>&1 &' \
     "the tracer runs in the background, off the job's output"
_has "$S" 'if [ "${GPU_MEM_TRACE:-0}" = "1" ]' "off unless GPU_MEM_TRACE=1"
_has "$(sed -n '/^wait "\$_docker_pid"$/,/^exit "\$_docker_rc"$/p' <<<"$S")" 'kill "$_trace_pid"' \
     "it is stopped when the container ends"
_has "$(sed -n '/^_stop_container() {$/,/^}$/p' <<<"$S")" 'kill "$_trace_pid"' "it is stopped when the job is cancelled"
_body="$(sed -n '/_trace_pid="";$/,/^docker run --rm/p' <<<"$S")"
_hasnot "$_body" "'" "no single quote inside the single-quoted srun body"
[ "$(sed -n '/_trace_pid="";$/=' <<<"$S" | head -n 1)" -lt "$(sed -n '/^docker run --rm/=' <<<"$S" | head -n 1)" ] \
    && { printf "  PASS  %s\n" "it starts before the container"; pass=$((pass+1)); } \
    || { printf "  FAIL  %s\n" "it starts before the container"; fail=$((fail+1)); }

echo "======================================================"
echo "  gpu_mem_trace_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
