#!/bin/bash
# clean_node.sh — ensure a node's GPUs are in a CLEAN state before deploying an engine.
# Run via srun on the target node (exclusive). Kills leftover serving containers/processes
# that squat VRAM (the #1 cause of OOM-at-init), then verifies VRAM is actually free.
#
# Usage (inside srun on the node):  clean_node.sh [max_used_gb_threshold]
#   default threshold = 5 GB; exits non-zero if any GPU still above it after cleanup.
set -uo pipefail
THRESH_GB="${1:-5}"

echo "[clean] $(hostname): removing ALL containers + stray GPU procs (reservation is exclusive to us)"
# The reservation is slurm-exclusive to us, but Docker containers are NOT slurm-tracked,
# so orphaned/stale containers from prior runs can linger and squat VRAM. Remove them ALL.
cids=$(docker ps -aq 2>/dev/null)
[ -n "$cids" ] && { docker kill $cids >/dev/null 2>&1; docker rm -f $cids >/dev/null 2>&1; } || true
# belt-and-suspenders: kill stray engine procs that may hold VRAM
pkill -9 -f "vllm|atom.entrypoints|sglang.launch|EngineCore|sglang.launch_server" 2>/dev/null || true
sleep 5

# 3) verify VRAM free on every GPU.
# rocm-smi prints one line per GPU: "GPU[N]  : VRAM Total Used Memory (B): <bytes>".
# Take the LAST integer on each line (the byte count), not the GPU index.
echo "[clean] verifying VRAM free (threshold ${THRESH_GB}GB)"
busy=0; i=0
while IFS= read -r line; do
  b=$(echo "$line" | grep -oE "[0-9]+" | tail -1)
  [ -z "$b" ] && continue
  gb=$(( b / 1000000000 ))
  if [ "$gb" -gt "$THRESH_GB" ]; then echo "  GPU$i: ${gb}GB USED (>threshold)"; busy=1; else echo "  GPU$i: ${gb}GB ok"; fi
  i=$((i+1))
done < <(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "Used Memory")
if [ "$busy" -ne 0 ]; then
  echo "[clean] WARNING: some GPUs still hold VRAM after cleanup — listing holders:"
  rocm-smi --showpids 2>/dev/null | grep -iE "vllm|atom|sglang|EngineCore" | head
  exit 1
fi
echo "[clean] $(hostname): all GPUs clean"
exit 0
