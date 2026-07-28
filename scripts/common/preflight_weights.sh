#!/bin/bash
# Phase 0 pre-flight: verify the model weights exist on every target node BEFORE
# sbatch. Local NVMe (/mnt/m2m_nobackup/models_blog) is per-node and NON-UNIFORM
# on OCI amd-rccl, so a disagg job whose nodelist includes a node missing the
# weights will fail deep into bring-up. Run from the login node.
#
# Usage:
#   MODEL_DIR=/mnt/m2m_nobackup/models_blog MODEL_NAME=DeepSeek-V3 \
#     NODELIST=useocpm2m-097-083,useocpm2m-097-087 \
#     bash scripts/common/preflight_weights.sh
#
# Confirmed local-NVMe DeepSeek-V3/R1 node set (survey 2026-07-25):
#   008 030 038 083 087 099 119 122   (ABSENT on 137)
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/m2m_nobackup/models_blog}"
MODEL_NAME="${MODEL_NAME:-}"
NODELIST="${NODELIST:-}"
PARTITION="${PARTITION:-amd-rccl}"
[ -n "$MODEL_NAME" ] || { echo "[preflight][ERROR] set MODEL_NAME" >&2; exit 2; }
[ -n "$NODELIST" ]   || { echo "[preflight][ERROR] set NODELIST (comma-separated)" >&2; exit 2; }

target="$MODEL_DIR/$MODEL_NAME"
echo "[preflight] checking $target on: $NODELIST"

# One task per node; each prints PRESENT/MISSING with its hostname.
out="$(srun -p "$PARTITION" --nodelist="$NODELIST" \
        --ntasks-per-node=1 --gres=gpu:1 --time=3 --overcommit bash -c \
        "if [ -d '$target' ] && [ -n \"\$(ls -A '$target' 2>/dev/null)\" ]; then \
            echo \"\$(hostname) PRESENT\"; else echo \"\$(hostname) MISSING\"; fi" 2>/dev/null)"

echo "$out" | sort
if echo "$out" | grep -q MISSING; then
    echo "[preflight][FAIL] some nodes lack $target. Pick from the confirmed set (008 030 038 083 087 099 119 122) or use MODEL_DIR=/shared_inference/models_blog (NFS, uniform)." >&2
    exit 1
fi
echo "[preflight][OK] all nodes have $target"
