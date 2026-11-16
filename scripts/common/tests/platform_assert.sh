#!/bin/bash
# Offline checks of the platform layer (compat/rocm.yaml platforms, scripts/common/platform/
# platform.py): cluster.sh takes GPUS_PER_NODE from the run's platform, and the GPU probe verifies
# a named platform rather than deciding it.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n        got: %s\n" "$3" "$2" "$1"; fail=$((fail+1)) ;; esac; }

cat > "$TMP/p.sh" <<'P'
. "$1/common/cluster.sh" >/dev/null 2>"$2/err"
echo "gpus=${GPUS_PER_NODE:-} acc=${MAD_PLATFORM_ACCELERATOR:-}"
cluster_verify_platform 2>&1; echo "rc=$?"
P
plat() { env -i PATH="$PATH" HOME="$TMP" MAD_SCRIPTS_DIR="$DIR" "$@" bash "$TMP/p.sh" "$DIR" "$TMP"; }

echo "=== GPUs per node from the platform ==="
_has "$(plat)" "gpus=8 acc=" "no platform named: the catalog's default platform, not verified"
_has "$(plat MAD_PLATFORM=mi455x-helios-tray)" "gpus=4 acc=cdna5" "a Helios tray: 4 GPUs per node"
_has "$(plat MAD_PLATFORM=mi455x-helios-tray GPUS_PER_NODE=2)" "gpus=2" "a value the run passed wins"
out="$(plat MAD_PLATFORM=nope)"
_has "$out" "rc=1" "an unknown platform is refused"
_has "$(cat "$TMP/err")" "unknown platform 'nope'; the catalog has:" "...naming the known ones"

echo ""
echo "=== the probes verify a named platform ==="
_has "$(plat MAD_PLATFORM=mi355x-ubb-8 MAD_GPU_ARCH=gfx950 MAD_GPU_COUNT=8)" "rc=0" "MI355X platform on gfx950 nodes with 8 GPUs"
_has "$(plat MAD_PLATFORM=mi355x-ubb-8 MAD_GPU_ARCH=gfx942)" "is gfx950, but these nodes are gfx942" "the wrong accelerator is refused"
_has "$(plat MAD_GPU_COUNT=4)" "uses 8 GPUs per node, but a node has 4" "a node with fewer GPUs than the run uses is refused"
_has "$(plat MAD_GPU_COUNT=32)" "rc=0" "more GPUs (a partition mode) is fine"
_has "$(plat MAD_GPU_ARCH=gfx950)" "rc=0" "unknown facts are not refused"

echo ""
echo "======================================================"
echo "  platform_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
