#!/bin/bash
# Run the full 6-class perf matrix (MI325X column) sequentially via perf_point.sh.
# Runs ON the proxy node. Usage: PLAT=mi325x OUT=~/k3_study_results/03_perf bash run_perf_matrix.sh
set -u
PLAT="${PLAT:-mi325x}"; OUT="${OUT:?outdir}"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
# class: ISL OSL "con-list"   (MI325X ceilings)
run(){ local isl=$1 osl=$2; shift 2; for c in "$@"; do
  echo "=== $PLAT isl$isl osl$osl con$c @ $(date +%T) ===";
  ROUTER=http://127.0.0.1:30000 bash "$HERE/perf_point.sh" "$isl" "$osl" "$c" "$PLAT" "$OUT" 2>&1 | tail -4; done; }
# latency floor
run 128 32      1 8 16
# throughput
run 1024 1024   8 32 64 128 256
# long-context
run 32000 512   32 64 128
# very-long
run 128000 512  16 32
# 300K frontier
run 300000 128  8 32
echo "=== perf matrix done -> $OUT ==="; ls -la "$OUT"/*.json 2>/dev/null | wc -l
