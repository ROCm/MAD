#!/bin/bash
# OSL sweep at fixed concurrency: proves the FIXED floor exists numerically.
# Hold con=8, ISL=2048, vary OSL. If wall = floor + k*OSL, the fit's y-intercept IS the
# fixed decode-wave floor (wall when OSL->0) and the slope is per-token decode cost.
# Runs detached on proxy. Usage: OUT=~/k3_study_results/03c_osl bash osl_sweep.sh
set -u
OUT="${OUT:?outdir}"; CON="${CON:-8}"; ISL="${ISL:-2048}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; mkdir -p "$OUT"
R=http://127.0.0.1:30000
curl -s -m 300 -o /dev/null "$R/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
echo "=== OSL sweep con=$CON ISL=$ISL @ $(date +%T) ==="
for osl in 16 32 64 128 256; do
  echo "--- osl$osl $(date +%T) ---"
  ROUTER=$R bash "$HERE/perf_point.sh" "$ISL" "$osl" "$CON" "mi325xosl" "$OUT"
done
echo "=== OSL SWEEP DONE $(date +%T) ==="
