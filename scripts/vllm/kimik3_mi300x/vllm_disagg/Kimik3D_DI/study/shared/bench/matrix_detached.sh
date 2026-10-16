#!/bin/bash
# Runs on proxy node, fully detached. Warms up once, then the full matrix. All output to log.
OUT=~/k3_study_results/03_perf; mkdir -p $OUT
R=http://127.0.0.1:30000
echo "=== warmup single request $(date +%T) ==="
curl -s -m 300 -o /dev/null "$R/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
echo "=== matrix start $(date +%T) ==="
run(){ local isl=$1 osl=$2; shift 2; for c in "$@"; do
  echo "--- isl$isl osl$osl con$c $(date +%T) ---"
  ROUTER=$R bash ~/perf_point.sh $isl $osl $c mi325x $OUT; done; }
run 128 32       1 8 16
run 1024 1024    8 32 64 128 256
run 32000 512    32 64 128
run 128000 512   16 32
run 300000 128   8 32
echo "=== MATRIX DONE $(date +%T) ==="
