#!/bin/bash
# Envelope sweep: hold ISL/OSL light+realistic, sweep concurrency across the saturation knee.
# Shows "running" (throughput rises, latency flat) -> knee -> "saturated" (throughput plateaus,
# latency climbs, completions drop). Runs detached on the proxy node.
# Usage: PLAT=mi325x OUT=~/k3_study_results/03b_envelope bash envelope_sweep.sh
set -u
PLAT="${PLAT:-mi325x}"; OUT="${OUT:?outdir}"; ISL="${ISL:-2048}"; OSL="${OSL:-128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; mkdir -p "$OUT"
R=http://127.0.0.1:30000
# one warmup
curl -s -m 300 -o /dev/null "$R/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
echo "=== envelope sweep ISL=$ISL OSL=$OSL @ $(date +%T) ==="
for c in 1 4 8 16 32 64 128; do
  echo "--- con$c $(date +%T) ---"
  ROUTER=$R bash "$HERE/perf_point.sh" "$ISL" "$OSL" "$c" "${PLAT}env" "$OUT"
done
echo "=== ENVELOPE DONE $(date +%T) ==="
