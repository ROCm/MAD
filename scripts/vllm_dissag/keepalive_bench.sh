#!/bin/bash
# Keepalive with LIGHT heartbeat traffic: holds the disagg server up AND sends a
# tiny request every ~20s so prefill discovery/ping stays registered (idle sleep
# lets the prefill ZMQ ping die ~2min in). Runs KEEPALIVE_MINS (default 90).
: "${KEEPALIVE_MINS:=90}"
PORT="${BENCHMARK_PORT:-30000}"
MODEL="/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8"
echo "[keepalive] light-traffic hold ${KEEPALIVE_MINS}min on :${PORT}"
_end=$(( $(date +%s) + KEEPALIVE_MINS*60 ))
i=0
while [ "$(date +%s)" -lt "$_end" ]; do
  curl -s -m 30 "http://127.0.0.1:${PORT}/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" >/dev/null 2>&1
  i=$((i+1))
  [ $((i % 3)) -eq 0 ] && echo "[keepalive] heartbeat $i, $(( (_end-$(date +%s))/60 ))min left"
  sleep 20
done
echo "[keepalive] done"
