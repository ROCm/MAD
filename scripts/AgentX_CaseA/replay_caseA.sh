#!/bin/bash
# =============================================================================
# AgentX Case-A conformance trace — generic replay driver.
#
# Replays the frozen Case-A trace against ANY OpenAI-compatible endpoint and
# writes a concurrency-sweep summary. The trace is engine- and topology-agnostic:
# to benchmark a 1-node aggregated server, a 2-node setup, or a disaggregated
# 1P1D router, point URL at that endpoint — nothing else changes.
#
#   URL=http://<host>:<port> ./replay_caseA.sh
#
# aiperf runs INSIDE a container on purpose: some hosts' /proc lacks per-process
# entries (/proc/self/stat), which crashes aiperf's psutil bare-metal. A container
# has a clean /proc. If your host runs aiperf fine bare-metal, set AIPERF to your
# local `aiperf` binary and IMG="" to skip the container.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL="${URL:?set URL to the endpoint under test, e.g. http://localhost:8801}"
SERVED="${SERVED:-GLM-5.2-MXFP4}"          # served-model-name the endpoint answers to
TOK="${TOK:-/models/GLM-5.2-MXFP4}"        # tokenizer path or HF id
CONCS="${CONCS:-1 2 4 8 16}"
DUR="${DUR:-300}"                          # seconds per concurrency point
CORP="${CORP:-$HERE/corpus}"               # trace dir (auto-unpacked if missing)
OUT="${OUT:-$HERE/results}"
SEED="${SEED:-42}"; NSESS="${NSESS:-200}"
IMG="${IMG:-rocm/atom-dev:latest}"         # any image with the aiperf fork installed
AIPERF="${AIPERF:-aiperf}"                 # aiperf binary path (inside IMG, or local)

mkdir -p "$OUT"

# --- ensure the trace exists (deterministic: seed 42 => identical every time) ---
if [ ! -d "$CORP" ] || [ -z "$(ls "$CORP"/*.json 2>/dev/null)" ]; then
  if [ -f "$HERE/caseA_conformance_corpus.tar.gz" ] && [ "$CORP" = "$HERE/corpus" ]; then
    # tarball contains a top-level corpus/ dir → unpack into $HERE so it lands at $HERE/corpus
    echo "[replay] unpacking frozen trace -> $CORP"
    tar xzf "$HERE/caseA_conformance_corpus.tar.gz" -C "$HERE"
  else
    echo "[replay] regenerating trace (seed=$SEED, n=$NSESS) -> $CORP"
    python3 "$HERE/gen_caseA_conformance.py" "$CORP" "$NSESS" "$SEED"
  fi
fi
NJSON=$(ls "$CORP"/*.json 2>/dev/null | wc -l)
echo "[replay] trace: $CORP ($NJSON sessions)  endpoint: $URL  model: $SERVED"

SUM="$OUT/summary.csv"
echo "conc,out_tok_s,req_s,ttft_p50,ttft_p90,ttft_p99,itl_p50,itl_p90,e2e_p50,e2e_p90,e2e_p99,isl_mean,osl_mean,cache_pct" > "$SUM"

run_aiperf(){  # $1=conc  $2=artifact-dir
  if [ -n "$IMG" ]; then
    docker run --rm --network host \
      -v "$HERE":"$HERE" $( [ -d /models ] && echo "-v /models:/models:ro" ) \
      -v /shared_nfs:/shared_nfs:ro 2>/dev/null \
      -e HOME="$HERE/.home" -e HF_HOME="$HERE/.home/hf" \
      -e AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 -e AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
      --entrypoint "$AIPERF" "$IMG" "${AIPERF_ARGS[@]}"
  else
    "$AIPERF" "${AIPERF_ARGS[@]}"
  fi
}

for C in $CONCS; do
  rd="$OUT/c${C}"; mkdir -p "$rd"
  echo "==================== C=$C ($(date +%H:%M:%S)) ===================="
  AIPERF_ARGS=(profile --scenario inferencex-agentx-mvp
    --custom-dataset-type weka_trace --input-file "$CORP"
    --url "$URL" --endpoint /v1/chat/completions --endpoint-type chat --streaming
    --model "$SERVED" --tokenizer "$TOK"
    --concurrency "$C" --benchmark-duration "$DUR" --unsafe-override
    --trajectory-start-min-ratio 0.90 --trajectory-start-max-ratio 0.98
    --tokenizer-trust-remote-code --use-server-token-count --no-gpu-telemetry
    --max-context-length 262144 --output-artifact-dir "$rd/art")
  run_aiperf "$C" "$rd/art" > "$rd.log" 2>&1
  csv="$rd/art/profile_export_aiperf.csv"
  if [ -f "$csv" ]; then
    awk -F, -v c=$C '
      /^Output Token Throughput \(/{ot=$2} /^Request Throughput/{rs=$2}
      /^Time to First Token/{t50=$10;t90=$12;t99=$14}
      /^Inter Token Latency/{i50=$10;i90=$12}
      /^Request Latency/{e50=$10;e90=$12;e99=$14}
      /^Input Sequence Length/{isl=$2} /^Output Sequence Length/{osl=$2}
      /^Theoretical Prefix Cache Hit/{ca=$2}
      END{printf "%s,%.1f,%.2f,%.0f,%.0f,%.0f,%.2f,%.2f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f\n",c,ot,rs,t50,t90,t99,i50,i90,e50,e90,e99,isl,osl,ca}
    ' "$csv" >> "$SUM"
    echo "  C=$C -> $(tail -1 "$SUM")"
  else
    echo "$C,FAILED" >> "$SUM"; echo "  C=$C FAILED (see $rd.log)"
  fi
done
echo "==================== DONE ($(date +%H:%M:%S)) ===================="
column -t -s, "$SUM" 2>/dev/null || cat "$SUM"
