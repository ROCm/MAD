#!/bin/bash
# =============================================================================
# GLM-5.2 long-context scaling sweep — TP8 / EP8 / EP16, 50K..750K tokens.
#
# Blog-grade study driver: for one LIVE config (router already up + registered),
# walks the size x concurrency matrix, gating each size on NIAH retrieval BEFORE
# trusting its perf, and records every cell (success OR capacity-ceiling failure)
# with full provenance to NFS + the logbook.
#
# Run INSIDE the vLLM image (needs `vllm bench serve`), pointed at a live router.
#
# Required env:
#   CFG            config label: tp8 | ep8 | ep16   (provenance only; drives np policy)
#   BASE           router base url, e.g. http://<router-ip>:10001
#   MODEL          served model path (must match /v1/models exactly)
#   NODES          node list string for provenance, e.g. "008+044/043+042"
# Optional env:
#   SIZES          token sizes (default "50000 100000 200000 300000 500000 750000")
#   CON_LIST       concurrency sweep (default "8 16 32 64")
#   OSL            output len (default 128)
#   OUT            result dir (default /shared_nfs/ravgupta_disagg205/results_longctx_<CFG>)
#   NIAH_PY        fixed NIAH harness (default /shared_nfs/ravgupta_disagg205/benchmark_niah_v3.py)
#   LOGBOOK        NFS logbook to append (default /shared_nfs/ravgupta_disagg205/EP16_LOGBOOK.md)
#   NIAH_MIN       accuracy gate threshold /10 (default 8)
#   IMAGE_REF      image tag for provenance (default vllm-mori-pr558:ionic)
#   GPU_UTIL,MAXLEN   recorded for provenance (informational)
# =============================================================================
set -uo pipefail

CFG="${CFG:?set CFG=tp8|ep8|ep16}"
BASE="${BASE:?set BASE=http://<router-ip>:10001}"
MODEL="${MODEL:?set MODEL=<served model path>}"
NODES="${NODES:-unknown}"
SIZES="${SIZES:-50000 100000 200000 300000 500000 750000}"
CON_LIST="${CON_LIST:-8 16 32 64}"
OSL="${OSL:-128}"
OUT="${OUT:-/shared_nfs/ravgupta_disagg205/results_longctx_${CFG}}"
NIAH_PY="${NIAH_PY:-/shared_nfs/ravgupta_disagg205/benchmark_niah_v3.py}"
LOGBOOK="${LOGBOOK:-/shared_nfs/ravgupta_disagg205/EP16_LOGBOOK.md}"
NIAH_MIN="${NIAH_MIN:-8}"
IMAGE_REF="${IMAGE_REF:-vllm-mori-pr558:ionic}"
GPU_UTIL="${GPU_UTIL:-?}"; MAXLEN="${MAXLEN:-?}"

mkdir -p "$OUT" 2>/dev/null || true
SUMMARY="$OUT/SUMMARY_longctx_${CFG}.txt"
CSV="$OUT/longctx_${CFG}.csv"
STAMP(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SUMMARY" >&2; }
book(){ printf '%s\n' "$*" >> "$LOGBOOK" 2>/dev/null || true; }

# CSV header (one row per size x con cell)
if [ ! -f "$CSV" ]; then
  echo "utc,cfg,nodes,image,size_tok,osl,con,np,niah_found,niah_gate,successful,failed,req_tput,out_tput,ttft_mean_ms,ttft_p50_ms,ttft_p99_ms,tpot_mean_ms,tpot_p50_ms,e2el_mean_ms,duration_s,outcome" > "$CSV"
fi

# ---- smoke test the route before ANY measurement -------------------------
log "###### START longctx CFG=$CFG base=$BASE nodes=$NODES image=$IMAGE_REF ######"
book ""
book "## LONGCTX SWEEP START — CFG=$CFG nodes=$NODES @ $(STAMP)"
book "image=$IMAGE_REF gpu_util=$GPU_UTIL max-model-len=$MAXLEN sizes=[$SIZES] con=[$CON_LIST]"
_smoke=$(curl -s -m 60 "$BASE/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}" 2>&1)
if ! echo "$_smoke" | grep -qi "paris\|choices"; then
  log "!!! SMOKE TEST FAILED — route bad, aborting. resp: ${_smoke:0:200}"
  book "ABORT: smoke test failed (bad route) @ $(STAMP)"
  exit 3
fi
log "smoke OK"

# ---- helper: run NIAH at one token size, return found/10 -----------------
niah_gate(){ local sz=$1 out="$OUT/niah_${sz}.txt"
  NIAH_URL="$BASE/v1/chat/completions" NIAH_MODEL="$MODEL" \
  NIAH_TOKENS="$sz" NIAH_TOKENIZER="$MODEL" NIAH_MAXTOK=256 NIAH_SEEDS="0,1" \
  NIAH_TIMEOUT=2400 NIAH_WARMUP=1 \
  timeout 3600 python3 "$NIAH_PY" > "$out" 2>&1
  # v2 prints "found=NN/10"; take the min across seeds as the gate
  local f
  f=$(grep -oE "found=[ ]*[0-9]+/10" "$out" | grep -oE "[0-9]+" | sort -n | head -1)
  echo "${f:-0}"
}

# ---- helper: one perf cell -> append CSV row + summary --------------------
bench_cell(){ local sz=$1 con=$2 np=$3 niah=$4 gate=$5 raw="$OUT/raw_${sz}_con${con}.log"
  # timeout scales with total tokens (prefill is ~linear): base 1800s per 2k tokens, capped high
  local tmo=$(( 1800 * (sz + OSL) / 2048 )); [ "$tmo" -lt 1800 ] && tmo=1800; [ "$tmo" -gt 21600 ] && tmo=21600
  log ">>> $CFG size=$sz con=$con np=$np (timeout ${tmo}s, niah=${niah}/10)"
  timeout "$tmo" vllm bench serve --backend openai --base-url "$BASE" --model "$MODEL" \
    --dataset-name random --random-input-len "$sz" --random-output-len "$OSL" --random-prefix-len 0 \
    --max-concurrency "$con" --num-prompts "$np" --ignore-eos --temperature 0 \
    --percentile-metrics ttft,tpot,e2el \
    --result-dir "$OUT" --result-filename "res_${sz}_con${con}.json" --save-result > "$raw" 2>&1
  local rc=${PIPESTATUS[0]} outcome="ok"
  # extract metrics
  local succ fail rtput otput ttftm ttftp50 ttftp99 tpotm tpotp50 e2el dur
  gv(){ grep -iE "$1" "$raw" | grep -oE "[0-9]+\.[0-9]+|[0-9]+" | tail -1; }
  succ=$(grep -iE "Successful requests" "$raw" | grep -oE "[0-9]+" | tail -1)
  if [ "$rc" -eq 124 ]; then outcome="STALL_TIMEOUT"
  elif [ -z "$succ" ]; then
    outcome="NO_METRICS"
    # classify capacity ceiling from the log
    grep -qiE "out of memory|OOM|HIP out of memory|mla_attention" "$raw" && outcome="OOM"
    grep -qiE "KV cache|no available|preempt|evict" "$raw" && outcome="KV_CEILING"
    grep -qiE "400|exceeds|maximum context|max_model_len" "$raw" && outcome="CTX_OVERFLOW"
  fi
  fail=$(grep -iE "Failed requests" "$raw" | grep -oE "[0-9]+" | tail -1)
  rtput=$(gv "Request throughput"); otput=$(gv "Output token throughput")
  ttftm=$(gv "Mean TTFT"); ttftp50=$(gv "Median TTFT"); ttftp99=$(gv "P99 TTFT")
  tpotm=$(gv "Mean TPOT"); tpotp50=$(gv "Median TPOT"); e2el=$(gv "Mean E2EL"); dur=$(gv "Benchmark duration")
  echo "$(STAMP),$CFG,$NODES,$IMAGE_REF,$sz,$OSL,$con,$np,$niah,$gate,${succ:-0},${fail:-},${rtput:-},${otput:-},${ttftm:-},${ttftp50:-},${ttftp99:-},${tpotm:-},${tpotp50:-},${e2el:-},${dur:-},$outcome" >> "$CSV"
  if [ "$outcome" = "ok" ]; then
    log "    OK succ=$succ out_tput=${otput} tpot_p50=${tpotp50} ttft_mean=${ttftm}"
    book "  [$CFG] size=$sz con=$con -> tok/s=${otput} TPOT=${tpotp50}ms TTFT=${ttftm}ms succ=$succ (niah ${niah}/10) @ $(STAMP)"
  else
    log "    !!! $outcome — CAPACITY CEILING at size=$sz con=$con. tail:"; tail -4 "$raw" | sed 's/^/    !! /' | tee -a "$SUMMARY" >&2
    book "  [$CFG] size=$sz con=$con -> **$outcome** (capacity ceiling; this is the config's limit) @ $(STAMP)"
  fi
}

# ---- main matrix: per size, NIAH-gate THEN perf con-sweep -----------------
for sz in $SIZES; do
  log "===== size=$sz tokens ====="
  gate="PASS"; niah=$(niah_gate "$sz")
  [ "$niah" -lt "$NIAH_MIN" ] && gate="LOW(<$NIAH_MIN)"
  log "NIAH size=$sz -> ${niah}/10 gate=$gate"
  book "  [$CFG] NIAH size=$sz -> ${niah}/10 ($gate) @ $(STAMP)"
  # per-shape warmup once (compile kernels for this size), low con
  timeout 3600 vllm bench serve --backend openai --base-url "$BASE" --model "$MODEL" \
    --dataset-name random --random-input-len "$sz" --random-output-len "$OSL" \
    --max-concurrency 2 --num-prompts 2 --ignore-eos --temperature 0 >/dev/null 2>&1 || true
  for con in $CON_LIST; do
    # np policy: small multiple of con, shrink at huge sizes to bound wall-clock
    np=$(( con * 3 ))
    [ "$sz" -ge 200000 ] && np=$(( con * 2 ))
    [ "$sz" -ge 500000 ] && np=$con
    bench_cell "$sz" "$con" "$np" "$niah" "$gate"
    # if this size hit a hard ceiling at low con, higher con will too — skip the rest of the row
    if tail -1 "$CSV" | grep -qE "OOM|KV_CEILING|CTX_OVERFLOW"; then
      log "    ceiling hit at con=$con; skipping higher con for size=$sz"
      book "  [$CFG] size=$sz: ceiling at con=$con, higher con skipped (would also fail) @ $(STAMP)"
      break
    fi
  done
done

log "###### DONE longctx CFG=$CFG ######"
book "## LONGCTX SWEEP DONE — CFG=$CFG @ $(STAMP)  (CSV: $CSV)"
