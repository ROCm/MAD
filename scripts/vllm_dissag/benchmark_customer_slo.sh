#!/bin/bash
# Customer-facing SLO benchmark for GLM-5.2 / HY4 on MI355X.
#
# WHY THIS EXISTS, AND WHY IT IS NOT benchmark_xPyD.sh
# ----------------------------------------------------
# benchmark_xPyD.sh is a SWEEP: it walks a grid of isl/osl x concurrency at
# --request-rate inf and reports the max throughput per cell. That is the right tool
# for tuning -- it answers "did my change help".
#
# It is the WRONG tool for answering "do we meet the customer's SLO", for four
# reasons, each of which this script fixes:
#
#   1. --request-rate inf is a saturation test, not a service test. It fires all
#      requests at once and measures how fast the queue drains. Under that regime
#      TTFT is dominated by queueing: at con=64/isl=28672 the measured TTFT is just
#      458,752 tok / 39,340 tok/s = 11.66 s -- an identity, not a property of the
#      model. The customer's "<7 s avg TTFT" is a statement about a SERVED request,
#      so it must be measured at a finite arrival rate.
#   2. The reported stat is the mean/max. The customer specified p50/p95/p99. Those
#      are different questions and vLLM will only print percentiles if asked
#      (--percentile-metrics / --metric-percentiles).
#   3. The pass/fail is left to a human reading a log. Here the SLO is expressed to
#      vLLM directly via --goodput, so "how many requests actually met the SLO" is a
#      number the server computes, and this script exits non-zero when it is missed.
#   4. --random-prefix-len 0 means every request is unique. The customer's use case
#      is AGENTIC and they explicitly flagged interest in long prefix caching -- an
#      agent loop re-sends a large, mostly-static context every turn. Measuring that
#      with zero shared prefix understates the real system by whatever the prefix
#      cache would have saved. See PREFIX_FRAC below.
#
# WORKLOAD, AS SPECIFIED BY THE CUSTOMER
# ---------------------------------------
#   use case       agentic workflow          -> shared prefix, multi-turn
#   model          GLM-5.2 (HY4 family)      -> FP8 and MXFP4
#   ISL/OSL        80K/1K   (256K context)
#                  200K/1K  (1M context)
#   concurrency    max 256 per DP rank
#   TTFT           < 7 s   avg
#   TPOT           < 50 ms avg  (equivalently "> 20 tok/s" per the sheet)
#   prefill        34,000 tok/s per rank   (MI355X)
#   decode            670 tok/s per rank   (MI355X)
#   framework      vLLM only
#
# Note the sheet gives TTFT/TPOT as *averages* but asks for p50/p95/p99 in the
# latency-SLO row. We report all of them and gate on the average, which is the
# stated acceptance criterion; the percentiles are reported so the tail is visible
# rather than hidden inside a mean.
#
# CAPACITY REALITY CHECK -- READ BEFORE SETTING CONCURRENCY
# ---------------------------------------------------------
# GLM-5.2 MLA KV is kv_lora_rank(512) + qk_rope_head_dim(64) = 576 elem/token/layer,
# FP8, 78 layers = 43.88 KiB/token. Per request that is:
#       28,672 tok ->  1.20 GiB      80,000 tok ->  3.35 GiB     200,000 tok -> 8.37 GiB
# On one 8x MI355X node (288 GB/GPU) the usable KV pool is roughly 930 GiB (FP8,
# util 0.80) after weights (88 GiB/GPU) and the 3.5 GiB MLA chunked-prefill
# workspace. So the KV-bound concurrency ceiling on ONE decode node is about:
#       ISL  28,672 -> ~777        ISL 80,000 -> ~279        ISL 200,000 -> ~111
# Concurrency 256 at 80K needs ~857 GiB and fits. **Concurrency 256 at 200K needs
# ~2,142 GiB and does NOT fit on a single decode node** -- it needs 3 (or MXFP4,
# which has ~1,440 GiB of pool and still only reaches ~172). This script does not
# silently truncate: it runs what you ask and reports what happened, but the 200K
# row is defaulted to a concurrency list that fits so that a failed run means a real
# failure and not an OOM you could have predicted with arithmetic.
#
# USAGE
#   MODEL_PATH=/models/GLM-5.2-FP8 BENCHMARK_PORT=8000 ./benchmark_customer_slo.sh
#
# Exit code 0 only if every scenario met both the TTFT and the TPOT SLO.

set -uo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
BENCHMARK_PORT="${BENCHMARK_PORT:-8000}"
HOST="${BENCHMARK_HOST:-127.0.0.1}"
LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/customer_slo}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

# --- SLO, from the customer sheet. Milliseconds, because --goodput wants ms. ---
SLO_TTFT_MS="${SLO_TTFT_MS:-7000}"
SLO_TPOT_MS="${SLO_TPOT_MS:-50}"
# Per-rank throughput targets (MI355X row of the sheet).
TGT_PREFILL_TOK_S="${TGT_PREFILL_TOK_S:-34000}"
TGT_DECODE_TOK_S="${TGT_DECODE_TOK_S:-670}"

# --- Number of DP ranks, to convert aggregate throughput into per-rank. ---
# The sheet's targets are PER RANK and vLLM reports an AGGREGATE, so getting this
# wrong scales every throughput verdict by 8x in whichever direction hurts.
DP_RANKS="${DP_RANKS:-8}"

# --- Agentic shared prefix -----------------------------------------------------
# Fraction of ISL that is a shared, cacheable system/tool preamble. vLLM's
# RandomDataset builds the prefix ONCE and prepends the same tokens to every
# request (datasets.py: "Generate prefix once"), so this is a genuine shared
# prefix that the prefix cache can hit -- not per-request random padding.
# 0.0 reproduces the old zero-sharing behaviour. 0.5 is a deliberate, declared
# assumption: the customer said "agentic" and "interested in long prefix caching"
# but did not give a reuse ratio. Sweep it rather than trusting one value.
PREFIX_FRAC="${PREFIX_FRAC:-0.5}"

# --- Arrival model -------------------------------------------------------------
# Poisson (burstiness 1.0) at a finite rate. NOT "inf". See reason 1 above.
BURSTINESS="${BURSTINESS:-1.0}"

# Scenarios: "label:isl:osl:concurrency-list". Concurrency lists are chosen to fit
# the KV pool (see the capacity note); override with SCENARIOS=... to push past it.
DEFAULT_SCENARIOS="\
256k-ctx:80000:1024:16 32 64 128 256|\
1m-ctx:200000:1024:8 16 32 64"
IFS='|' read -ra SCENARIOS <<< "${SCENARIOS:-$DEFAULT_SCENARIOS}"

ITERS="${SLO_ITERS:-1}"
RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/slo_json}"
mkdir -p "$RESULT_DIR"

echo "=============================================================="
echo " Customer SLO benchmark -- GLM-5.2 / MI355X"
echo "   model        : $MODEL_PATH"
echo "   SLO          : TTFT <= ${SLO_TTFT_MS} ms avg, TPOT <= ${SLO_TPOT_MS} ms avg"
echo "   per-rank tgt : prefill ${TGT_PREFILL_TOK_S} tok/s, decode ${TGT_DECODE_TOK_S} tok/s (DP=${DP_RANKS})"
echo "   prefix       : ${PREFIX_FRAC} of ISL shared (agentic reuse)"
echo "   arrival      : Poisson, burstiness ${BURSTINESS}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Shape warmup. Without this the first measured cell of a shape absorbs residual
# JIT and reports a wildly inflated TPOT (observed 302 ms vs ~89 ms steady state).
# Deliberately at low concurrency and few prompts so it is cheap.
# ---------------------------------------------------------------------------
warmup_shape() {
    local isl=$1 osl=$2 pfx=$3
    echo "[WARMUP] isl=$isl osl=$osl prefix=$pfx"
    timeout "${WARMUP_TIMEOUT:-3600}" vllm bench serve \
        --model "$MODEL_PATH" --backend vllm --host "$HOST" --port "$BENCHMARK_PORT" \
        --dataset-name random \
        --random-input-len "$isl" --random-output-len "$osl" --random-prefix-len "$pfx" \
        --num-prompts 4 --max-concurrency 2 --request-rate inf --ignore-eos \
        >>"${LOG}_warmup.log" 2>&1
}

# ---------------------------------------------------------------------------
# One measured cell.
#
# Request rate is derived, not guessed: to sustain `con` in flight with a
# per-request latency of about (TTFT + OSL*TPOT), Little's Law gives
#     rate = con / latency
# Feeding a rate materially above that just rebuilds the infinite-rate queue and
# we are back to measuring drain time. We target the SLO latency, i.e. we ask
# "can the system serve the offered load AT the SLO", which is the actual question.
# ---------------------------------------------------------------------------
run_cell() {
    local label=$1 isl=$2 osl=$3 con=$4 pfx=$5 iter=$6

    local slo_lat_s
    slo_lat_s=$(python3 -c "print(($SLO_TTFT_MS + $osl*$SLO_TPOT_MS)/1000.0)")
    local rate
    rate=$(python3 -c "print(round($con/$slo_lat_s, 4))")

    # Enough prompts that the measurement is not dominated by ramp-in/ramp-out.
    # 4x concurrency, floored at 32, capped so a 200K cell stays affordable.
    local prompts=$(( con * 4 )); [ "$prompts" -lt 32 ] && prompts=32
    [ "$prompts" -gt "${MAX_PROMPTS:-512}" ] && prompts="${MAX_PROMPTS:-512}"

    # Timeout scales with the work: prompts * (isl+osl) tokens, assuming a
    # pessimistic floor rate, plus slack.
    local tmo
    tmo=$(python3 -c "print(int(900 + $prompts*($isl+$osl)/3000.0))")

    local jf="$RESULT_DIR/${label}_con${con}_iter${iter}.json"
    echo "[RUNNING] $label isl=$isl osl=$osl con=$con prefix=$pfx rate=${rate}/s prompts=$prompts (timeout ${tmo}s)"

    timeout "$tmo" vllm bench serve \
        --model "$MODEL_PATH" --backend vllm --host "$HOST" --port "$BENCHMARK_PORT" \
        --dataset-name random \
        --random-input-len "$isl" --random-output-len "$osl" --random-prefix-len "$pfx" \
        --num-prompts "$prompts" \
        --max-concurrency "$con" \
        --request-rate "$rate" \
        --burstiness "$BURSTINESS" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --goodput "ttft:${SLO_TTFT_MS}" "tpot:${SLO_TPOT_MS}" \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "$(basename "$jf")" \
        2>&1 | tee -a "${LOG}.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ]; then
        echo "[STALL] $label con=$con timed out after ${tmo}s" | tee -a "${LOG}.log" "${LOG}_stalls.log"
    fi
    echo "$jf"
}

for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r label isl osl conlist <<< "$scenario"
    pfx=$(python3 -c "print(int($isl*$PREFIX_FRAC))")
    echo; echo "########## scenario $label  (isl=$isl osl=$osl, shared prefix=$pfx) ##########"
    warmup_shape "$isl" "$osl" "$pfx"
    for iter in $(seq 1 "$ITERS"); do
        for con in $conlist; do
            run_cell "$label" "$isl" "$osl" "$con" "$pfx" "$iter" >/dev/null
            sleep 10
        done
    done
done

echo; echo "########## VERDICT ##########"
python3 "$(dirname "$0")/slo_report.py" \
    --result-dir "$RESULT_DIR" \
    --slo-ttft-ms "$SLO_TTFT_MS" --slo-tpot-ms "$SLO_TPOT_MS" \
    --target-prefill "$TGT_PREFILL_TOK_S" --target-decode "$TGT_DECODE_TOK_S" \
    --dp-ranks "$DP_RANKS" \
    --csv "${LOG}_slo.csv"
exit $?
