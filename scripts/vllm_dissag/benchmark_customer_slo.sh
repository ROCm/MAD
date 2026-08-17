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
# ISL IS A DISTRIBUTION, NOT AN AVERAGE -- see WORKLOAD_MODE
# ----------------------------------------------------------
# The sheet's ISL row is "avg: 80K (256K context)". Two numbers, and an earlier
# version of this script honoured only the first: a fixed --random-input-len 80000,
# which never once touched the 256K window. WORKLOAD_MODE=dist (the default) fixes
# that by generating an explicit right-skewed length distribution whose mean is the
# stated average and whose p99 reaches the stated window, then driving it through
# --dataset-name custom. See gen_workload.py for why --random-range-ratio provably
# cannot express this (uniform, symmetric, and validated to r < 1.0, so the widest
# possible support is [0, 2*mean]; 80K->256K would need r=2.28).
#
# WORKLOAD_MODE=fixed keeps the old single-length behaviour. It is still useful --
# a fixed shape isolates one variable and is the right thing for A/B tuning -- but
# it is not the customer's workload.
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

# --- Which server are we pointed at? -------------------------------------------
# `vllm bench serve --backend vllm` speaks vLLM's /v1/completions and asks for
# per-token timing extensions that a generic OpenAI-compatible server does not have.
# Point this at SGLang, TGI, or a hosted endpoint and it fails at the first request.
# Override for those:
#     BENCH_BACKEND=openai-chat  BENCH_EXTRA_ARGS="--endpoint /v1/chat/completions"
# --ignore-eos is likewise a vLLM extension. It exists so OSL is EXACT: without it the
# model stops when it wants to and the measured TPOT is an average over a length we did
# not choose, which makes the 50 ms/token verdict meaningless. Keep it on vLLM. On a
# server that rejects it, set IGNORE_EOS=0 and treat the OSL -- and therefore the TPOT
# -- as approximate; say so when reporting, do not quote it as the sheet's number.
BENCH_BACKEND="${BENCH_BACKEND:-vllm}"
IGNORE_EOS="${IGNORE_EOS:-1}"
read -r -a BENCH_EXTRA <<< "${BENCH_EXTRA_ARGS:-}"
_eos_arg=(); [ "$IGNORE_EOS" = "1" ] && _eos_arg=(--ignore-eos)

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

# --- Workload shape -------------------------------------------------------------
#   dist  : generate a right-skewed ISL distribution (mean = the sheet's avg, p99 =
#           the sheet's context window) and drive it via --dataset-name custom.
#           This is the customer's stated workload.
#   fixed : one ISL, equal to the mean. Old behaviour; good for A/B tuning, but it
#           silently never exercises the 256K/1M window.
WORKLOAD_MODE="${WORKLOAD_MODE:-dist}"
# p99 ISL as a fraction of the context window. 1.0 = the p99 request fills the
# window, which is the reading under which "256K context" is a claim worth testing.
TAIL_FRAC="${TAIL_FRAC:-1.0}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/workloads}"
mkdir -p "$WORKLOAD_DIR"

# --- Sampling error, and how to shrink it -----------------------------------
# A lognormal with these parameters has CV 0.64 (256K row) / 1.08 (1M row), so the
# realised mean of ONE sample carries a standard error of CV/sqrt(n): 4.0% at n=256 and
# 9.5% at n=128. Measured over 12 seeds the realised mean spanned 14.2% and 35.4%
# respectively. That is inherent to drawing few samples from a heavy tail.
#
# RESAMPLE_PER_ITER=1 draws a FRESH sample each iteration (seed = SEED_BASE + iter-1),
# so N iterations pool to n*N and the error falls as 1/sqrt(N): 10 iterations take 4.0%
# -> 1.3% and 9.5% -> 3.0%. That is what benchmark_avg_80K_ten.sh does.
#
# RESAMPLE_PER_ITER=0 (the default) reuses ONE sample across iterations. Then the
# iterations measure server variance on a FIXED workload -- which is the right thing
# when A/B testing a code change, and the wrong thing when characterising the
# customer's average, because repeating the same sample cannot reduce its sampling
# error no matter how many times you run it.
RESAMPLE_PER_ITER="${RESAMPLE_PER_ITER:-0}"
SEED_BASE="${SEED_BASE:-0}"

# PREREQUISITE, and it is not optional: the SERVER must have been started with
# --max-model-len >= the context window of every scenario you run (262144, or
# 1048576 for the 1M row). GLM-5.2's config declares max_position_embeddings
# 1048576, so the model supports it, but vLLM sizes the server from --max-model-len
# and will reject a longer request with a 400 rather than truncate it. A rejected
# request does not appear in the latency stats -- it appears as a smaller "completed"
# count -- so this failure mode reads as a suspiciously good result, not as an error.
# slo_report.py prints `completed`; check it against num-prompts.

# Scenarios: "label:isl:osl:context-window:concurrency-list".
#   isl            = the sheet's stated AVERAGE input length
#   context-window = the sheet's stated window; the generated p99 lands here
# Concurrency lists are chosen to fit the KV pool (see the capacity note); override
# with SCENARIOS=... to push past it.
#
# The concurrency lists are TIGHTER than the fixed-ISL ones were, deliberately. With
# a skewed distribution the KV charge is per ACTUAL length, so a cell that holds a
# few p99 requests costs far more than con x mean. At 80K/256K, con=128 is 430 GiB at
# the mean but 1,404 GiB if the in-flight set skews long -- past a single node's ~930
# GiB. con=256 does not fit either way and is left out rather than scheduled to OOM.
DEFAULT_SCENARIOS="\
256k-ctx:80000:1024:262144:16 32 64|\
1m-ctx:200000:1024:1048576:8 16 32"
IFS='|' read -ra SCENARIOS <<< "${SCENARIOS:-$DEFAULT_SCENARIOS}"

ITERS="${SLO_ITERS:-1}"
RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/slo_json}"
mkdir -p "$RESULT_DIR"

echo "=============================================================="
echo " Customer SLO benchmark -- GLM-5.2 / MI355X"
echo "   model        : $MODEL_PATH"
echo "   SLO          : TTFT <= ${SLO_TTFT_MS} ms avg, TPOT <= ${SLO_TPOT_MS} ms avg"
echo "   per-rank tgt : prefill ${TGT_PREFILL_TOK_S} tok/s, decode ${TGT_DECODE_TOK_S} tok/s (DP=${DP_RANKS})"
echo "   workload     : ${WORKLOAD_MODE} (dist = skewed ISL, p99 at ${TAIL_FRAC} x window)"
echo "   prefix       : ${PREFIX_FRAC} of ISL shared (agentic reuse)"
echo "   arrival      : Poisson, burstiness ${BURSTINESS}"
if [ "$RESAMPLE_PER_ITER" = "1" ]; then
echo "   sampling     : FRESH draw per iteration, seeds ${SEED_BASE}..$(( SEED_BASE + ITERS - 1 ))"
echo "                  (${ITERS} iters pool to n x ${ITERS}; SE falls as 1/sqrt(iters))"
else
echo "   sampling     : ONE fixed draw, seed ${SEED_BASE}, reused across ${ITERS} iteration(s)"
echo "                  (iters measure server variance, NOT sampling error)"
fi
echo "=============================================================="

# ---------------------------------------------------------------------------
# Shape warmup. Without this the first measured cell of a shape absorbs residual
# JIT and reports a wildly inflated TPOT (observed 302 ms vs ~89 ms steady state).
# Deliberately at low concurrency and few prompts so it is cheap.
# ---------------------------------------------------------------------------
# Caveat in dist mode: this warms the MEAN shape, not the p99 one. The first request
# that lands near the context window will therefore still pay some one-off cost. That
# is left as-is rather than warming at 262K/1M, which would cost more than the
# measurement itself; it shows up as a fat p99 TTFT on iter 1, so run SLO_ITERS>=2 and
# read iter 2 if the tail looks anomalous.
warmup_shape() {
    local isl=$1 osl=$2 pfx=$3
    echo "[WARMUP] isl=$isl osl=$osl prefix=$pfx"
    timeout "${WARMUP_TIMEOUT:-3600}" vllm bench serve \
        --model "$MODEL_PATH" --backend "$BENCH_BACKEND" --host "$HOST" --port "$BENCHMARK_PORT" \
        --dataset-name random \
        --random-input-len "$isl" --random-output-len "$osl" --random-prefix-len "$pfx" \
        --num-prompts 4 --max-concurrency 2 --request-rate inf \
        "${_eos_arg[@]}" ${BENCH_EXTRA[@]+"${BENCH_EXTRA[@]}"} \
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
    local label=$1 isl=$2 osl=$3 con=$4 pfx=$5 iter=$6 wl=$7

    local slo_lat_s
    slo_lat_s=$(python3 -c "print(($SLO_TTFT_MS + $osl*$SLO_TPOT_MS)/1000.0)")
    local rate
    rate=$(python3 -c "print(round($con/$slo_lat_s, 4))")

    # Enough prompts that the measurement is not dominated by ramp-in/ramp-out.
    # 4x concurrency, floored at 32, capped so a 200K cell stays affordable.
    local prompts=$(( con * 4 )); [ "$prompts" -lt 32 ] && prompts=32
    [ "$prompts" -gt "${MAX_PROMPTS:-512}" ] && prompts="${MAX_PROMPTS:-512}"

    # Timeout scales with the work: prompts * (isl+osl) tokens, assuming a
    # pessimistic floor rate, plus slack. In dist mode `isl` is the MEAN, and a
    # right-skewed sample does materially more work than mean*n suggests once the
    # long tail lands -- sizing on the mean produces timeouts that look like server
    # stalls. Size on the actual token count in the JSONL instead.
    local tmo work
    if [ "$WORKLOAD_MODE" = "dist" ] && [ -s "$wl" ]; then
        work=$(python3 -c "
import json
n=$prompts; t=0; c=0
for line in open('$wl'):
    if c>=n: break
    d=json.loads(line)
    t += int(d.get('input_tokens') or len(d['prompt'])//3)
    t += int(d.get('output_tokens') or $osl)
    c += 1
# oversampled if the file has fewer rows than requested prompts
print(int(t*n/c) if c else 0)")
    else
        work=$(( prompts * (isl + osl) ))
    fi
    tmo=$(python3 -c "print(int(900 + $work/3000.0))")

    # Dataset selection. In dist mode the lengths (and the shared prefix) are already
    # baked into the JSONL, so --random-* would be ignored anyway; passing them would
    # be a lie in the log. --custom-output-len -1 tells CustomDataset to honour the
    # per-request "output_tokens" field instead of overriding it, and
    # --skip-chat-template stops it prepending role tokens that would shift every
    # length gen_workload.py just placed.
    local -a ds_args
    if [ "$WORKLOAD_MODE" = "dist" ]; then
        ds_args=(--dataset-name custom --dataset-path "$wl"
                 --custom-output-len -1 --skip-chat-template)
    else
        ds_args=(--dataset-name random
                 --random-input-len "$isl" --random-output-len "$osl"
                 --random-prefix-len "$pfx")
    fi

    local jf="$RESULT_DIR/${label}_con${con}_iter${iter}.json"
    echo "[RUNNING] $label mode=$WORKLOAD_MODE isl=$isl osl=$osl con=$con prefix=$pfx rate=${rate}/s prompts=$prompts (timeout ${tmo}s)"

    timeout "$tmo" vllm bench serve \
        --model "$MODEL_PATH" --backend "$BENCH_BACKEND" --host "$HOST" --port "$BENCHMARK_PORT" \
        "${ds_args[@]}" \
        --num-prompts "$prompts" \
        --max-concurrency "$con" \
        --request-rate "$rate" \
        --burstiness "$BURSTINESS" \
        "${_eos_arg[@]}" ${BENCH_EXTRA[@]+"${BENCH_EXTRA[@]}"} \
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

# ---------------------------------------------------------------------------
# Generate (or reuse) one workload sample. Echoes the path on stdout; empty on
# failure. The filename carries the seed so two seeds cannot collide in the cache --
# without that suffix the `[ -s ]` reuse check would hand back seed 0's file for
# every iteration and the pooling below would average one sample with itself.
# ---------------------------------------------------------------------------
gen_wl() {
    local label=$1 isl=$2 osl=$3 ctxwin=$4 nprompt=$5 seed=$6 conlist=$7
    local wl="$WORKLOAD_DIR/${label}_s${seed}.jsonl"
    if [ -s "$wl" ]; then
        echo "[REUSE] $label seed=$seed workload $wl" >&2
        echo "$wl"; return 0
    fi
    python3 "$(dirname "$0")/gen_workload.py" \
        --mean-isl "$isl" --context-window "$ctxwin" --tail-frac "$TAIL_FRAC" \
        --osl "$osl" --num-prompts "$nprompt" --prefix-frac "$PREFIX_FRAC" \
        --tokenizer "$MODEL_PATH" --trust-remote-code \
        --concurrency "$conlist" --seed "$seed" \
        --out "$wl" >&2 2>&1
    [ -s "$wl" ] && echo "$wl"
}

for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r label isl osl ctxwin conlist <<< "$scenario"
    pfx=$(python3 -c "print(int($isl*$PREFIX_FRAC))")
    wl=""

    echo; echo "########## scenario $label  (avg isl=$isl osl=$osl window=$ctxwin) ##########"

    if [ "$WORKLOAD_MODE" = "dist" ]; then
        # Size the sample for the LARGEST concurrency in the list and reuse it across
        # the cells of one iteration. Regenerating per CELL would change the sample
        # between concurrencies and make the concurrency sweep apples-to-oranges --
        # that comparison needs the workload held fixed. Regenerating per ITERATION
        # is the opposite case and is what RESAMPLE_PER_ITER turns on.
        maxcon=0; for c in $conlist; do [ "$c" -gt "$maxcon" ] && maxcon=$c; done
        nprompt=$(( maxcon * 4 )); [ "$nprompt" -lt 32 ] && nprompt=32
        [ "$nprompt" -gt "${MAX_PROMPTS:-512}" ] && nprompt="${MAX_PROMPTS:-512}"
        if [ "$RESAMPLE_PER_ITER" != "1" ]; then
            wl=$(gen_wl "$label" "$isl" "$osl" "$ctxwin" "$nprompt" "$SEED_BASE" "$conlist")
            if [ -z "$wl" ]; then
                echo "[SKIP] $label: workload generation failed -- see ${LOG}.log" \
                    | tee -a "${LOG}.log" "${LOG}_stalls.log"
                continue
            fi
        fi
        # In dist mode the prefix is inside the JSONL; the vLLM-side prefix arg is
        # unused. Zero it so the warmup does not build a second, different one.
        pfx=0
    fi

    warmup_shape "$isl" "$osl" "$pfx"
    for iter in $(seq 1 "$ITERS"); do
        if [ "$WORKLOAD_MODE" = "dist" ] && [ "$RESAMPLE_PER_ITER" = "1" ]; then
            seed=$(( SEED_BASE + iter - 1 ))
            wl=$(gen_wl "$label" "$isl" "$osl" "$ctxwin" "$nprompt" "$seed" "$conlist")
            if [ -z "$wl" ]; then
                echo "[SKIP] $label iter=$iter: workload generation failed" \
                    | tee -a "${LOG}.log" "${LOG}_stalls.log"
                continue
            fi
            echo "[SAMPLE] $label iter=$iter seed=$seed -> $(basename "$wl")"
        fi
        for con in $conlist; do
            run_cell "$label" "$isl" "$osl" "$con" "$pfx" "$iter" "$wl" >/dev/null
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
