#!/bin/bash
# Customer-facing SLO benchmark for GLM-5.2 / HY4.
#
# Written against the customer's MI355X sheet; the defaults below are retargeted to
# MI308X (gfx942, 192 GB/GPU), which is the platform this branch runs. Every
# platform-dependent number is an env override -- see PLATFORM CONSTANTS.
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
#   prefill        34,000 tok/s per rank   (customer's MI355X ask)
#   decode            670 tok/s per rank   (customer's MI355X ask)
#   framework      vLLM only
#
# The two throughput rows are the customer's numbers FOR MI355X and are reported here
# unchanged, because a target is not a measurement -- rescaling them to MI308X would
# quietly replace what the customer asked for with what we expect to deliver. They are
# reported as a ratio, not gated. TTFT/TPOT are latency SLOs and are platform-neutral,
# so those two ARE gated.
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
# CAPACITY REALITY CHECK -- READ BEFORE SETTING CONCURRENCY OR A CONTEXT WINDOW
# ------------------------------------------------------------------------------
# GLM-5.2 MLA KV is kv_lora_rank(512) + qk_rope_head_dim(64) = 576 elem/token/layer,
# FP8, 78 layers. The arithmetic gives 43.88 KiB/token; the value MEASURED on MI308X
# from the engine's own `GPU KV cache size` line is 46.58 KiB/token, 6.2% higher --
# vLLM rounds the per-layer allocation up to whole blocks. Use the measured one: the
# arithmetic value makes every ceiling below look 6% roomier than it is.
#       28,672 tok -> 1.27 GiB   80,000 tok -> 3.55 GiB
#      200,000 tok -> 8.89 GiB  262,144 tok -> 11.65 GiB  1,048,576 tok -> 46.58 GiB
#
# On MI308X (192 GB/GPU) the usable per-rank KV pool, measured at boot, is
#       EP8  (1P/1D, weights 107.33 GiB/GPU, util 0.80): 35.71 GiB per rank, 8 ranks
#       EP16 (2P/2D, weights  65.12 GiB/GPU, util 0.72): 64.19 GiB per rank, 16 ranks
# EP16 wins twice over: the MoE experts shard 16 ways instead of 8, which FREES
# 42 GiB/GPU of weights, and there are twice as many GPUs holding KV. Aggregate
# decode-tier pool is therefore ~286 GiB at EP8 and ~1,027 GiB at EP16.
#
# Node-wide KV-bound concurrency ceiling, at the MEAN length:
#       ISL  28,672 -> EP8 ~224   EP16 ~806
#       ISL  80,000 -> EP8  ~80   EP16 ~289
#       ISL 200,000 -> EP8  ~32   EP16 ~116
# and at a request that FILLS the window, which is what a p99 request does:
#       262,144 tok -> EP8 2.5-3.1 per rank   EP16 4.9-5.5 per rank
#     1,048,576 tok -> EP8 0.0-0.8 per rank   EP16 0.6-1.4 per rank
# The ranges are the two defensible treatments of the chunked-prefill workspace,
# which scales with max-model-len; they are not error bars. Read them as: 262K is
# comfortable at EP16 and tight at EP8, so **EP16 is necessary for the 256K row, not
# merely preferable**; and **1M does not fit on 4 nodes in either topology** -- the
# 1m-ctx scenario is retained for completeness and is expected to fail here.
# probe_context_ceiling.sh boots each rung and settles this empirically.
#
# SECOND PREREQUISITE, EASY TO MISS: models.yaml caps the server at
# `--max-model-len ${GLM_MAX_MODEL_LEN:-65536}`. The 256K row needs
# GLM_MAX_MODEL_LEN=262144 exported at launch; without it every prompt above 65,536
# is rejected with a 400 (see the note below on why that reads as a GOOD result).
#
# This script does not silently truncate: it runs what you ask and reports what
# happened. The default concurrency lists are sized to fit so that a failure is a
# real failure and not an OOM you could have predicted with arithmetic.
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
# --- PLATFORM CONSTANTS --------------------------------------------------------
# Per-rank throughput targets. These are the customer's MI355X asks, kept verbatim so
# the report shows distance-from-target rather than distance-from-expectation. They
# are reported, never gated. Override when benchmarking against a different ask.
TGT_PREFILL_TOK_S="${TGT_PREFILL_TOK_S:-34000}"
TGT_DECODE_TOK_S="${TGT_DECODE_TOK_S:-670}"

# --- Number of DP ranks, to convert aggregate throughput into per-rank. ---
# The sheet's targets are PER RANK and vLLM reports an AGGREGATE, so getting this
# wrong scales every throughput verdict by the DP degree in whichever direction hurts.
# 8 is EP8 (1P/1D). **At EP16 (2P/2D) the decode tier has 16 ranks -- leaving this at
# 8 overstates every per-rank number by exactly 2x.** Set DP_RANKS=16 for EP16 runs.
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

# --- Sampling error: what one sample can and cannot tell you -----------------
# A lognormal with these parameters has CV 0.64 (256K row) / 1.08 (1M row), so the
# realised mean of ONE sample carries a standard error of CV/sqrt(n): 4.0% at n=256 and
# 9.5% at n=128. Measured over 12 seeds the realised mean spanned 14.2% and 35.4%
# respectively. That is inherent to drawing few samples from a heavy tail, and it does
# NOT go away by default here.
#
# The default configuration does not try to make it go away. It REPORTS it: every run
# writes <workload>.meta.json with the achieved mean, its deviation from target, and the
# seed, and pool_workload.py recomputes the percentiles from the raw lengths. So the
# claim we make is "this trace averaged 78,412 tokens, 2.0% below the 80,000 target,
# p99 262,144" -- a measured statement about the trace that was actually served, not an
# unqualified claim to have hit the target.
#
# If you DO want the target pinned, the mechanism is still here and it is the only one
# that works: raise SLO_ITERS and set RESAMPLE_PER_ITER=1, which draws a fresh sample
# per measured iteration (seed = SEED_BASE + iter - 1) so they pool to n*N and the error
# falls as 1/sqrt(N) -- ten measured samples take 4.0% -> 1.3% and 9.5% -> 3.0%. It
# costs ten times the wall clock, which is why it is not the default.
#
# RESAMPLE_PER_ITER=0 (the default) reuses ONE sample across iterations. That is
# REQUIRED for the warm-up iteration to do its job: warming the caches with a different
# trace than the one you then measure warms the wrong thing. It also makes repeated
# measured iterations a measure of server variance on a fixed workload, which is the
# right question when A/B testing a code change.
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
#
# ON THIS BRANCH THAT PREREQUISITE IS NOT MET BY DEFAULT. models.yaml lines 441/447
# pass `--max-model-len ${GLM_MAX_MODEL_LEN:-65536}`, so an unconfigured server caps
# at 65,536 and *every* request in the 256k-ctx row above that is 400'd. Launch with
#     export GLM_MAX_MODEL_LEN=262144
# and confirm the server actually booted at it -- raising max-model-len also grows the
# chunked-prefill workspace, so it can OOM at boot rather than fail at request time.
# probe_context_ceiling.sh walks the ladder and reports the highest rung that boots.

# Scenarios: "label:isl:osl:context-window:concurrency-list".
#   isl            = the sheet's stated AVERAGE input length
#   context-window = the sheet's stated window; the generated p99 lands here
# Concurrency lists are chosen to fit the KV pool (see the capacity note); override
# with SCENARIOS=... to push past it.
#
# The concurrency lists are TIGHTER than the fixed-ISL ones were, deliberately. With
# a skewed distribution the KV charge is per ACTUAL length, so a cell that holds a
# few p99 requests costs far more than con x mean. At 80K/256K on MI308X EP16 the
# aggregate pool is ~1,027 GiB: con=64 is 227 GiB at the mean but 745 GiB if the
# in-flight set skews long -- which fits, while con=128 (1,490 GiB skewed) does not.
# At EP8 the pool is ~286 GiB and even con=32 skews past it, so the 256k row is an
# EP16 row. Sized from measured pools, not from the MI355X sheet's 930 GiB.
DEFAULT_SCENARIOS="\
256k-ctx:80000:1024:262144:16 32 64|\
1m-ctx:200000:1024:1048576:8 16 32"
IFS='|' read -ra SCENARIOS <<< "${SCENARIOS:-$DEFAULT_SCENARIOS}"

# --- Warm-up pass, then measurement -------------------------------------------
# Default shape of a run is TWO passes over the scenario, not one:
#
#   pass 0  warm-up   -- the full concurrency list, the real trace, results DISCARDED
#   pass 1  measured  -- the same trace again, results kept and judged
#
# One pass is not enough and the reason is not subtle. The first pass over a shape
# absorbs every one-off cost in the stack: Triton/aiter JIT for the shapes it has not
# seen, the decode cudagraph captures, the MoRI dispatch buffers, and the first fill of
# the prefix cache. Measured on this platform that inflated TPOT from ~89 ms steady
# state to 302 ms. A single-pass run reports that as the answer.
#
# The warm-up uses the SAME trace as the measurement -- same seed, same file. Warming
# with a different draw warms the wrong prompt lengths and, worse, leaves the shared
# agentic prefix cold, which is the thing PREFIX_FRAC exists to exercise. That is also
# why RESAMPLE_PER_ITER must stay 0 for this to be meaningful; see the note above.
#
# Set SLO_WARMUP_ITERS=0 for a cold-start measurement (report it as such), or >1 if a
# shape needs more than one pass to settle. Warm-up results go to $RESULT_DIR/_warmup
# so slo_report.py -- which globs $RESULT_DIR/*.json -- can never pick them up.
WARMUP_ITERS="${SLO_WARMUP_ITERS:-1}"
ITERS="${SLO_ITERS:-1}"
RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/slo_json}"
WARMUP_RESULT_DIR="$RESULT_DIR/_warmup"
mkdir -p "$RESULT_DIR" "$WARMUP_RESULT_DIR"

echo "=============================================================="
echo " Customer SLO benchmark -- GLM-5.2"
echo "   model        : $MODEL_PATH"
echo "   SLO          : TTFT <= ${SLO_TTFT_MS} ms avg, TPOT <= ${SLO_TPOT_MS} ms avg"
echo "   per-rank tgt : prefill ${TGT_PREFILL_TOK_S} tok/s, decode ${TGT_DECODE_TOK_S} tok/s (DP=${DP_RANKS})"
echo "   workload     : ${WORKLOAD_MODE} (dist = skewed ISL, p99 at ${TAIL_FRAC} x window)"
echo "   prefix       : ${PREFIX_FRAC} of ISL shared (agentic reuse)"
echo "   arrival      : Poisson, burstiness ${BURSTINESS}"
echo "   passes       : ${WARMUP_ITERS} warm-up (discarded) + ${ITERS} measured"
if [ "$WARMUP_ITERS" -eq 0 ]; then
echo "                  !! SLO_WARMUP_ITERS=0 -- the first measured cell of each shape"
echo "                     absorbs JIT/cudagraph/prefix-cache cost. Report as COLD."
fi
if [ "$RESAMPLE_PER_ITER" = "1" ]; then
echo "   sampling     : FRESH draw per measured iteration, seeds ${SEED_BASE}..$(( SEED_BASE + ITERS - 1 ))"
echo "                  (${ITERS} iters pool to n x ${ITERS}; SE falls as 1/sqrt(iters))"
else
echo "   sampling     : ONE fixed draw, seed ${SEED_BASE}, warm-up and measurement share it"
echo "                  (sampling error is REPORTED via .meta.json, not averaged away)"
fi
echo "=============================================================="

# ---------------------------------------------------------------------------
# Shape warmup. Without this the first measured cell of a shape absorbs residual
# JIT and reports a wildly inflated TPOT (observed 302 ms vs ~89 ms steady state).
# Deliberately at low concurrency and few prompts so it is cheap.
# ---------------------------------------------------------------------------
# Caveat in dist mode: this warms the MEAN shape, not the p99 one, at con=2 with 4
# prompts. It costs seconds and cannot warm the cudagraph batch sizes for con=64, the
# long-tail prefill path, or the shared prefix. That is what the warm-up PASS below is
# for -- the full concurrency list over the real trace, results discarded. This cheap
# poke stays because it fails fast: if the server cannot serve 4 prompts, finding out
# here beats finding out an hour into a 262K sweep.
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
    # 8th arg selects where the result JSON lands: $RESULT_DIR for a measured cell,
    # $WARMUP_RESULT_DIR for a warm-up one. slo_report.py globs the former only, so
    # routing is what keeps warm-up numbers out of the verdict -- not a filename
    # convention someone can later break by renaming a file.
    local rdir=${8:-$RESULT_DIR}
    # And the log tag: benchmark_parser.py anchors on "[RUNNING]", so a warm-up cell
    # must NOT carry that tag or every downstream parse silently averages the JIT pass
    # into the result.
    local tag="[RUNNING]"; [ "$rdir" = "$WARMUP_RESULT_DIR" ] && tag="[WARMUP-PASS]"

    # Request rate. Default = Little's-Law service rate (con / SLO_latency): a SERVICE
    # test -- arrivals throttled to the SLO offered load, so TTFT is queue-free and the
    # gate means "meets SLO at the customer's rate". Override SLO_REQUEST_RATE=inf for a
    # SATURATION snapshot: all `prompts` fired at once. With prompts==con (set
    # SLO_PROMPTS_PER_CON=1) that is exactly `con` concurrent requests, no queue, giving
    # true con-concurrent TTFT/TPOT/throughput fast -- but report it as SATURATION, not
    # the SLO gate (queue wait / batch contention differ from the throttled service test).
    local slo_lat_s rate
    slo_lat_s=$(python3 -c "print(($SLO_TTFT_MS + $osl*$SLO_TPOT_MS)/1000.0)")
    rate=$(python3 -c "print(round($con/$slo_lat_s, 4))")
    [ -n "${SLO_REQUEST_RATE:-}" ] && rate="${SLO_REQUEST_RATE}"

    # Enough prompts that the measurement is not dominated by ramp-in/ramp-out.
    # SLO_PROMPTS_PER_CON x concurrency (default 4), floored at 16, capped so a heavy
    # long-context cell stays affordable. Set SLO_PROMPTS_PER_CON=2 for heavy rows: still
    # >=32 samples at con=16 (a solid median) but ~2x faster than 4x. It does NOT change
    # KV pressure (that is max-concurrency); it only changes total offered + wall clock.
    local _ppc="${SLO_PROMPTS_PER_CON:-4}"
    local prompts=$(( con * _ppc ))
    # The 16-prompt floor keeps a THROTTLED service test from being dominated by
    # ramp-in/out. It must NOT apply to a saturation snapshot (SLO_REQUEST_RATE=inf)
    # with _ppc=1: there the whole point is prompts==con so all `con` fire at once and
    # NONE queue (1 req/rank, queue-free TTFT). Flooring 8->16 would re-queue 8 behind
    # the first 8 and silently corrupt exactly the TTFT we set out to measure.
    if [ "${SLO_REQUEST_RATE:-}" != "inf" ] && [ "$prompts" -lt 16 ]; then prompts=16; fi
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

    local jf="$rdir/${label}_con${con}_iter${iter}.json"
    echo "$tag $label mode=$WORKLOAD_MODE isl=$isl osl=$osl con=$con prefix=$pfx rate=${rate}/s prompts=$prompts (timeout ${tmo}s)"

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
        --save-result --result-dir "$rdir" \
        --result-filename "$(basename "$jf")" \
        2>&1 | tee -a "${LOG}.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ]; then
        # A warm-up stall is still worth knowing about -- it usually means the measured
        # pass is about to stall too -- but it is tagged so it cannot be mistaken for a
        # failed measurement.
        local stall_what="measured"
        [ "$rdir" = "$WARMUP_RESULT_DIR" ] && stall_what="warm-up"
        echo "[STALL] $label con=$con ($stall_what) timed out after ${tmo}s" \
            | tee -a "${LOG}.log" "${LOG}_stalls.log"
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
        nprompt=$(( maxcon * ${SLO_PROMPTS_PER_CON:-4} ))
        # Same saturation exception as the measure pass: rate=inf + _ppc=1 must keep
        # nprompt==maxcon (queue-free 1 req/rank), so skip the 16-floor in that mode.
        if [ "${SLO_REQUEST_RATE:-}" != "inf" ] && [ "$nprompt" -lt 16 ]; then nprompt=16; fi
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

    # Warm-up PASSES over the real trace, at the real concurrencies. warmup_shape above
    # is a 4-prompt con=2 poke at the MEAN shape; it costs seconds and cannot warm the
    # cudagraph batch sizes for con=64, the long-tail prefill path, or the shared prefix.
    # This does, at the price of doubling the run -- which is the trade the default makes.
    #
    # Under RESAMPLE_PER_ITER=1 the per-iteration trace does not exist yet at this point,
    # so generate seed SEED_BASE here -- which is exactly the trace measured iteration 1
    # will use, so the warm-up still warms the right thing rather than a neighbouring draw.
    if [ "$WORKLOAD_MODE" = "dist" ] && [ "$RESAMPLE_PER_ITER" = "1" ] \
       && [ "$WARMUP_ITERS" -gt 0 ] && [ -z "$wl" ]; then
        wl=$(gen_wl "$label" "$isl" "$osl" "$ctxwin" "$nprompt" "$SEED_BASE" "$conlist")
    fi

    # Two run shapes, selected by SLO_INTERLEAVE:
    #
    #   interleave (default): for each concurrency, warm it up THEN measure it, before
    #   moving to the next -- warmup8 -> measure8 -> warmup16 -> measure16 -> ...
    #   The batch-N cudagraph is captured immediately before the batch-N measurement, and
    #   -- the reason this is the default now -- if a high concurrency crashes the engine
    #   (the gloo DP-sync cascade seen 2026-08-18), every LOWER concurrency has already been
    #   measured and saved before the crash. A staged run loses them all.
    #   Only valid for the 1-warmup + 1-measured default (RESAMPLE_PER_ITER=0); it shares
    #   the single trace, which is what makes per-con warm-up warm the right thing.
    #
    #   staged (SLO_INTERLEAVE=0, or forced when ITERS>1/RESAMPLE_PER_ITER=1): all warm-up
    #   passes first, then all measured iterations -- the pooling modes need this because a
    #   measured iteration draws a fresh seed and the warm-up must precede the whole pool.
    _interleave="${SLO_INTERLEAVE:-1}"
    if [ "$ITERS" -gt 1 ] || [ "$RESAMPLE_PER_ITER" = "1" ]; then _interleave=0; fi

    if [ "$_interleave" = "1" ]; then
        if [ "$WORKLOAD_MODE" = "dist" ] && [ -z "$wl" ]; then
            echo "[SKIP] $label: no workload" | tee -a "${LOG}.log"
        else
            for con in $conlist; do
                for w in $(seq 1 "$WARMUP_ITERS"); do
                    echo "[WARMUP-PASS] $label con=$con warm $w/$WARMUP_ITERS -- discarded"
                    run_cell "$label" "$isl" "$osl" "$con" "$pfx" "w$w" "$wl" \
                             "$WARMUP_RESULT_DIR" >/dev/null
                    sleep 10
                done
                run_cell "$label" "$isl" "$osl" "$con" "$pfx" "1" "$wl" >/dev/null
                sleep 10
            done
        fi
    else
        for w in $(seq 1 "$WARMUP_ITERS"); do
            if [ "$WORKLOAD_MODE" = "dist" ] && [ -z "$wl" ]; then
                echo "[SKIP] $label warm-up pass $w: no workload" | tee -a "${LOG}.log"
                break
            fi
            echo "[WARMUP-PASS] $label pass $w/$WARMUP_ITERS -- results discarded"
            for con in $conlist; do
                run_cell "$label" "$isl" "$osl" "$con" "$pfx" "w$w" "$wl" \
                         "$WARMUP_RESULT_DIR" >/dev/null
                sleep 10
            done
        done

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
    fi
done

echo; echo "########## VERDICT ##########"
python3 "$(dirname "$0")/slo_report.py" \
    --result-dir "$RESULT_DIR" \
    --slo-ttft-ms "$SLO_TTFT_MS" --slo-tpot-ms "$SLO_TPOT_MS" \
    --target-prefill "$TGT_PREFILL_TOK_S" --target-decode "$TGT_DECODE_TOK_S" \
    --dp-ranks "$DP_RANKS" \
    --csv "${LOG}_slo.csv"
exit $?
