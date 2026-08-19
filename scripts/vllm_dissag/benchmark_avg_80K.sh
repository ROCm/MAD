#!/bin/bash
# ---------------------------------------------------------------------------
# Customer row 1: avg ISL 80K / OSL 1K, 256K context.
#
# RUN IT (EP16, con 16 & 32 -- the customer operating point). GLM_PERSIST_GATE=0 is
# REQUIRED at this context on an aiter-#4076-fixed image (else the >=65K prefill aborts;
# see models.yaml + docs/niah_longcontext_glm52_mi308x.html):
#
#   PREFILL=node0,node1 DECODE=node2,node3 EP=16 DP_RANKS=16 \
#   MODEL_NAME=GLM-5.2-FP8 MODEL_PATH=/models/GLM-5.2-FP8 IMAGE=...v027-bnxt238 \
#   GLM_PERSIST_GATE=0 GLM_MAX_MODEL_LEN=280000 \
#   GLM_PREFILL_BATCHED_TOKENS=16384 KV_CACHE_MEMORY_BYTES=45000000000 \
#   BENCHMARK_SCRIPT_FILE=benchmark_avg_80K.sh bash launch_disagg_skyriver.sh
#
# Different concurrencies: override the 5th colon-field of SCENARIOS (space-separated).
#   SCENARIOS="256k-ctx:80000:1024:262144:16 32 64"   # add 64
#   SCENARIOS="256k-ctx:80000:1024:262144:8"          # single point
# Each cell sends con*4 requests (con=16 -> 64, con=32 -> 128), max-concurrency=con.
#
# TWO PASSES: one warm-up, one measured
# --------------------------------------
# The sheet gives an AVERAGE input length and a context window. Those pin a right-skewed
# lognormal (see gen_workload.py); this wrapper draws ONE sample from it and runs the
# whole concurrency list over that sample TWICE:
#
#   pass 0  warm-up, results DISCARDED -- absorbs Triton/aiter JIT for shapes the engine
#           has not seen, the decode cudagraph captures, the MoRI dispatch buffers, and
#           the first fill of the shared agentic prefix cache. Measured on this platform
#           that one-off cost inflated TPOT from ~89 ms steady state to 302 ms. A
#           single-pass run reports 302 ms as the answer.
#   pass 1  measured, at the real concurrencies, judged against the SLO.
#
# Both passes use the SAME trace -- same seed, same JSONL. That is the point. Warming
# with a different draw warms the wrong prompt lengths and leaves the shared agentic
# prefix cold, which is precisely the thing PREFIX_FRAC exists to exercise.
#
# WHAT ONE SAMPLE DOES NOT TELL YOU, AND HOW TO REPORT IT HONESTLY
# ------------------------------------------------------------------
# One sample does not pin the mean:
#
#     CV = sqrt(exp(sigma^2)-1) = 0.637 at sigma 0.5833
#     SE(mean)/mean = CV/sqrt(n) = 4.0% at n=256
#
# and measured over 12 seeds the realised mean actually spanned 72,884-84,222 -- 14.2%.
# So the mean of this run is a property of THIS SEED, not of the workload.
#
# That is REPORTED, not hidden. gen_workload.py writes <trace>.meta.json with the
# achieved mean, its deviation from target and the seed, and pool_workload.py recomputes
# the percentiles from the raw lengths. Quote it that way:
#
#     "our trace averaged 78,412 tokens, 2.0% below the 80,000 target, p99 262,144"
#
# NOT "we ran the customer's 80K average" -- that second claim is the one that gets
# challenged, and it cannot be defended from a single seed.
#
# If the target must be pinned, the mechanism is still here and it is the only one that
# works: ten DISTINCT seeds pool to n=2,560 and the error falls as 1/sqrt(10) to 1.3%.
#
#     SLO_ITERS=10 RESAMPLE_PER_ITER=1 ./benchmark_avg_80K.sh
#
# It costs ten times the wall clock, which is why it is not the default. (Reaching 1.3%
# inside a single run would need n=1,014 requests at ~80K tokens each -- more than
# MAX_PROMPTS and far more wall clock than ten smaller runs, for the same answer. The
# other alternative, stratified inverse-CDF placement, pins the mean without any of this
# but caps the sample at the (n-0.5)/n quantile, so p99 would reach only ~225K instead of
# 262,144 -- and the window is the number the customer actually stated.)
#
# This is a THIN WRAPPER. Every measurement decision (Little's-Law request rate, the
# goodput thresholds, the timeout sizing, the warm-up routing) lives in
# benchmark_customer_slo.sh and has exactly one implementation. Two copies of that logic
# would drift, and the drift would show up as an unexplained delta between the 80K and
# 200K rows.
#
# TWO PREREQUISITES ON MI308X, BOTH SET AT LAUNCH, NOT HERE
# ----------------------------------------------------------
#   export GLM_MAX_MODEL_LEN=262144   # models.yaml defaults to 65536; without this
#                                     # every request over 65,536 is 400'd, which
#                                     # presents as a SUSPICIOUSLY GOOD result (fewer
#                                     # completed requests, better latencies) not as
#                                     # an error. slo_report.py prints `completed` --
#                                     # check it against num-prompts.
#   EP16 (2P/2D), with DP_RANKS=16 and GPU_MEMORY_UTILIZATION=0.72
#                                     # A 262,144-token p99 request is 11.65 GiB of KV.
#                                     # Per-rank pool is 35.71 GiB at EP8 vs 64.19 at
#                                     # EP16, so this row is 2.5-3.1 requests/rank at
#                                     # EP8 and 4.9-5.5 at EP16. EP16 is NECESSARY
#                                     # here, not merely faster. Leaving DP_RANKS at
#                                     # its default 8 doubles every per-rank figure.
# Run probe_context_ceiling.sh first to confirm 262144 actually boots.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One scenario only: label:isl:osl:context-window:concurrency-list.
# con 16 32 = customer operating point; each cell sends con*4 requests (64, 128).
# The 2026-08-18 "gloo DP-sync cascade at con>=32" was ROOT-CAUSED (2026-08-19) to the
# obsolete persistent-kernel-gate patcher forcing the illegal non-persistent MLA path
# (asm_mla.cu:945) at long context -- NOT concurrency. Run with GLM_PERSIST_GATE=0 (this
# image's aiter has the #4076 fix) and it survives. Override the scenario via SCENARIOS=.
export SCENARIOS="${SCENARIOS:-256k-ctx:80000:1024:262144:16 32}"

# One warm-up pass over the real trace, then one measured pass. See the header.
export SLO_WARMUP_ITERS="${SLO_WARMUP_ITERS:-1}"
export SLO_ITERS="${SLO_ITERS:-1}"

# 0 = warm-up and measurement share one trace, which is what makes the warm-up warm the
# right thing. Only set 1 alongside SLO_ITERS>1, where pooling distinct seeds is the
# point; the engine then warms on seed SEED_BASE, the same trace iteration 1 measures.
export RESAMPLE_PER_ITER="${RESAMPLE_PER_ITER:-0}"
export SEED_BASE="${SEED_BASE:-0}"

# Give it a distinct log/result root so a 200K run in the same job cannot overwrite it.
# /run_logs is the ONLY host-backed path in the container (-v ${LOG_PATH}:/run_logs).
# Anything written elsewhere lands on the container's writable overlay, survives the
# run, and is destroyed with the container -- results silently gone.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg80K}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg80K_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg80K_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

echo "=============================================================="
echo " avg 80K / 1K, 256K context"
echo " ${SLO_WARMUP_ITERS} warm-up pass (discarded) + ${SLO_ITERS} measured pass"
if [ "${RESAMPLE_PER_ITER}" = "1" ]; then
echo " seeds ${SEED_BASE}..$(( SEED_BASE + SLO_ITERS - 1 )); pooled n = 256 x ${SLO_ITERS}"
echo " expected SE of the mean 4.0% -> $(python3 -c "print('%.1f' % (4.0/($SLO_ITERS**0.5)))")%"
else
echo " one fixed trace, seed ${SEED_BASE}; its mean carries ~4.0% standard error"
echo " REPORT the .meta.json deviation -- do not claim the 80,000 target was hit."
echo " To pin it: SLO_ITERS=10 RESAMPLE_PER_ITER=1 $(basename "$0")"
fi
echo "=============================================================="

# Recover the real status: `| tee` in the callee always exits 0, and a green exit on a
# run that never produced a result is the failure mode that wastes a day.
set +e
bash "$HERE/benchmark_customer_slo.sh"
rc=$?
set -e

# Summarise the trace that was actually served. Runs even if the benchmark failed -- the
# workload that WAS generated still describes what was offered, and knowing that is worth
# more than a bare non-zero exit code. With the default single seed this restates the
# .meta.json; with SLO_ITERS>1 + RESAMPLE_PER_ITER=1 it pools for real, recomputing
# percentiles from raw lengths rather than averaging summaries.
# Derive the glob from the scenario LABEL rather than hardcoding "256k-ctx". gen_wl()
# names its files "<label>_s<seed>.jsonl", so a hardcoded pattern would silently match
# nothing the moment anyone overrode SCENARIOS -- and "No workload files matched" at the
# very end of a job is a bad place to learn that.
echo
IFS='|' read -ra _scn <<< "$SCENARIOS"
for _s in "${_scn[@]}"; do
    _label="${_s%%:*}"
    python3 "$HERE/pool_workload.py" \
        --glob "$WORKLOAD_DIR/${_label}_s*.jsonl" \
        --label "avg 80K / 256K context [$_label]" \
        --out "${LOG}_pooled_workload_${_label}.json" || true
done

exit $rc
