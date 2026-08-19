#!/bin/bash
# ---------------------------------------------------------------------------
# Customer row 1, LIGHTER TAIL VARIANT: avg ISL 80K / OSL 1K, 128K context (not 256K).
#
# WHY THIS VARIANT EXISTS
# -----------------------
# The full customer spec is avg-80K with a 256K (262,144) tail -- see benchmark_avg_80K.sh.
# That full-tail workload has a heavy per-request p99 KV cost (11.64 GiB at 262K) and a
# large prefill-activation + cudagraph footprint sized from GLM_MAX_MODEL_LEN=280000, which
# on EP8 (110 GiB weights/GPU) leaves too little headroom -> boot OOM. EP16 (68 GiB/GPU)
# fits it, but then EP8-vs-EP16 cannot be compared on the SAME config.
#
# This variant caps the tail at 131,072 (128K):
#   * per-request p99 KV halves (262K->131K = 11.64 -> 5.82 GiB)
#   * the window is 131072, so prefill activation + cudagraph shrink enough that BOTH EP8
#     and EP16 boot on ONE identical config -> a fair apples-to-apples topology comparison.
#
# THE TRADE-OFF (state it honestly in any report): this TRUNCATES the customer's stated
# 256K tail. Requests that would have been 131K-262K are clamped at 131K, so this measures
# the mean/p50/p95 operating point -- which DOMINATES the TTFT/TPOT numbers -- but NOT the
# extreme-tail behaviour. Use benchmark_avg_80K.sh (256K tail) for the full-spec SLO on
# EP16; use THIS for the EP8-vs-EP16 comparison and for the lighter operating point.
#
# Mean is still 80,000 (the sheet's stated average); only the tail/window changes. The
# lognormal is re-solved for mean=80000, p99=131072 (a smaller sigma, less skew) -- see
# gen_workload.py. Everything else (two-pass warm-up, Little's-Law rate, goodput gates,
# per-rank targets) is identical to benchmark_avg_80K.sh; this is a thin wrapper that only
# changes the context-window field of SCENARIOS.
#
# RUN IT (con 16, same config both topologies -- GLM_PERSIST_GATE=0 required):
#   # EP16:
#   PREFILL=node0,node1 DECODE=node2,node3 EP=16 DP_RANKS=16 \
#   MODEL_NAME=GLM-5.2-FP8 MODEL_PATH=/models/GLM-5.2-FP8 IMAGE=...v027-bnxt238 \
#   GLM_PERSIST_GATE=0 GLM_MAX_MODEL_LEN=140000 GLM_PREFILL_BATCHED_TOKENS=16384 \
#   BENCHMARK_SCRIPT_FILE=benchmark_avg80k_ctx128k.sh bash launch_disagg_skyriver.sh
#   # EP8: PREFILL=node0 DECODE=node1 EP=8 DP_RANKS=8 ... (same everything else)
#
# Concurrency: default con 16 (2 req/rank at EP8, 1/rank at EP16 -- an equal-OFFERED-load
# comparison). Override via SCENARIOS=. SLO_PROMPTS_PER_CON=2 for 2x (32 requests) instead
# of the 4x default.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# label:isl:osl:context-window:concurrency-list.  Window (4th field) = 131072 (128K tail).
export SCENARIOS="${SCENARIOS:-128k-ctx:80000:1024:131072:16}"

# One warm-up pass over the real trace, then one measured pass (see benchmark_avg_80K.sh).
export SLO_WARMUP_ITERS="${SLO_WARMUP_ITERS:-1}"
export SLO_ITERS="${SLO_ITERS:-1}"
export RESAMPLE_PER_ITER="${RESAMPLE_PER_ITER:-0}"
export SEED_BASE="${SEED_BASE:-0}"

# 2x requests per concurrency by default here (heavier long-context cells) -- still >=32
# samples at con=16 for a solid median, ~2x faster than the 4x default. Override freely.
export SLO_PROMPTS_PER_CON="${SLO_PROMPTS_PER_CON:-2}"

# Distinct log/result root so it cannot collide with the 256K-tail run's outputs.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg80K_128ktail}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg80K_128ktail_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg80K_128ktail_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

echo "=============================================================="
echo " avg 80K / 1K, 128K context (LIGHTER TAIL -- truncated from the 256K spec)"
echo " ${SLO_WARMUP_ITERS} warm-up pass (discarded) + ${SLO_ITERS} measured pass"
echo " con-list from SCENARIOS; ${SLO_PROMPTS_PER_CON}x requests per concurrency"
echo " NOTE: this clamps the tail at 131072 -- report it as a 128K-tail measurement,"
echo "       NOT the customer's full 256K SLO (use benchmark_avg_80K.sh for that)."
echo "=============================================================="

# Recover the real status: `| tee` in the callee always exits 0.
set +e
bash "$HERE/benchmark_customer_slo.sh"
rc=$?

# Pool the achieved workload (single trace by default -> reports that sample's percentiles).
python3 "$HERE/pool_workload.py" \
    --glob "$WORKLOAD_DIR/128k-ctx_s*.jsonl" \
    --label "avg80K-128ktail" \
    --out "$(dirname "$LOG")/avg80K_128ktail_pooled.json" 2>/dev/null || true

exit $rc
