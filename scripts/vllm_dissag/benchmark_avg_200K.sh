#!/bin/bash
# ---------------------------------------------------------------------------
# Customer row 2: avg ISL 200K / OSL 1K, 1M context.
#
# Same design as benchmark_avg_80K.sh -- one warm-up pass over the real trace, then one
# measured pass over the same trace. Read the rationale there. What differs is the
# magnitude of the sampling problem this row has, and it is worse:
#
#     sigma 0.8778 -> CV 1.078 -> SE(mean)/mean = 9.5% at n=128
#
# and measured over 12 seeds the realised mean spanned 166,447-237,295 -- 35.4%. A single
# sample's mean here is genuinely not quotable as "the customer's 200K average"; quote
# the achieved mean from <trace>.meta.json instead. If the target must be pinned, ten
# distinct seeds pool to n=1,280 and bring 9.5% to 3.0%:
#
#     SLO_ITERS=10 RESAMPLE_PER_ITER=1 ./benchmark_avg_200K.sh
#
# Note the n is HALF the 80K row's (128 vs 256), because concurrency is halved -- these
# requests are ~2.5x larger and the KV pool, not the statistics, sets the ceiling. At
# the p99 a single request holds 46.58 GiB of KV (46.58 KiB/token measured on MI308X
# x 1,048,576 tokens), so con=32 with a tail-heavy in-flight set is far past budget.
# The larger SE on this row is a consequence of that hardware limit, not of a choice --
# which is exactly why it gets reported rather than hidden.
#
# THIS ROW IS NOT EXPECTED TO RUN ON MI308X / 4 NODES. READ BEFORE SCHEDULING IT.
# ------------------------------------------------------------------------------
# Two independent blockers, either of which is fatal:
#
#   1. CAPACITY. A single 1,048,576-token request costs 46.58 GiB of KV. The measured
#      per-rank pool is 35.71 GiB at EP8 and 64.19 GiB at EP16, so ONE p99 request
#      needs 0.0-0.8 ranks' worth at EP8 and 0.6-1.4 at EP16 -- i.e. a single p99
#      request does not reliably fit in a rank at either topology, before any
#      concurrency at all. EP32 would fit (~1.09M tokens) but needs 2 decode nodes
#      x 16 + at least 1 prefill node > the 4 available. This is a node-count limit,
#      not a physics one.
#   2. MAX-MODEL-LEN. models.yaml passes `--max-model-len ${GLM_MAX_MODEL_LEN:-65536}`.
#      An earlier version of this header claimed the 1M window was inherited from
#      max_position_embeddings and needed no flag -- that was TRUE upstream and is
#      FALSE on this branch. Without GLM_MAX_MODEL_LEN=1048576 every request over
#      65,536 is REJECTED with 400, vanishes from the latency stats, and shows up only
#      as a lower `completed` count -- i.e. as a suspiciously GOOD result. And with it,
#      the server is expected to OOM at boot per (1).
#
# Kept in the tree because it is the customer's second stated row and the arithmetic
# for why it does not fit belongs next to it. Run benchmark_avg_80K.sh instead;
# run probe_context_ceiling.sh if you want the empirical boot ladder.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCENARIOS="${SCENARIOS:-1m-ctx:200000:1024:1048576:8 16 32}"

# One warm-up pass over the real trace, then one measured pass. See benchmark_avg_80K.sh.
export SLO_WARMUP_ITERS="${SLO_WARMUP_ITERS:-1}"
export SLO_ITERS="${SLO_ITERS:-1}"
export RESAMPLE_PER_ITER="${RESAMPLE_PER_ITER:-0}"
export SEED_BASE="${SEED_BASE:-0}"

# /run_logs is the only host-backed mount in the container; see the note in the 80K
# wrapper. Writing elsewhere loses the results when the container exits.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg200K}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg200K_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg200K_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

echo "=============================================================="
echo " avg 200K / 1K, 1M context"
echo " ${SLO_WARMUP_ITERS} warm-up pass (discarded) + ${SLO_ITERS} measured pass"
if [ "${RESAMPLE_PER_ITER}" = "1" ]; then
echo " seeds ${SEED_BASE}..$(( SEED_BASE + SLO_ITERS - 1 )); pooled n = 128 x ${SLO_ITERS}"
echo " expected SE of the mean 9.5% -> $(python3 -c "print('%.1f' % (9.5/($SLO_ITERS**0.5)))")%"
else
echo " one fixed trace, seed ${SEED_BASE}; its mean carries ~9.5% standard error"
echo " REPORT the .meta.json deviation -- do not claim the 200,000 target was hit."
echo " To pin it: SLO_ITERS=10 RESAMPLE_PER_ITER=1 $(basename "$0")"
fi
echo "=============================================================="

set +e
bash "$HERE/benchmark_customer_slo.sh"
rc=$?
set -e

# Glob derived from the scenario label, not hardcoded -- see the note in the 80K wrapper.
echo
IFS='|' read -ra _scn <<< "$SCENARIOS"
for _s in "${_scn[@]}"; do
    _label="${_s%%:*}"
    python3 "$HERE/pool_workload.py" \
        --glob "$WORKLOAD_DIR/${_label}_s*.jsonl" \
        --label "avg 200K / 1M context [$_label]" \
        --out "${LOG}_pooled_workload_${_label}.json" || true
done

exit $rc
