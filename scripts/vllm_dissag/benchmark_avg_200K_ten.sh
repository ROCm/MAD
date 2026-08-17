#!/bin/bash
# ---------------------------------------------------------------------------
# Customer row 2: avg ISL 200K / OSL 1K, 1M context. Ten runs.
#
# Same design as benchmark_avg_80K_ten.sh -- read the rationale there. What differs is
# the magnitude of the problem this row has, and it is worse:
#
#     sigma 0.8778 -> CV 1.078 -> SE(mean)/mean = 9.5% at n=128
#
# and measured over 12 seeds the realised mean spanned 166,447-237,295 -- 35.4%. A
# single run here is genuinely not quotable. Ten distinct seeds pool to n=1,280 and
# bring it to 3.0%.
#
# Note the n is HALF the 80K row's (128 vs 256), because concurrency is halved -- these
# requests are ~2.5x larger and the KV pool, not the statistics, sets the ceiling. At
# the p99 a single request holds 43.88 GiB of KV (576 B/token x 78 layers x 1,048,576
# tokens), so con=32 with a tail-heavy in-flight set is already near a node's budget.
# The larger SE on this row is a consequence of that hardware limit, not of a choice --
# which is exactly why it gets reported rather than hidden.
#
# PREREQUISITE: the server must admit 1,048,576-token requests. GLM-5.2 declares
# max_position_embeddings 1048576 and models.yaml sets no --max-model-len, so it
# inherits the full window and this is already satisfied. If someone later adds a
# --max-model-len below 1048576, the p99 requests will be REJECTED with 400, vanish
# from the latency stats, and show up only as a lower `completed` count -- i.e. as a
# suspiciously good result. slo_report.py prints `completed`; check it.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCENARIOS="${SCENARIOS:-1m-ctx:200000:1024:1048576:8 16 32}"

export SLO_ITERS="${SLO_ITERS:-10}"
export RESAMPLE_PER_ITER=1
export SEED_BASE="${SEED_BASE:-0}"

# /run_logs is the only host-backed mount in the container; see the note in the 80K
# wrapper. Writing elsewhere loses the results when the container exits.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg200K_ten}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg200K_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg200K_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

echo "=============================================================="
echo " avg 200K / 1K, 1M context -- ${SLO_ITERS} runs, seeds ${SEED_BASE}..$(( SEED_BASE + SLO_ITERS - 1 ))"
echo " pooled n = 128 x ${SLO_ITERS}; expected SE of the mean 9.5% -> $(python3 -c "print('%.1f' % (9.5/($SLO_ITERS**0.5)))")%"
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
