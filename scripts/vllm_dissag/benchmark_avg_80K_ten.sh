#!/bin/bash
# ---------------------------------------------------------------------------
# Customer row 1: avg ISL 80K / OSL 1K, 256K context. Ten runs.
#
# WHY TEN, and why this file exists at all
# -----------------------------------------
# The sheet gives an AVERAGE input length and a context window. Those pin a
# right-skewed distribution (see gen_workload.py), and the realised mean of a single
# few-hundred-request draw from a skewed distribution wobbles:
#
#     CV = sqrt(exp(sigma^2)-1) = 0.637 at sigma 0.5833
#     SE(mean)/mean = CV/sqrt(n) = 4.0% at n=256
#
# and measured over 12 seeds the realised mean actually spanned 72,884-84,222 -- 14.2%.
# Quoting one run's mean to the customer would be quoting a property of that seed.
#
# Ten runs at DISTINCT seeds pool to n=2,560 and the error falls as 1/sqrt(10), to
# 1.3%. That is a number worth putting on a slide. Reaching 1.3% inside a SINGLE run
# would need n=1,014 requests at ~80K tokens each -- more than MAX_PROMPTS and far more
# wall clock than ten smaller runs, for the same statistical answer.
#
# The alternative -- stratified inverse-CDF placement -- pins the mean without any of
# this, but caps the sample at the (n-0.5)/n quantile, so the p99 would reach only
# ~225K instead of 262,144. The window number is the thing the customer actually
# stated; we do not trade it away for tidiness in a number they did not state.
#
# This is a THIN WRAPPER. Every measurement decision (Little's-Law request rate, the
# goodput thresholds, the timeout sizing, the warmup) lives in benchmark_customer_slo.sh
# and has exactly one implementation. Two copies of that logic would drift, and the
# drift would show up as an unexplained delta between the 80K and 200K rows.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One scenario only: label:isl:osl:context-window:concurrency-list.
export SCENARIOS="${SCENARIOS:-256k-ctx:80000:1024:262144:16 32 64}"

# Ten iterations, each drawing a FRESH sample. RESAMPLE_PER_ITER=1 is the whole point:
# without it the harness caches one JSONL and replays it, and averaging one sample with
# itself ten times shrinks nothing.
export SLO_ITERS="${SLO_ITERS:-10}"
export RESAMPLE_PER_ITER=1
export SEED_BASE="${SEED_BASE:-0}"

# Give it a distinct log/result root so a 200K run in the same job cannot overwrite it.
# /run_logs is the ONLY host-backed path in the container (-v ${LOG_PATH}:/run_logs).
# Anything written elsewhere lands on the container's writable overlay, survives the
# run, and is destroyed with the container -- ten runs of results, silently gone.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg80K_ten}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg80K_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg80K_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

echo "=============================================================="
echo " avg 80K / 1K, 256K context -- ${SLO_ITERS} runs, seeds ${SEED_BASE}..$(( SEED_BASE + SLO_ITERS - 1 ))"
echo " pooled n = 256 x ${SLO_ITERS}; expected SE of the mean 4.0% -> $(python3 -c "print('%.1f' % (4.0/($SLO_ITERS**0.5)))")%"
echo "=============================================================="

# Recover the real status: `| tee` in the callee always exits 0, and a green exit on a
# run that never produced a result is the failure mode that wastes a day.
set +e
bash "$HERE/benchmark_customer_slo.sh"
rc=$?
set -e

# Pool the ten samples into the one distribution figure we report upstream. Runs even
# if the benchmark failed -- the workloads that WERE generated still describe what was
# offered, and knowing that is worth more than a bare non-zero exit code.
# Derive the glob from the scenario LABEL rather than hardcoding "256k-ctx". gen_wl()
# names its files "<label>_s<seed>.jsonl", so a hardcoded pattern would silently match
# nothing the moment anyone overrode SCENARIOS -- and "No workload files matched" at the
# very end of a ten-run job is a bad place to learn that.
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
