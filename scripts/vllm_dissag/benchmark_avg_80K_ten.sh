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
#
# FIRST-PASS MODE (the current defaults)
# --------------------------------------
# Everything above describes the FINAL number. It is not what the defaults below do,
# and the difference is deliberate: this row has never been run at all, so the first
# question is "does 80K/256K even hold up", not "what is its mean to 1.3%". Ten fresh
# seeds x 3 concurrencies is several hours; spending that before knowing whether c32
# stays inside the KV pool is spending it on a run we might have to discard wholesale.
#
# So the defaults are a characterisation pass:
#
#   * c16 and c32 only. c64 at 80K needs 64 x 80,000 x 43.88 KiB = 214 GiB of KV
#     RESIDENT AT ONCE. That fits only at decode util 0.80 (817 GiB pool); at the 0.50
#     we are running it does not, and the cell would fail for a capacity reason that
#     tells us nothing about the model. Add it back once util is raised.
#   * SLO_ITERS=2 with CELL_WARMUP=1 -- concurrency becomes the outer loop, so each
#     cell runs twice back-to-back and iteration 1 is a rehearsal at its OWN
#     concurrency. warmup_shape in the callee runs --max-concurrency 2, which warms the
#     80K SHAPE but leaves batch composition, block-table pressure and the MoRI-IO
#     transfer pattern at 32-in-flight cold. Those costs have to land somewhere; this
#     puts them on a run slo_report.py discards (--min-iter 2).
#   * RESAMPLE_PER_ITER=0 -- warm and measure MUST see the same sample. Resampling here
#     would rehearse a different 128-prompt draw than the one scored, and pay a second
#     multi-minute tokenisation to build it.
#
# Pooled n is 192 (64 + 128 measured prompts), SE of the realised mean ~4.6%. That is
# too loose to quote and is not meant to be quoted -- it is enough to see whether TTFT
# and TPOT are in the right order of magnitude and whether the cells complete.
#
# To get the reportable number, restore the ten-seed run explicitly:
#   SLO_ITERS=10 RESAMPLE_PER_ITER=1 CELL_WARMUP=0 \
#     SCENARIOS='256k-ctx:80000:1024:262144:16 32 64' bash benchmark_avg_80K_ten.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One scenario only: label:isl:osl:context-window:concurrency-list.
export SCENARIOS="${SCENARIOS:-256k-ctx:80000:1024:262144:16 32}"

export SLO_ITERS="${SLO_ITERS:-2}"
export RESAMPLE_PER_ITER="${RESAMPLE_PER_ITER:-0}"
export CELL_WARMUP="${CELL_WARMUP:-1}"
export SEED_BASE="${SEED_BASE:-0}"

# Give it a distinct log/result root so a 200K run in the same job cannot overwrite it.
# /run_logs is the ONLY host-backed path in the container (-v ${LOG_PATH}:/run_logs).
# Anything written elsewhere lands on the container's writable overlay, survives the
# run, and is destroyed with the container -- ten runs of results, silently gone.
export LOG="${LOG:-/run_logs/${SLURM_JOB_ID:-local}/avg80K_ten}"
export RESULT_DIR="${RESULT_DIR:-$(dirname "$LOG")/avg80K_slo_json}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$(dirname "$LOG")/avg80K_workloads}"
mkdir -p "$RESULT_DIR" "$WORKLOAD_DIR" "$(dirname "$LOG")"

# Compute the banner from the ACTUAL cell list rather than a hardcoded 256. The callee
# sets prompts = max(32, min(con*4, MAX_PROMPTS)), so n depends on which concurrencies
# are in SCENARIOS -- the old fixed "256 x ITERS" was wrong for every cell set we have
# ever run, and it overstated the precision of the answer. Under CELL_WARMUP the first
# iteration of each cell is discarded, so it does not count toward n either.
_banner=$(python3 - "$SCENARIOS" "$SLO_ITERS" "${CELL_WARMUP:-0}" "${MAX_PROMPTS:-512}" <<'PY'
import sys
scn, iters, warm, cap = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
scored = iters - 1 if warm else iters
n = 0
for s in scn.split("|"):
    for c in s.split(":")[4].split():
        n += max(32, min(int(c) * 4, cap)) * max(scored, 0)
cv = 0.637                      # lognormal mu=11.1197 sigma=0.5833, mean 80k / p99 262144
se = 100.0 * cv / n ** 0.5 if n else float("nan")
print("%d|%.1f|%s" % (n, se, "discarding iter 1 of each cell" if warm else "all iters scored"))
PY
)
IFS='|' read -r _n _se _note <<< "$_banner"
echo "=============================================================="
echo " avg 80K / 1K, 256K context -- ${SLO_ITERS} iters/cell, seed base ${SEED_BASE}"
echo " cells: ${SCENARIOS##*:}   (${_note})"
echo " pooled measured n = ${_n}; SE of the realised mean ~${_se}%"
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
