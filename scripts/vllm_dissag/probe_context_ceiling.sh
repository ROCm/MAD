#!/bin/bash
# ---------------------------------------------------------------------------
# Context-window ceiling probe for GLM-5.2 on MI308X.  BOOT ONLY -- no benchmark.
#
# WHY THIS EXISTS
# ---------------
# The customer sheet asks for a 256K window (row 1) and a 1M window (row 2).
# models.yaml caps us at --max-model-len ${GLM_MAX_MODEL_LEN:-65536}, and that cap is
# a MEMORY cap, not a policy: uncapped, every EngineCore aborts at init with
#
#     46.58 GiB KV cache is needed, which is larger than the available
#     KV cache memory (1.23 GiB)
#
# because TWO things grow with the window out of ONE fixed pool:
#
#   * the DSA indexer workspace, sized ~max_model_len*40, reserved BEFORE any KV;
#   * the KV for a single max-length request, at 46.6 KiB/token on this build.
#
# So doubling the window doubles the per-request KV *and* shrinks the pool it has to
# come from.  You cannot reason your way to the ceiling from either number alone --
# hence a probe.
#
# WHAT THE ARITHMETIC PREDICTS (and why it is not good enough)
# ------------------------------------------------------------
# Fitting the workspace slope to the two EP8 points we have measured
# (65,536 -> 35.71 GiB available; 1,048,576 -> 1.23 GiB available) gives ~2.3 GiB of
# workspace per 64K of window, and therefore:
#
#   window      1 req      EP8 fits        EP16 fits
#   65,536        2.91     12.27x          21.85x
#   131,072       5.82      5.74-6.13x     10.53-10.93x
#   196,608       8.73      3.56-4.09x      6.76-7.28x
#   262,144      11.64      2.47-3.07x      4.87-5.46x
#   524,288      23.29      0.84-1.53x      2.04-2.73x
#   1,048,576    46.58      0.03-0.77x      0.63-1.37x
#
# The RANGES are not error bars -- they are the two defensible treatments of the one
# soft term.  LOWER (pessimistic) = workspace grows linearly at 2.30 GiB per 64K, as
# fitted.  UPPER (optimistic) = workspace is negligible and the whole pool is KV.  The
# truth lies between.  Where both ends agree on the verdict, the verdict does not
# depend on the assumption -- and that is every row here EXCEPT 1,048,576, which is
# "will not boot" at one end and "boots, serves exactly one request" at the other.
# Operationally those are the same answer, but the model cannot tell them apart.
#
# Two honest caveats about that table:
#
#   1. It is a TWO-POINT LINEAR FIT, and the slope was derived from the 1M failure
#      itself -- so the EP8 1M row is reproduced by construction, not predicted.
#   2. The EP16 column is pure extrapolation.  No EP16 boot above 65,536 has ever
#      been observed on this cluster.
#
# That is precisely why this script boots the thing instead of quoting the table.
# The table sets expectations; the probe sets facts.
#
# THE HEADLINE PREDICTION, TO BE CONFIRMED OR KILLED
# ---------------------------------------------------
#   * 1M does NOT fit usefully, even at EP16 (0.63-1.37x).  The pessimistic end says it
#     will not boot; the optimistic end says it boots and serves exactly one request.
#     Both are the same operational answer: 1M is not a row we can run.
#   * 262,144 DOES fit at EP16, at 4.87-5.46x -- comfortable rather than marginal, and
#     comfortable under BOTH workspace assumptions.  That is the customer's row-1 window.
#   * 262,144 at EP8 is 2.47-3.07x -- too tight to run con=16 against.
#
# If that holds, 256K is a row we can actually run, and it is a row where EP16 is not
# merely better than EP8 but NECESSARY.  That is the topology question the campaign
# report called decisive, answered with a measurement.
#
# WHY MORE EP RANKS *DO* RAISE THE CEILING -- AND WHY WE STILL CANNOT REACH 1M
# ----------------------------------------------------------------------------
# An earlier version of this header claimed more nodes could not raise the single-request
# ceiling.  That was WRONG, and the reasoning is worth keeping so it is not re-derived.
#
# The true premise: with DP attention + MLA a request's KV lives entirely on ONE decode
# rank, so the binding constraint is per-rank.  The false conclusion: that per-rank is
# therefore fixed.  It is not.  MoE expert weights SHARD with the EP degree, so going
# EP8 -> EP16 cut per-rank weights 107.33 -> 65.12 GiB and handed the difference to KV:
# the pool grew 35.71 -> 64.19 GiB (1.8x) DESPITE a LOWER utilisation (0.80 -> 0.72).
# More EP ranks => thinner per-rank weights => a bigger pool => genuinely longer single
# requests.  EP32 would put ~44.02 GiB/rank of weights against a ~84.73 GiB pool, which
# is ~1,094,703 tokens at concurrency 1 -- 1M would in fact fit.
#
# We cannot run it: EP32 needs 4 decode nodes, disaggregated P/D needs at least one
# prefill node, and the cluster has 4 nodes total.  1M is out of reach on THIS CLUSTER
# for want of hardware, not for want of physics.  (Asymptotically, EP->infinity gives a
# 105.84 GiB pool ~ 2.27M tokens; the ceiling is finite but it is well above 1M.)
#
# USAGE
#   EP=16 ./probe_context_ceiling.sh              # 4 nodes, the interesting case
#   EP=8  ./probe_context_ceiling.sh              # 2 nodes, cheap reference
#   RUNGS="65536 131072 262144" EP=16 ./probe_context_ceiling.sh
#   DRY_RUN=1 EP=16 ./probe_context_ceiling.sh    # print the launches, run nothing
#
# Each rung is a full boot: ~10 min at EP8, ~20 min at EP16 (weights + MoRI heap +
# DSA JIT + cudagraph capture).  Budget ~80 min for a 4-rung EP16 ladder.  There is
# no way to shortcut this -- the numbers we need are printed by the memory profiler
# after weights are resident, so the boot IS the measurement.
# ---------------------------------------------------------------------------
set -uo pipefail        # deliberately NOT -e: a rung that fails to boot is DATA.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP="${EP:-16}"
# Ladder stops at 262144 on purpose.  524288 is predicted to fail at EP8 and to fit
# only 2.1x at EP16, and 1048576 is predicted to fail outright on both -- spending
# 40 min of boots to confirm an arithmetic near-certainty is worse value than
# spending it on the rung the customer can actually use.  Override RUNGS to chase
# them anyway once 262144 is known to boot.
RUNGS="${RUNGS:-65536 131072 196608 262144}"
OUT="${OUT:-/models/common/logs/ctx_ceiling_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'window\tstatus\tavail_kv_gib\tkv_tokens\tconcurrency_at_window\tnote\n' > "$SUMMARY"

echo "=============================================================="
echo " GLM-5.2 context-window ceiling probe -- EP${EP}, boot only"
echo " rungs   : $RUNGS"
echo " output  : $OUT"
echo "=============================================================="
echo
echo "Predicted (2-point fit, EP16 column is extrapolation -- see header):"
python3 - <<'PY'
KIB = 46.58/1048576.0
W64 = (35.71-1.23)/15.0
import os
ep = os.environ.get("EP","16")
pool = (35.71 if ep=="8" else 64.19) + W64
for L in [int(x) for x in os.environ.get("RUNGS","65536 131072 196608 262144").split()]:
    need = L*KIB
    # Two bounds on the one soft term, not an error bar: LO assumes the fitted linear
    # indexer workspace, HI assumes it is negligible.  A rung is only "PREDICTED FAIL"
    # if it fails even under HI; a rung whose bounds straddle 1.0 is UNDECIDED and is
    # exactly the rung worth spending a boot on.
    lo = max(pool - W64*(L/65536.0), 0.0)/need
    hi = pool/need
    verdict = "OK" if lo >= 1 else ("PREDICTED FAIL" if hi < 1 else "UNDECIDED -- boot decides")
    print("   window %9s  1req %6.2f GiB  -> %5.2f-%5.2f concurrent  %s"
          % ("{:,}".format(L), need, lo, hi, verdict))
PY
echo

for win in $RUNGS; do
    echo "--------------------------------------------------------------"
    echo " RUNG: --max-model-len $win"
    echo "--------------------------------------------------------------"

    # GLM_PREFILL_BATCHED_TOKENS must stay >= the longest prompt or asm_mla.cu:945
    # turns a long prompt into a worker SIGKILL rather than a slow request (see
    # models.yaml).  It must ALSO stay <= max-model-len.  40960 satisfies both for
    # every rung here; it is left alone rather than scaled, because this probe is
    # about the window and changing two variables would make a failure ambiguous.
    launch_env=(
        "GLM_MAX_MODEL_LEN=$win"
    )
    if [ "$EP" = "16" ]; then
        launch_env+=( "GPU_MEMORY_UTILIZATION=0.72" "MORI_SHMEM_HEAP_SIZE=34359738368" )
    else
        launch_env+=( "GPU_MEMORY_UTILIZATION=0.80" "MORI_SHMEM_HEAP_SIZE=17179869184" )
    fi
    # Display only. The real topology comes from PREFILL/DECODE, which the launcher
    # reads (launch_disagg_skyriver.sh:69-70); echo the same values here rather than
    # baking in site hostnames. Unset -> the launcher's own defaults apply.
    nodes="${PREFILL:-<PREFILL unset>} -> ${DECODE:-<DECODE unset>}"

    echo "  env  : ${launch_env[*]}"
    echo "  nodes: $nodes"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN=1 -- not launching"
        printf '%s\tDRY_RUN\t-\t-\t-\t-\n' "$win" >> "$SUMMARY"
        continue
    fi

    rung_log="$OUT/rung_${win}.log"
    # BENCHMARK_SCRIPT_FILE=keepalive_bench.sh: boot, hold, exit.  We want the memory
    # profiler's verdict, not a benchmark -- and running a real benchmark here would
    # multiply an already-long ladder by the benchmark's wall clock for no extra
    # information about capacity.
    env "${launch_env[@]}" \
        BENCHMARK_SCRIPT_FILE=keepalive_bench.sh \
        EP="$EP" \
        bash "$HERE/launch_disagg_skyriver.sh" > "$rung_log" 2>&1
    rc=$?

    # The two lines that answer the question.  Read from the DECODE role: decode holds
    # the pool that serves generation, prefill KV is transient and handed off, so
    # scoring prefill would overstate capacity (same reasoning as diag/kv_capacity.sh).
    avail=$(grep -h "Available KV cache memory" "$rung_log" 2>/dev/null | head -1 \
            | grep -oE '[-0-9.]+ GiB' | grep -oE '[-0-9.]+' || true)
    ktok=$(grep -h "GPU KV cache size" "$rung_log" 2>/dev/null | head -1 \
            | grep -oE '[0-9,]+ tokens' | tr -d ', tokens' || true)

    if [ -z "${avail:-}" ]; then
        note="no KV line -- boot failed before profiling (rc=$rc)"
        status="FAIL"
    elif python3 -c "import sys; sys.exit(0 if float('$avail') > 0 else 1)"; then
        status="OK"; note="rc=$rc"
    else
        # A NEGATIVE available-KV is the signature failure of this exact cap: the
        # indexer workspace consumed more than the pool held.  It is the reason the
        # 65536 cap exists at all.
        status="FAIL"; note="negative available KV -- indexer workspace exceeded pool"
    fi

    conc="-"
    if [ -n "${ktok:-}" ] && [ "$ktok" -gt 0 ] 2>/dev/null; then
        conc=$(python3 -c "print('%.2f' % ($ktok/$win))")
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$win" "$status" "${avail:--}" "${ktok:--}" "$conc" "$note" >> "$SUMMARY"
    echo "  -> $status  availKV=${avail:--} GiB  tokens=${ktok:--}  concurrent=${conc}"

    # Stop climbing once a rung fails: every higher rung reserves strictly more
    # workspace and needs strictly more KV, so it cannot succeed where this one did
    # not.  Continuing would burn ~20 min per rung to confirm monotonicity.
    if [ "$status" = "FAIL" ]; then
        echo
        echo "  Ceiling found below $win. Higher rungs cannot pass -- stopping."
        break
    fi
done

echo
echo "=============================================================="
column -t -s $'\t' "$SUMMARY"
echo "=============================================================="
echo "summary: $SUMMARY"
echo
echo "Read the highest OK rung as the usable window for EP${EP}, then set"
echo "GLM_MAX_MODEL_LEN to it in the scenario wrapper. Note the concurrency column:"
echo "a rung that boots with <2x concurrency is a rung you cannot run con=8 against."
