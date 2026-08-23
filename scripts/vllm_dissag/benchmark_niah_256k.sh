#!/bin/bash
# ---------------------------------------------------------------------------
# NIAH retrieval ladder to 256K. Selected with BENCHMARK_SCRIPT=niah_256k.
#
# WHY A SEPARATE WRAPPER FROM benchmark_niah_long.sh
# --------------------------------------------------
# Not because the ladder is shorter -- NIAH_TOKENS would have handled that. Because a
# shorter ladder can AFFORD DIFFERENT DEFAULTS, and those defaults are the point.
#
# The 950K wrapper runs NIAH_SEEDS=0 NIAH_WARMUP=0. That is a budget decision, stated
# there: three seeds with warmup on that ladder is ~13 h against a 24 h job that also
# has to boot the engine and run everything else. At a 256K ceiling the whole ladder is
# ~55 min with three seeds AND warmup, so the budget reason evaporates and the defaults
# should flip back to the ones that actually answer the question:
#
#     NIAH_SEEDS=0,1,2   each seed re-rolls BOTH the filler and the needle offsets, so
#                        three seeds is what separates "retrieval works" from "retrieval
#                        happened to work at these 10 offsets". With one seed a 10/10 and
#                        a 7/10 are not distinguishable from layout luck.
#     NIAH_WARMUP=1      the first request at a new context length pays kernel autotune.
#                        On the 950K ladder that cost is amortised over hours and the
#                        generous timeout absorbs it; here a cold 262K request landing on
#                        the scored path is a large fraction of that rung's time and can
#                        read as a false timeout.
#
# Copying the wrapper rather than overriding at the call site keeps that reasoning
# attached to the run instead of living in someone's shell history.
#
# THE LADDER
# ----------
# 8,192 / 32,768 / 65,536 / 131,072 / 262,144. Doubling, so each rung is a controlled
# step and a failure has a bracket rather than a cliff. 262,144 is the customer's stated
# window for the 80K-average row -- it is the number being claimed, so it is the number
# tested. 8,192 is the control: if it is not 10/10 the problem is the harness or the
# prompt, not long context, and nothing above it is interpretable.
#
# RUNTIME (measured TPOT 38.6 ms; TTFT from the quadratic prefill model, 4.80 s @ 28,672)
#
#      8,192  TTFT ~0.4s     rung  4.0 min
#     32,768  TTFT ~6.3s     rung  4.4 min
#     65,536  TTFT ~25s      rung  5.6 min
#    131,072  TTFT ~100s     rung 10.6 min
#    262,144  TTFT ~401s     rung 30.7 min
#                            ----------------
#                            total ~55 min
#
# Those rung times are worst case: they assume the model emits the full NIAH_MAXTOK of
# 2,048 tokens. It answers with a short list of animals, so expect well under.
#
# PREREQUISITE THAT HAS ACTUALLY BITTEN US
# ----------------------------------------
# --max-model-len must admit 262,144 + NIAH_MAXTOK. GLM-5.2-FP8 declares
# max_position_embeddings 1,048,576 with rope_scaling null, so the MODEL does not block
# this -- but our tenant-workaround overlay set --max-model-len 131072, and every rung
# above it becomes an HTTP 400. vLLM REJECTS an over-length request rather than
# truncating, and that arrives at the client as a transport error that reads exactly
# like a dead server. The check below refuses to start in that case instead of
# generating an hour of misleading NO-RESULT lines.
# ---------------------------------------------------------------------------
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-/run_logs/${SLURM_JOB_ID:-local}}"
mkdir -p "$LOGDIR"
NIAH_LOG="${LOGDIR}/niah256k_${SLURM_JOB_ID:-local}_${timestamp}_xP${xP:-0}_yD${yD:-0}_${MODEL_NAME:-model}.log"

LADDER="${NIAH_TOKENS:-8192,32768,65536,131072,262144}"
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}"

# Per-request timeout for the LONGEST rung; benchmark_niah.py scales shorter rungs down
# quadratically (NIAH_TIMEOUT_SCALE=1, its default) with a 300 s floor. Sizing on the
# same quadratic prefill model that produced the table above, plus the decode tail, with
# a 3x margin -- so a genuinely hung 262K rung fails in ~20 min rather than consuming
# the job, while a merely slow one still completes.
NIAH_TIMEOUT_256K="${NIAH_TIMEOUT:-$(python3 -c "
tops = [int(x) for x in '${LADDER}'.split(',')]
top  = max(tops)
ttft = 4.80 * (top / 28672.0) ** 2      # quadratic: sparse indexer scans all prior keys
dec  = ${NIAH_MAXTOK} * 0.0386          # measured TPOT at c64
print(max(1800, int(3 * (ttft + dec))))")}"

echo "=============================================================="
echo " NIAH retrieval ladder to 256K (TOKENS)"
echo "   port     : ${BENCHMARK_PORT}"
echo "   model    : ${MODEL_PATH}"
echo "   ladder   : ${LADDER}"
echo "   seeds    : ${NIAH_SEEDS:-0,1,2}   warmup: ${NIAH_WARMUP:-1}"
echo "   maxtok   : ${NIAH_MAXTOK}   timeout: ${NIAH_TIMEOUT_256K}s (longest rung)"
echo "   log      : ${NIAH_LOG}"
echo "=============================================================="

# ---- readiness: probe with a REAL completion -------------------------------------
# /v1/models is wrong in BOTH directions on this router -- it has returned 200 while
# every request 503'd, and 503 while completions succeeded. Its listing path and its
# forwarding path consult different state. A completion is the only probe that confirms
# router + backend + P->D together.
_ready=0
for _i in $(seq 1 60); do
    _probe=$(curl -s --max-time 120 "http://127.0.0.1:${BENCHMARK_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The CEO of AMD is\",\"max_tokens\":8,\"temperature\":0}" 2>&1)
    if echo "$_probe" | grep -q '"text"'; then
        _ready=1; echo "[niah256k] end-to-end probe OK after ~$((_i*10))s"; break
    fi
    sleep 10
done
if [ "$_ready" != 1 ]; then
    echo "[niah256k] FATAL: no successful completion in 600s -- not testing a dead server."
    echo "[niah256k] last response: ${_probe:0:400}"
    exit 1
fi

# ---- context-window check --------------------------------------------------------
# Ask the server what it will admit, rather than trusting the config we think we set.
# max_model_len is reported by /v1/models per served model. If the top rung plus the
# answer does not fit, every request above the cap 400s and the ladder produces an hour
# of NO-RESULT that looks like a retrieval failure. Fail loudly here instead.
_top=$(python3 -c "print(max(int(x) for x in '${LADDER}'.split(',')))")
_need=$(( _top + NIAH_MAXTOK ))
_cap=$(curl -s --max-time 30 "http://127.0.0.1:${BENCHMARK_PORT}/v1/models" 2>/dev/null \
       | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
    print(max(int(m.get('max_model_len') or 0) for m in d.get('data', [])) or 0)
except Exception:
    print(0)" 2>/dev/null)
_cap="${_cap:-0}"
if [ "$_cap" -gt 0 ] && [ "$_cap" -lt "$_need" ]; then
    echo "[niah256k] FATAL: server max_model_len=${_cap} < ${_need} needed"
    echo "           (top rung ${_top} + NIAH_MAXTOK ${NIAH_MAXTOK})."
    echo "           Every rung above the cap would return HTTP 400, which this harness"
    echo "           reports as NO-RESULT and reads like broken retrieval. Raise"
    echo "           --max-model-len in the models.yaml overlay for BOTH roles, or lower"
    echo "           the ladder. Refusing to run a test whose failures cannot be trusted."
    exit 2
elif [ "$_cap" = 0 ]; then
    echo "[niah256k] WARN: could not read max_model_len from /v1/models; proceeding."
    echo "           If rungs above some length all fail identically, suspect the cap."
else
    echo "[niah256k] context check OK: max_model_len=${_cap} >= ${_need}"
fi

# Invoked directly, NOT via benchmark_niah.sh: that wrapper ends in `| tee`, whose exit
# status is tee's (always 0), so a FAILING ladder would report success.
set -o pipefail
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_TOKENIZER="${NIAH_TOKENIZER:-${MODEL_PATH}}" \
NIAH_TOKENS="${LADDER}" \
NIAH_MAXTOK="${NIAH_MAXTOK}" \
NIAH_SEEDS="${NIAH_SEEDS:-0,1,2}" \
NIAH_WARMUP="${NIAH_WARMUP:-1}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT_256K}" \
NIAH_MIN_SCORE="${NIAH_MIN_SCORE:-8.0}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${NIAH_LOG}"
_rc=${PIPESTATUS[0]}
set +o pipefail

echo "==== NIAH 256K ladder exit=${_rc} -> ${NIAH_LOG} ===="
# Exit-code contract (shared with benchmark_niah.py):
#   3 = some rung produced NO usable result (dead server / timeout)
#   4 = every rung scored, but some mean < NIAH_MIN_SCORE (retrieval broken)
#   0 = pass
exit "${_rc}"
