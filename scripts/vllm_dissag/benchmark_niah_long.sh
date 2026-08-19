#!/bin/bash
# ---------------------------------------------------------------------------
# Long-context NIAH ladder, in TOKENS, to 950K. Selected with BENCHMARK_SCRIPT=niah_long.
#
# WHAT THIS ANSWERS
# -----------------
# "Does retrieval still work at the context length we are claiming?" -- separately at
# each rung of the ladder and, within a rung, at each needle DEPTH. The customer sheet
# claims a 256K window on one row and 1M on the other; a throughput number from a
# server that has silently stopped attending past 500K is not a result.
#
# WHY TOKENS AND NOT WORDS
# ------------------------
# benchmark_niah.py historically took NIAH_WORDS. Words are not tokens: for this filler
# the ratio is ~1.3, so "950,000" read as words is ~1.24M tokens -- ABOVE the 1,048,576
# window, and vLLM REJECTS an over-length request with a 400 rather than truncating it.
# That arrives at the client as a transport error and reads as a dead server. NIAH_TOKENS
# calibrates against the real tokenizer instead, and approaches the target from below.
#
# PREREQUISITE, already satisfied: the server must admit these lengths. GLM-5.2 declares
# max_position_embeddings 1048576 and models.yaml sets no --max-model-len, so it inherits
# the full window. If anyone adds a smaller --max-model-len, every rung above it becomes
# a 400 and this script will report NO-RESULT -- which is the correct, loud failure.
#
# RUNTIME, WHICH IS THE REAL CONSTRAINT
# -------------------------------------
# Prefill here is ~QUADRATIC in context length: the sparse indexer scores each 8,192-token
# prefill chunk against all preceding keys (index_topk=2048 caps the DECODE key set, not
# the prefill scan). Scaling from a measured 4.80 s TTFT at 28,672 tokens:
#
#      32,768 ->  0.1 min     393,216 -> 15.0 min
#      65,536 ->  0.4 min     524,288 -> 26.7 min
#     131,072 ->  1.7 min     786,432 -> 60.2 min
#     262,144 ->  6.7 min     950,000 -> 87.8 min
#
# Summed, the default ladder is ~3.3 h at ONE seed with no warmup, ~6.6 h with warmup,
# and ~13.2 h at three seeds with warmup -- against a 24 h job that also has to boot the
# engine (~20 min) and run everything else. Note how top-heavy that is: the last TWO
# rungs are 74% of the total, so dropping 950,000 alone buys back 44% of the ladder if
# the budget gets tight. The defaults here are therefore deliberately NOT the defaults
# of benchmark_niah.py:
#
#     NIAH_SEEDS=0    (not 0,1,2)  -- 3x the cost for variance information we can buy
#                                     more cheaply by adding rungs
#     NIAH_WARMUP=0                -- a warmup pass DOUBLES the ladder. The bottom rungs
#                                     compile the shapes that matter; a cold 950K request
#                                     is charged to the scored request, which is why
#                                     NIAH_TIMEOUT below is sized so generously.
#
# Override either when you have the budget: NIAH_SEEDS=0,1,2 NIAH_WARMUP=1.
#
# "AT DIFFERENT SLOTS"
# --------------------
# Each rung places 10 needles at ~9% depth intervals, and benchmark_niah.py now reports
# the DEPTH of every miss plus a decile histogram. That is the difference between "7/10
# at 950K" (uninterpretable) and "7/10 at 950K, all three misses past 80% depth" (the
# tail of the context is being dropped). Seeds beyond 0 jitter those depths; rungs vary
# the absolute token offsets, so the ladder itself already sweeps chunk boundaries.
# ---------------------------------------------------------------------------
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-/run_logs/${SLURM_JOB_ID:-local}}"
mkdir -p "$LOGDIR"
NIAH_LOG="${LOGDIR}/niah_long_${SLURM_JOB_ID:-local}_${timestamp}_xP${xP:-0}_yD${yD:-0}_${MODEL_NAME:-model}.log"

# The ladder. Rungs are chosen to bracket the two window claims (262,144 and 1,048,576)
# and to put several points between them, because the interesting failures are gradual.
# 950,000 is the top rung the user asked for; it sits just under the 1,048,576 window
# with enough headroom that tokenizer calibration error cannot push it over.
LADDER="${NIAH_TOKENS:-32768,65536,131072,262144,393216,524288,786432,950000}"

# Per-request timeout. The 1800 s default is fine to 262K and hopeless at 950K, and a
# timeout at the top rung is indistinguishable in the log from a server that hung. Size
# it on the quadratic model with a 3x margin, floored at the old default.
NIAH_TIMEOUT_LONG="${NIAH_TIMEOUT:-$(python3 -c "
tops = [int(x) for x in '${LADDER}'.split(',')]
# 4.80 s measured at 28,672 tokens, quadratic, 3x margin.
print(max(1800, int(3 * 4.80 * (max(tops)/28672.0)**2)))")}"

echo "=============================================================="
echo " NIAH long-context ladder (TOKENS)"
echo "   port     : ${BENCHMARK_PORT}"
echo "   model    : ${MODEL_PATH}"
echo "   ladder   : ${LADDER}"
echo "   seeds    : ${NIAH_SEEDS:-0}   warmup: ${NIAH_WARMUP:-0}"
echo "   timeout  : ${NIAH_TIMEOUT_LONG}s per request"
echo "=============================================================="

# Poll with a REAL completion, not /v1/models. That endpoint is wrong in BOTH directions
# on this router: it has returned 200 while every request 503'd, and returns 503 while
# completions succeed -- its listing path and its forwarding path consult different
# state. A completion is the only probe that confirms router + backend + P->D together.
_ready=0
for _i in $(seq 1 60); do
    _probe=$(curl -s --max-time 120 "http://127.0.0.1:${BENCHMARK_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The CEO of AMD is\",\"max_tokens\":8,\"temperature\":0}" 2>&1)
    if echo "$_probe" | grep -q '"text"'; then
        _ready=1; echo "[niah-long] end-to-end probe OK after ~$((_i*10))s"; break
    fi
    sleep 10
done
if [ "$_ready" != 1 ]; then
    echo "[niah-long] FATAL: no successful completion in 600s -- not testing a dead server."
    echo "[niah-long] last response: ${_probe:0:400}"
    exit 1
fi

# Invoked directly, NOT via benchmark_niah.sh: that wrapper ends in `| tee`, whose exit
# status is tee's (always 0), so a FAILING ladder would report success.
set -o pipefail
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_TOKENIZER="${NIAH_TOKENIZER:-${MODEL_PATH}}" \
NIAH_TOKENS="${LADDER}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}" \
NIAH_SEEDS="${NIAH_SEEDS:-0}" \
NIAH_WARMUP="${NIAH_WARMUP:-0}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT_LONG}" \
NIAH_MIN_SCORE="${NIAH_MIN_SCORE:-8.0}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${NIAH_LOG}"
_rc=${PIPESTATUS[0]}
set +o pipefail

echo "==== NIAH long ladder exit=${_rc} -> ${NIAH_LOG} ===="
# Exit-code contract (shared with benchmark_niah.py):
#   3 = some rung produced NO usable result (dead server / timeout)
#   4 = every rung scored, but some mean < NIAH_MIN_SCORE (retrieval broken)
#   0 = pass
exit "${_rc}"
