#!/bin/bash
# NIAH accuracy first, then the ISL/OSL perf sweep -- both against the SAME live
# server, in that order. Selected with BENCHMARK_SCRIPT=niah_perf.
#
# WHY COMBINED: engine boot is ~20 min (weights + MoRI symmetric heap + DSA indexer
# JIT + cudagraph capture). Two launches pay that twice AND measure accuracy and
# throughput on two different boots, so a perf regression could never be attributed.
#
# WHY GATED: if NIAH cannot retrieve a needle, the 28k/1k throughput number is
# measuring a broken server. Reporting it would be worse than reporting nothing, so
# the perf half is skipped unless NIAH passes. Override with NIAH_GATE=0.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="/run_logs/${SLURM_JOB_ID}"
NIAH_LOG="${LOGDIR}/niah_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}.log"

echo "==== PHASE 1/2: NIAH long-context retrieval ===="
echo "port=${BENCHMARK_PORT}  model=${MODEL_PATH}  sizes=${NIAH_WORDS:-2000,8000,20000,35000}"

# Poll until the router serves. On a fresh boot it registers a few seconds after the
# workers report ready, so a blind sleep either wastes time or starts too early.
# Poll with a REAL completion, not /v1/models. That endpoint is wrong in BOTH
# directions on this router: it returned 200 while every request 503'd (the 11:55
# run: 20 perf cells of 0.00 tok/s recorded as SUCCESS), and it returns 503
# "No prefill servers available" right now while completions succeed end to end.
# Its listing path and its forwarding path consult different state. A completion
# is the only probe that confirms router + backend + P->D KV transfer together.
_ready=0
for _i in $(seq 1 60); do
    _probe=$(curl -s --max-time 120 "http://127.0.0.1:${BENCHMARK_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The CEO of AMD is\",\"max_tokens\":8,\"temperature\":0}" 2>&1)
    if echo "$_probe" | grep -q '"text"'; then
        _ready=1; echo "[niah] end-to-end probe OK after ~$((_i*10))s"; break
    fi
    sleep 10
done
if [ "$_ready" != 1 ]; then
    echo "[niah] FATAL: no successful completion in 600s -- not benchmarking a dead server."
    echo "[niah] last response: ${_probe:0:400}"
    exit 1
fi
echo "[niah] probe: $(echo "$_probe" | head -c 200)"


# Invoked directly, NOT via benchmark_niah.sh: that wrapper ends in `| tee`, whose
# exit status is tee's (always 0), so a FAILING NIAH would report success and the
# gate below would be inert. Pipe to tee here but recover the real status.
set -o pipefail
NIAH_URL="http://127.0.0.1:${BENCHMARK_PORT}/v1/chat/completions" \
NIAH_MODEL="${MODEL_PATH}" \
NIAH_WORDS="${NIAH_WORDS:-2000,8000,20000,35000}" \
NIAH_MAXTOK="${NIAH_MAXTOK:-2048}" \
NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}" \
NIAH_WARMUP="${NIAH_WARMUP:-1}" \
  python3 "${DIR}/benchmark_niah.py" 2>&1 | tee -a "${NIAH_LOG}"
_niah_rc=${PIPESTATUS[0]}
set +o pipefail

echo "==== NIAH exit=${_niah_rc} -> ${NIAH_LOG} ===="

if [ "${NIAH_GATE:-1}" = "1" ] && [ "${_niah_rc}" != "0" ]; then
    echo "==== PHASE 2/2 SKIPPED: NIAH did not pass (exit ${_niah_rc}) ===="
    echo "A 28k/1k throughput number from a server that cannot retrieve a needle is"
    echo "not a result. Fix accuracy first, or re-run with NIAH_GATE=0 to force perf."
    exit "${_niah_rc}"
fi

echo "==== PHASE 2/2: ISL/OSL perf sweep ===="
# Defaults are the user's spec: 28k in / 1k out at concurrency 16, 32, 64.
# SHAPE_WARMUP=1 matters here -- benchmark_xPyD.sh's own comment records that without
# it the first measured cell absorbs residual JIT and reports 302 ms TPOT against an
# ~89 ms steady state, i.e. a 3.4x error on the headline metric.
export BENCHMARK_COMBINATIONS="${PERF_COMBINATIONS:-28672/1024}"
export BENCHMARK_CON="${PERF_CON:-16 32 64}"
export SHAPE_WARMUP="${SHAPE_WARMUP:-1}"
bash "${DIR}/benchmark_xPyD.sh"
