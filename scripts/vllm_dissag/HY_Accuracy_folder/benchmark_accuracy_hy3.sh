#!/bin/bash
# =============================================================================
# benchmark_accuracy_hy3.sh - consolidated accuracy benchmark for Hy3 disagg/EP
# =============================================================================
# Runs the self-contained accuracy tests (no external dataset / network needed)
# against a live vLLM disagg server, and emits one combined JSON + a verdict line.
# Accuracy is the central risk for WideEP (EP32 silently emits `!!!` while perf
# benchmarks still report 200 OK), so this is meant to run at EVERY config.
#
# Tiers (all greedy / temperature=0 -> deterministic, comparable across configs):
#   1. KNOWN-ANSWER  (accuracy_eval.py --known) : ~20 factual/math/code prompts scored
#                     against ground truth. Gross-correctness gate; EP32 garbage -> 0%.
#   2. NIAH          (niah_probe.py)            : needle-in-haystack retrieval at
#                     54k/128k/256k x depths -> long-context KV-path correctness.
#   3. EQUIVALENCE   (accuracy_probe.py compare): greedy exact-match vs an EP16 golden
#                     (optional; only if a golden json is provided / present).
#
# Usage:
#   benchmark_accuracy_hy3.sh                 # uses env defaults below
#   benchmark_accuracy_hy3.sh <url> <tag> [golden_json]
#
# Env (overridable):
#   ACC_URL        server url            (default http://127.0.0.1:${BENCHMARK_PORT:-30000})
#   ACC_TAG        label for outputs     (default ${MODEL_NAME}_xP${xP}yD${yD} or 'run')
#   ACC_OUT_DIR    output dir            (default /shared_inference/$USER/Tencent_HY3/accuracy)
#   NIAH_LENGTHS   context lengths       (default "54000 128000 256000")
#   NIAH_DEPTHS    needle depths         (default "0.1 0.5 0.9")
#   ACC_GOLDEN     golden json for equiv (default: skip if unset/missing)
#   ACC_SKIP_NIAH  =1 to skip NIAH (fast mode)
# =============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

ACC_URL="${1:-${ACC_URL:-http://127.0.0.1:${BENCHMARK_PORT:-30000}}}"
ACC_TAG="${2:-${ACC_TAG:-${MODEL_NAME:-run}_xP${xP:-1}yD${yD:-1}}}"
ACC_GOLDEN="${3:-${ACC_GOLDEN:-}}"
# Container runs as root ($USER empty); USER_NAME is plumbed in by the slurm. Fall back safely.
_ACC_USER="${USER:-${USER_NAME:-ravgupta}}"
ACC_OUT_DIR="${ACC_OUT_DIR:-/shared_inference/${_ACC_USER}/Tencent_HY3/accuracy}"
NIAH_LENGTHS="${NIAH_LENGTHS:-54000 128000 256000}"
NIAH_DEPTHS="${NIAH_DEPTHS:-0.1 0.5 0.9}"
mkdir -p "$ACC_OUT_DIR"

echo "============================================================"
echo "  Hy3 ACCURACY BENCHMARK"
echo "  url=$ACC_URL  tag=$ACC_TAG  out=$ACC_OUT_DIR"
echo "============================================================"

# Resolve served model id. The vllm-router (port 30000) does NOT proxy /v1/models
# (only /v1/completions), so /v1/models fails behind the router. Prefer the known
# served name = MODEL_PATH (full path on this stack), then ACC_MODEL override, and
# only fall back to /v1/models (works when hitting a raw vLLM serve port directly).
MODEL="${ACC_MODEL:-${MODEL_PATH:-}}"
if [ -z "$MODEL" ]; then
    MODEL=$(python3 -c "import json,urllib.request;print(json.loads(urllib.request.urlopen('$ACC_URL/v1/models',timeout=30).read())['data'][0]['id'])" 2>/dev/null)
fi
if [ -z "$MODEL" ]; then
    echo "ERROR: could not resolve model id (set ACC_MODEL or MODEL_PATH). url=$ACC_URL" >&2
    exit 1
fi
# Sanity: confirm the server answers before the full suite. The FIRST disagg
# request can be very slow (final cold kernel warmup), so retry up to ~10 min
# with a long per-attempt timeout rather than failing on a one-shot 60s ping.
if ! python3 -c "
import json,urllib.request,time
ok=False
# Probe with a REAL (non-stream) completion — matches the working benchmark path.
# EP32 cold pipeline can take ~100s+ for the first generation, so use a generous
# per-attempt timeout and budget. Sanity errors are NOT suppressed (kept visible).
for i in range(8):  # 8 x 240s = 32 min budget
    try:
        # Real completion, matching the proven-working warmup curl (vllm_disagg_mori_ep.sh:590).
        # The FIRST inference after 'Ready' triggers last-mile AITER clang/gfx942 JIT (can take
        # minutes) — the generous per-attempt timeout + budget rides through that cold compile,
        # which is the actual failure mode (NOT request shape; model/stream both work once warm).
        b=json.dumps({'prompt':'Who is AMD CEO?','max_tokens':10,'temperature':0,'top_k':1,'stream':False}).encode()
        r=urllib.request.Request('$ACC_URL/v1/completions',data=b,headers={'Content-Type':'application/json','x-request-id':f'sanity-{i}'})
        resp=urllib.request.urlopen(r,timeout=240).read()
        json.loads(resp)['choices'][0]['text']; ok=True; break
    except Exception as e:
        print(f'[sanity] attempt {i+1} not ready: {repr(e)[:120]}',flush=True); time.sleep(10)
print('ok' if ok else 'fail')
" | grep -q '^ok'; then
    echo "ERROR: server not answering /v1/completions at $ACC_URL after retries (model=$MODEL)" >&2
    exit 1
fi
echo "served model: $MODEL"

# ---- Tier 1: known-answer (scored, fast) ----
echo; echo "--- Tier 1: known-answer correctness ---"
python3 "$DIR/accuracy_eval.py" --url "$ACC_URL" --model "$MODEL" --known \
    --out "$ACC_OUT_DIR/acc_${ACC_TAG}.json"
T1=$?

# ---- Tier 2: NIAH long-context retrieval ----
if [ "${ACC_SKIP_NIAH:-0}" != "1" ]; then
    echo; echo "--- Tier 2: NIAH long-context retrieval ($NIAH_LENGTHS) ---"
    python3 "$DIR/niah_probe.py" --url "$ACC_URL" --model "$MODEL" \
        --lengths $NIAH_LENGTHS --depths $NIAH_DEPTHS \
        --out "$ACC_OUT_DIR/niah_${ACC_TAG}.json"
else
    echo "--- Tier 2: NIAH skipped (ACC_SKIP_NIAH=1) ---"
fi

# ---- Tier 3: equivalence vs golden (optional) ----
if [ -n "$ACC_GOLDEN" ] && [ -f "$ACC_GOLDEN" ]; then
    echo; echo "--- Tier 3: greedy equivalence vs golden ($ACC_GOLDEN) ---"
    python3 "$DIR/accuracy_probe.py" compare --url "$ACC_URL" --model "$MODEL" \
        --golden "$ACC_GOLDEN" --out "$ACC_OUT_DIR/equiv_${ACC_TAG}.json"
else
    echo "--- Tier 3: equivalence skipped (no golden; set ACC_GOLDEN to enable) ---"
fi

# ---- combined verdict ----
echo; echo "=== VERDICT [$ACC_TAG] ==="
python3 - "$ACC_OUT_DIR" "$ACC_TAG" <<'PY'
import json, os, sys
d, tag = sys.argv[1], sys.argv[2]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
acc  = load(f"{d}/acc_{tag}.json")
niah = load(f"{d}/niah_{tag}.json")
equiv= load(f"{d}/equiv_{tag}_summary.json") or load(f"{d}/equiv_{tag}.json")
parts=[]
if acc and "known" in acc:
    k=acc["known"]; parts.append(f"known={100*k['rate']:.0f}% ({k['passed']}/{k['total']})")
if niah:
    parts.append(f"niah={100*niah.get('rate',0):.0f}% ({niah.get('passed',0)}/{niah.get('total',0)})")
if equiv and "exact" in equiv:
    parts.append(f"equiv={100*equiv['exact']/equiv['total']:.0f}% ({equiv['exact']}/{equiv['total']})")
verdict = " | ".join(parts) if parts else "no results"
# pass/fail heuristic: known>=80% AND (no niah or niah>=80%)
ok = bool(acc and acc.get("known",{}).get("rate",0) >= 0.8)
if niah is not None: ok = ok and niah.get("rate",0) >= 0.8
print(f"  {verdict}")
print(f"  STATUS: {'PASS' if ok else 'FAIL/REVIEW'}")
PY
echo "outputs: $ACC_OUT_DIR/{acc,niah,equiv}_${ACC_TAG}.json"
