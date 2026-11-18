#!/bin/bash
# Two-pass prefix-reuse benchmark for tiered KV offloading (KV_OFFLOAD), on the
# robust `vllm bench serve` (asyncio) path. No multiprocessing / mp.Queue.
#
# WHY TWO PASSES. Each conversation has a UNIQUE large prefix (N concurrent convs =
# N distinct prefixes, so their KV overflows the HBM cap and evicts — a single shared
# prefix would dedup and never evict). Two passes emulate a 2-turn conversation:
#   Pass A (prime):   prompt = prefix + q1          , tiny output (OSL1)
#   Pass B (measure): prompt = prefix + q1 + q2      , output OSL2
# The reused span is prefix+q1. When the resident set (num_prompts distinct primed
# convs) exceeds the HBM cap, primed KV evicts; on Pass B:
#   KV_OFFLOAD on  -> evicted prefix RELOADS from CPU  -> low turn-2 TTFT
#   KV_OFFLOAD off -> evicted prefix RECOMPUTED        -> high turn-2 TTFT
# Below the crossover everything fits HBM and all arms match (baseline).
#
# KEY DESIGN (vs the old np=con harness): num_prompts (np) is the HBM-PRESSURE knob
# (all np primed convs stay resident in the prefix cache), con is the SATURATION knob
# (in-flight requests). Hold con LOW (de-saturated) and drive np UP to cross the
# HBM->CPU boundary — so turn-2 TTFT reflects reload-vs-recompute, not queue wait.
#
# SWEEP is an explicit point list:  MT_SWEEP="con:np:isl con:np:isl ..."
#   e.g.  MT_SWEEP="8:64:32768 8:128:32768 8:256:32768 8:384:32768"
# If MT_SWEEP is unset, falls back to the legacy MT_CON_LIST x MT_PREFIX_LEN_LIST
# grid with MT_NUM_PROMPTS (np defaults to con).  Select via BENCHMARK_SCRIPT=multiturn.
#
# INSTRUMENTATION. Around each measure pass we snapshot the prefill-role vLLM Prometheus
# /metrics (PREFILL_METRICS_URL, exported by the connector) and
# record the GPU prefix-cache and external(offload) hit-rate deltas into perf.csv — so
# every run self-reports whether reuse/offload actually fired. Missing URLs => skipped.
#
# Env knobs (defaults):
#   MT_SWEEP      ""      explicit "con:np:isl ..." points (else legacy grid)
#   MT_Q1_LEN     1024    turn-1 user question (part of the reused span)
#   MT_Q2_LEN     2048    turn-2 user question (fresh prefill on the measure pass)
#   MT_OSL1       16      turn-1 output (prime; kept small)
#   MT_OSL2       128     turn-2 output (measure; small — signal is TTFT, keeps runtime low)
#   MT_SEED       42
#   legacy grid:  MT_CON_LIST, MT_PREFIX_LEN(_LIST), MT_NUM_PROMPTS
set -u

timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-2584}"
RESULT_DIR="/run_logs/${SLURM_JOB_ID}"
LOG="${RESULT_DIR}/multiturn_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

MT_Q1_LEN="${MT_Q1_LEN:-1024}"
MT_Q2_LEN="${MT_Q2_LEN:-2048}"
MT_OSL1="${MT_OSL1:-16}"
MT_OSL2="${MT_OSL2:-128}"
MT_SEED="${MT_SEED:-42}"

# Prefill metrics endpoint (exported by the connector; may be empty).
PREFILL_METRICS_URL="${PREFILL_METRICS_URL:-}"
DECODE_METRICS_URL="${DECODE_METRICS_URL:-}"

# ── build the sweep point list ────────────────────────────────────────────────
# Primary: MT_SWEEP="con:np:isl ...". Fallback: legacy grid (preserves old runs).
MT_SWEEP="${MT_SWEEP:-}"
if [ -z "$MT_SWEEP" ]; then
  _CON_LIST="${MT_CON_LIST:-8 16 32 64 128 256 512}"
  _PREFIX="${MT_PREFIX_LEN:-32768}"
  _PREFIX_LIST="${MT_PREFIX_LEN_LIST:-$_PREFIX}"; _PREFIX_LIST="${_PREFIX_LIST//,/ }"
  _NP="${MT_NUM_PROMPTS:-}"
  for _isl in $_PREFIX_LIST; do
    for _con in $_CON_LIST; do
      MT_SWEEP="$MT_SWEEP ${_con}:${_NP:-$_con}:${_isl}"
    done
  done
fi
# normalize (commas -> spaces, collapse whitespace)
MT_SWEEP="$(echo "${MT_SWEEP//,/ }" | tr -s ' ' | sed 's/^ *//;s/ *$//')"

echo "==== Two-pass offload benchmark ${LOG} ====" | tee -a ${LOG}.log
echo "sweep='${MT_SWEEP}'" | tee -a ${LOG}.log
echo "q1=${MT_Q1_LEN} q2=${MT_Q2_LEN} osl1=${MT_OSL1} osl2=${MT_OSL2} seed=${MT_SEED} KV_OFFLOAD=${KV_OFFLOAD:-none}" | tee -a ${LOG}.log
echo "metrics: prefill='${PREFILL_METRICS_URL:-<none>}' decode='${DECODE_METRICS_URL:-<none>}'" | tee -a ${LOG}.log
echo "UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}.log

echo "[setup] checking numpy..." | tee -a ${LOG}.log
pip install -q --user numpy 2>&1 | grep -v 'already satisfied' | tee -a ${LOG}.log || true

# ── unique ISLs + max np per ISL (dataset pool size) ──────────────────────────
_ISLS=""
for _pt in $MT_SWEEP; do
  _i="${_pt##*:}"
  case " $_ISLS " in *" $_i "*) ;; *) _ISLS="$_ISLS $_i" ;; esac
done

_maxnp_for_isl() {  # echo max np among points with isl=$1
  local isl="$1" mx=0 c n i
  for _pt in $MT_SWEEP; do
    IFS=: read -r c n i <<<"$_pt"
    [ "$i" = "$isl" ] && [ "$n" -gt "$mx" ] && mx="$n"
  done
  echo "$mx"
}

# ── metrics snapshot helper (robust; no-op if url empty or unreachable) ────────
_metrics_snapshot() {  # $1=url  $2=outfile
  local url="$1" out="$2"
  : > "$out"
  [ -z "$url" ] && return 0
  python3 - "$url" "$out" <<'PYEOF' 2>/dev/null || true
import sys, json, urllib.request
url, out = sys.argv[1], sys.argv[2]
keys = ("vllm:prefix_cache_queries","vllm:prefix_cache_hits",
        "vllm:gpu_prefix_cache_queries","vllm:gpu_prefix_cache_hits",
        "vllm:external_prefix_cache_queries","vllm:external_prefix_cache_hits")
# kept with labels (full series name) so per-role offload store/load and per-source hits survive
lab_prefixes = ("vllm:kv_offload", "vllm:prompt_tokens_cached_by_source", "vllm:prompt_tokens_by_source",
                # per-engine (= per DP rank) request/token counters + histogram _sum/_count (no _bucket): which ranks served
                "vllm:request_success", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                "vllm:request_prefill_time_seconds", "vllm:time_to_first_token_seconds")
try:
    txt = urllib.request.urlopen(url, timeout=5).read().decode()
except Exception:
    json.dump({}, open(out, "w")); sys.exit(0)
agg = {}
for ln in txt.splitlines():
    if not ln or ln[0] == "#":
        continue
    name = ln.split("{", 1)[0].split(" ", 1)[0]
    base = name[:-6] if name.endswith("_total") else name
    if base in keys:
        try:
            agg[base] = agg.get(base, 0.0) + float(ln.rsplit(" ", 1)[1])
            if "{" in ln:  # per-engine series, kept alongside the sum
                agg[ln.rsplit(" ", 1)[0]] = float(ln.rsplit(" ", 1)[1])
        except Exception:
            pass
    elif name.startswith(lab_prefixes) and not name.endswith(("_bucket", "_created")):
        try:
            agg[ln.rsplit(" ", 1)[0]] = float(ln.rsplit(" ", 1)[1])
        except Exception:
            pass
json.dump(agg, open(out, "w"))
PYEOF
}

# ── generate datasets once per unique ISL ─────────────────────────────────────
for MT_PREFIX_LEN in $_ISLS; do
  _N="$(_maxnp_for_isl "$MT_PREFIX_LEN")"
  ISL_TAG="isl${MT_PREFIX_LEN}"
  PRIME_JSONL="${RESULT_DIR}/mt_prime_${ISL_TAG}.jsonl"
  MEASURE_JSONL="${RESULT_DIR}/mt_measure_${ISL_TAG}.jsonl"
  echo "" | tee -a ${LOG}.log
  echo "======== dataset ${ISL_TAG}  pool=${_N} convs ========" | tee -a ${LOG}.log

  export _MODEL_PATH="$MODEL_PATH" _N="$_N" _PREFIX="$MT_PREFIX_LEN" _Q1="$MT_Q1_LEN" \
         _Q2="$MT_Q2_LEN" _OSL1="$MT_OSL1" _OSL2="$MT_OSL2" _SEED="$MT_SEED" \
         _PRIME="$PRIME_JSONL" _MEASURE="$MEASURE_JSONL"
  python3 - <<'GEN_EOF' 2>&1 | tee -a ${LOG}.log
import os, json
import numpy as np
from transformers import AutoTokenizer

mp_          = os.environ["_MODEL_PATH"]
N            = int(os.environ["_N"])
L1           = int(os.environ["_PREFIX"]) + int(os.environ["_Q1"])   # reused span
L2           = int(os.environ["_Q2"])
osl1, osl2   = int(os.environ["_OSL1"]), int(os.environ["_OSL2"])
seed         = int(os.environ["_SEED"])

tok = AutoTokenizer.from_pretrained(mp_, trust_remote_code=True)
allowed = np.array(sorted(set(range(tok.vocab_size)) - set(tok.all_special_ids)))
rng = np.random.default_rng(seed)

first_prime = first_measure = None
with open(os.environ["_PRIME"], "w") as fp, open(os.environ["_MEASURE"], "w") as fm:
    for i in range(N):
        ids = allowed[rng.integers(0, len(allowed), size=L1 + L2)]
        prime_text = tok.decode(ids[:L1].tolist(),        skip_special_tokens=True)
        q2_text    = tok.decode(ids[L1:L1 + L2].tolist(), skip_special_tokens=True)
        fp.write(json.dumps({"prompt": prime_text, "output_tokens": osl1}) + "\n")
        measure_text = prime_text + q2_text
        fm.write(json.dumps({"prompt": measure_text, "output_tokens": osl2}) + "\n")
        if i == 0:
            first_prime, first_measure = prime_text, measure_text

# self-check: measure must re-tokenize to a prefix whose leading tokens EXACTLY match
# prime's (block granularity 16). Re-encode both as the server will and report alignment.
pi = tok.encode(first_prime,   add_special_tokens=False)
mi = tok.encode(first_measure, add_special_tokens=False)
aligned = 0
for a, b in zip(pi, mi):
    if a != b:
        break
    aligned += 1
print(f"[selfcheck] prime reencodes to {len(pi)} tok, measure to {len(mi)} tok; "
      f"aligned prefix = {aligned} tok = {aligned // 16} full 16-tok blocks "
      f"({'OK' if aligned >= L1 - 16 else 'MISALIGNED <L1'})")
print(f"[gen] {N} convs -> prime(reuse span {L1} tok, osl {osl1}) + "
      f"measure(+q2 {L2}, osl {osl2})")
GEN_EOF
done

if [ "${MT_DRYRUN:-0}" = "1" ]; then
  echo "[MT_DRYRUN] dataset generation complete; skipping warmup/sweep." | tee -a ${LOG}.log
  for MT_PREFIX_LEN in $_ISLS; do
    wc -l "${RESULT_DIR}/mt_prime_isl${MT_PREFIX_LEN}.jsonl" \
          "${RESULT_DIR}/mt_measure_isl${MT_PREFIX_LEN}.jsonl" | tee -a ${LOG}.log
  done
  exit 0
fi

sleep 10

# ── warmup (small; does not touch the measured prefixes) ───────────────────────
vllm bench serve \
    --model "$MODEL_PATH" --backend vllm --host 127.0.0.1 --port "$BENCHMARK_PORT" \
    --dataset-name random --random-input-len 32 --random-output-len 32 --random-prefix-len 0 \
    --num-prompts 16 --request-rate inf --ignore-eos --max-concurrency 1 \
    2>&1 | tee -a ${LOG}.log >/dev/null

# ── one pass of one sweep point ───────────────────────────────────────────────
_run_pass() {
    local label="$1" con="$2" np="$3" seed="$4" file="$5" tag="$6"
    echo "[RUNNING] ${label} con=${con} np=${np} seed=${seed} ${tag}" | tee -a ${LOG}.log
    vllm bench serve \
        --model "$MODEL_PATH" --backend vllm --host 127.0.0.1 --port "$BENCHMARK_PORT" \
        --dataset-name custom --dataset-path "$file" \
        --num-prompts "$np" --request-rate inf --ignore-eos \
        --seed "$seed" --max-concurrency "$con" \
        --save-result --result-filename "${RESULT_DIR}/${tag}_${label}.json" \
        2>&1 | tee -a ${LOG}.log
}

idx=0
for _pt in $MT_SWEEP; do
    IFS=: read -r con np isl <<<"$_pt"
    seed=$((MT_SEED + idx))
    ISL_TAG="isl${isl}"
    TAG="isl${isl}_np${np}_con${con}"
    PRIME_JSONL="${RESULT_DIR}/mt_prime_${ISL_TAG}.jsonl"
    MEASURE_JSONL="${RESULT_DIR}/mt_measure_${ISL_TAG}.jsonl"
    echo "" | tee -a ${LOG}.log
    echo "==== point ${TAG} (seed ${seed}) ====" | tee -a ${LOG}.log

    _run_pass prime "$con" "$np" "$seed" "$PRIME_JSONL" "$TAG"
    sleep 10

    # snapshot metrics around the measure pass -> hit/retrieve deltas
    _metrics_snapshot "$PREFILL_METRICS_URL" "${RESULT_DIR}/${TAG}_pmetrics_pre.json"
    _metrics_snapshot "$DECODE_METRICS_URL" "${RESULT_DIR}/${TAG}_dmetrics_pre.json"
    _run_pass measure "$con" "$np" "$seed" "$MEASURE_JSONL" "$TAG"
    _metrics_snapshot "$PREFILL_METRICS_URL" "${RESULT_DIR}/${TAG}_pmetrics_post.json"
    _metrics_snapshot "$DECODE_METRICS_URL" "${RESULT_DIR}/${TAG}_dmetrics_post.json"

    sleep 5
    idx=$((idx + 1))
done

# ── parse measure-pass TTFT + cache-hit deltas -> perf.csv ────────────────────
export _RESULT_DIR="$RESULT_DIR" _SWEEP="$MT_SWEEP" _MODEL="$MODEL_NAME" \
       _OFFLOAD="${KV_OFFLOAD:-none}" _BACKEND="$([ "${KV_OFFLOAD:-none}" = none ] && echo none || echo "${OFFLOAD_BACKEND:-none}")"
python3 - <<'PARSE_EOF' 2>&1 | tee -a ${LOG}.log
import os, json, csv
import numpy as np

rd      = os.environ["_RESULT_DIR"]
points  = os.environ["_SWEEP"].split()
model   = os.environ["_MODEL"]
offld   = os.environ["_OFFLOAD"]
backend = os.environ["_BACKEND"]
perf    = f"{rd}/perf.csv"

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def ttft(d, k):
    return round(d.get(k), 1) if d and d.get(k) is not None else -1.0

def p99(d):
    t = d.get("ttfts") if d else None
    return round(float(np.percentile(t, 99)) * 1000, 1) if t else -1.0

def hitrate(pre, post, qk, hk):
    """delta hit-rate over the measure pass for counter pair (queries, hits)."""
    if not pre or not post:
        return -1.0, -1.0, -1.0
    dq = post.get(qk, 0.0) - pre.get(qk, 0.0)
    dh = post.get(hk, 0.0) - pre.get(hk, 0.0)
    rate = round(100.0 * dh / dq, 1) if dq > 0 else -1.0
    return round(dq, 0), round(dh, 0), rate

rows = []
for pt in points:
    con, npr, isl = pt.split(":")
    tag = f"isl{isl}_np{npr}_con{con}"
    m = load(f"{rd}/{tag}_measure.json")
    p = load(f"{rd}/{tag}_prime.json")
    if not m:
        print(f"[warn] missing measure json for {tag}", flush=True)
        continue
    pre  = load(f"{rd}/{tag}_pmetrics_pre.json")
    post = load(f"{rd}/{tag}_pmetrics_post.json")
    gq, gh, grate = hitrate(pre, post, "vllm:prefix_cache_queries", "vllm:prefix_cache_hits")
    if grate < 0:  # some builds name the gpu counters differently
        gq, gh, grate = hitrate(pre, post, "vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_hits")
    xq, xh, xrate = hitrate(pre, post, "vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_hits")
    rows.append({
        "model":                  model,
        "kv_offload":             offld,
        "offload_backend":        backend,
        "prefix_len":             isl,
        "np":                     npr,
        "con":                    con,
        "measure_ttft_mean_ms":   ttft(m, "mean_ttft_ms"),
        "measure_ttft_p50_ms":    ttft(m, "median_ttft_ms"),
        "measure_ttft_p99_ms":    p99(m),
        "prime_ttft_mean_ms":     ttft(p, "mean_ttft_ms"),
        "prime_ttft_p50_ms":      ttft(p, "median_ttft_ms"),
        "prime_ttft_p99_ms":      p99(p),
        "measure_req_throughput": round((m.get("request_throughput") or -1), 3),
        "measure_out_tok_s":      round((m.get("output_throughput") or -1), 1),
        "total_token_throughput": round((m.get("total_token_throughput") or -1), 1),
        "gpu_prefix_hit_rate":    grate,   # % of measure-pass prefix queries served from HBM
        "ext_offload_hit_rate":   xrate,   # % served from the CPU offload tier (reload fired)
        "gpu_prefix_hits":        gh,
        "ext_offload_hits":       xh,
    })
    r = rows[-1]
    print(f"  {tag:>28}  T1 p50={r['prime_ttft_p50_ms']}ms  T2 p50={r['measure_ttft_p50_ms']}ms "
          f"p99={r['measure_ttft_p99_ms']}ms  gpu_hit={r['gpu_prefix_hit_rate']}% "
          f"offload_hit={r['ext_offload_hit_rate']}%", flush=True)

if rows:
    hdr = (not os.path.exists(perf)) or os.path.getsize(perf) == 0
    with open(perf, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if hdr:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {perf}", flush=True)
else:
    print("[error] no measure result JSONs found", flush=True)
PARSE_EOF
