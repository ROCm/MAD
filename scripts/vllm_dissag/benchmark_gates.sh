#!/bin/bash
# =============================================================================
# benchmark_gates.sh - accuracy and KV-transfer gates for a P/D deployment, in this order:
#   1. every-rank KV-transfer probe   benchmark_read_probe.py
#   2. GSM8K 5-shot, all 1,319        lm-eval[api]==0.4.13 local-completions, 64 concurrent,
#                                     max_gen_toks 512 (exact_match, strict-match filter)
#   3. NIAH                           benchmark_niah_corpus.py single: 64k/256k/512k x depth
#                                     0/0.5/1 x 2 needles = 18 requests
#   4. multi-turn needle              benchmark_niah_corpus.py multi at ~246k and ~490k:
#                                     4 sessions x 2 questions each
# Selected via BENCHMARK_SCRIPT=gates; standalone with GATES_BASE_URL and METRICS_ENDPOINTS.
# Every request must be disaggregated (the slurm proxies always are; on llm-d use an
# always-disaggregate decider for this run). perf.csv gets one row per gate.
#
# Env: GATES_BASE_URL (default http://127.0.0.1:${BENCHMARK_PORT}), SERVED_MODEL_NAME and
#   TOKENIZER (default ${MODEL_PATH}), METRICS_ENDPOINTS (default from AGENTIC_SERVER_METRICS),
#   GATES="probe gsm8k niah needle" (subset to run), GSM8K_LIMIT (unset = all), MT_CORPUS.
# =============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp=$(date "+%Y%m%d_%H%M%S")
URL="${GATES_BASE_URL:-http://127.0.0.1:${BENCHMARK_PORT:-30000}}"; URL="${URL%/}"
MODEL="${SERVED_MODEL_NAME:-${MODEL_PATH:-}}"
TOK="${TOKENIZER:-${MODEL_PATH:-}}"
GATES="${GATES:-probe gsm8k niah needle}"
OUT="${GATES_RUN_DIR:-/run_logs/${SLURM_JOB_ID:-local}/gates_${timestamp}_xP${xP:-1}_yD${yD:-1}_${MODEL_NAME:-model}}"
PERF_CSV="${GATES_PERF_CSV:-$(dirname "$OUT")/perf.csv}"
CORPUS="${MT_CORPUS:-/tmp/pg1184.txt}"
if [[ -z "${METRICS_ENDPOINTS:-}" && -n "${AGENTIC_SERVER_METRICS:-}" ]]; then
  read -r _p _d _ <<< "${AGENTIC_SERVER_METRICS}"
  METRICS_ENDPOINTS="prefill=${_p}${_d:+ decode=${_d}}"
fi
[[ -n "$MODEL" && -n "$TOK" ]] || { echo "[gates] set MODEL_PATH (or SERVED_MODEL_NAME and TOKENIZER)" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/gates.log") 2>&1
echo "==== P/D gates (${GATES}) url=${URL} model=${MODEL} out=${OUT} ===="

for _i in $(seq 1 60); do
  curl -sf -m 5 -o /dev/null "${URL}/v1/models" && break
  sleep 5
done

rc=0
row() { # value metric status
  python3 - "$PERF_CSV" "${MODEL_NAME:-$MODEL}" "$1" "$2" "$3" <<'PY'
import csv, os, sys
sys.path.insert(0, os.environ["GATES_DIR"])
import parse_to_csv as p
path, model, value, metric, status = sys.argv[1:]
new = not (os.path.exists(path) and os.path.getsize(path) > 0)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=p.PERF_CSV_FIELDS)
    if new:
        w.writeheader()
    r = {"model": model, "performance": value, "metric": metric, "status": status}
    r.update(p._get_run_metadata("vllm"))
    w.writerow(r)
PY
}
export GATES_DIR="$DIR"

if [[ " $GATES " == *" probe "* ]]; then
  python3 "$DIR/benchmark_read_probe.py" --url "$URL" --model "$MODEL" \
    --metrics-endpoints "${METRICS_ENDPOINTS:-}" --out "$OUT/probe"
  p=$?; [[ $p == 0 ]] && st=SUCCESS || { st=FAILURE; rc=1; }
  row "$([[ $p == 0 ]] && echo 1 || echo 0)" "kv transfer probe pass (every rank)" "$st"
fi

if [[ " $GATES " == *" gsm8k "* ]]; then
  python3 -m venv --system-site-packages "$OUT/lm-eval-venv" \
    && "$OUT/lm-eval-venv/bin/pip" -q install "lm-eval[api]==0.4.13"
  "$OUT/lm-eval-venv/bin/lm_eval" --model local-completions \
    --model_args "base_url=${URL}/v1/completions,model=${MODEL},tokenizer=${TOK},tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=64,max_retries=3,max_gen_toks=512" \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 ${GSM8K_LIMIT:+--limit "$GSM8K_LIMIT"} \
    --output_path "$OUT/gsm8k"
  acc="$(python3 - "$OUT/gsm8k" <<'PY'
import glob, json, sys
fs = sorted(glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True))
r = json.load(open(fs[-1]))["results"]["gsm8k"] if fs else {}
v = r.get("exact_match,strict-match")
print("" if v is None else f"{100 * v:.2f}")
PY
)"
  if [[ -n "$acc" ]]; then row "$acc" "gsm8k 5-shot exact_match strict-match (%)" SUCCESS
  else row 0 "gsm8k 5-shot exact_match strict-match (%)" FAILURE; rc=1; fi
  echo "[gates] GSM8K strict-match: ${acc:-none}"
fi

if [[ " $GATES " == *" niah "* || " $GATES " == *" needle "* ]]; then
  [[ -s "$CORPUS" ]] || curl -fsSL --retry 3 https://www.gutenberg.org/ebooks/1184.txt.utf-8 -o "$CORPUS"
  echo "99245b8f3f6b637a3670d37305bb1ba52003f05797b6638571562999fd623f9c  $CORPUS" | sha256sum -c - \
    || { echo "[gates] corpus SHA256 mismatch: $CORPUS" >&2; exit 1; }
  common=(--url "$URL" --model "$MODEL" --tokenizer "$TOK" --corpus "$CORPUS")
  [[ " $GATES " == *" niah "* ]] && python3 "$DIR/benchmark_niah_corpus.py" single "${common[@]}" \
    --out "$OUT/niah" --lengths 65536,262144,524288 --depths 0,0.5,1.0 --needles 2 --concurrency 6
  if [[ " $GATES " == *" needle "* ]]; then
    for s in 246000 490000; do
      python3 "$DIR/benchmark_niah_corpus.py" multi "${common[@]}" --out "$OUT/needle-$s" \
        --start "$s" --turns 3 --new-tokens 10000 --k 3 --sessions 4 --tag "mt$s"
    done
  fi
  python3 "$DIR/parse_to_csv.py" "$OUT/gates.log" --niah-mt --perf-csv "$OUT/niah-perf.csv" \
    --model-name "${MODEL_NAME:-$MODEL}" \
    && { if [[ -s "$PERF_CSV" ]]; then tail -n +2 "$OUT/niah-perf.csv"; else cat "$OUT/niah-perf.csv"; fi; } >> "$PERF_CSV"
fi

echo "==== gates done (rc=${rc}) -> ${OUT} ===="
exit $rc
