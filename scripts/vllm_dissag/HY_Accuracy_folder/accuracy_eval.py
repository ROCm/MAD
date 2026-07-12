#!/usr/bin/env python3
"""Accuracy evaluation for a running vLLM (disagg/EP) server — SCORED, not just consistency.

Two tiers, both greedy (temperature=0) so results are deterministic and comparable
across EP/KV configs:

  Tier 1  KNOWN-ANSWER set : ~20 fixed factual/math/code prompts with EXPECTED answers,
          scored by substring/normalized match. Fast (seconds). Catches gross
          corruption (the EP32 `!!!` case scores 0%) AND wrong-but-fluent output.

  Tier 2  GSM8K            : grade-school math word problems pulled from HuggingFace
          (`datasets` is in the image; no lm-eval needed). N-shot prompting, extract the
          final integer, exact-match against the gold answer -> real accuracy %.

Unlike accuracy_probe.py (which checks if config A == config B), this scores against
GROUND TRUTH, so a single config gets an absolute number (e.g. "EP16 = 92% GSM8K").

Usage:
  # Tier 1 only (fast)
  python3 accuracy_eval.py --url http://127.0.0.1:30000 --model <served-name> --known --out acc_ep16.json
  # Tier 1 + GSM8K (40 problems, 5-shot)
  python3 accuracy_eval.py --url http://127.0.0.1:30000 --model <served-name> \
      --known --gsm8k 40 --gsm8k-shots 5 --out acc_ep16.json

Notes:
  * --model must be the SERVED name. On this stack that is the full model PATH
    (e.g. /mnt/m2m_nobackup/models_blog/Hy3-preview) unless --served-model-name was set.
    Use --auto-model to query /v1/models and use the first id.
  * GSM8K download needs network + `datasets`. If offline, pass --gsm8k-file <jsonl>
    with {"question","answer"} rows.
"""
import argparse, json, re, sys, urllib.request

# ---------------------------------------------------------------------------
# Tier 1 — known-answer probes. `match` semantics:
#   list of acceptable substrings (case-insensitive); pass if ANY appears in output.
# ---------------------------------------------------------------------------
KNOWN = [
    {"prompt": "The capital of France is", "accept": ["paris"]},
    {"prompt": "The capital of Japan is", "accept": ["tokyo"]},
    {"prompt": "The largest planet in the solar system is", "accept": ["jupiter"]},
    {"prompt": "Water freezes at a temperature of", "accept": ["0", "zero", "32"]},
    {"prompt": "Question: What is 2+2? Answer:", "accept": ["4", "four"]},
    {"prompt": "Question: What is 2*3? Answer:", "accept": ["6", "six"]},
    {"prompt": "Question: What is 10-7? Answer:", "accept": ["3", "three"]},
    {"prompt": "Question: What is the square root of 144? Answer:", "accept": ["12", "twelve"]},
    {"prompt": "The chemical symbol for gold is", "accept": ["au"]},
    {"prompt": "The author of 'Romeo and Juliet' is", "accept": ["shakespeare"]},
    {"prompt": "Complete the sequence: 2, 4, 8, 16,", "accept": ["32"]},
    {"prompt": "The boiling point of water at sea level in Celsius is", "accept": ["100"]},
    {"prompt": "Translate 'Hello' to French:", "accept": ["bonjour"]},
    {"prompt": "Roses are red, violets are", "accept": ["blue"]},
    {"prompt": "The first president of the United States was", "accept": ["washington"]},
    {"prompt": "The speed of light is approximately", "accept": ["300", "3", "186", "299"]},
    {"prompt": "In Python, the keyword to define a function is", "accept": ["def"]},
    {"prompt": "The opposite of 'hot' is", "accept": ["cold"]},
    {"prompt": "How many days are in a week? Answer:", "accept": ["7", "seven"]},
    {"prompt": "The chemical formula for water is", "accept": ["h2o", "h₂o"]},
]

_REQ_N = [0]

def call(url, model, prompt, max_tokens, stop=None):
    # Use the EXACT protocol of `vllm bench serve` (the proven-stable client that
    # ran 128 concurrent reqs with 0 crashes), which raw probes did NOT:
    #   (1) stream=True + parse SSE chunks
    #   (2) an explicit x-request-id header (prefix like the bench client's
    #       request_id_prefix). The MoRIIO router embeds the prefill/decode routing
    #       into the request id; without a client-provided id the bare auto-id gets
    #       no kv_transfer injection -> decode KeyError remote_host -> crash.
    _REQ_N[0] += 1
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    if stop:
        payload["stop"] = stop
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "x-request-id": f"acc-{_REQ_N[0]:06d}"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=data, headers=headers)
    text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                text += json.loads(chunk)["choices"][0]["text"]
            except Exception:
                pass
    return text

def resolve_model(url, model, auto):
    if model and not auto:
        return model
    req = urllib.request.Request(url.rstrip("/") + "/v1/models")
    with urllib.request.urlopen(req, timeout=60) as r:
        ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
    if not ids:
        raise RuntimeError("no model ids from /v1/models")
    return ids[0]

def _scored_hit(out, accept):
    # Strict-but-fair match: score the FIRST LINE only (the answer; we stop at \n),
    # with word boundaries so a short answer like "3" must appear AS a token, not
    # buried in coincidental garbage. Scoring the whole first line (not a fixed char
    # window) avoids cutting off valid longer answers like "...is 12.". Avoids both
    # failure modes of bare substring-anywhere: (1) over-generated extra Q/A after the
    # answer (excluded by stop=\n), (2) degenerate garbage with a stray digit
    # (excluded by the word boundary).
    head = out.strip().splitlines()[0].lower() if out.strip() else ""
    for a in accept:
        al = a.lower()
        if re.search(r"(^|[^a-z0-9])" + re.escape(al) + r"([^a-z0-9]|$)", head):
            return True
    return False

# ---- Tier 1 ----
def run_known(url, model):
    rows, ok = [], 0
    for item in KNOWN:
        try:
            # stop at newline / next "Question:" so we score the FIRST answer only,
            # not 32 tokens of continuation.
            out = call(url, model, item["prompt"], 32, stop=["\n", "Question:"])
            hit = _scored_hit(out, item["accept"])
        except Exception as e:
            out = f"ERROR: {repr(e)[:100]}"; hit = False
        ok += int(hit)
        rows.append({"prompt": item["prompt"], "accept": item["accept"],
                     "got": out.strip()[:80], "pass": hit})
    return {"passed": ok, "total": len(KNOWN), "rate": ok / len(KNOWN), "rows": rows}

# ---- Tier 2 GSM8K ----
GSM_ANS = re.compile(r"####\s*(-?[\d,]+)")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def load_gsm8k(n, path=None):
    if path:
        rows = [json.loads(l) for l in open(path) if l.strip()][:n]
        return [(r["question"], r["answer"]) for r in rows]
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"]; a = ds[i]["answer"]
        m = GSM_ANS.search(a)
        gold = m.group(1).replace(",", "") if m else None
        out.append((q, gold))
    return out

def build_fewshot(shots):
    ex = [
        ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. "
         "How many clips did she sell altogether in April and May?",
         "In April she sold 48. In May she sold 48/2 = 24. Altogether 48+24 = 72. The answer is 72."),
        ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
         "How much did she earn?",
         "Per minute she earns 12/60 = $0.2. For 50 minutes she earned 50*0.2 = $10. The answer is 10."),
        ("Betty is saving for a $100 wallet. She has half of the money she needs. Her parents give her $15 "
         "and her grandparents twice as much as her parents. How much more does she need?",
         "Half of 100 is 50. Grandparents give 2*15 = 30. Total now 50+15+30 = 95. She needs 100-95 = 5. The answer is 5."),
        ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
         "Each time he writes 3*2 = 6 pages. Twice a week is 6*2 = 12 pages. Per year 12*52 = 624. The answer is 624."),
        ("Ten more than twice a number is 30. What is the number?",
         "Twice the number plus 10 is 30, so twice the number is 20, so the number is 10. The answer is 10."),
    ][:shots]
    s = ""
    for q, a in ex:
        s += f"Question: {q}\nAnswer: {a}\n\n"
    return s

def extract_answer(text):
    # prefer "The answer is X", else last number
    m = re.search(r"answer is\s*\$?(-?[\d,]+\.?\d*)", text, re.I)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None

def run_gsm8k(url, model, n, shots, path=None):
    data = load_gsm8k(n, path)
    fewshot = build_fewshot(shots)
    rows, ok = [], 0
    for q, gold in data:
        prompt = fewshot + f"Question: {q}\nAnswer:"
        try:
            out = call(url, model, prompt, 320, stop=["Question:", "\n\n"])
            pred = extract_answer(out)
            hit = (pred is not None and gold is not None and
                   abs(float(pred) - float(gold)) < 1e-6)
        except Exception as e:
            out = f"ERROR: {repr(e)[:80]}"; pred = None; hit = False
        ok += int(hit)
        rows.append({"gold": gold, "pred": pred, "pass": hit, "got": out.strip()[:120]})
    return {"passed": ok, "total": len(data), "rate": ok / len(data) if data else 0, "rows": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--auto-model", action="store_true", help="resolve model id from /v1/models")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--gsm8k", type=int, default=0, help="num GSM8K problems (0=skip)")
    ap.add_argument("--gsm8k-shots", type=int, default=5)
    ap.add_argument("--gsm8k-file", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = resolve_model(a.url, a.model, a.auto_model)
    report = {"model": model, "url": a.url}
    print(f"[acc] model={model}")

    if a.known:
        r = run_known(a.url, model)
        report["known"] = r
        print(f"[acc] KNOWN-ANSWER: {r['passed']}/{r['total']} ({100*r['rate']:.1f}%)")
        for row in r["rows"]:
            if not row["pass"]:
                print(f"    FAIL {row['prompt'][:40]!r} -> {row['got'][:40]!r}")

    if a.gsm8k > 0:
        r = run_gsm8k(a.url, model, a.gsm8k, a.gsm8k_shots, a.gsm8k_file)
        report["gsm8k"] = r
        print(f"[acc] GSM8K ({a.gsm8k_shots}-shot): {r['passed']}/{r['total']} ({100*r['rate']:.1f}%)")

    json.dump(report, open(a.out, "w"), indent=2)
    print(f"[acc] full report -> {a.out}")
    # overall verdict line for quick scan
    parts = []
    if "known" in report: parts.append(f"known={100*report['known']['rate']:.0f}%")
    if "gsm8k" in report: parts.append(f"gsm8k={100*report['gsm8k']['rate']:.0f}%")
    print(f"[acc] VERDICT {model}: " + " ".join(parts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
