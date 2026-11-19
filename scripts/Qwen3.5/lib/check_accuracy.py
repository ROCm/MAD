#!/usr/bin/env python3
# Sanity/accuracy check: send 5 real prompts to a running OpenAI-compatible server,
# verify coherent + expected output. Catches corrupt-weight / bad-quant failures that
# throughput benchmarks (which use --ignore-eos on random tokens) miss.
#
# Usage: PORT, MODEL, OUT(json) via env. Exits 0 if all coherent, 1 otherwise.
import json, os, sys, time, urllib.request

PORT = os.environ.get("PORT", "8000")
MODEL = os.environ["MODEL"]
OUT = os.environ.get("OUT", "/results/sanity.json")
HERE = os.path.dirname(os.path.abspath(__file__))
prompts = json.load(open(os.path.join(HERE, "prompts.json")))

def chat(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://0.0.0.0:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], time.time() - t0

results = []
allpass = True
for p in prompts:
    try:
        text, dt = chat(p["prompt"], p["max_tokens"])
        low = text.lower()
        hits = [k for k in p["expect_contains"] if k.lower() in low]
        # coherent = non-empty, not pure repetition/garbage
        coherent = len(text.strip()) > 0 and len(set(text.split())) > 1 or len(text.strip()) <= 8
        ok = len(hits) > 0 and coherent
        results.append({"id": p["id"], "ok": ok, "hits": hits,
                        "expected": p["expect_contains"], "latency_s": round(dt, 2),
                        "output": text.strip()[:200]})
        if not ok:
            allpass = False
    except Exception as e:
        results.append({"id": p["id"], "ok": False, "error": str(e)[:150]})
        allpass = False

summary = {"model": MODEL, "all_pass": allpass,
           "passed": sum(1 for r in results if r.get("ok")),
           "total": len(results), "results": results}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=2)

print(f"=== SANITY: {summary['passed']}/{summary['total']} prompts passed ===")
for r in results:
    mark = "PASS" if r.get("ok") else "FAIL"
    out = r.get("output", r.get("error", ""))
    print(f"  [{mark}] {r['id']:12s} -> {out[:90]!r}")
sys.exit(0 if allpass else 1)
