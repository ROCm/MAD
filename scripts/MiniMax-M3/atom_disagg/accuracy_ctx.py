#!/usr/bin/env python3
# accuracy_ctx.py — context-varied accuracy probe for the MiniMax-M3 ATOM disagg pipeline.
#
# Why: a PD-disaggregated pipeline can be correct at short context but corrupt the KV that
# mooncake transfers prefill->decode at long context. A 5-short-prompt gate misses that.
# This embeds a known "needle" fact inside filler of increasing size and checks the router
# (which spans prefill+decode+mooncake) returns it verbatim — exercising the FULL pipeline
# at each context length.
#
# Usage: ROUTER_URL, MODEL, OUT(json), [CTX_SIZES="512 2048 8192 16384"] via env.
#        Exits 0 if all context sizes pass, 1 otherwise.
import json, os, sys, time, urllib.request

ROUTER = os.environ.get("ROUTER_URL", "http://0.0.0.0:8000")
MODEL  = os.environ["MODEL"]
OUT    = os.environ.get("OUT", "/out/accuracy_ctx.json")
# approx tokens of filler context to test at (the needle adds a few more)
CTX_SIZES = [int(x) for x in os.environ.get("CTX_SIZES", "512 2048 8192 16384").split()]

# A deterministic needle: a unique code the model must echo back. Greedy (temp 0).
SECRET = "PLATYPUS-7731-AZURE"
FILLER_SENTENCE = ("The quarterly logistics report notes that warehouse throughput "
                   "remained within nominal parameters across all regional hubs. ")

def make_prompt(approx_tokens):
    # ~1.3 tokens/word for English; build filler then plant the needle in the MIDDLE
    words_needed = int(approx_tokens / 1.3)
    filler_words = FILLER_SENTENCE.split()
    reps = max(1, words_needed // len(filler_words))
    block = (" ".join(filler_words) + " ") * reps
    half = len(block) // 2
    needle = f" IMPORTANT: the access code is {SECRET}. Remember it exactly. "
    ctx = block[:half] + needle + block[half:]
    q = ("\n\nQuestion: What is the exact access code mentioned above? "
         "Reply with ONLY the code, nothing else.")
    return ctx + q

def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{ROUTER}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], time.time() - t0

results = []; allpass = True
for ctx in CTX_SIZES:
    try:
        prompt = make_prompt(ctx)
        text, dt = ask(prompt)
        found = SECRET in text.upper()
        ok = found
        results.append({"ctx_tokens": ctx, "ok": ok, "found_needle": found,
                        "reply": text.strip()[:80], "latency_s": round(dt, 2),
                        "prompt_chars": len(prompt)})
        print(f"  ctx~{ctx:>6}tok: {'PASS' if ok else 'FAIL'} "
              f"(needle={'yes' if found else 'NO'}, {dt:.1f}s) reply='{text.strip()[:40]}'")
        allpass = allpass and ok
    except Exception as e:
        results.append({"ctx_tokens": ctx, "ok": False, "error": str(e)[:120]})
        print(f"  ctx~{ctx:>6}tok: ERROR {str(e)[:80]}")
        allpass = False

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"model": MODEL, "router": ROUTER, "all_pass": allpass, "results": results},
          open(OUT, "w"), indent=2)
print(f"[accuracy_ctx] {'ALL PASS' if allpass else 'FAILURES'} -> {OUT}")
sys.exit(0 if allpass else 1)
