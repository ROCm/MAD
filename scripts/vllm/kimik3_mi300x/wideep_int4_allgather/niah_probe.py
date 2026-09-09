#!/usr/bin/env python3
"""Lightweight NIAH (needle-in-a-haystack) + basic accuracy probe for a served
Kimi-K3 endpoint. No deps beyond stdlib (urllib). Run from any node that can
reach the head IP:port.

Usage:
  python3 niah_probe.py --url http://<head-ip>:8000 --model kimi-k3 \
      --ctx 8000 --depths 0.1,0.5,0.9
"""
import argparse, json, time, urllib.request, urllib.error

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Grand vistas of rolling hills stretched to the horizon. ")

def chat(url, model, prompt, max_tokens=256, timeout=180):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    ch = d["choices"][0]["message"]
    txt = (ch.get("content") or "") + " " + (ch.get("reasoning") or "")
    return txt.strip(), d.get("usage", {}), dt

def make_haystack(approx_words, needle, depth):
    # Build filler to approx word count; insert needle at fractional depth.
    words = []
    while len(words) < approx_words:
        words.extend(FILLER.split())
    n = len(words)
    pos = int(n * depth)
    needle_words = needle.split()
    words[pos:pos] = needle_words
    return " ".join(words)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--ctx", type=int, default=8000, help="approx haystack words")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    args = ap.parse_args()

    print(f"== sanity ==")
    txt, usage, dt = chat(args.url, args.model, "Reply with only: OK", max_tokens=8)
    print(f"  resp={txt!r} usage={usage} {dt:.1f}s")

    secret = "The secret passcode is HELIOTROPE-7492."
    question = ("\n\nQuestion: What is the secret passcode? "
                "Answer with only the passcode.")
    passed = 0; total = 0
    for depth in [float(x) for x in args.depths.split(",")]:
        hay = make_haystack(args.ctx, secret, depth)
        prompt = hay + question
        try:
            txt, usage, dt = chat(args.url, args.model, prompt, max_tokens=256)
        except Exception as e:
            print(f"  depth={depth}: ERROR {e}")
            total += 1; continue
        ok = "HELIOTROPE-7492" in txt.upper()
        passed += ok; total += 1
        print(f"  depth={depth:>4}: {'PASS' if ok else 'FAIL'} "
              f"(prompt_tok={usage.get('prompt_tokens')}, {dt:.1f}s) resp={txt[:60]!r}")
    print(f"== NIAH {passed}/{total} passed ==")

if __name__ == "__main__":
    main()
