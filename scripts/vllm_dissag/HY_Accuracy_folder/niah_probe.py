#!/usr/bin/env python3
"""NIAH (needle-in-a-haystack) long-context retrieval probe.

Embeds a known fact ("the magic number") at varied DEPTHS inside long filler of
target token LENGTHS, then asks the model to retrieve it. Objective pass/fail
(did the exact number come back) — no LLM judge needed. This is the only test
that exercises the long-context KV path the way a 256K workload does, so it's the
direct accuracy answer for the 256K concern, run per EP/KV config.

Token length is approximated from the served tokenizer if reachable, else ~4 chars/token.

Usage:
  python3 niah_probe.py --url http://127.0.0.1:30000 --model <name> \
      --lengths 54000 128000 256000 --depths 0.1 0.5 0.9 --out niah_ep32.json
"""
import argparse, json, sys, urllib.request

FILLER = ("The grass was green and the sky was clear. People walked along the "
          "quiet streets as the day went on without anything unusual happening. ")
NEEDLE_TMPL = "\n\n>>> IMPORTANT: The magic access code for this session is {code}. Remember it. <<<\n\n"
QUESTION = "\n\nQuestion: What is the magic access code for this session? Answer with only the number.\n\nAnswer:"

def approx_tokens_to_chars(ntok):  # ~4 chars/token for English filler
    return ntok * 4

def build_haystack(target_tokens, depth, code):
    total_chars = approx_tokens_to_chars(target_tokens)
    needle = NEEDLE_TMPL.format(code=code)
    body_chars = max(0, total_chars - len(needle) - len(QUESTION))
    reps = body_chars // len(FILLER) + 1
    body = (FILLER * reps)[:body_chars]
    cut = int(len(body) * depth)
    return body[:cut] + needle + body[cut:] + QUESTION

_REQ_N = [0]

def gen(url, model, prompt, max_tokens=32):
    # Stream + x-request-id (vllm bench serve protocol) so the MoRIIO router injects
    # KV routing; raw non-streaming/no-id requests crash decode (KeyError remote_host).
    _REQ_N[0] += 1
    b = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                    "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=b,
                                 headers={"Content-Type": "application/json",
                                          "x-request-id": f"niah-{_REQ_N[0]:06d}"})
    text = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                text += json.loads(chunk)["choices"][0]["text"]
            except Exception:
                pass
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[54000, 128000, 256000])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    code = "8675309"   # fixed, distinctive 7-digit needle
    rows, passed = [], 0
    for L in a.lengths:
        for d in a.depths:
            prompt = build_haystack(L, d, code)
            try:
                out = gen(a.url, a.model, prompt)
                ok = code in out
            except Exception as e:
                out = f"ERROR: {repr(e)[:120]}"; ok = False
            passed += int(ok)
            rows.append({"length": L, "depth": d, "pass": ok, "got": out.strip()[:80]})
            print(f"  len={L:>7} depth={d:.1f}  {'PASS' if ok else 'FAIL'}  got={out.strip()[:40]!r}")
    total = len(rows)
    summary = {"passed": passed, "total": total, "rate": passed / total if total else 0, "rows": rows}
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"[niah] retrieval {passed}/{total} ({100*passed/total:.0f}%) -> {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
