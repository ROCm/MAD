#!/usr/bin/env python3
"""Needle-in-a-haystack sweep including long context (validated to 200K tokens).

Hides 10 animal names at even intervals in a filler-word haystack and asks the model to
list them back. Reports found/10, end-to-end latency, and the server-reported
prompt_tokens per length, and can dump the whole run to JSON.

Usage:
    niah_200k.py <base_url> [lengths_csv] [out_json]
      base_url    e.g. http://127.0.0.1:20005   (prefill/serve port, or the router)
      lengths_csv comma-separated WORD counts   (default: 2k..200k)
      out_json    optional path to write results

    NIAH_MODEL=<path-or-name>   override the served model id (default below)

Lengths are given in WORDS to stay comparable with earlier published runs. On this filler
the GLM tokenizer lands ~1 token/word, so words ~= tokens (the script prints the actual
prompt_tokens so you can check).

NOTE: the FIRST request after a server boot pays cold Triton JIT and can take >80s with a
prefill instance running eager. Warm the server (or use a generous timeout) before
treating any latency number here as steady-state.
"""
import json, os, sys, time, random, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:20005"
LENGTHS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                            "2000,8000,16000,20000,28000,35000,64000,100000,150000,200000".split(","))]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
URL = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("NIAH_MODEL", "/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8")

FILLER = ("table chair window bottle pencil garden river mountain coffee planet "
          "engine guitar pillow ticket basket candle market silver button orange").split()
ANIMALS = ["elephant", "giraffe", "kangaroo", "penguin", "dolphin",
           "tiger", "rhinoceros", "octopus", "crocodile", "panda"]
SYS = ("You read a word list and pick out the animals. Reply with a single "
       "comma-separated list of lowercase animal names. Output nothing else.")


def hay(n, seed=0):
    rng = random.Random(seed)
    w = [rng.choice(FILLER) for _ in range(n)]
    step = max(n // (len(ANIMALS) + 1), 1)
    for i, a in enumerate(ANIMALS):
        w[min((i + 1) * step, len(w) - 1)] = a
    return " ".join(w)


def run(n, seed=0, timeout=1800):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "Find the animals in this list:\n\n" + hay(n, seed)},
        ],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        m = r["choices"][0]["message"]
        txt = ((m.get("content") or "") + " " + (m.get("reasoning") or "")).lower()
        found = sorted(a for a in ANIMALS if a in txt)
        u = r.get("usage") or {}
        rec = {"words": n, "seed": seed, "found": len(found), "latency_s": round(time.time() - t, 1),
               "prompt_tokens": u.get("prompt_tokens"), "animals": found}
        print("words=%7d tok=%-7s found=%2d/10 (%6.1fs) %s" % (
            n, rec["prompt_tokens"], rec["found"], rec["latency_s"], found), flush=True)
        return rec
    except Exception as e:
        rec = {"words": n, "seed": seed, "found": -1, "latency_s": round(time.time() - t, 1),
               "error": str(e)[:200]}
        print("words=%7d ERROR (%.1fs) %s" % (n, rec["latency_s"], rec["error"]), flush=True)
        return rec


results = [run(n) for n in LENGTHS]
if OUT:
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT, flush=True)
