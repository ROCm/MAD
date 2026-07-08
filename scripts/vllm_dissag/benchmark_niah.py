#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Env:
#   NIAH_URL     endpoint (default http://127.0.0.1:30000/v1/chat/completions)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in words (default 2000,8000,20000,35000)
#   NIAH_MAXTOK  max_tokens for the answer (default 2048)
#   NIAH_SEEDS   comma list of needle-layout seeds (default 0,1,2); summary reports
#                mean/min/max across seeds to separate real accuracy from variance
#   NIAH_TIMEOUT per-request timeout seconds (default 1800)
import os, sys, json, random, urllib.request

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]
MAXTOK = int(os.environ.get("NIAH_MAXTOK", "2048"))
TIMEOUT = float(os.environ.get("NIAH_TIMEOUT", "1800"))
# Needle layout is seeded, so a single run is deterministic (bit-exact repro on the
# same stack). Run multiple seeds to distinguish real accuracy from single-needle
# variance; the summary reports mean/min/max across seeds. Default 0,1,2.
SEEDS = [int(x) for x in os.environ.get("NIAH_SEEDS", "0,1,2").split(",") if x.strip()]

FILLER = (
    "table chair window bottle pencil garden river mountain coffee planet "
    "engine guitar pillow ticket basket candle market silver button orange "
    "rocket napkin ladder pepper carpet helmet jacket mirror anchor pocket "
    "branch copper saddle tunnel violin wallet zipper meadow cactus pebble"
).split()
ANIMALS = ["elephant", "giraffe", "kangaroo", "penguin", "dolphin",
           "tiger", "rhinoceros", "octopus", "crocodile", "panda"]

SYSTEM = (
    "You read a word list and pick out the animals. Reply with a single "
    "comma-separated list of lowercase animal names. Output nothing else."
)


def make_haystack(n_words, seed=0):
    rng = random.Random(seed)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    step = max(n_words // (len(ANIMALS) + 1), 1)
    for i, animal in enumerate(ANIMALS):
        words[min((i + 1) * step, len(words) - 1)] = animal
    return " ".join(words)


def run(n_words, seed=0):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the animals in this list:\n\n" + make_haystack(n_words, seed)},
        ],
        "temperature": 0.0,
        "max_tokens": MAXTOK,
        # Thinking models (e.g. GLM-5.1) emit chain-of-thought into a separate
        # reasoning field and leave `content` empty until the final answer; with a
        # small max_tokens the answer never appears in `content` and the score is a
        # false 0/10. Disable thinking so the answer lands in `content` directly.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
    except Exception as e:
        print("words=%6d  ERROR  %s" % (n_words, e), flush=True)
        return None
    # Score content plus any reasoning field (some servers surface CoT as
    # `reasoning` or `reasoning_content`) so a thinking model is never mis-scored.
    text = ((msg.get("content") or "") + " "
            + (msg.get("reasoning_content") or "") + " "
            + (msg.get("reasoning") or "")).lower()
    found = sorted(a for a in ANIMALS if a in text)
    print("words=%6d  seed=%d  found=%2d/10  %s" % (n_words, seed, len(found), found), flush=True)
    return len(found)


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    print("=== NIAH retrieval test ===", flush=True)
    print("url=%s  model=%s  sizes=%s  seeds=%s" % (URL, MODEL, WORDS, SEEDS), flush=True)
    results = {}  # n_words -> list of scores across seeds (None on error)
    for n in WORDS:
        results[n] = [run(n, s) for s in SEEDS]
    print("=== NIAH summary (mean/min/max across %d seed(s)) ===" % len(SEEDS), flush=True)
    for n in WORDS:
        vals = [v for v in results[n] if v is not None]
        if not vals:
            print("  words=%6d  ERR" % n, flush=True)
            continue
        mean = sum(vals) / len(vals)
        print("  words=%6d  mean=%.1f/10  min=%d  max=%d  (n=%d)"
              % (n, mean, min(vals), max(vals), len(vals)), flush=True)


if __name__ == "__main__":
    main()
