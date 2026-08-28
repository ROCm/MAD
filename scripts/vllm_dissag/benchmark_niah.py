#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Env:
#   NIAH_URL     endpoint (default http://127.0.0.1:30000/v1/chat/completions)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in words (default 2000,8000,20000,35000)
#   NIAH_MAXTOK  max_tokens for the answer (default 8192)
#   NIAH_TIMEOUT per-request timeout seconds (default 1800)
#
# Reasoning models need headroom. Scoring reads the reasoning trace as well as
# the answer, so if max_tokens runs out mid-trace the response is truncated and
# only the earliest needles appear -- indistinguishable from a genuine retrieval
# miss unless finish_reason is recorded. Observed on Kimi-K3 at 2048: two of four
# context sizes scored 1/10, both listing only ANIMALS[0], the needle placed
# first in the haystack. Hence the larger default and the finish= field below.
import os, sys, json, random, urllib.request

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]
MAXTOK = int(os.environ.get("NIAH_MAXTOK", "8192"))
TIMEOUT = float(os.environ.get("NIAH_TIMEOUT", "1800"))

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


def run(n_words):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the animals in this list:\n\n" + make_haystack(n_words)},
        ],
        "temperature": 0.0,
        "max_tokens": MAXTOK,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            choice = json.loads(r.read())["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason") or "unknown"
    except Exception as e:
        print("words=%6d  ERROR  %s" % (n_words, e), flush=True)
        return None
    text = ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).lower()
    found = sorted(a for a in ANIMALS if a in text)
    # finish is reported so a truncated response ("length") is distinguishable
    # from a real retrieval miss. Without it a run that exhausted max_tokens
    # mid-reasoning scores low and reads as a model failure.
    print("words=%6d  found=%2d/10  finish=%s  %s"
          % (n_words, len(found), finish, found), flush=True)
    return (len(found), finish)


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    print("=== NIAH retrieval test ===", flush=True)
    print("url=%s  model=%s  sizes=%s" % (URL, MODEL, WORDS), flush=True)
    results = {}
    for n in WORDS:
        results[n] = run(n)
    print("=== NIAH summary ===", flush=True)
    for n in WORDS:
        v = results[n]
        if v is None:
            print("  words=%6d  found=ERR/10" % n, flush=True)
        else:
            count, finish = v
            note = "  (TRUNCATED: raise NIAH_MAXTOK)" if finish == "length" else ""
            print("  words=%6d  found=%s/10  finish=%s%s"
                  % (n, count, finish, note), flush=True)


if __name__ == "__main__":
    main()
