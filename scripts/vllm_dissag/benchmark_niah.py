#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (long-context sparse-MLA decode collapse),
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
#   NIAH_WARMUP  1 (default) = send one throwaway request per context length BEFORE
#                scoring, so the first-hit JIT/kernel-autotune compile happens outside
#                the scored/gated window. On a freshly-booted node the first request of
#                a shape can take minutes to compile; without warmup that lands on the
#                first scored request -> false 0/10 or timeout. Warmup failures are
#                tolerated (logged, not fatal). Set 0 to disable.
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
WARMUP = os.environ.get("NIAH_WARMUP", "1") == "1"
# Warmup uses a generous timeout (cold compile of a long-context shape can take minutes)
# and never fails the run — its only job is to trigger compilation before scoring.
WARMUP_TIMEOUT = max(TIMEOUT, 1800.0)

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


def _request(n_words, seed, max_tokens, timeout):
    """POST one NIAH request; return (message_dict, error_str). Exactly one is non-None."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the animals in this list:\n\n" + make_haystack(n_words, seed)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Thinking models emit chain-of-thought into a separate reasoning field
        # and leave `content` empty until the final answer; with a
        # small max_tokens the answer never appears in `content` and the score is a
        # false 0/10. Disable thinking so the answer lands in `content` directly.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"], None
    except Exception as e:
        return None, str(e)


def warmup(n_words):
    """One throwaway request per length so first-hit compile happens off the scored path.
    Never fatal: a warmup timeout just means the shape is still compiling; the scored
    request will pay whatever remains (bounded by NIAH_TIMEOUT)."""
    _, err = _request(n_words, seed=0, max_tokens=8, timeout=WARMUP_TIMEOUT)
    status = "ok" if err is None else ("timeout/err: %s" % err)
    print("words=%6d  [warmup] %s" % (n_words, status), flush=True)


def run(n_words, seed=0):
    # Sentinel: None = timeout/transport error (NOT a wrong answer); int = score 0..10.
    msg, err = _request(n_words, seed, MAXTOK, TIMEOUT)
    if err is not None:
        print("words=%6d  seed=%d  TIMEOUT/ERROR  %s" % (n_words, seed, err), flush=True)
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
    print("url=%s  model=%s  sizes=%s  seeds=%s  warmup=%s" % (URL, MODEL, WORDS, SEEDS, WARMUP), flush=True)
    # Warmup pass: compile every shape once before scoring, so cold JIT never lands on a
    # scored/gated request (the common cause of false 0/10 or timeout on a fresh boot).
    if WARMUP:
        print("=== NIAH warmup (one throwaway request per length) ===", flush=True)
        for n in WORDS:
            warmup(n)
    results = {}  # n_words -> list of scores across seeds (None = timeout/error, not a wrong answer)
    for n in WORDS:
        results[n] = [run(n, s) for s in SEEDS]
    print("=== NIAH summary (mean/min/max across %d seed(s)) ===" % len(SEEDS), flush=True)
    for n in WORDS:
        scored = results[n]
        vals = [v for v in scored if v is not None]
        n_to = sum(1 for v in scored if v is None)  # timeouts/errors, excluded from mean
        if not vals:
            print("  words=%6d  NO-RESULT (%d/%d timed out or errored — likely cold compile; "
                  "raise NIAH_TIMEOUT or keep NIAH_WARMUP=1)" % (n, n_to, len(scored)), flush=True)
            continue
        mean = sum(vals) / len(vals)
        extra = ("  [%d timeout/err excluded]" % n_to) if n_to else ""
        print("  words=%6d  mean=%.1f/10  min=%d  max=%d  (n=%d)%s"
              % (n, mean, min(vals), max(vals), len(vals), extra), flush=True)


if __name__ == "__main__":
    main()
