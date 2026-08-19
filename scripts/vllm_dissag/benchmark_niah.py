#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Env:
#   NIAH_URL     endpoint (default http://127.0.0.1:30000/v1/chat/completions)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in WORDS (default 2000,8000,20000,35000)
#   NIAH_TOKENS  comma list of context sizes in TOKENS. Overrides NIAH_WORDS when set.
#                Use this whenever the number is customer-facing: "950K context" is a
#                claim about TOKENS, and words are not tokens -- for this filler the
#                ratio is around 1.3 tok/word, so 950,000 words would be ~1.24M tokens
#                and would be REJECTED by the server, not truncated. The rejection
#                surfaces as a transport error and would be read as a server fault.
#   NIAH_TOKENIZER  tokenizer path used to calibrate NIAH_TOKENS (default: NIAH_MODEL).
#                Needs transformers + trust_remote_code. If it cannot be loaded the run
#                falls back to NIAH_TOKENS_PER_WORD and says so LOUDLY -- an uncalibrated
#                length is an approximate length and must not be quoted as exact.
#   NIAH_TOKENS_PER_WORD  fallback ratio when no tokenizer is available (default 1.30).
#   NIAH_MAXTOK  max_tokens for the answer (default 2048)
#   NIAH_SEEDS   comma list of layout seeds (default 0,1,2); summary reports
#                mean/min/max across seeds to separate real accuracy from variance.
#                Each seed varies BOTH the filler words and the needle offsets
#                (seed 0 = the historical evenly-spaced layout, kept for
#                comparability). Varying offsets is what gives coverage of
#                position-dependent failures -- lost-in-the-middle, RoPE
#                extrapolation, and corruption at prefill CHUNK BOUNDARIES.
#   NIAH_TIMEOUT per-request timeout seconds (default 1800). With NIAH_TIMEOUT_SCALE=1
#                (the default) this is the budget for the LONGEST rung and shorter rungs
#                scale down quadratically, floored at 300 s -- so a dead server fails in
#                minutes on the 32K rung instead of burning the 950K budget eight times.
#   NIAH_TIMEOUT_SCALE  0 to apply NIAH_TIMEOUT flat to every rung.
#   NIAH_WARMUP  1 (default) = send one throwaway request per context length BEFORE
#                scoring, so the first-hit JIT/kernel-autotune compile happens outside
#                the scored/gated window. On a freshly-booted node the first request of
#                a shape can take minutes to compile; without warmup that lands on the
#                first scored request -> false 0/10 or timeout. Warmup failures are
#                tolerated (logged, not fatal). Set 0 to disable.
import os, sys, json, random, time, urllib.request

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]
TOKENS = [int(x) for x in os.environ.get("NIAH_TOKENS", "").split(",") if x.strip()]
TOKENIZER = os.environ.get("NIAH_TOKENIZER", "") or MODEL
TOK_PER_WORD = float(os.environ.get("NIAH_TOKENS_PER_WORD", "1.30"))
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
# Scale the per-request timeout with the context length instead of applying one flat
# value to every rung. NIAH_TIMEOUT then means "the budget for the LONGEST rung".
#
# This matters in one direction only, and it is the expensive one: a flat timeout sized
# for 950K (~4.4 h) applied to the 32K rung means a dead server burns 4.4 h before the
# first rung even reports, and eight rungs would exceed the job wall clock without
# producing a single line. Scaling quadratically -- the same model that sets the budget,
# since the sparse indexer scans all preceding keys per prefill chunk -- keeps a hung
# 32K rung to minutes while still allowing 950K its hours. Floored at 300 s so short
# rungs still tolerate a cold compile.
TIMEOUT_SCALE = os.environ.get("NIAH_TIMEOUT_SCALE", "1") == "1"


def timeout_for(n_words, max_words):
    if not TIMEOUT_SCALE or not max_words:
        return TIMEOUT
    return max(300.0, TIMEOUT * (n_words / float(max_words)) ** 2)

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
    """Return (text, depths) where depths[i] is the fractional position of ANIMALS[i].

    The depths are returned, not just used, because "which slot was missed" is the
    actual diagnostic. A bare 7/10 says retrieval is degraded; 7/10 with the three
    misses all past 80% depth says the tail of the context is being dropped, which
    points at a completely different bug than three scattered misses."""
    rng = random.Random(seed)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    step = max(n_words // (len(ANIMALS) + 1), 1)
    depths = []
    for i, animal in enumerate(ANIMALS):
        base = (i + 1) * step
        # Needle offsets must vary with the seed, or the seeds only reshuffle
        # filler and every seed probes the SAME 10 positions. That hides exactly
        # the failure mode this stack is suspected of: prefill splits at
        # max_num_batched_tokens, and a needle pinned on a bad chunk boundary is
        # then missed identically by all seeds and reported as a confident mean.
        #
        # seed 0 reproduces the historical evenly-spaced layout bit-for-bit so
        # earlier seed-0 results stay comparable. Jitter is < step and slot i+1
        # starts at (i+2)*step, so needles can never collide or reorder; all 10
        # always survive.
        off = 0 if seed == 0 else rng.randrange(step)
        pos = min(base + off, len(words) - 1)
        words[pos] = animal
        depths.append(pos / float(max(len(words) - 1, 1)))
    return " ".join(words), depths


# --- token targeting --------------------------------------------------------
# make_haystack takes WORDS. The customer's number is TOKENS. Converting with a fixed
# ratio is not good enough at the top of the ladder: at 950,000 tokens a 3% ratio error
# is 28,500 tokens, and if it errs high against a 1,048,576 window the request is
# REJECTED with a 400 rather than truncated -- which arrives here as a transport error
# and reads as a dead server. So we calibrate against the real tokenizer and converge.
_TOKENIZER_OBJ = None
_TOKENIZER_FAILED = False


def _get_tokenizer():
    global _TOKENIZER_OBJ, _TOKENIZER_FAILED
    if _TOKENIZER_OBJ is not None or _TOKENIZER_FAILED:
        return _TOKENIZER_OBJ
    try:
        from transformers import AutoTokenizer
        _TOKENIZER_OBJ = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    except Exception as e:
        _TOKENIZER_FAILED = True
        print("WARNING: could not load tokenizer %r (%s).\n"
              "         Falling back to NIAH_TOKENS_PER_WORD=%.2f. Context lengths are\n"
              "         then APPROXIMATE -- do not quote them as exact token counts."
              % (TOKENIZER, e, TOK_PER_WORD), flush=True)
    return _TOKENIZER_OBJ


def words_for_tokens(target_tokens):
    """Words such that the user message lands at ~target_tokens. Returns (n_words, got).

    Bounded loop, not a solve: the filler is drawn with replacement so the tokens/word
    ratio varies slightly with n, and one rescale does not land. Four passes converge to
    well under 1% in practice. We deliberately approach from BELOW on the last step --
    overshooting a context window is a 400, undershooting is just a slightly shorter
    test."""
    tok = _get_tokenizer()
    if tok is None:
        return max(int(target_tokens / TOK_PER_WORD), 16), None
    n = max(int(target_tokens / TOK_PER_WORD), 16)
    got = None
    for _ in range(4):
        text, _d = make_haystack(n, seed=0)
        got = len(tok(SYSTEM + "Find the animals in this list:\n\n" + text,
                      add_special_tokens=False)["input_ids"])
        if got == 0:
            break
        if abs(got - target_tokens) <= max(64, int(0.002 * target_tokens)):
            break
        n = max(int(n * target_tokens / float(got)), 16)
    # Final guard: never hand back a length that came out ABOVE target, for the
    # rejection reason above.
    if got and got > target_tokens:
        n = max(int(n * target_tokens / float(got)) - 1, 16)
        text, _d = make_haystack(n, seed=0)
        got = len(tok(SYSTEM + "Find the animals in this list:\n\n" + text,
                      add_special_tokens=False)["input_ids"])
    return n, got


def _request(n_words, seed, max_tokens, timeout):
    """POST one NIAH request; return (message_dict, error_str, depths).
    Exactly one of message_dict / error_str is non-None."""
    hay, depths = make_haystack(n_words, seed)
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the animals in this list:\n\n" + hay},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Thinking models (e.g. GLM-5.1) emit chain-of-thought into a separate
        # reasoning field and leave `content` empty until the final answer; with a
        # small max_tokens the answer never appears in `content` and the score is a
        # false 0/10. Disable thinking so the answer lands in `content` directly.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"], None, depths
    except Exception as e:
        return None, str(e), depths


def warmup(n_words, label="", timeout=None):
    """One throwaway request per length so first-hit compile happens off the scored path.
    Never fatal: a warmup timeout just means the shape is still compiling; the scored
    request will pay whatever remains (bounded by NIAH_TIMEOUT)."""
    t0 = time.time()
    _, err, _d = _request(n_words, seed=0, max_tokens=8,
                          timeout=max(timeout or 0.0, WARMUP_TIMEOUT))
    status = "ok" if err is None else ("timeout/err: %s" % err)
    print("%s  [warmup] %s  (%.1fs)" % (label or ("words=%6d" % n_words), status,
                                        time.time() - t0), flush=True)


def run(n_words, seed=0, label="", timeout=None):
    """Returns (score, misses) -- score is None on timeout/error (NOT a wrong answer),
    misses is the list of DEPTHS (0.0-1.0) at which needles were not retrieved."""
    label = label or ("words=%6d" % n_words)
    t0 = time.time()
    msg, err, depths = _request(n_words, seed, MAXTOK, timeout or TIMEOUT)
    dt = time.time() - t0
    if err is not None:
        print("%s  seed=%d  TIMEOUT/ERROR after %.0fs  %s" % (label, seed, dt, err),
              flush=True)
        return None, []
    # Score content plus any reasoning field (some servers surface CoT as
    # `reasoning` or `reasoning_content`) so a thinking model is never mis-scored.
    text = ((msg.get("content") or "") + " "
            + (msg.get("reasoning_content") or "") + " "
            + (msg.get("reasoning") or "")).lower()
    found = sorted(a for a in ANIMALS if a in text)
    # Depth of every needle the model did NOT return. This is the "different slots"
    # answer: a uniform miss pattern and a tail-clustered one score identically but
    # mean completely different things.
    misses = [depths[i] for i, a in enumerate(ANIMALS) if a not in text]
    mtxt = ("  missed@depth=" + ",".join("%.0f%%" % (100 * d) for d in sorted(misses))) \
        if misses else ""
    print("%s  seed=%d  found=%2d/10  %.0fs%s" % (label, seed, len(found), dt, mtxt),
          flush=True)
    return len(found), misses


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    print("=== NIAH retrieval test ===", flush=True)

    # Build the ladder. Token mode is the customer-facing one; word mode is kept so
    # existing NIAH_WORDS invocations behave exactly as before.
    #   sizes  = list of (n_words, label, target_tokens_or_None)
    if TOKENS:
        print("token mode: calibrating word counts for %s tokens via %r"
              % (TOKENS, TOKENIZER), flush=True)
        sizes = []
        for t in TOKENS:
            n, got = words_for_tokens(t)
            if got is None:
                lbl = "tok~%7d" % t
                print("  target %8d tok -> %8d words (UNCALIBRATED, ratio %.2f)"
                      % (t, n, TOK_PER_WORD), flush=True)
            else:
                lbl = "tok=%7d" % got
                print("  target %8d tok -> %8d words -> %8d tok actual (%+.2f%%)"
                      % (t, n, got, 100.0 * (got - t) / t), flush=True)
            sizes.append((n, lbl, t))
    else:
        sizes = [(n, "words=%6d" % n, None) for n in WORDS]

    print("url=%s  model=%s  seeds=%s  warmup=%s" % (URL, MODEL, SEEDS, WARMUP), flush=True)
    # Runtime warning, not a limit. The sparse indexer scores each prefill chunk against
    # all preceding keys, so prefill cost grows roughly QUADRATICALLY in context length:
    # relative to a measured 28,672-token TTFT of 4.80 s, 950K tokens is ~875x, i.e. of
    # order an hour for ONE request. A 3-seed ladder to 950K with warmup does not fit
    # beside anything else in a 24 h job. Say so before burning the allocation rather
    # than after.
    if TOKENS and max(TOKENS) > 262144:
        est = sum((t / 28672.0) ** 2 * 4.80 * (len(SEEDS) + (1 if WARMUP else 0))
                  for t in TOKENS) / 60.0
        print("NOTE: prefill is ~quadratic in context length here (sparse indexer scans\n"
              "      all preceding keys per chunk). Rough upper-bound wall clock for this\n"
              "      ladder: %.0f min at %d seed(s)%s. It is an UPPER bound -- the 4.80s\n"
              "      reference at 28,672 tok also contains fixed cost that does not scale."
              % (est, len(SEEDS), " + warmup" if WARMUP else ""), flush=True)

    # Warmup pass: compile every shape once before scoring, so cold JIT never lands on a
    # scored/gated request (the common cause of false 0/10 or timeout on a fresh boot).
    max_words = max(n for n, _l, _t in sizes)
    if TIMEOUT_SCALE and len(sizes) > 1:
        print("per-rung timeouts (NIAH_TIMEOUT=%.0fs is the budget for the LONGEST rung;\n"
              "  shorter rungs scale down quadratically so a hung short rung fails fast):"
              % TIMEOUT, flush=True)
        for n, lbl, _t in sizes:
            print("    %s  timeout=%.0fs" % (lbl, timeout_for(n, max_words)), flush=True)

    if WARMUP:
        print("=== NIAH warmup (one throwaway request per length) ===", flush=True)
        for n, lbl, _t in sizes:
            warmup(n, lbl, timeout_for(n, max_words))
    results = {}   # n_words -> list of scores across seeds (None = timeout/error)
    misses = {}    # n_words -> pooled list of missed depths across seeds
    for n, lbl, _t in sizes:
        tmo = timeout_for(n, max_words)
        rs, ms = [], []
        for s in SEEDS:
            sc, md = run(n, s, lbl, tmo)
            rs.append(sc)
            ms.extend(md)
        results[n], misses[n] = rs, ms
    WORDS[:] = [n for n, _l, _t in sizes]  # what the summary below iterates
    LABEL = {n: l for n, l, _t in sizes}
    print("=== NIAH summary (mean/min/max across %d seed(s)) ===" % len(SEEDS), flush=True)
    # EXIT-CODE CONTRACT (benchmark_niah_perf.sh:64 gates the perf phase on this):
    #   3 = some length had NO usable result at all (dead server / timeout)
    #   4 = every length scored, but some mean < NIAH_MIN_SCORE (retrieval broken)
    #   0 = all lengths scored >= NIAH_MIN_SCORE
    # Previously main() always fell through to an implicit `return None` -> exit 0,
    # so the gate could never fire and a dead server produced "0.00 tok/s SUCCESS".
    min_score = float(os.environ.get("NIAH_MIN_SCORE", "8.0"))
    any_noresult = False
    any_lowscore = False
    for n in WORDS:
        scored = results[n]
        lbl = LABEL.get(n, "words=%6d" % n)
        vals = [v for v in scored if v is not None]
        n_to = sum(1 for v in scored if v is None)  # timeouts/errors, excluded from mean
        if not vals:
            any_noresult = True
            print("  %s  NO-RESULT (%d/%d timed out or errored — likely cold compile; "
                  "raise NIAH_TIMEOUT or keep NIAH_WARMUP=1)" % (lbl, n_to, len(scored)),
                  flush=True)
            continue
        mean = sum(vals) / len(vals)
        extra = ("  [%d timeout/err excluded]" % n_to) if n_to else ""
        verdict = "" if mean >= min_score else "  <-- BELOW NIAH_MIN_SCORE=%.1f" % min_score
        if mean < min_score:
            any_lowscore = True
        print("  %s  mean=%.1f/10  min=%d  max=%d  (n=%d)%s%s"
              % (lbl, mean, min(vals), max(vals), len(vals), extra, verdict), flush=True)
        # Where the misses were. A decile histogram over pooled seeds: needles are
        # placed at ~9% intervals so each decile holds about one slot per seed, and a
        # column that is dark while the others are clean localises the failure to a
        # depth band instead of leaving it as a bare score.
        if misses.get(n):
            hist = [0] * 10
            for d in misses[n]:
                hist[min(int(d * 10), 9)] += 1
            print("        miss depth 0%%..100%%: [%s]  (%d miss(es) over %d seed(s))"
                  % (" ".join("%d" % h for h in hist), len(misses[n]), len(vals)),
                  flush=True)
    if any_noresult:
        print("=== NIAH VERDICT: FAIL (no usable result at one or more lengths) ===",
              flush=True)
        return 3
    if any_lowscore:
        print("=== NIAH VERDICT: FAIL (retrieval below NIAH_MIN_SCORE=%.1f) ===" % min_score,
              flush=True)
        return 4
    print("=== NIAH VERDICT: PASS (all lengths >= %.1f/10) ===" % min_score, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
