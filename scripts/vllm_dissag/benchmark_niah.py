#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Uses /v1/completions with stream=True + x-request-id so the MoRIIO disagg
# router injects KV routing. Raw /v1/chat/completions without these fields
# crashes the decode engine with KeyError('remote_host').
#
# Env:
#   NIAH_URL     endpoint base (default http://127.0.0.1:30000)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in words (default 2000,8000,20000,35000)
#   NIAH_METHOD  hybrid (default) = PR-176 system+user via the DeepSeek chat
#                template PLUS a restatement after the haystack. pr176 = faithful
#                PR-176 (instruction before the haystack only; collapses on DSV4
#                at 8k+). product = the bare continuation the
#                GLM/DSV3/Hy3/DSV4 product rows were graded on. The haystack is
#                identical in all three, so only the ask changes.
#   NIAH_MAXTOK  max_tokens for the answer (default 512 hybrid|pr176 / 64 product)
#   NIAH_MIN_TOKENS  ban EOS for the first N tokens (default 0 = off, diagnostic)
#   NIAH_SEEDS   comma list of needle-layout seeds (default 0,1,2); summary reports
#                mean/min/max across seeds to separate real accuracy from variance
#   NIAH_TIMEOUT per-request timeout seconds (default 1800)
#   NIAH_WARMUP  1 (default) = for each length, one throwaway request THEN the scored
#                request (warmup N → score N). Do not warm 16k–35k before scoring 2k:
#                a later EngineCore 500 makes earlier lengths look like found=0/10.
#                Set 0 to disable warmup. NIAH_HALT_ON_FAIL=1 (default) stops the
#                ladder after a warmup/score timeout or SSE error so a dead engine
#                does not keep printing ok / 0/10.
#   NIAH_PAIR_SLEEP_S          seconds to wait after finishing a length before the
#                              next pair (default 30). Lets KV/JIT settle; 0 disables.
#   NIAH_WARMUP_SCORE_SLEEP_S  seconds between warmup ok and the scored request
#                              (default 5). 0 disables.
#   NIAH_DS_WRAP  1 = prefix <User>/<Assistant> (not a product row)
#   NIAH_TERSE    1 = comma-separated names in order of appearance (diagnostic)
#   NIAH_STOP_BLANK  1 = request stop=["\\n\\n"] (diagnostic; cuts phase-2 dumps)
import os, re, sys, json, time, urllib.request
from datetime import datetime, timezone

from niah_score import (
    MAX_DECOYS,
    _band_bounds,
    make_haystack,
    score_answer,
)

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]

# --- Prompt method -----------------------------------------------------------
# hybrid  (default) PR-176's system instruction AND a restatement placed after
#         the haystack, so generation begins next to the task.
# pr176             Faithful PR-176: instruction only BEFORE the haystack.
#         Kept unmodified so Track A / Track B stay exactly comparable.
# product           The historical bare continuation used for GLM,
#         DSV3, Hy3 and the DSV4 product rows.
#
# Why hybrid is the default (measured 27 Aug):
#   Pro 8k, n=3, same haystack -- product 7,7,8 | pr176 0,0,0 | hybrid+tail 0,7,7
# Faithful pr176 loses the task entirely above 2k: Pro emitted 512 tokens of
# filler WITH REPEATS and no animal at all. Flash fails the same way, one
# word then EOS. But measurement showed the cause is NOT just instruction distance --
# at 2k the hybrid tail sits right next to generation and Pro STILL prefixes ~39
# filler words before the animals. What the product stem uniquely does is
# constrain the first generated token. hybrid therefore carries PR-176's terse
# instruction (which fixed Flash 2k, 0/10 -> 10/10 x3) AND the product stem as
# an assistant-turn prime. See HYBRID_TAIL / HYBRID_PRIME.
#
# The haystack is byte-identical in all three modes (make_haystack md5 matches
# PR-176 at 2k/8k/16k/35k), so the method changes the ask, never the exam.
# Default is `product`, the bare continuation, because it is the only mode that
# is model-agnostic: `hybrid`/`pr176` render the DeepSeek chat template here
# (the MoRIIO router needs /v1/completions, so the server cannot apply it), and
# those `<｜User｜>` / `<｜Assistant｜>` tokens are literal text to a non-DeepSeek
# tokenizer. The DSV4 wrapper sets NIAH_METHOD explicitly, so it never relies on
# this default; every other model gets a prompt its own tokenizer understands.
METHOD = os.environ.get("NIAH_METHOD", "product").strip().lower()
if METHOD not in ("hybrid", "pr176", "product"):
    print("NIAH_METHOD must be hybrid, pr176 or product (got %r)" % METHOD,
          file=sys.stderr)
    sys.exit(2)
# Both chat-template modes share prompt scaffolding and the recall headline.
_CHAT = METHOD in ("hybrid", "pr176")

# PR-176 verbatim (pr-176 benchmark_niah.py:50-53). Do not paraphrase — the
# "Output nothing else" clause is what stops a chatty model burning its budget
# on a preamble, which is exactly how Flash scored 0/10 at 2k with a 64-token budget.
PR176_SYSTEM = (
    "You read a word list and pick out the animals. Reply with a single "
    "comma-separated list of lowercase animal names. Output nothing else."
)
PR176_USER_PREFIX = "Find the animals in this list:\n\n"

# hybrid only. Placed AFTER the haystack so the tokens adjacent to generation
# state the task instead of being more filler.
HYBRID_TAIL = (
    "\n\nList every animal name that appears in the word list above. "
    "Reply with a single comma-separated list of lowercase animal names. "
    "Output nothing else."
)
# hybrid only. Assistant-turn prefill, byte-identical to the product stem.
#
# The tail alone is NOT enough. Under every chat-style prompt
# DSV4-Pro first enumerates the unique FILLER vocabulary and only then reaches
# the animals: at 2k it always finishes the ~40-word list and scores 10/10 (on a
# 49-item answer), but at 8k it sometimes loops back to the start of the list
# and burns the whole 512-token cap, scoring 0/10. Tail-only recovered 2 of 3
# draws at 8k (0,7,7) versus 0 of 3 for faithful pr176 (0,0,0) — better, still
# not the product baseline of 7,7,8.
#
# Instruction POSITION was therefore only part of the story: at 2k the tail sits
# directly next to generation and the dump happens anyway. What the product stem
# uniquely does is constrain the FIRST generated token — "Animals found in the
# above list: " leaves no room for a preamble. Priming the assistant turn with
# that exact string gives PR-176's instruction the product stem's first-token
# constraint.
HYBRID_PRIME = "Animals found in the above list: "

# Text appended to the very end of the prompt, i.e. the tokens immediately before
# generation. Default "" (off) leaves every prompt byte-identical to the rows
# already scored.
#
# Logging the top-5 candidates for Flash's first generated token and the
# three rungs separate cleanly:
#
#   2k  token `1` p~0.19, margin 0.13-0.50, ` elephant` in the top 5   -> 10.0/10
#   8k  token `0` p~0.22, margin 0.38-0.88, ` dolphin`/` kang`/` oct`  ->  6.0/10
#   16k token `0` p 0.40-0.71, margin 1.50-2.88, NO animal, ` none`    ->  0.0/10
#
# At 8k the needles are not merely present, they OUTRANK the prior: ` kang` and
# ` oct` sit at ranks 2-3 while the generic ` dog`/` cat` trail at 4-5, which a
# model that had not read the haystack could not produce. Retrieval works there;
# `0` wins by 0.375 nats over ` dolphin` (a 1.45x likelihood ratio) purely on
# FORMAT -- the stem "Animals found in the above list: " can be continued either
# as a count or as a list, and Flash is losing a coin flip between the two. That
# coin flip is what makes 8k swing 0-8/10 between otherwise identical runs, and
# it is why perturbing index_topk appeared to "fix" the rung without any ranking
# ever having been wrong.
#
# At 2k the same near-tie lands on `1`, which opens a numbered list and scores
# 10/10. Priming that token disambiguates the stem so the rung measures retrieval
# instead of format-guessing. It cannot help 16k, where no animal appears among
# the candidates at all -- that is a genuine retrieval failure and the two must
# not be conflated.
#
# Diagnostic until a rerun says otherwise: any row with this set is a different
# prompt and is NOT comparable to the product rows above.
LIST_PRIME = os.environ.get("NIAH_LIST_PRIME", "")

# pr176 needs room for a 10-name list after any preamble; product is the
# historical 64-token DSV4 handicap. 27 Aug: 512 rather than PR-176's 2048
# because DSV4-Pro runs ~1 s/token, so 2048 x 18 requests overruns the 8h
# assoc walltime. It is a cap, not a target — raise with NIAH_MAXTOK.
MAXTOK = int(os.environ.get("NIAH_MAXTOK", "512" if _CHAT else "64"))
# Product NIAH is the BARE completion for every model, DeepSeek-V4 included:
# GLM, DSV3 and Hy3 were all graded on it, so a
# wrapped DSV4 score is not comparable to any of them. Opt in with
# NIAH_DS_WRAP=1 to A/B the <User>/<Assistant> curl prompt.
#
# This used to default to wrapping when the var was unset and the model was
# DSV4, which silently wrapped two runs and cost both as product
# cells. Unset now means bare, matching the harness export.
DS_WRAP = os.environ.get("NIAH_DS_WRAP", "").strip() == "1"
# Diagnostic only. Product rows stay the GLM/DSV3/Hy3 stem. Set 1 to ask for
# a comma-separated list in haystack order so 10 names can fit in 64 tokens.
TERSE = os.environ.get("NIAH_TERSE", "").strip() == "1"
# Diagnostic only. stop=["\\n\\n"] cuts phase-2 zoo dumps; numbered lists survive.
#
# NOTE this cannot fire on the runaway answers measured: those answers
# are a SINGLE line of space-separated numbered items, with no blank line anywhere.
# Use NIAH_STOP for those.
STOP_BLANK = os.environ.get("NIAH_STOP_BLANK", "").strip() == "1"
# Pipe-separated stop strings. Default "" (off) leaves the request body unchanged,
# so every row already scored stays comparable.
#
# THE TERMINATION BUG. Both models emit the real needles first and then cannot
# stop, filling the whole token budget. It is not a retrieval failure and it is
# not model-specific -- only the runaway CONTENT differs:
#
#   Flash 16k  `... 7. butterfly 8. flamingo 9. crocodile 10. penguin 11. elephant
#              12. giraffe ...` -- the retrieved list looped verbatim to the cap,
#              or an A->Z alphabet game (`umbrella`, `x-ray fish` are not animals).
#   Pro   35k  a CORRECT five-needle answer, then a ```json restatement, then
#              training-data leakage (`<|end_of_file|> data/llama-3.1-8b-it/...`).
#
# So Flash invents animals (list_precision 0.05-0.14) while Pro leaks pretraining
# text (list_precision 0.23-1.00). Both inflate `items` to 60-118 for a 10-item
# answer and both make `decoys` unreadable as a fabrication signal.
#
# Recommended set, and why each entry is safe:
#   11.                      caps a numbered list at TEN items. The grader is
#                            recall@10 / p@10, so nothing past item 10 is scored
#                            anyway -- this truncates only ungraded text. Never
#                            fires on comma-separated answers, which were already
#                            short and clean.
#   ```                      cuts the JSON/code-fence restatement.
#   <|end_of_file|> and      cut the training-data leakage seen here and in
#   <|begin_of_file_name|>
#
# Do NOT add a bare `**`: it is a live first-token candidate on Pro (` **`), so it
# would truncate legitimate bold-formatted answers.
#
# Stop strings are excluded from the returned text (include_stop_str_in_output is
# False by default), so the answer is not polluted by the marker itself.
STOP = [s for s in os.environ.get("NIAH_STOP", "").split("|") if s]
# Suppress EOS for the first N sampled tokens (vLLM SamplingParams.min_tokens).
#
# Flash 2P/2D hybrid 8k: prefill returned prompt_tokens=8719 (the whole
# prompt), completion_tokens=1, text="", finish_reason="stop" in all three draws
# -- the FIRST sampled token was EOS, so decode had nothing to continue and the
# answer was chars=0. The same cell at 2k answered 10/10 with a clean list, and
# the write seal was attn=41 skipped=0, so this is neither transport nor
# truncation: at 8k Flash simply prefers end-of-sequence to any first word.
# The pr176 one-word answer is the milder form of the same thing (one word, then EOS with 510
# tokens unused).
#
# min_tokens bans EOS until N tokens are out, which turns an empty answer into a
# measurement of what Flash would have said. Default 0 (off) so product/pr176
# rows and every score above stay comparable -- this is a probe, not a fix.
MIN_TOKENS = int(os.environ.get("NIAH_MIN_TOKENS", "0"))
# Top-N candidates to log for the first NIAH_LOGPROBS_STEPS generated tokens.
# 0 (default) = off and the request body is unchanged, so scored rows are
# unaffected.
#
# Every Flash failure at 8k+ opens with the SAME token: `0`. Colocated 16k is the
# two-char `'0 '`; 28k/35k are `0 ## 2.2.2.2...`; 8k clustered is
# `0 Animals found in the above list:` looped to the cap. So Flash answers the
# product stem with "zero animals" and then fills the budget. The degenerate loop
# may therefore be a SYMPTOM of having committed to a two-character answer with
# 512 tokens left, not an independent bug.
#
# What separates the two diagnoses is the MARGIN on that first token:
#   p(`0`) near 1.0        -> Flash genuinely concludes the list has no animals.
#                             A comprehension/attention failure, and sampling is
#                             innocent.
#   p(`0`) barely over the
#   first animal name      -> a near-tie decided by numerical noise. That also
#                             explains index_topk 512->2048 flipping the rung
#                             (the indexer feeds attention feeds the logits, so
#                             perturbing it flips a coin-flip), WITHOUT the
#                             ranking ever having been wrong.
# Diagnostic only; pair with 2k (where Flash answers correctly) as the control.
LOGPROBS = int(os.environ.get("NIAH_LOGPROBS", "0"))
LOGPROBS_STEPS = int(os.environ.get("NIAH_LOGPROBS_STEPS", "5"))
TIMEOUT = float(os.environ.get("NIAH_TIMEOUT", "1800"))
# Needle layout is seeded, so a single run is deterministic (bit-exact repro on the
# same stack). Run multiple seeds to distinguish real accuracy from single-needle
# variance; the summary reports mean/min/max across seeds. Default 0,1,2.
SEEDS = [int(x) for x in os.environ.get("NIAH_SEEDS", "0,1,2").split(",") if x.strip()]
WARMUP = os.environ.get("NIAH_WARMUP", "1") == "1"
HALT_ON_FAIL = os.environ.get("NIAH_HALT_ON_FAIL", "1") == "1"
PAIR_SLEEP = float(os.environ.get("NIAH_PAIR_SLEEP_S", "30"))
WARMUP_SCORE_SLEEP = float(os.environ.get("NIAH_WARMUP_SCORE_SLEEP_S", "5"))
# Same cap as the scored request. Do not floor at 1800: that made NIAH_TIMEOUT=240
# a lie (an 8k warmup sat ~30 min). Override with NIAH_WARMUP_TIMEOUT.
WARMUP_TIMEOUT = float(os.environ.get("NIAH_WARMUP_TIMEOUT", str(TIMEOUT)))


def _stop_list():
    """Stop strings for the request, or None to omit the field entirely.

    NIAH_STOP first, then NIAH_STOP_BLANK's "\\n\\n", de-duplicated with order
    preserved. Returning None (not []) matters: an empty list is still a body
    field, and the point of the default is a byte-identical request.
    """
    out = []
    for s in list(STOP) + (["\n\n"] if STOP_BLANK else []):
        if s not in out:
            out.append(s)
    return out or None


def _collect_logprobs(obj, out):
    """Append (chosen_token, [(tok, logprob), ...] desc) per streamed token.

    vLLM's /v1/completions logprobs payload is
    {"tokens": [...], "token_logprobs": [...], "top_logprobs": [{tok: lp}, ...]}.
    Fields are optional and a chunk may carry several tokens, so every access is
    defensive: this is a diagnostic and must never break a run.
    """
    try:
        lp = obj["choices"][0].get("logprobs")
        if not lp:
            return
        toks = lp.get("tokens") or []
        tops = lp.get("top_logprobs") or []
        for i, tok in enumerate(toks):
            if len(out) >= LOGPROBS_STEPS:
                return
            cand = tops[i] if i < len(tops) and tops[i] else {}
            ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
            out.append((tok, ranked))
    except Exception:
        pass


def make_prompt(n_words, seed=0):
    """Build a /v1/completions prompt.

    NIAH_METHOD=hybrid (default) is PR-176's prompt plus a restatement after the
    haystack. NIAH_METHOD=pr176 is faithful PR-176, instruction before the
    haystack only. NIAH_METHOD=product is the bare continuation used by the
    GLM/DSV3/Hy3/DSV4 product rows; there, chat tokens appear only if
    NIAH_DS_WRAP=1 and NIAH_TERSE=1 is a diagnostic stem.
    """
    haystack = make_haystack(n_words, seed)
    if _CHAT:
        # PR-176 posts these as `messages` to /v1/chat/completions and lets the
        # server apply the template. We cannot: the MoRIIO router needs
        # /v1/completions + stream + x-request-id (see module header), so render
        # the template here. DeepSeek puts the system block bare, ahead of the
        # first user turn. NIAH_DS_WRAP / NIAH_TERSE do not apply in this mode.
        tail = HYBRID_TAIL if METHOD == "hybrid" else ""
        prime = HYBRID_PRIME if METHOD == "hybrid" else ""
        return (
            PR176_SYSTEM
            + "\n\n<｜User｜>"
            + PR176_USER_PREFIX
            + haystack
            + tail
            + "<｜Assistant｜>"
            + prime
            + LIST_PRIME
        )
    if TERSE:
        ident = (
            "The following is a long word list. Identify every animal name "
            "hidden in the list. Reply with a comma-separated list of those "
            "names only, in order of appearance, and nothing else.\n\n"
        )
    else:
        ident = (
            "The following is a long word list. Read it carefully and identify "
            "every animal name hidden in the list.\n\n"
        )
    body = ident + haystack + "\n\n" + "Animals found in the above list: "
    if DS_WRAP:
        return "<｜User｜>" + body + "<｜Assistant｜>" + LIST_PRIME
    return body + LIST_PRIME


_REQ_N = [0]


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fail_side(err):
    """Map client error text to proxy / prefill / decode for grep."""
    if not err:
        return "ok"
    e = str(err).lower()
    # The proxy now returns JSON naming the leg, so trust it over any guessing.
    m = re.search(r'"fail_side"\s*:\s*"(prefill|decode|proxy)"', e)
    if m:
        which = m.group(1)
        if "upstream_unreachable" in e:
            return "%s — upstream unreachable (dead/not listening), FAIL_SOURCE=%s" % (
                which,
                which,
            )
        return "%s — FAIL_SOURCE=%s in proxy log" % (which, which)
    if "prefill" in e:
        return "prefill — FAIL_SOURCE=prefill in proxy log"
    if "timed out" in e or "timeout" in e:
        return (
            "client-timeout — grep proxy_NODE0.log FAIL_SOURCE=prefill|decode|proxy "
            "and which of 'prefill POST HEADERS' vs 'decode POST HEADERS' is missing"
        )
    if "enginecore" in e or "internalservererror" in e:
        return "decode — EngineCore 500 (FAIL_SOURCE=decode in proxy log)"
    if "sse error" in e:
        return "decode — SSE error object (FAIL_SOURCE=decode)"
    if "decode" in e:
        return "decode — FAIL_SOURCE=decode in proxy log"
    if "connection" in e or "refused" in e:
        return "proxy — connection (not listening :10001)"
    return "proxy-or-unknown — grep FAIL_SOURCE= in proxy_NODE0.log"


# Per-length failure strings, so the summary can tell an upstream rejection from
# a cold-compile timeout instead of guessing.
_FAIL_ERRS = {}
_REJECT_RE = re.compile(
    r"\b4\d\d\b|BadRequestError|Bad Request|must be less than|"
    r"Unsupported|invalid|ValidationError",
    re.I,
)


def _request(n_words, seed, max_tokens, timeout):
    """POST one NIAH request via /v1/completions (disagg-compatible).

    Uses stream=True + x-request-id so the MoRIIO router injects KV routing.
    Returns ({"content": text}, error_str) — exactly one is non-None.
    """
    _REQ_N[0] += 1
    rid = "niah-%06d" % _REQ_N[0]
    prompt = make_prompt(n_words, seed)
    body = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        # GLM/thinking models: keep the answer in `content`. Ignored elsewhere.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if LOGPROBS > 0:
        body["logprobs"] = LOGPROBS
    _stops = _stop_list()
    if _stops:
        body["stop"] = _stops
    if MIN_TOKENS > 0:
        # Both legs get it: in WRITE mode prefill samples token 1, and that is
        # the token that came back as EOS on Flash's 8k rungs.
        body["min_tokens"] = min(MIN_TOKENS, max_tokens)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL.rstrip("/") + "/v1/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-request-id": rid,
        },
    )
    t0 = time.time()
    print(
        "[niah] start %s words=%d seed=%d max_tokens=%d timeout=%.0fs x-request-id=%s"
        % (_utc(), n_words, seed, max_tokens, timeout, rid),
        flush=True,
    )
    try:
        text = ""
        http_status = None
        ctype = None
        lp_steps = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            http_status = getattr(r, "status", None)
            ctype = r.headers.get("Content-Type", "")
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    dt = time.time() - t0
                    print(
                        "[niah] fail %s words=%d dt=%.1fs http=%s ctype=%s "
                        "fail_side=%s x-request-id=%s err=%s"
                        % (
                            _utc(),
                            n_words,
                            dt,
                            http_status,
                            ctype,
                            _fail_side("sse error: %s" % msg),
                            rid,
                            msg,
                        ),
                        flush=True,
                    )
                    return None, "sse error: %s" % msg
                if LOGPROBS > 0 and len(lp_steps) < LOGPROBS_STEPS:
                    _collect_logprobs(obj, lp_steps)
                try:
                    text += obj["choices"][0]["text"]
                except Exception:
                    pass
        if lp_steps:
            for i, (tok, top) in enumerate(lp_steps):
                # margin = how far the winner leads the runner-up, in logprob.
                # Small margin means noise can flip the token.
                margin = (top[0][1] - top[1][1]) if len(top) > 1 else float("inf")
                print(
                    "[niah] logprob step=%d token=%r margin=%.4f  top=%s"
                    % (
                        i,
                        tok,
                        margin,
                        ", ".join("%r:%.4f" % (t, v) for t, v in top[:LOGPROBS]),
                    ),
                    flush=True,
                )
        dt = time.time() - t0
        print(
            "[niah] done %s words=%d dt=%.1fs http=%s ctype=%s chars=%d "
            "x-request-id=%s fail_side=ok"
            % (_utc(), n_words, dt, http_status, ctype, len(text), rid),
            flush=True,
        )
        if not text.strip():
            print(
                "[niah] empty-text words=%d x-request-id=%s — no choices[0].text; "
                "grep proxy first_chunk / FAIL_SOURCE= for %s"
                % (n_words, rid, rid),
                flush=True,
            )
        return {"content": text}, None
    except Exception as e:
        dt = time.time() - t0
        # urllib's HTTPError str() is just "HTTP Error 500:" -- the body carries
        # the proxy's fail_side/type, and an error once logged a bare colon without it.
        detail = str(e)
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:500]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        if body:
            detail = "%s body=%s" % (detail, body)
        print(
            "[niah] fail %s words=%d dt=%.1fs fail_side=%s x-request-id=%s err=%s"
            % (_utc(), n_words, dt, _fail_side(detail), rid, detail),
            flush=True,
        )
        return None, detail


def warmup(n_words):
    """Throwaway request for this length only. Returns True on HTTP+SSE success."""
    _, err = _request(n_words, seed=0, max_tokens=8, timeout=WARMUP_TIMEOUT)
    status = "ok" if err is None else ("timeout/err: %s" % err)
    extra = "" if err is None else ("  fail_side=%s" % _fail_side(err))
    print("words=%6d  [warmup] %s%s" % (n_words, status, extra), flush=True)
    return err is None


def run(n_words, seed=0):
    # Sentinel: None = timeout/transport error (NOT a wrong answer); int = score 0..10.
    msg, err = _request(n_words, seed, MAXTOK, TIMEOUT)
    if err is not None:
        print(
            "words=%6d  seed=%d  TIMEOUT/ERROR  fail_side=%s  %s"
            % (n_words, seed, _fail_side(err), err),
            flush=True,
        )
        _FAIL_ERRS.setdefault(n_words, []).append(err)
        return None
    # Score content plus any reasoning field (some servers surface CoT as
    # `reasoning` or `reasoning_content`) so a thinking model is never mis-scored.
    raw = (
        (msg.get("content") or "")
        + " "
        + (msg.get("reasoning_content") or "")
        + " "
        + (msg.get("reasoning") or "")
    )
    s = score_answer(raw)
    found, decoys, precision = s["found"], s["decoys"], s["precision"]
    print("words=%6d  seed=%d  found=%2d/10  %s" % (n_words, seed, len(found), found), flush=True)
    preview = (msg.get("content") or "").replace("\n", " ")[:400]
    print("words=%6d  seed=%d  answer=%r" % (n_words, seed, preview), flush=True)
    print(
        "words=%6d  seed=%d  decoys=%2d precision=%.2f verdict=%s  %s"
        % (n_words, seed, len(decoys), precision, s["verdict"], decoys[:12]),
        flush=True,
    )
    # Dumping is invisible to found/decoys/p@10 — a Pro 2k answer scored a perfect
    # row on 39 filler words plus the ten animals. Read this line next to it.
    print(
        "words=%6d  seed=%d  items=%d extra=%d list_precision=%.2f filler=%d  %s"
        % (n_words, seed, s["items"], s["extra"], s["list_precision"],
           len(s["filler"]), s["filler"][:12]),
        flush=True,
    )
    print(
        "words=%6d  seed=%d  retrieval_walk=%s kangaroo=%s penguin=%s dolphin=%s"
        % (n_words, seed, s["retrieval_walk"], s["kangaroo"],
           s["penguin"], s["dolphin"]),
        flush=True,
    )
    print(
        "words=%6d  seed=%d  streak=%d r@10=%.2f p@10=%.2f product_pass=%s "
        "prefix_verdict=%s junk=%s negated=%s"
        % (
            n_words,
            seed,
            s["streak"],
            s["r_at_k"],
            s["p_at_k"],
            s["product_pass"],
            s["prefix_verdict"],
            s["junk"] or "",
            s["negated"][:8],
        ),
        flush=True,
    )
    return (len(found), len(decoys), bool(s["junk"]))


def no_scored_answers(results):
    """True iff no seed of any length produced an answer (timeout/error/skip).

    A scored 0/10 is still a result. An all-NO-RESULT ladder is a harness
    failure, which validate used to report as niah=0.
    """
    return not any(v is not None for scored in results.values() for v in scored)


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    print("=== NIAH retrieval test ===", flush=True)
    # method= is printed from the same constant make_prompt() branches on, so it
    # cannot disagree with the prompt actually sent (the ds_wrap= guarantee).
    print("url=%s  model=%s  method=%s  maxtok=%d  sizes=%s  seeds=%s  warmup=%s "
          "halt_on_fail=%s pair_sleep=%.0fs warmup_score_sleep=%.0fs  "
          "ds_wrap=%s terse=%s stop_blank=%s min_tokens=%d"
          % (URL, MODEL, METHOD, MAXTOK, WORDS, SEEDS, WARMUP, HALT_ON_FAIL,
             PAIR_SLEEP, WARMUP_SCORE_SLEEP, DS_WRAP, TERSE, STOP_BLANK,
             MIN_TOKENS),
          flush=True)
    # Printed from niah_score's own parsed bounds, not the raw env string, so it
    # reports the layout actually built (the ds_wrap= guarantee).
    # Printed from the same helper the request body uses, so it cannot disagree
    # with what was actually sent (the ds_wrap= guarantee).
    _stops = _stop_list()
    if _stops:
        print("[niah] stop=%r: caps the runaway continuation both models show "
              "(needles first, then a verbatim loop / A-Z game on Flash, or a "
              "```json restatement + training-data leakage on Pro). Truncates "
              "only text past item 10, which recall@10 / p@10 never scored."
              % (_stops,), flush=True)
    if LIST_PRIME:
        print("[niah] list_prime=%r appended to the stem: the first token is given, "
              "not sampled. Flash's 8k stem is a 0.375-nat coin flip "
              "between `0` (count) and ` dolphin` (list) with the needles already "
              "ranked above the prior. Diagnostic — NOT comparable to product rows."
              % LIST_PRIME, flush=True)
    if LOGPROBS > 0:
        print("[niah] LOGPROBS=%d steps=%d: logging the top candidates for the "
              "first %d generated tokens. Diagnostic; the request body carries "
              "an extra field." % (LOGPROBS, LOGPROBS_STEPS, LOGPROBS_STEPS),
              flush=True)
    _band = _band_bounds()
    if _band is not None:
        print("[niah] NEEDLE_BAND=%.3f-%.3f: all ten needles confined to that "
              "fraction of the haystack. DIAGNOSTIC haystack -- found=/decoys= "
              "are NOT comparable to product rows." % _band, flush=True)
    if MIN_TOKENS > 0:
        print("[niah] min_tokens=%d: EOS banned for the first %d tokens ("
              "Flash 8k returned completion_tokens=1 text='' finish_reason=stop). "
              "Diagnostic — not comparable with min_tokens=0 rows."
              % (MIN_TOKENS, MIN_TOKENS), flush=True)
    if _CHAT:
        if METHOD == "hybrid":
            print("[niah] method=hybrid: PR-176 system+user via DeepSeek template "
                  "PLUS a restatement AFTER the haystack. Not a faithful PR-176 "
                  "score — use method=pr176 for cross-track comparison.",
                  flush=True)
        else:
            print("[niah] method=pr176: FAITHFUL PR-176, instruction before the "
                  "haystack only. Known to collapse on DSV4 at 8k+ (Pro: "
                  "0/10 x3, 512 tokens of filler); use method=hybrid to score.",
                  flush=True)
        print("[niah] ds_wrap/terse ignored. Headline metric is found= (recall), "
              "same as PR-176; decoys/precision/verdict still printed because "
              "recall alone cannot see fabrication (Pro 28k: found=9.3 mean at "
              "decoys=11.7).", flush=True)
        if DS_WRAP or TERSE:
            print("[niah] WARN NIAH_DS_WRAP/NIAH_TERSE are set but do not apply "
                  "under method=%s; use NIAH_METHOD=product for those." % METHOD,
                  flush=True)
    else:
        print("[niah] method=product: bare continuation, comparable to GLM / "
              "DSV3 / Hy3 and the DSV4 product rows. "
              "Not comparable to PR-176 numbers.", flush=True)
    # Pair each length: warmup N then score N. One overnight run warmed 16k (EngineCore
    # 500) before scoring 2k, so 2k printed found=0/10 on a dead engine.
    results = {}  # n_words -> list of scores across seeds (None = timeout/error)
    halted = False
    for i, n in enumerate(WORDS):
        if halted:
            results[n] = [None] * len(SEEDS)
            print("words=%6d  SKIP (ladder halted)" % n, flush=True)
            continue
        if i > 0 and PAIR_SLEEP > 0:
            print(
                "[niah] sleep %.0fs between pairs (after words=%d, before words=%d)"
                % (PAIR_SLEEP, WORDS[i - 1], n),
                flush=True,
            )
            time.sleep(PAIR_SLEEP)
        if WARMUP:
            print("=== pair words=%d (warmup then score) ===" % n, flush=True)
            if not warmup(n):
                results[n] = [None] * len(SEEDS)
                print("words=%6d  SKIP score (warmup failed)" % n, flush=True)
                if HALT_ON_FAIL:
                    print("NIAH_HALT_ON_FAIL=1 — stopping remaining lengths", flush=True)
                    halted = True
                continue
            if WARMUP_SCORE_SLEEP > 0:
                print(
                    "[niah] sleep %.0fs between warmup and score words=%d"
                    % (WARMUP_SCORE_SLEEP, n),
                    flush=True,
                )
                time.sleep(WARMUP_SCORE_SLEEP)
        results[n] = [run(n, s) for s in SEEDS]
        if HALT_ON_FAIL and all(v is None for v in results[n]):
            print("NIAH_HALT_ON_FAIL=1 — stopping remaining lengths (score failed)", flush=True)
            halted = True
    print("=== NIAH summary (mean/min/max across %d seed(s)) ===" % len(SEEDS), flush=True)
    for n in WORDS:
        scored = results[n]
        vals = [v for v in scored if v is not None]
        n_to = sum(1 for v in scored if v is None)  # timeouts/errors, excluded from mean
        if not vals:
            # A run died on a prefill 400 (min_tokens > the proxy's rewritten
            # max_tokens=1) at dt=0.0s, and the old blanket "likely cold compile,
            # raise NIAH_TIMEOUT" sent the reader at the one knob that could not
            # help. An upstream rejection and a cold-compile timeout need
            # opposite responses, so name which one happened.
            errs = _FAIL_ERRS.get(n, [])
            rejected = [e for e in errs if _REJECT_RE.search(e or "")]
            if rejected and len(rejected) == len(errs):
                print("  words=%6d  NO-RESULT (%d/%d REJECTED upstream, not a timeout — "
                      "fix the request, NIAH_TIMEOUT will not help): %s"
                      % (n, n_to, len(scored), (rejected[0] or "")[:200]), flush=True)
            else:
                print("  words=%6d  NO-RESULT (%d/%d timed out or errored — likely cold compile; "
                      "raise NIAH_TIMEOUT or keep NIAH_WARMUP=1)"
                      % (n, n_to, len(scored)), flush=True)
            continue
        hits = [v[0] for v in vals]
        dec = [v[1] for v in vals]
        mean = sum(hits) / len(hits)
        mean_dec = sum(dec) / len(dec)
        junked = any(len(v) > 2 and v[2] for v in vals)
        # A length only counts as read if every seed stayed under the decoy cap.
        if junked:
            tag = "INVALID"
        elif mean >= 8 and max(dec) <= MAX_DECOYS:
            tag = "RETRIEVAL"
        elif max(dec) > MAX_DECOYS:
            tag = "GUESSING"
        elif mean >= 5:
            tag = "PARTIAL"
        else:
            tag = "NONE"
        extra = ("  [%d timeout/err excluded]" % n_to) if n_to else ""
        print("  words=%6d  mean=%.1f/10  min=%d  max=%d  decoys=%.1f  %s  (n=%d)%s"
              % (n, mean, min(hits), max(hits), mean_dec, tag, len(hits), extra),
              flush=True)
    if no_scored_answers(results):
        print("=== NIAH FAILED: every request was timeout/error/skip "
              "(no scored answer) ===", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
