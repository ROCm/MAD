#!/usr/bin/env python3
"""Warm multi-turn sessions with an exact retained context and exact new tokens per turn.

Writes a ShareGPT-style list that vLLM's benchmarks/multi_turn/benchmark_serving_multi_turn.py
reads with -i. Used by benchmark_multiturn_retained.sh, one file per round.

  benchmark_multiturn_retained.py --tokenizer DIR --corpus FILE --isl 286000 --new-tokens 30000 \
      --turns 3 --conversations 16 --round 1 --out sessions.json [--osl 1000] [--sticky-mod 8]

--isl is the target mean input length of the measured turns (retained context + new tokens).
--turns counts user turns including the warm-up turn (measured turns + 1).

Why not the client's own generator: it gives every round the same user turns and takes each
session's context from a fresh corpus span, so long sessions need a corpus far larger than the
public text. Here:
  * turn 1 (prefix P0 + N new tokens) depends only on (isl, new_tokens, i), so it is identical
    across rounds and only the first round pays the cold prefill;
  * turns 2..T draw N new tokens per (round, turn), so every warm turn carries N truly new tokens
    plus the previous answer;
  * the corpus only has to hold one session's span; sessions read it at different offsets with
    wrap-around, and a unique first line keeps every session's block hashes distinct.
Session ids can be balanced over the router's sticky ranks (--sticky-mod): the hash is the one
used by the tested llm-d Envoy Lua sticky filter (x = (x*31 + byte) mod 2^31-1). Ids stay valid
names for any other router.
"""
import argparse
import json
import random
import sys

BASE_PROMPT = "Please rewrite the following text and add more content: "
MAX_MODEL_LEN = 560000


def sticky_hash(s: str) -> int:
    x = 0
    for ch in s.encode():
        x = (x * 31 + ch) % 2147483647
    return x


def balanced_id(i: int, mod: int) -> str:
    """Session id that hashes to rank i % mod; plain CONV_ID_<i> when mod <= 1."""
    if mod <= 1:
        return f"CONV_ID_{i}"
    for j in range(10000):
        cid = f"CONV_ID_{i}_{j}"
        if sticky_hash(cid) % mod == i % mod:
            return cid
    raise SystemExit("no balanced id")


def shape(isl: int, new_tokens: int, turns: int, osl: int, max_model_len: int = MAX_MODEL_LEN) -> dict:
    """Turn-1 length, retained prefix P0 and last-turn length for a target mean measured ISL."""
    grow, t_meas = new_tokens + osl, turns - 1
    isl1 = int(isl - grow * (t_meas + 1) / 2)
    p0 = isl1 - new_tokens
    isl_last = isl1 + t_meas * grow
    if p0 <= 0:
        raise ValueError(f"P0={p0} <= 0: --isl too small for --new-tokens/--turns")
    if isl_last + osl > max_model_len:
        raise ValueError(f"last turn {isl_last}+{osl} exceeds max-model-len {max_model_len}")
    return dict(isl_warm=isl1, p0=p0, t_meas=t_meas, isl_last=isl_last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--isl", type=int, required=True, help="target mean ISL of the measured turns")
    ap.add_argument("--new-tokens", type=int, required=True)
    ap.add_argument("--turns", type=int, required=True)
    ap.add_argument("--conversations", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--osl", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sticky-mod", type=int, default=8, help="DP ranks behind the sticky router (0: off)")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        sh = shape(a.isl, a.new_tokens, a.turns, a.osl, a.max_model_len)
    except ValueError as e:
        raise SystemExit(str(e))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    ids = tok.encode(open(a.corpus).read(), add_special_tokens=False)
    n, p0, t_meas = a.new_tokens, sh["p0"], sh["t_meas"]

    def cyc(start, length):
        reps = (start + length) // len(ids) + 1
        return (ids * reps)[start:start + length]

    answer = tok.decode(ids[: a.osl])
    out = []
    for i in range(a.conversations):
        cid = balanced_id(i, a.sticky_mod)
        head = f"Session {cid}. "
        rng = random.Random(f"{a.seed}:{a.isl}:{n}:{i}")
        ps = rng.randrange(0, len(ids))
        us = ps + p0
        msgs = [{"role": "user", "content": head + f"{i}, " + tok.decode(cyc(ps, p0)) + BASE_PROMPT
                 + tok.decode(cyc(us, n))},
                {"role": "assistant", "content": answer}]
        rest = len(ids) - p0 - n
        shift = ((a.round - 1) * t_meas * n) % max(rest - t_meas * n, 1)
        for k in range(t_meas):
            s = (us + n + shift + k * n) % len(ids)
            msgs.append({"role": "user", "content": head + BASE_PROMPT + tok.decode(cyc(s, n))})
            msgs.append({"role": "assistant", "content": answer})
        out.append({"id": cid, "messages": msgs})
    json.dump(out, open(a.out, "w"))
    print(json.dumps(dict(out=a.out, conversations=a.conversations, isl_target=a.isl, new_tokens=n,
                          round=a.round, corpus_tokens=len(ids), **sh)))


if __name__ == "__main__":
    sys.exit(main())
