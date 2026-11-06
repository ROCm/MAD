#!/usr/bin/env python3
"""Needle tests on token-exact corpus text (chat completions, temperature 0).

The needle is "The secret code of the <name> vault is NNNNNN."; the question asks for the code.
  single  lengths x depths x needles: each request has its own haystack span, so no two requests
          share a prefix. Used by benchmark_niah.sh with NIAH_MODE=corpus.
  multi   warm multi-turn sessions: turn 1 holds ~START tokens with K needles at spread depths and
          asks for "OK"; each later turn appends NEW_TOKENS of fresh text and asks for one needle,
          so the answer depends on KV computed in turn 1 and carried by the prefix cache (and, in
          P/D, by the KV connector). Used with NIAH_MODE=multiturn.
X-Session-ID is sent on every request so a sticky router keeps a session on one rank.

  benchmark_niah_corpus.py single --url http://host:port --model M --tokenizer DIR --corpus F \
      --out DIR [--lengths 65536,262144,524288 --depths 0,0.5,1.0 --needles 2 --concurrency 6]
  benchmark_niah_corpus.py multi --url ... --start 246000 [--turns 3 --new-tokens 10000 --k 3 --sessions 4]

Prints one JSON line per request and a summary line
  niah_single lengths=<L> found=<F>/<ASKED> errors=<E>      (single)
  niah_mt start=<START> found=<F>/<ASKED> errors=<E>        (multi)
which parse_to_csv.py --niah-mt turns into perf.csv. Rows go to <out>/<mode>.jsonl.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request

NAMES = ["amber", "cobalt", "crimson", "jade", "ivory", "onyx", "scarlet", "teal", "violet", "umber"]
QUESTION = "\n\nWhat is the secret code of the {name} vault? Answer with the number only."


def chat(url, model, messages, session, max_tokens, timeout):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Session-ID": session})
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as e:  # noqa: BLE001 - recorded as a failed request
        return {"error": repr(e)[:500], "s": round(time.time() - t, 1)}
    m = r["choices"][0]["message"]
    return {"s": round(time.time() - t, 1), "usage": r.get("usage"), "finish": r["choices"][0]["finish_reason"],
            "content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or ""}


def found(res, code):
    return "error" not in res and code in re.sub(r"[,\s]", "", res["content"])


def needle(rng):
    name = rng.choice(NAMES)
    code = str(rng.randrange(100000, 999999))
    return name, code, f"\n\nThe secret code of the {name} vault is {code}.\n\n"


class Hay:
    def __init__(self, tok, corpus):
        self.tok = tok
        self.ids = tok.encode(open(corpus).read(), add_special_tokens=False)

    def span(self, rng, n):
        o = rng.randrange(0, len(self.ids))       # wrap-around: the corpus is shorter than 512k
        return (self.ids * ((o + n) // len(self.ids) + 1))[o:o + n]

    def build(self, rng, length, depth_sentences):
        ids = self.span(rng, length)
        parts, last = [], 0
        for depth, sent in sorted(depth_sentences):
            cut = int(len(ids) * depth)
            parts += [self.tok.decode(ids[last:cut]), sent]
            last = cut
        parts.append(self.tok.decode(ids[last:]))
        return "".join(parts)


def run_single(a, hay):
    rng = random.Random(a.seed)
    jobs = []
    for length in [int(x) for x in a.lengths.split(",")]:
        for depth in [float(x) for x in a.depths.split(",")]:
            for k in range(a.needles):
                name, code, sent = needle(rng)
                body = length - 200 - a.max_tokens   # room for the question and the answer
                text = hay.build(random.Random(rng.random()), body, [(depth, sent)])
                jobs.append(dict(length=length, depth=depth, k=k, name=name, code=code,
                                 prompt=text + QUESTION.format(name=name)))
    rows = []
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(chat, a.url, a.model, [{"role": "user", "content": j["prompt"]}],
                          f"niah-{a.tag}-{i}", a.max_tokens, a.timeout): j for i, j in enumerate(jobs)}
        for f in cf.as_completed(futs):
            j, res = futs[f], f.result()
            row = {k: j[k] for k in ("length", "depth", "k", "name", "code")}
            row.update(res)
            row["found"] = found(res, j["code"])
            row["reasoning"] = row.get("reasoning", "")[-300:]
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("length", "depth", "k", "found", "s", "finish")}), flush=True)
    return rows


def run_multi(a, hay):
    rng = random.Random(a.seed)

    def session(si):
        srng = random.Random(rng.random() + si)
        sid = f"niahmt-{a.tag}-{si}"
        needles = [needle(srng) for _ in range(a.k)]
        depths = [(i + 0.5) / a.k for i in range(a.k)]
        doc = hay.build(srng, a.start, [(d, n[2]) for d, n in zip(depths, needles)])
        msgs = [{"role": "user", "content": doc + "\n\nRead the document above. Reply with OK."}]
        out = []
        for t in range(a.turns):
            name = code = None
            if t > 0:
                name, code, _ = needles[(t - 1) % a.k]
                msgs.append({"role": "user", "content": "More material follows.\n\n"
                             + hay.tok.decode(hay.span(srng, a.new_tokens)) + QUESTION.format(name=name)})
            res = chat(a.url, a.model, msgs, sid, a.max_tokens if t else 64, a.timeout)
            row = {"session": sid, "turn": t + 1, "name": name, "code": code, **res}
            if code:
                row["found"] = found(res, code)
            row["reasoning"] = row.get("reasoning", "")[-300:]
            out.append(row)
            print(json.dumps({k: row.get(k) for k in ("session", "turn", "found", "s", "usage")}), flush=True)
            msgs.append({"role": "assistant", "content": res.get("content", "")})
        return out

    rows = []
    with cf.ThreadPoolExecutor(a.sessions) as ex:
        for r in ex.map(session, range(a.sessions)):
            rows.extend(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="65536,262144,524288", help="single: request lengths in tokens")
    ap.add_argument("--depths", default="0,0.5,1.0", help="single: needle depths")
    ap.add_argument("--needles", type=int, default=2, help="single: requests per length and depth")
    ap.add_argument("--concurrency", type=int, default=6, help="single: requests in flight")
    ap.add_argument("--start", type=int, default=246000, help="multi: turn-1 context tokens")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=10000)
    ap.add_argument("--k", type=int, default=3, help="needles per session")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    hay = Hay(AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True), a.corpus)
    t0 = time.time()
    rows = run_single(a, hay) if a.mode == "single" else run_multi(a, hay)
    with open(os.path.join(a.out, f"{a.mode}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    asked = [r for r in rows if "found" in r]
    summ = {"args": vars(a), "wall_s": round(time.time() - t0), "found": sum(r["found"] for r in asked),
            "asked": len(asked), "errors": sum("error" in r for r in rows)}
    json.dump(summ, open(os.path.join(a.out, f"{a.mode}-summary.json"), "w"), indent=1)
    key = f"niah_single lengths={a.lengths}" if a.mode == "single" else f"niah_mt start={a.start}"
    print(f"{key} found={summ['found']}/{summ['asked']} errors={summ['errors']}", flush=True)


if __name__ == "__main__":
    main()
