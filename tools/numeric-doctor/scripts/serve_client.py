#!/usr/bin/env python3
"""Send a file of requests to a serving endpoint and report what came back (debug-numeric, servers: `vllm serve`,
`sglang.launch_server`; run by scripts/serve_run.sh).

    python3 scripts/serve_client.py --url http://127.0.0.1:<port> --requests <requests.jsonl> \
        [--api openai|sglang] [--concurrency 1] [--max-tokens 32] [--repeat 1]

requests.jsonl, one request per line: {"prompt": "<text>"} or {"prompt_token_ids": [..]}, optionally "max_tokens",
"temperature" (default 0: greedy), "seed". The user's failing requests, or a replay of their load; the client
never invents traffic.

  --api openai   POST /v1/completions (vLLM's OpenAI server: logprobs=2, return_tokens_as_token_ids)
  --api sglang   POST /generate (SGLang's native endpoint: input_ids / text, return_logprob, top_logprobs_num=2)

Per request it prints `Logprobs are not finite at request <i>` when a returned logprob is NaN / Inf / missing
where the server should have one, or when the server answers with an error about a non-finite value (vLLM's
OpenAI server cannot serialise a NaN logprob: HTTP 400, 2026-11-24) -- the <NAN_SIGNAL> line of a server run --
and `request <i> tokens <first 8>` every --log-every requests. With ND_DIFF_OUT it writes <dir>/outputs.jsonl in
output_diff.py's format (prompt hash, token ids, chosen logprob, top-1 / top-2 margin), so differential mode compares
servers like offline runs;
with ND_DIFF_REF (+ ND_DIFF_THR) it prints `ND_DIFF_HIT` lines against a reference run (diff_tap's check mode).
Last line: `SERVE_CLIENT_RESULT requests=<n> errors=<e> nonfinite=<k> hits=<h>`. Exit 0 unless every request
failed. Standard library only (urllib), no torch.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import re

HERE = os.path.dirname(os.path.abspath(__file__))
NAN_ERROR = re.compile(r"(?i)\bnan\b|\binf(inity)?\b|not JSON compliant|out of range float")
sys.path.insert(0, HERE)


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())             # json accepts NaN / Infinity, which servers may emit


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def openai_request(base, r, args):
    body = {"model": args.model, "max_tokens": r.get("max_tokens", args.max_tokens),
            "temperature": r.get("temperature", 0.0), "logprobs": 2, "return_tokens_as_token_ids": True,
            "ignore_eos": True, "min_tokens": r.get("max_tokens", args.max_tokens)}
    if "seed" in r:
        body["seed"] = r["seed"]
    body["prompt"] = r["prompt_token_ids"] if "prompt_token_ids" in r else r["prompt"]
    out = post(base + "/v1/completions", body, args.timeout)
    lp = out["choices"][0].get("logprobs") or {}
    toks = [int(str(t).split(":")[-1]) if str(t).startswith("token_id:") else t for t in lp.get("tokens") or []]
    chosen = [_f(x) for x in lp.get("token_logprobs") or []]
    per = []
    for d in lp.get("top_logprobs") or []:
        per.append({int(str(k).split(":")[-1]) if str(k).startswith("token_id:") else k: _f(v)
                    for k, v in (d or {}).items()})
    return toks, chosen, per


def sglang_request(base, r, args):
    sp = {"max_new_tokens": r.get("max_tokens", args.max_tokens), "temperature": r.get("temperature", 0.0),
          "ignore_eos": True}
    body = {"sampling_params": sp, "return_logprob": True, "top_logprobs_num": 2}
    if "prompt_token_ids" in r:
        body["input_ids"] = r["prompt_token_ids"]
    else:
        body["text"] = r["prompt"]
    out = post(base + "/generate", body, args.timeout)
    mi = out.get("meta_info", {})
    otl = mi.get("output_token_logprobs") or []
    toks = out.get("output_ids") or [e[1] for e in otl]
    chosen = [_f(e[0]) for e in otl]
    per = [{int(e[1]): _f(e[0]) for e in (pos or [])} for pos in (mi.get("output_top_logprobs") or [])]
    return toks, chosen, per


def record(i, r, toks, chosen, per):
    key = r.get("prompt_token_ids", r.get("prompt"))
    inp = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]
    margin = []
    for d in per:
        vals = sorted((v for v in d.values() if v is not None and math.isfinite(v)), reverse=True)
        margin.append(vals[0] - vals[1] if len(vals) > 1 else None)
    lp = [c if c is None or math.isfinite(c) else c for c in chosen]
    return {"kind": "gen", "i": i, "inp": inp, "tokens": [int(t) if str(t).lstrip("-").isdigit() else t for t in toks],
            "lp": lp, "margin": margin}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--api", choices=("openai", "sglang"), default="openai")
    p.add_argument("--model", default="", help="the served model name (openai API; default: the server's first)")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--repeat", type=int, default=1, help="send the whole file this many times")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    base = args.url.rstrip("/")
    reqs = [json.loads(x) for x in open(args.requests) if x.strip()] * args.repeat
    if args.api == "openai" and not args.model:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
            args.model = json.loads(r.read().decode())["data"][0]["id"]
    send = openai_request if args.api == "openai" else sglang_request

    out_dir, ref = os.environ.get("ND_DIFF_OUT"), os.environ.get("ND_DIFF_REF")
    fout = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fout = open(os.path.join(out_dir, "outputs.jsonl"), "a")
        fout.write(json.dumps({"kind": "meta", "framework": "server:" + args.api, "url": base,
                               "requests": os.path.abspath(args.requests)}) + "\n")
    judge = thr = refrecs = None
    if ref:
        import output_diff
        _, refrecs = output_diff.load(ref)
        thr_path = os.environ.get("ND_DIFF_THR", "")
        thr = (json.load(open(thr_path)).get("thresholds") if thr_path and os.path.isfile(thr_path)
               else output_diff.thresholds("bfloat16", {}, 0.0))
        judge = output_diff.judge_gen

    n_err = n_bad = hits = 0
    t0 = time.time()

    def one(i):
        try:
            return i, send(base, reqs[i], args), None
        except urllib.error.HTTPError as e:              # keep the server's reason: it may BE the NaN
            try:
                body = e.read().decode(errors="replace")
            except Exception:                            # noqa: BLE001
                body = ""
            return i, None, RuntimeError("HTTP %d: %s" % (e.code, body[:300]))
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
            return i, None, e

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        results = list(ex.map(one, range(len(reqs))))
    for i, res, err in results:                      # in request order: the records' indices are stable
        if err is not None:
            n_err += 1
            print("request %d failed: %s" % (i, str(err)[:300]), flush=True)
            # a server that cannot serialise a NaN / Inf logprob answers with an error instead of the numbers
            # (vLLM's OpenAI server: HTTP 400 "Out of range float values are not JSON compliant"): the NaN signal
            if NAN_ERROR.search(str(err)):
                n_bad += 1
                print("Logprobs are not finite at request %d (the server could not return them)" % i, flush=True)
            continue
        toks, chosen, per = res
        if not chosen or any(c is None or not math.isfinite(c) for c in chosen):
            n_bad += 1
            print("Logprobs are not finite at request %d" % i, flush=True)
        rec = record(i, reqs[i], toks, chosen, per)
        if fout:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
        if judge and ("gen", i) in refrecs:
            v, d = judge(rec, refrecs[("gen", i)], thr["gen"])
            if v == "DIVERGE":
                hits += 1
                print("ND_DIFF_HIT kind=gen i=%d pos=%s gap=%.4g threshold=%.4g"
                      % (i, d.get("pos", "-"), d.get("gap", float("nan")), thr["gen"]), flush=True)
        if args.log_every and i % args.log_every == 0:
            print("request %d tokens %s" % (i, toks[:8]), flush=True)
    print("SERVE_CLIENT_RESULT requests=%d errors=%d nonfinite=%d hits=%d seconds=%.1f"
          % (len(reqs), n_err, n_bad, hits, time.time() - t0), flush=True)
    return 1 if reqs and n_err == len(reqs) else 0


if __name__ == "__main__":
    sys.exit(main())
