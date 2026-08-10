#!/usr/bin/env python3
"""Concurrency / throughput bench for the EP16 2P/2D disagg serve.

Fires N identical requests concurrently at the router and reports wall time,
throughput (req/s and output tok/s), and per-request latency percentiles.
This is the metric disagg is FOR (DP8 = 8 concurrent replicas), unlike
single-stream NIAH.

Usage: concurrency_bench.py <ctx_tokens> <concurrency> [max_out]
  e.g. concurrency_bench.py 20000 8 64
"""
import sys, json, time, urllib.request, concurrent.futures as cf

ROUTER = "http://10.32.82.3:30000/v1/completions"
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank while "
          "the morning sun rises over the distant mountains and birds sing. ")

def make_prompt(ctx_tokens):
    body = (FILLER * (ctx_tokens * 4 // len(FILLER) + 1))[:ctx_tokens * 4]
    return body + "\n\nSummarize the above in one word:"

def one(prompt, max_out):
    data = json.dumps({"model": "kimi-k3", "prompt": prompt,
                       "max_tokens": max_out, "temperature": 0}).encode()
    req = urllib.request.Request(ROUTER, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
        dt = time.time() - t0
        n = out.get("usage", {}).get("completion_tokens", max_out)
        return dt, n, True
    except Exception as e:
        return time.time() - t0, 0, False

def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs); i = min(len(xs) - 1, int(p / 100 * len(xs)))
    return xs[i]

def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_out = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    prompt = make_prompt(ctx)
    print(f"ctx={ctx}tok concurrency={conc} max_out={max_out}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one(prompt, max_out), range(conc)))
    wall = time.time() - t0
    lats = [d for d, n, ok in res if ok]
    okn = sum(1 for _, _, ok in res if ok)
    outtok = sum(n for _, n, ok in res if ok)
    print(f"  ok={okn}/{conc}  wall={wall:.1f}s", flush=True)
    print(f"  throughput: {okn/wall:.3f} req/s | {outtok/wall:.1f} out-tok/s", flush=True)
    print(f"  latency: mean={sum(lats)/len(lats):.1f}s p50={pct(lats,50):.1f}s "
          f"p99={pct(lats,99):.1f}s min={min(lats):.1f}s max={max(lats):.1f}s", flush=True)

if __name__ == "__main__":
    main()
