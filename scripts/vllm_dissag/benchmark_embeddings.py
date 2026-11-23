#!/usr/bin/env python3
"""Throughput and latency of an embeddings endpoint (OpenAI-compatible /v1/embeddings).

Sends EMB_REQUESTS requests of EMB_BATCH inputs of about EMB_WORDS words each, EMB_CONCURRENCY at
a time, and prints one summary per concurrency, which parse_to_csv.py --embeddings reads:

    EMBEDDINGS concurrency=<c> requests_per_s=<r> inputs_per_s=<i> p50_ms=<p50> p99_ms=<p99> n=<n> errors=<e>

Env: EMB_URL, EMB_MODEL, EMB_CONCURRENCY (comma list, "1,8,32"), EMB_REQUESTS (200), EMB_BATCH (8),
EMB_WORDS (128), EMB_TIMEOUT (120). Standard library only.
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request

WORDS = ("model serve token cache layer expert router prefill decode memory fabric rack node "
         "kernel tensor batch latency throughput accuracy recipe platform").split()


def one(url, model, batch, words, timeout, rng):
    inputs = [" ".join(rng.choice(WORDS) for _ in range(words)) for _ in range(batch)]
    req = urllib.request.Request(url, json.dumps({"model": model, "input": inputs}).encode(),
                                 {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)["data"]
    if len(data) != batch:
        raise ValueError(f"{len(data)} embeddings for {batch} inputs")
    return time.monotonic() - t


def run(url, model, conc, n, batch, words, timeout):
    rng = random.Random(conc)
    lat, err = [], 0
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        for fut in cf.as_completed([ex.submit(one, url, model, batch, words, timeout, rng) for _ in range(n)]):
            try:
                lat.append(fut.result())
            except Exception as e:  # noqa: BLE001 -- counted, never timed
                err += 1
                print(f"[embeddings] request failed: {e}")
    wall = time.monotonic() - t0
    lat.sort()
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3 if lat else 0.0  # noqa: E731
    print(f"EMBEDDINGS concurrency={conc} requests_per_s={len(lat) / wall:.2f} "
          f"inputs_per_s={len(lat) * batch / wall:.1f} p50_ms={pct(0.5):.1f} p99_ms={pct(0.99):.1f} "
          f"n={n} errors={err}")
    return err == 0


def main():
    g = os.environ.get
    ok = [run(g("EMB_URL"), g("EMB_MODEL"), int(c), int(g("EMB_REQUESTS", "200")), int(g("EMB_BATCH", "8")),
              int(g("EMB_WORDS", "128")), float(g("EMB_TIMEOUT", "120")))
          for c in g("EMB_CONCURRENCY", "1,8,32").split(",") if c]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
