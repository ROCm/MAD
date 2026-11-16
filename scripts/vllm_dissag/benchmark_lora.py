#!/usr/bin/env python3
"""Multi-LoRA serving: chat requests spread over the adapters a server holds (each is a model name
of its own), per concurrency. One summary per concurrency, which parse_to_csv.py --lora reads:

    LORA adapters=<k> concurrency=<c> requests_per_s=<r> p50_ms=<a> p99_ms=<b> n=<n> errors=<e>

Env: LORA_URL (chat completions), LORA_ADAPTERS (comma-separated names, as in the role's
lora.modules), LORA_CONCURRENCY ("1,8,32"), LORA_REQUESTS (100), LORA_MAX_TOKENS (64),
LORA_TIMEOUT (300). Standard library only.
"""
import concurrent.futures as cf
import itertools
import json
import os
import sys
import time
import urllib.request


def one(url, model, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Write one sentence about GPUs."}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}),
                                timeout=timeout) as r:
        json.load(r)["choices"][0]
    return time.monotonic() - t


def run(url, adapters, conc, n, max_tokens, timeout):
    names = itertools.cycle(adapters)
    lat, err = [], 0
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        for fut in cf.as_completed([ex.submit(one, url, next(names), max_tokens, timeout) for _ in range(n)]):
            try:
                lat.append(fut.result())
            except Exception as e:  # noqa: BLE001 -- counted, never timed
                err += 1
                print(f"[lora] request failed: {e}")
    wall = time.monotonic() - t0
    lat.sort()
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3 if lat else 0.0  # noqa: E731
    print(f"LORA adapters={len(adapters)} concurrency={conc} requests_per_s={len(lat) / wall:.2f} "
          f"p50_ms={pct(0.5):.1f} p99_ms={pct(0.99):.1f} n={n} errors={err}")
    return err == 0


def main():
    g = os.environ.get
    adapters = [a for a in g("LORA_ADAPTERS", "").split(",") if a]
    if not adapters:
        print("LORA_ADAPTERS is empty: name the adapters the role serves (lora.modules)", file=sys.stderr)
        return 2
    ok = [run(g("LORA_URL"), adapters, int(c), int(g("LORA_REQUESTS", "100")), int(g("LORA_MAX_TOKENS", "64")),
              float(g("LORA_TIMEOUT", "300"))) for c in g("LORA_CONCURRENCY", "1,8,32").split(",") if c]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
