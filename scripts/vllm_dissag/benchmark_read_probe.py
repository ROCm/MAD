#!/usr/bin/env python3
"""Every-rank KV-transfer probe for a P/D deployment (MoRI-IO READ or WRITE).

Sends REPS rounds of short chat requests whose X-Session-ID values are balanced over the sticky
router's RANKS (two ids per rank), and snapshots every engine's /metrics before and after.
Passes when:
  - every request returned 200;
  - prefill generated exactly one token per request (the one-token prefill leg);
  - decode computed exactly one prompt token per request locally and received the rest by KV
    transfer (vllm:prompt_tokens_by_source{source="external_kv_transfer"} > 0);
  - every decode engine served at least one request.
GPU memory headroom after the first transfer is not visible in /metrics; check it separately.

  benchmark_read_probe.py --url http://host:port --model M \
      --metrics-endpoints "prefill=host:port decode=host:port" [--ranks 8 --reps 2] [--out DIR]
Prints "read_probe requests=N ok=N pass=True|False" and writes <out>/read-probe.json.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_multiturn_retained import balanced_id  # noqa: E402
from multiturn_metrics import csum, delta, lab, parse_prom  # noqa: E402


def snapshot(endpoints, path):
    with open(path, "w") as f:
        for kv in endpoints.split():
            role, ep = kv.split("=", 1)
            f.write(f"# EP role={role} ep={ep}\n")
            try:
                body = urllib.request.urlopen(f"http://{ep}/metrics", timeout=30).read().decode()
                f.write("".join(line + "\n" for line in body.splitlines() if not line.startswith("#")))
            except Exception as e:  # noqa: BLE001 - recorded, and the checks then fail
                f.write(f"# EP-ERROR {ep} {e!r}\n")


def ask(url, model, sid, text):
    body = {"model": model, "max_tokens": 64, "temperature": 0,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Session-ID": sid})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status
    except Exception:  # noqa: BLE001
        return 0


def evaluate(d, n_ok):
    """Gate checks on counter deltas d (multiturn_metrics.parse_prom/delta format)."""
    p_gen = csum(d, "prefill", "vllm:generation_tokens_total")
    d_local = csum(d, "decode", "vllm:prompt_tokens_by_source_total", source="local_compute")
    d_ext = csum(d, "decode", "vllm:prompt_tokens_by_source_total", source="external_kv_transfer")
    engines = {}
    for (r, _, n, labels), v in d.items():
        if r == "decode" and n == "vllm:request_success_total":
            e = lab(labels, "engine")
            engines[e] = engines.get(e, 0) + v
    checks = {
        "prefill_one_token_per_request": n_ok > 0 and p_gen == n_ok,
        "decode_one_local_token_per_request": n_ok > 0 and d_local == n_ok,
        "decode_external_kv_present": d_ext > 0,
        "every_decode_engine_served": bool(engines) and all(v > 0 for v in engines.values()),
    }
    return checks, dict(prefill_generated=p_gen, decode_local_compute=d_local,
                        decode_external_kv_transfer=d_ext, decode_requests_by_engine=engines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics-endpoints", default=os.environ.get("METRICS_ENDPOINTS", ""))
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if "decode=" not in a.metrics_endpoints or "prefill=" not in a.metrics_endpoints:
        raise SystemExit("--metrics-endpoints needs prefill= and decode= endpoints")
    os.makedirs(a.out, exist_ok=True)
    before, after = os.path.join(a.out, "prom-probe-before.prom"), os.path.join(a.out, "prom-probe-after.prom")
    snapshot(a.metrics_endpoints, before)
    ids = [balanced_id(i, a.ranks) for i in range(2 * a.ranks)]
    codes = []
    for rep in range(a.reps):
        with cf.ThreadPoolExecutor(len(ids)) as ex:
            codes += list(ex.map(lambda x: ask(a.url, a.model, x[1],
                                               f"Probe {rep} {x[0]}. What is 17 times 23? Reply with the number."),
                                 enumerate(ids)))
    time.sleep(5)
    snapshot(a.metrics_endpoints, after)
    n_ok = sum(c == 200 for c in codes)
    checks, counts = evaluate(delta(parse_prom(before), parse_prom(after)), n_ok)
    checks["all_requests_ok"] = n_ok == len(codes)
    ok = all(checks.values())
    json.dump(dict(requests=len(codes), ok=n_ok, checks=checks, **counts, passed=ok),
              open(os.path.join(a.out, "read-probe.json"), "w"), indent=1)
    print(json.dumps(dict(checks=checks, **counts)))
    print(f"read_probe requests={len(codes)} ok={n_ok} pass={ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
