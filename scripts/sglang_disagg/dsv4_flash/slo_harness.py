#!/usr/bin/env python3
# SLO harness: streaming TTFT/TPOT percentiles for DSV4-Flash via sglang router.
# Measures per-request TTFT (first token) + TPOT (mean inter-token), reports
# p50/p95/p99 TTFT, p95 TPOT, aggregate output tok/s, E2E percentiles.
import json, urllib.request, time, threading, os, sys, statistics

URL = os.environ.get("ENDPOINT", "http://192.168.200.55:2322") + "/v1/completions"
M = os.environ.get("MODEL", "/models/DeepSeek-V4-Flash-FP8-E4M3")
ISL = int(os.environ.get("ISL", "100000"))
OSL = int(os.environ.get("OSL", "1100"))
# unique-ish filler to avoid radix cache sharing across requests
FILL = "The quick brown fox jumps over the lazy dog. "

def make_prompt(isl, salt):
    reps = isl // 9 + 1
    base = (f"[req{salt}] " + FILL * reps)
    return base[: isl * 5]

def one(prompt, osl, res, i):
    t0 = time.time()
    ttft = None; nevent = 0; usage_tok = 0
    body = json.dumps({"model": M, "prompt": prompt, "max_tokens": osl,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=1200)
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            u = obj.get("usage")
            if u and u.get("completion_tokens"): usage_tok = u["completion_tokens"]
            txt = obj.get("choices", [{}])[0].get("text", "") if obj.get("choices") else ""
            if txt:
                now = time.time()
                if ttft is None: ttft = now - t0
                nevent += 1
        e2e = time.time() - t0
        ntok = usage_tok or nevent   # real output tokens (usage) — correct under MTP bursts
        # eff_tpot = decode wall time / REAL output tokens — burst-insensitive, correct for
        # speculative decode (MTP). This is the TRUE per-output-token latency over decode.
        eff_tpot = ((e2e - ttft) / max(ntok - 1, 1)) * 1000.0 if ttft is not None else 0.0
        res[i] = {"ttft": ttft, "eff_tpot": eff_tpot, "ntok": ntok, "e2e": e2e}
    except Exception as e:
        res[i] = {"err": str(e)[:80]}

def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs); k = (len(xs)-1) * p/100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f]) * (k-f)

def run(con):
    prompts = [make_prompt(ISL, i) for i in range(con)]
    res = [None]*con; ths=[]; t0=time.time()
    for i in range(con):
        th = threading.Thread(target=one, args=(prompts[i], OSL, res, i)); th.start(); ths.append(th)
    for th in ths: th.join()
    wall = time.time()-t0
    ok = [r for r in res if r and "err" in r is False or (r and "ttft" in r and r["ttft"] is not None)]
    ok = [r for r in res if r and r.get("ttft") is not None]
    errs = [r for r in res if r and "err" in r]
    if not ok:
        print(f"CON={con}: ALL FAILED. sample_err={errs[0]['err'] if errs else '?'}", flush=True); return
    ttfts=[r["ttft"] for r in ok]
    eff=[r["eff_tpot"] for r in ok if r["eff_tpot"]>0]
    ntoks=sum(r["ntok"] for r in ok); e2es=[r["e2e"] for r in ok]
    print(f"CON={con} ISL={ISL} OSL={OSL}: {len(ok)}/{con} ok"
          f" | TTFT p50={pct(ttfts,50):.2f}s p95={pct(ttfts,95):.2f}s p99={pct(ttfts,99):.2f}s"
          f" | TPOT p50={pct(eff,50):.0f}ms p95={pct(eff,95):.0f}ms"
          f" | E2E p50={pct(e2es,50):.1f}s p95={pct(e2es,95):.1f}s"
          f" | agg_out={ntoks/wall:.0f} tok/s | wall={wall:.1f}s"
          + (f" | ERRS={len(errs)}" if errs else ""), flush=True)

if __name__ == "__main__":
    cons = [int(x) for x in os.environ.get("CONS", "12,24,36").split(",")]
    print(f"# SLO sweep ISL={ISL} OSL={OSL} cons={cons} endpoint={URL}", flush=True)
    for c in cons:
        run(c)
