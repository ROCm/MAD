#!/usr/bin/env python3
"""Reduce one warm multi-turn (or cold `vllm bench serve --save-detailed`) run directory.

  multiturn_metrics.py <run_dir> --mode multiturn|bench_serve --gpus N --isl ISL --osl OSL
      [--t-meas T] [--layout pd|coloc] [--itl-targets-ms 100,150,250,500] [--out result.json]

Inputs (written by benchmark_multiturn_retained.sh / benchmark_long_context.sh LC_DETAILED=1):
  multiturn    round-*/stats.json (client --stats-json-output), optional round-*/itl.jsonl
               (per-chunk gaps from the patched client), round-*/prom-{before,after}.prom
  bench_serve  bench_serve.json (--save-detailed), prom-{before,after}.prom
  both         samples.tsv (scrape_engine_metrics.sh sample), t_end (wall-clock end)

Client side comes from per-request data only (the tools' printed aggregates differ: bench serve's
TPOT excludes 1-token outputs, the multi-turn client hides percentiles under 100 samples).
Server side comes from the engines' /metrics: counter deltas between snapshots (rounds 2.. of a
multi-turn run, so round 1's cold prefill is excluded), histogram quantiles by interpolation.

Throughput is output tokens emitted inside the steady window (all sessions active) divided by
the window and by --gpus (every GPU of every role: 16 for 1P1D at 8 GPUs per node).
Goodput at an ITL target T: the run passes when the engine decode-step ITL p99 (histogram
vllm:inter_token_latency_seconds, decode role for P/D) <= T and client TTFT p90 <= the TTFT SLO
(default max(10 s, ISL / 10k tokens/s)); goodput is tok/s/GPU when it passes, else 0.
Prints one "MT_RESULT {...}" line; parse_to_csv.py --result-json turns result.json into perf.csv.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

PCTS = (50, 90, 99, 99.9)


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    return s[f] if f == len(s) - 1 else s[f] + (s[f + 1] - s[f]) * (k - f)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def stats(prefix, xs, out):
    out[f"mean_{prefix}_ms"] = mean(xs)
    for p in PCTS:
        out[f"p{p:g}_{prefix}_ms"] = pct(xs, p)


def _read(d, name):
    p = os.path.join(d, name)
    return float(open(p).read().strip()) if os.path.exists(p) else None


def _rounds(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "round-*")), key=lambda p: int(p.rsplit("-", 1)[1]))


# ----------------------------------------------------------------------------- client loaders
def load_bench_serve(run_dir):
    j = json.load(open(os.path.join(run_dir, "bench_serve.json")))
    errs = j.get("errors") or [""] * len(j["start_times"])
    reqs = []
    for i, st in enumerate(j["start_times"]):
        ok = not errs[i] and j["output_lens"][i] > 0
        reqs.append(dict(start=st, end=st + j["latencies"][i], ttft_ms=j["ttfts"][i] * 1e3,
                         itl_ms=[x * 1e3 for x in j["itls"][i]], out=j["output_lens"][i],
                         inp=j["input_lens"][i], ok=ok, conv=None))
    return [dict(reqs=reqs, t_end=_read(run_dir, "t_end"))]


def load_multiturn(run_dir):
    rounds = []
    for rd in _rounds(run_dir):
        f = os.path.join(rd, "stats.json")
        if not os.path.exists(f):
            continue
        chunks = {}
        itl_f = os.path.join(rd, "itl.jsonl")
        if os.path.exists(itl_f):
            for line in open(itl_f):
                r = json.loads(line)
                chunks[(r["conversation_id"], round(r["start_time_ms"], 3))] = r.get("itl_ms")
        reqs = []
        for s in json.load(open(f)):
            st = s["start_time_ms"] / 1e3
            reqs.append(dict(start=st, end=st + s["latency_ms"] / 1e3, ttft_ms=s["ttft_ms"],
                             tpot_ms=s["tpot_ms"], out=s["output_num_tokens"], inp=s["input_num_tokens"],
                             itl_ms=s.get("itl_ms") or chunks.get((s["conversation_id"], round(s["start_time_ms"], 3))),
                             ok=True, conv=s["conversation_id"]))
        rounds.append(dict(reqs=reqs, t_end=_read(rd, "t_end")))
    return rounds


# ----------------------------------------------------------------------------- windows
def steady_window(rnd, multiturn, t_meas):
    reqs = [r for r in rnd["reqs"] if r["ok"]]
    if not reqs:
        return None
    if not multiturn:
        return min(r["start"] for r in reqs), max(r["start"] for r in reqs)
    by = defaultdict(list)
    for r in reqs:
        by[r["conv"]].append(r)
    t1 = min(max(x["end"] for x in v) for v in by.values())   # first session done
    t0 = min(r["start"] for r in reqs)
    if t_meas >= 4:                                            # skip the synchronized first turn
        t0 = max(min(x["end"] for x in v) for v in by.values())
    return (t0, t1) if t1 > t0 else (min(r["start"] for r in reqs), t1)


def tokens_in(r, t0, t1):
    """Output tokens emitted inside [t0, t1]: chunk times if known, else uniform over decode."""
    first = r["start"] + r["ttft_ms"] / 1e3
    n = r["out"]
    if n <= 0:
        return 0.0
    if r.get("itl_ms"):
        t, cnt = first, (1.0 if t0 <= first <= t1 else 0.0)
        per = (n - 1) / len(r["itl_ms"])
        for g in r["itl_ms"]:
            t += g / 1e3
            if t0 <= t <= t1:
                cnt += per
        return cnt
    if r["end"] <= first:
        return float(n) if t0 <= first <= t1 else 0.0
    lo, hi = max(t0, first), min(t1, r["end"])
    return max(0.0, hi - lo) / (r["end"] - first) * n


# ----------------------------------------------------------------------------- server side
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([-+eE0-9.naNInf]+)')


def parse_prom(path):
    """-> {(role, ep, name, labels): value}"""
    out, role, ep = {}, None, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.startswith("# EP "):
            kv = dict(x.split("=", 1) for x in line[5:].split())
            role, ep = kv.get("role"), kv.get("ep")
            continue
        m = LINE.match(line)
        if m:
            out[(role, ep, m.group(1), m.group(3) or "")] = float(m.group(4))
    return out


def delta(a, b):
    return {k: b[k] - a.get(k, 0.0) for k in b}


def lab(labels, key):
    m = re.search(rf'{key}="([^"]*)"', labels)
    return m.group(1) if m else None


def csum(d, role, name, **want):
    return sum(v for (r, _, n, l), v in d.items()
               if r == role and n == name and all(lab(l, k) == w for k, w in want.items()))


def hist(d, role, name):
    """Merged cumulative buckets over all engines of a role -> [(le, count)]."""
    b = defaultdict(float)
    for (r, _, n, l), v in d.items():
        if r == role and n == name + "_bucket":
            le = lab(l, "le")
            b[float("inf") if le == "+Inf" else float(le)] += v
    return sorted(b.items())


def hist_q(buckets, q):
    if not buckets or buckets[-1][1] <= 0:
        return None
    target, prev_le, prev_c = q * buckets[-1][1], 0.0, 0.0
    for le, c in buckets:
        if c >= target:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((target - prev_c) / max(c - prev_c, 1e-12))
        prev_le, prev_c = le, c
    return prev_le


def hist_frac_above(buckets, x):
    """Share of samples above x; None unless x is a bucket edge (no interpolation)."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    below = next((c for le, c in buckets if abs(le - x) < 1e-9), None)
    return None if below is None else 1 - below / buckets[-1][1]


def counter_deltas(run_dir, multiturn):
    if not multiturn:
        return delta(parse_prom(os.path.join(run_dir, "prom-before.prom")),
                     parse_prom(os.path.join(run_dir, "prom-after.prom")))
    rds = _rounds(run_dir)
    d = defaultdict(float)
    for rd in (rds[1:] if len(rds) > 1 else rds):
        for k, v in delta(parse_prom(os.path.join(rd, "prom-before.prom")),
                          parse_prom(os.path.join(rd, "prom-after.prom"))).items():
            d[k] += v
    return d


def server_metrics(d, roles, itl_role, targets_ms):
    res = {}
    for role in roles:
        q = csum(d, role, "vllm:prefix_cache_queries_total")
        res[f"{role}_prefix_cache_hit_rate"] = csum(d, role, "vllm:prefix_cache_hits_total") / q if q else None
        src = {s: csum(d, role, "vllm:prompt_tokens_by_source_total", source=s)
               for s in ("local_compute", "local_cache_hit", "external_kv_transfer")}
        tot = sum(src.values())
        for s, v in src.items():
            res[f"{role}_prompt_frac_{s}"] = v / tot if tot else None
        res[f"{role}_preemptions"] = csum(d, role, "vllm:num_preemptions_total")
        drafts = csum(d, role, "vllm:spec_decode_num_drafts_total")
        dtok = csum(d, role, "vllm:spec_decode_num_draft_tokens_total")
        acc = csum(d, role, "vllm:spec_decode_num_accepted_tokens_total")
        res[f"{role}_spec_acceptance_rate"] = acc / dtok if dtok else None
        res[f"{role}_spec_acceptance_length"] = 1 + acc / drafts if drafts else None
    itl = hist(d, itl_role, "vllm:inter_token_latency_seconds")
    for p in PCTS:
        res[f"step_itl_p{p:g}_ms"] = hist_q(itl, p / 100) * 1e3 if itl else None
    res["step_itl_count"] = itl[-1][1] if itl else 0
    for t in targets_ms:
        res[f"step_itl_frac_gt_{t:g}ms"] = hist_frac_above(itl, t / 1000) if itl else None
    res["step_itl_role"] = itl_role
    return res


def gauges(run_dir, windows):
    res, vals = {}, defaultdict(list)
    sp = os.path.join(run_dir, "samples.tsv")
    if not (os.path.exists(sp) and windows):
        return res
    per_ts = defaultdict(float)
    for row in csv.DictReader(open(sp), delimiter="\t"):
        ts = float(row["ts"])
        if not any(a <= ts <= b for a, b in windows):
            continue
        key = row["metric"].split(":", 1)[1]
        if key == "num_requests_kv_fetch_by_stage":
            key += "_" + (lab(row["labels"], "stage") or "")
        if key == "kv_cache_usage_perc":
            vals[(row["role"], key)].append(float(row["value"]))
        else:
            per_ts[(row["role"], key, ts)] += float(row["value"])
    for (role, key, _), v in per_ts.items():
        vals[(role, key)].append(v)
    for (role, key), vs in vals.items():
        res[f"{role}_{key}_mean"] = mean(vs)
        res[f"{role}_{key}_max"] = max(vs)
    return res


# ----------------------------------------------------------------------------- main
def build(run_dir, mode, gpus, isl, osl, t_meas=2, layout="pd", targets_ms=(100, 150, 250, 500),
          ttft_slo_ms=None):
    multiturn = mode == "multiturn"
    rounds = load_multiturn(run_dir) if multiturn else load_bench_serve(run_dir)
    sel, toks, span, walls = [], 0.0, 0.0, []
    for rnd in rounds:
        w = steady_window(rnd, multiturn, t_meas)
        if not w:
            continue
        t0, t1 = w
        span += t1 - t0
        for r in rnd["reqs"]:
            if r["ok"]:
                toks += tokens_in(r, t0, t1)
                if t0 <= r["start"] < t1:
                    sel.append(r)
        if rnd.get("t_end"):
            off = rnd["t_end"] - max(r["end"] for r in rnd["reqs"] if r["ok"])  # client clock -> wall
            walls.append((t0 + off, t1 + off))
    rec = dict(mode=mode, layout=layout, gpus=gpus, isl_target=isl, osl=osl, t_meas=t_meas,
               rounds=len(rounds), window_s=span, completed=len(sel),
               failed=sum(1 for rnd in rounds for r in rnd["reqs"] if not r["ok"]))
    rec["mean_input_tokens"] = mean([r["inp"] for r in sel])
    rec["osl_exact_frac"] = (sum(r["out"] == osl for r in sel) / len(sel)) if sel else None
    rec["output_throughput"] = toks / span if span else None
    rec["out_tok_s_per_gpu"] = rec["output_throughput"] / gpus if rec["output_throughput"] else None
    stats("ttft", [r["ttft_ms"] for r in sel], rec)
    stats("e2el", [(r["end"] - r["start"]) * 1e3 for r in sel], rec)
    has_itl = bool(sel) and all(r.get("itl_ms") for r in sel if r["out"] > 1)
    if has_itl:
        stats("itl", [g for r in sel for g in r["itl_ms"]], rec)
    rec["client_itl_source"] = "client-chunks" if has_itl else None
    roles = ["prefill", "decode"] if layout == "pd" else ["coloc"]
    itl_role = "decode" if layout == "pd" else "coloc"
    rec.update(server_metrics(counter_deltas(run_dir, multiturn), roles, itl_role, targets_ms))
    rec.update(gauges(run_dir, walls))
    rec["slo_ttft_ms"] = ttft_slo_ms if ttft_slo_ms else max(10.0, isl / 10000) * 1000
    p90 = rec.get("p90_ttft_ms")
    p99 = rec.get("step_itl_p99_ms")
    for t in targets_ms:
        ok = p99 is not None and p90 is not None and p99 <= t and p90 <= rec["slo_ttft_ms"]
        rec[f"slo{t:g}_pass"] = ok
        rec[f"slo{t:g}_goodput_tok_s_per_gpu"] = rec["out_tok_s_per_gpu"] if ok else 0.0
        if has_itl and span:
            good = [r for r in sel if r["ttft_ms"] <= rec["slo_ttft_ms"] and (pct(r["itl_ms"], 99) or 0) <= t]
            rec[f"slo{t:g}_req_goodput_tok_s_per_gpu"] = sum(r["out"] for r in good) / span / gpus
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mode", choices=["multiturn", "bench_serve"], required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--t-meas", type=int, default=2)
    ap.add_argument("--layout", choices=["pd", "coloc"], default="pd")
    ap.add_argument("--itl-targets-ms", default="100,150,250,500")
    ap.add_argument("--ttft-slo-ms", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    rec = build(a.run_dir, a.mode, a.gpus, a.isl, a.osl, a.t_meas, a.layout,
                [float(x) for x in a.itl_targets_ms.split(",")], a.ttft_slo_ms)
    out = a.out or os.path.join(a.run_dir, "result.json")
    open(out, "w").write(json.dumps(rec, indent=1, default=str) + "\n")
    keys = ("completed", "failed", "osl_exact_frac", "out_tok_s_per_gpu", "p90_ttft_ms", "step_itl_p99_ms",
            "slo150_goodput_tok_s_per_gpu")
    print("MT_RESULT " + json.dumps({k: rec.get(k) for k in keys}, default=str))


if __name__ == "__main__":
    main()
