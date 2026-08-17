#!/usr/bin/env python3
"""
Pool the per-iteration workload samples into ONE reportable distribution.

Why this exists
---------------
The customer sheet gives an average ISL (80K for the 256K row, 200K for the 1M row)
and a context window whose p99 we are asked to reach. Those two numbers pin a
right-skewed distribution -- and drawing a few hundred samples from a right-skewed
distribution gives a realised mean that wobbles a lot:

    CV = sqrt(exp(sigma^2) - 1),  SE(mean)/mean = CV / sqrt(n)

    256K row: sigma 0.5833, CV 0.637, n=256 -> 4.0%
    1M   row: sigma 0.8778, CV 1.078, n=128 -> 9.5%

Measured across 12 seeds the realised mean actually spanned 14.2% and 35.4%. So a
single run's mean is not a number worth quoting on its own, and "our trace averaged
72,884 tokens" is a fact about that seed, not about the workload.

Ten runs at DISTINCT seeds pool to n*10, and the error falls as 1/sqrt(10):
4.0% -> 1.3% and 9.5% -> 3.0%. This script does that pooling and prints the pooled
figure, which IS worth quoting.

What it pools, and what it deliberately does not
------------------------------------------------
  * It reads the ACHIEVED input_tokens straight out of each JSONL, not the summary in
    the .meta.json sidecar. Percentiles do not average: the mean of ten p99s is not
    the p99 of the pool, and for a heavy tail it is materially lower. Pooling the raw
    lengths and re-computing is the only way to get a p99 that means anything.
  * The mean DOES average (all samples are the same size), but it is recomputed from
    the pool anyway so that one code path produces every number.
  * It refuses to pool files whose targets differ. Averaging an 80K sample with a
    200K one would produce a number describing no workload at all.

It reports deviation as a PERCENTAGE against the target, because that is the form the
number has to travel in. "78,412 tokens, 2.0% below the 80,000 target" is reportable;
"78412" alone invites the reader to assume it was meant to be exact.
"""

import argparse
import glob
import json
import math
import os
import sys


def percentile(sorted_vals, q):
    """Nearest-rank, matching gen_workload.py so the two agree exactly."""
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1,
                   int(math.ceil(q / 100.0 * len(sorted_vals))) - 1))
    return sorted_vals[k]


def read_lengths(path):
    """Achieved input_tokens per request. Falls back to the char count only if the
    field is missing, which it never is for files this repo writes -- but a silent
    zero would quietly drag the pooled mean down, so the fallback is explicit."""
    lens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("input_tokens")
            if v is None:
                v = len(d.get("prompt", "")) // 3
            lens.append(int(v))
    return lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True,
                    help="e.g. '<workload-dir>/256k-ctx_s*.jsonl'")
    ap.add_argument("--label", default="", help="name for the printed report")
    ap.add_argument("--out", default="", help="write the pooled report as JSON here")
    a = ap.parse_args()

    paths = sorted(glob.glob(a.glob))
    if not paths:
        print("No workload files matched %s" % a.glob)
        return 2

    pool, per_run, targets, sigmas = [], [], set(), set()
    for p in paths:
        lens = read_lengths(p)
        if not lens:
            print("  warning: %s had no rows -- skipped" % os.path.basename(p))
            continue
        meta_p = p + ".meta.json"
        seed = None
        if os.path.exists(meta_p):
            try:
                with open(meta_p) as f:
                    m = json.load(f)
                t = m.get("target", {})
                targets.add((t.get("mean_isl"), t.get("context_window")))
                sigmas.add(round(m.get("lognormal", {}).get("sigma", 0.0), 6))
                seed = m.get("seed")
            except (OSError, ValueError):
                pass
        s = sorted(lens)
        per_run.append({"file": os.path.basename(p), "seed": seed, "n": len(s),
                        "mean": sum(s) / float(len(s)), "p99": percentile(s, 99)})
        pool.extend(lens)

    if not pool:
        print("No usable samples.")
        return 2

    if len(targets) > 1:
        print("REFUSING to pool: the samples target different workloads: %s"
              % sorted(targets))
        return 2

    target_mean, window = (list(targets)[0] if targets else (None, None))

    s = sorted(pool)
    n = len(s)
    mean = sum(s) / float(n)
    pooled = {
        "n_runs": len(per_run), "n_total": n,
        "mean": mean, "min": s[0],
        "p50": percentile(s, 50), "p95": percentile(s, 95),
        "p99": percentile(s, 99), "max": s[-1],
        "at_window": sum(1 for v in s if window and v >= window),
    }

    print("=" * 74)
    print("Pooled achieved workload distribution%s"
          % ("  [%s]" % a.label if a.label else ""))
    print("=" * 74)
    print("Per-run realised means (this spread IS the sampling error, not a bug):")
    for r in sorted(per_run, key=lambda x: (x["seed"] is None, x["seed"])):
        dev = ""
        if target_mean:
            dev = "  [%+.1f%%]" % (100.0 * (r["mean"] - target_mean) / target_mean)
        print("  seed=%-4s n=%-5d mean=%-9.0f p99=%-9d %s   %s"
              % (r["seed"], r["n"], r["mean"], r["p99"], dev, r["file"]))
    lo = min(r["mean"] for r in per_run)
    hi = max(r["mean"] for r in per_run)
    if lo > 0:
        print("  spread across runs: %.0f - %.0f  (%.1f%% of the low end)"
              % (lo, hi, 100.0 * (hi - lo) / lo))

    print()
    print("POOLED over %d runs, n=%d requests:" % (len(per_run), n))
    print("  mean = %.0f" % mean)
    print("  p50  = %d      p95 = %d      p99 = %d      max = %d"
          % (pooled["p50"], pooled["p95"], pooled["p99"], pooled["max"]))
    if target_mean:
        pooled["mean_dev_pct"] = 100.0 * (mean - target_mean) / float(target_mean)
        print("  vs target mean %d: %+.1f%%" % (target_mean, pooled["mean_dev_pct"]))
    if window:
        print("  vs context window %d: p99 reaches %.1f%% of it, %d requests at the window"
              % (window, 100.0 * pooled["p99"] / window, pooled["at_window"]))
    if sigmas and len(sigmas) == 1:
        sigma = list(sigmas)[0]
        cv = math.sqrt(math.exp(sigma * sigma) - 1.0)
        se1 = 100.0 * cv / math.sqrt(per_run[0]["n"])
        sen = 100.0 * cv / math.sqrt(n)
        pooled["expected_se_pct_of_mean"] = sen
        print()
        print("  Expected SE of the mean: %.1f%% for one run (n=%d), %.1f%% pooled (n=%d)."
              % (se1, per_run[0]["n"], sen, n))
        print("  A pooled mean within ~2 SE (%.1f%%) of target is a normal draw." % (2 * sen))

    print()
    print("REPORT THIS, not the target: it is what was actually served.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "target_mean_isl": target_mean,
                       "context_window": window, "per_run": per_run,
                       "pooled": pooled}, f, indent=2, sort_keys=True)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
