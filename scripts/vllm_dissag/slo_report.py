#!/usr/bin/env python3
"""
Turn vLLM bench-serve result JSONs into a customer-facing SLO verdict.

Reads the --save-result JSONs written by benchmark_customer_slo.sh and answers the
only two questions the customer actually asked:

    does it meet the SLO, and if not, by how much

Design notes, because each of these was a real trap:

  * The customer's throughput targets are PER RANK ("prefill: 34000tokens/s
    (per-rank)"). vLLM reports AGGREGATE. Dividing by --dp-ranks is therefore not
    cosmetic -- reporting the aggregate against a per-rank target overstates the
    result by exactly the DP degree.

  * "Prefill throughput" is not a field vLLM emits. We derive it as
    total_input_tokens / duration, which is prompt tokens processed per wall second.
    That is an END-TO-END rate that includes decode time for the same requests, so
    it is a *lower bound* on the prefill engine's rate, not a kernel measurement.
    Labelled as such below rather than quietly presented as the engine number.

  * Goodput is the honest headline. mean_ttft can pass while a third of requests
    miss the SLO; request_goodput counts only the requests that met BOTH ttft and
    tpot thresholds, so goodput/throughput is the fraction of traffic actually
    served acceptably.

  * The gate is on the MEAN, because the acceptance row of the customer sheet says
    "avg ttft: <7s" and "avg tpot: 50ms/token(avg)". Percentiles are reported so
    the tail is visible, but they are not gated -- the customer did not set them.
"""

import argparse
import csv
import glob
import json
import os
import sys


def _fmt(v, spec="%.1f"):
    return "n/a" if v is None else spec % v


def load(result_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print("  warning: could not read %s (%s) -- skipped" % (path, e))
            continue
        if "mean_ttft_ms" not in d:
            # Not a bench-serve result (or a run that died before reporting).
            print("  warning: %s has no metrics -- run likely failed; skipped"
                  % os.path.basename(path))
            continue
        # Recover the label/concurrency from the filename we chose in the runner:
        #   <label>_con<N>_iter<M>.json
        base = os.path.basename(path)[:-5]
        label, con, itr = "?", 0, 1
        try:
            parts = base.split("_")
            label = parts[0]
            for p in parts[1:]:
                if p.startswith("con"):
                    con = int(p[3:])
                elif p.startswith("iter"):
                    itr = int(p[4:])
        except (ValueError, IndexError):
            pass
        d["_label"], d["_con"], d["_iter"], d["_file"] = label, con, itr, base
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--slo-ttft-ms", type=float, default=7000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=50.0)
    ap.add_argument("--target-prefill", type=float, default=34000.0)
    ap.add_argument("--target-decode", type=float, default=670.0)
    ap.add_argument("--dp-ranks", type=int, default=8)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    rows = load(a.result_dir)
    if not rows:
        print("No parseable results in %s -- nothing to judge." % a.result_dir)
        return 2

    hdr = ("%-10s %5s | %8s %8s %8s | %7s %7s %7s | %7s | %9s %8s"
           % ("scenario", "con", "ttft_avg", "p95", "p99",
              "tpot_avg", "p95", "p99", "good%", "prefill/r", "dec/r"))
    print(hdr)
    print("-" * len(hdr))

    out = []
    failures = []
    for d in sorted(rows, key=lambda r: (r["_label"], r["_con"], r["_iter"])):
        ttft = d.get("mean_ttft_ms")
        tpot = d.get("mean_tpot_ms")
        dur = d.get("duration") or 0.0
        n = a.dp_ranks

        # See module docstring: end-to-end lower bound, not a kernel rate.
        prefill_pr = (d.get("total_input_tokens", 0) / dur / n) if dur else None
        decode_pr = (d.get("output_throughput") or 0.0) / n

        thr = d.get("request_throughput") or 0.0
        good = d.get("request_goodput")
        goodpct = (100.0 * good / thr) if (good is not None and thr) else None

        ttft_ok = ttft is not None and ttft <= a.slo_ttft_ms
        tpot_ok = tpot is not None and tpot <= a.slo_tpot_ms
        mark = "" if (ttft_ok and tpot_ok) else "   <-- MISS"
        if not (ttft_ok and tpot_ok):
            why = []
            if not ttft_ok:
                why.append("TTFT %s ms > %.0f (%.2fx)"
                           % (_fmt(ttft), a.slo_ttft_ms,
                              (ttft / a.slo_ttft_ms) if ttft else float("nan")))
            if not tpot_ok:
                why.append("TPOT %s ms > %.0f (%.2fx)"
                           % (_fmt(tpot), a.slo_tpot_ms,
                              (tpot / a.slo_tpot_ms) if tpot else float("nan")))
            failures.append("%s con=%d: %s" % (d["_label"], d["_con"], "; ".join(why)))

        print("%-10s %5d | %8s %8s %8s | %7s %7s %7s | %7s | %9s %8s%s"
              % (d["_label"], d["_con"],
                 _fmt(ttft), _fmt(d.get("p95_ttft_ms")), _fmt(d.get("p99_ttft_ms")),
                 _fmt(tpot, "%.2f"), _fmt(d.get("p95_tpot_ms"), "%.2f"),
                 _fmt(d.get("p99_tpot_ms"), "%.2f"),
                 _fmt(goodpct, "%.1f"), _fmt(prefill_pr, "%.0f"),
                 _fmt(decode_pr, "%.0f"), mark))

        out.append({
            "scenario": d["_label"], "concurrency": d["_con"], "iter": d["_iter"],
            "input_tokens": d.get("total_input_tokens"),
            "output_tokens": d.get("total_output_tokens"),
            "duration_s": "%.1f" % dur if dur else "",
            "mean_ttft_ms": _fmt(ttft), "p50_ttft_ms": _fmt(d.get("p50_ttft_ms")),
            "p95_ttft_ms": _fmt(d.get("p95_ttft_ms")),
            "p99_ttft_ms": _fmt(d.get("p99_ttft_ms")),
            "mean_tpot_ms": _fmt(tpot, "%.2f"),
            "p50_tpot_ms": _fmt(d.get("p50_tpot_ms"), "%.2f"),
            "p95_tpot_ms": _fmt(d.get("p95_tpot_ms"), "%.2f"),
            "p99_tpot_ms": _fmt(d.get("p99_tpot_ms"), "%.2f"),
            "mean_e2el_ms": _fmt(d.get("mean_e2el_ms")),
            "goodput_pct": _fmt(goodpct, "%.1f"),
            "prefill_tok_s_per_rank": _fmt(prefill_pr, "%.0f"),
            "decode_tok_s_per_rank": _fmt(decode_pr, "%.0f"),
            "ttft_slo_met": ttft_ok, "tpot_slo_met": tpot_ok,
        })

    print()
    print("SLO: TTFT <= %.0f ms avg, TPOT <= %.0f ms avg" % (a.slo_ttft_ms, a.slo_tpot_ms))
    print("Per-rank targets (DP=%d): prefill %.0f tok/s, decode %.0f tok/s"
          % (a.dp_ranks, a.target_prefill, a.target_decode))
    print("NOTE prefill/r is total_input_tokens/duration/DP -- an end-to-end LOWER BOUND")
    print("     that includes decode time, not an isolated prefill-engine measurement.")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print("wrote %s" % a.csv)

    print()
    if failures:
        print("VERDICT: FAIL -- %d of %d cells missed SLO" % (len(failures), len(out)))
        for f_ in failures:
            print("  - %s" % f_)
        return 1
    print("VERDICT: PASS -- all %d cells met TTFT and TPOT SLO" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
