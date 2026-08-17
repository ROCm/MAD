#!/usr/bin/env python3
"""Aggregate the 8 per-rail MoRI-IO tables in a rank-0 log into one view.

benchmark.py prints one table per initiator process (one per rail/GPU). Each
table is that RAIL's bandwidth. The fabric number people actually want is the
SUM across rails -- 8 NICs moving KV concurrently -- and the spread across
rails, because a single slow rail is what a disagg prefill->decode handoff
actually waits on. Reporting only rail 0 understates the fabric 8x; reporting
only the sum hides a lame rail. So: print both.

Latency is aggregated as MAX, not mean. A batched transfer completes when its
slowest rail completes, so max is the number that bounds a KV handoff.
"""
import re, sys, json
from collections import defaultdict

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                 r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
HDR = re.compile(r"RDMA Benchmark:\s*Initiator Rank\s*(\d+)")

def parse(path):
    rails = defaultdict(dict)   # rail -> msgsize -> dict
    cur = None
    for line in open(path, errors="replace"):
        h = HDR.search(line)
        if h:
            cur = int(h.group(1)); continue
        m = ROW.match(line.strip())
        if m and cur is not None:
            sz, batch, tot, maxbw, avgbw, minlat, avglat = m.groups()
            rails[cur][int(sz)] = dict(batch=int(batch), total_mb=float(tot),
                                       max_bw=float(maxbw), avg_bw=float(avgbw),
                                       min_lat=float(minlat), avg_lat=float(avglat))
    return rails

def main(path):
    rails = parse(path)
    if not rails:
        sys.exit(f"no tables found in {path}")
    ids = sorted(rails)
    sizes = sorted(rails[ids[0]])
    print(f"source : {path}")
    print(f"rails  : {len(ids)} (initiator ranks {ids[0]}..{ids[-1]})\n")

    w = ("| MsgSize | Batch | Per-rail total | AGG Avg BW | AGG Max BW | "
         "Slowest rail | Fastest rail | Max Avg Lat |")
    print(w)
    print("|---------|-------|----------------|------------|------------|"
          "--------------|--------------|-------------|")
    out = []
    for s in sizes:
        avgs = [rails[r][s]["avg_bw"] for r in ids]
        maxs = [rails[r][s]["max_bw"] for r in ids]
        lats = [rails[r][s]["avg_lat"] for r in ids]
        tot  = rails[ids[0]][s]["total_mb"]
        agg_avg, agg_max = sum(avgs), sum(maxs)
        lo, hi = min(avgs), max(avgs)
        hname = f"{s//1024} KiB" if s < 1048576 else f"{s//1048576} MiB"
        print(f"| {hname:>7} | {rails[ids[0]][s]['batch']:>5} | {tot:>10.2f} MB | "
              f"{agg_avg:>7.1f} GB/s | {agg_max:>7.1f} GB/s | "
              f"{lo:>7.2f} GB/s | {hi:>7.2f} GB/s | {max(lats)/1000:>8.2f} ms |")
        out.append(dict(msg_bytes=s, batch=rails[ids[0]][s]["batch"],
                        per_rail_total_mb=tot, agg_avg_bw_gbs=round(agg_avg,2),
                        agg_max_bw_gbs=round(agg_max,2),
                        rail_min_avg_bw_gbs=round(lo,2), rail_max_avg_bw_gbs=round(hi,2),
                        rail_spread_pct=round(100*(hi-lo)/hi,1),
                        max_avg_lat_us=round(max(lats),2),
                        aggregate_total_mb=round(tot*len(ids),2)))

    peak = max(out, key=lambda r: r["agg_avg_bw_gbs"])
    print(f"\npeak aggregate: {peak['agg_avg_bw_gbs']:.1f} GB/s avg "
          f"({peak['agg_max_bw_gbs']:.1f} GB/s max) at "
          f"{peak['msg_bytes']//1048576} MiB blocks")
    print(f"worst rail spread: "
          f"{max(r['rail_spread_pct'] for r in out):.1f}% "
          f"(at {min(out, key=lambda r: r['agg_avg_bw_gbs'])['msg_bytes']} B)")

    js = path.rsplit(".", 1)[0] + "_aggregate.json"
    json.dump(dict(source=path, rails=len(ids), rows=out), open(js, "w"), indent=2)
    print(f"\nwrote {js}")

if __name__ == "__main__":
    main(sys.argv[1])
