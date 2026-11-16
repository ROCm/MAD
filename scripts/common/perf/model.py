#!/usr/bin/env python3
"""A first-order performance model, to rank layouts before spending GPU hours on them.

Roofline per GPU from the capability catalog (compat/rocm.yaml: peak dense TFLOPS per format, HBM
bandwidth, scale-up and scale-out link speeds) and a model's architecture (perf/models.yaml):

  prefill  tokens/s/GPU = efficiency x peak / (2 x active params)                 (compute bound)
  decode   one step, `batch` tokens per GPU on a pool of G GPUs (EP over the pool):
           compute  2 x active params x batch                                 / peak
           memory   (weights the pool's tokens touch / G + batch x context x KV)  / HBM bandwidth
           comm     MoE all-to-all, batch x hidden x experts x 2 x act bytes      / link, over
                    the scale-up link when the EP group fits one domain, else the NICs
           step = max(compute, memory) + comm;  tokens/s/GPU = batch / step

It ignores overlap, kernels, scheduling and the KV transfer between pools: it ranks, it does not
predict. `efficiency` (measured / modelled, per platform) calibrates it once runs exist
(perf/calibration.yaml, when present).

    model.py estimate --model M --platform P [--weights F --compute F --kv F --gpus G --batch B --context L]
    model.py search   --model M --platforms P1,P2 [--context L --batch B,..]   ranked decode layouts
"""
import argparse
import json
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "..", "..", "..", "compat", "rocm.yaml")


def _load(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def bytes_per(catalog, fmt):
    """Bytes per element of a format, with its block scale."""
    f = (catalog.get("formats") or {})[fmt]
    bits = f["bits"]
    if f.get("block"):
        bits += 8 / f["block"]          # one 8-bit scale per block
    return bits / 8


def peak(catalog, acc, fmt):
    """Peak TFLOPS for computing in `fmt` (its family's entry), or None."""
    table = (catalog["accelerators"][acc].get("peak_tflops") or {})
    fam = catalog["formats"][fmt]["family"]
    key = {"bf16": "bf16", "fp16": "bf16", "fp8": "fp8", "mx": fmt, "fp4": "mxfp4"}.get(fam, fmt)
    return table.get(key)


def calibration(platform):
    """Measured / modelled for a platform (perf/calibration.yaml, written from real runs), else 1."""
    path = os.path.join(HERE, "calibration.yaml")
    return float(((_load(path) if os.path.isfile(path) else {}).get(platform) or {}).get("efficiency", 1.0))


def estimate(catalog, model, platform, weights="fp8_e4m3", compute="fp8_e4m3", kv="fp8_e4m3",
             gpus=8, batch=64, context=4096, efficiency=None):
    if efficiency is None:
        efficiency = calibration(platform)
    plat = catalog["platforms"][platform]
    acc = plat["accelerator"]
    prod = catalog["accelerators"][acc]["products"][plat["product"]]
    pk = peak(catalog, acc, compute)
    if pk is None:
        raise ValueError(f"{platform}: no peak TFLOPS for {compute} (compat/rocm.yaml accelerators.{acc}.peak_tflops)")
    flops_per_s = pk * 1e12 * efficiency
    hbm = prod["hbm_tbps"] * 1e12
    act, total = model["active_params_b"] * 1e9, model["total_params_b"] * 1e9
    wb, kb, ab = bytes_per(catalog, weights), bytes_per(catalog, kv), bytes_per(catalog, compute)
    # `batch` is per GPU (each data-parallel rank's); the pool steps batch x gpus tokens at once.
    # Weights a GPU reads per step: its share of what the pool's tokens reach -- all of a dense
    # model (sharded), for MoE the experts the pool's batch touches, spread over the EP group.
    pool = batch * gpus
    touched = total if act == total else min(total, act * pool)
    t_compute = 2 * act * batch / flops_per_s
    t_memory = (touched * wb / gpus + batch * context * model["kv_elems_per_token"] * kb) / (hbm * efficiency)
    t_comm, link = 0.0, None
    if model.get("experts_per_token"):
        domain = (plat.get("scale_up") or {}).get("gpus_per_domain", plat["gpus_per_node"])
        if gpus <= domain:
            link, bw = "scale-up", plat["scale_up"]["gbps_per_gpu"] * 1e9
        else:
            link, bw = "scale-out", plat["nics_per_gpu"] * plat["nic_gbps"] / 8 * 1e9
        t_comm = batch * model["hidden"] * model["experts_per_token"] * 2 * ab / bw
    step = max(t_compute, t_memory) + t_comm
    return {"platform": platform, "gpus": gpus, "batch": batch, "context": context,
            "weights": weights, "compute": compute, "kv": kv,
            "decode_tokens_per_s_per_gpu": batch / step,
            "decode_bound": "compute" if t_compute >= t_memory else "memory",
            "all_to_all_link": link, "step_ms": step * 1e3,
            "prefill_tokens_per_s_per_gpu": flops_per_s / (2 * act),
            "fits": total * wb / gpus <= prod["hbm_gb"] * 1e9 * 0.9}


def search(catalog, model, platforms, context=4096, batches=(16, 64, 256), formats=None):
    """Decode layouts over platforms x pool sizes (whole domains and nodes) x batch x the formats
    each accelerator computes natively, best first; those whose weights do not fit are dropped."""
    out = []
    for name in platforms:
        plat = catalog["platforms"][name]
        acc = plat["accelerator"]
        native = [f for f in catalog["accelerators"][acc].get("matmul") or [] if peak(catalog, acc, f)]
        by_family = {}
        for f in native:                               # one format per family (the first listed)
            by_family.setdefault(catalog["formats"][f]["family"], f)
        fmts = [f for f in (formats or by_family.values()) if f in native]
        kv8 = by_family.get("fp8") or "bf16"          # the KV cache in the GPU's native fp8
        domain = (plat.get("scale_up") or {}).get("gpus_per_domain", plat["gpus_per_node"])
        sizes = sorted({plat["gpus_per_node"] * k for k in (1, 2, 4, 8)} | {domain, 2 * domain})
        for g in sizes:
            for b in batches:
                for f in fmts:
                    e = estimate(catalog, model, name, weights=f, compute=f, kv=kv8,
                                 gpus=g, batch=b, context=context)
                    if e["fits"]:
                        out.append(e)
    return sorted(out, key=lambda e: -e["decode_tokens_per_s_per_gpu"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--platform", required=True)
    for k, d in (("weights", "fp8_e4m3"), ("compute", "fp8_e4m3"), ("kv", "fp8_e4m3")):
        e.add_argument(f"--{k}", default=d)
    for k, d in (("gpus", 8), ("batch", 64), ("context", 4096)):
        e.add_argument(f"--{k}", type=int, default=d)
    s = sub.add_parser("search")
    s.add_argument("--platforms", required=True)
    s.add_argument("--context", type=int, default=4096)
    s.add_argument("--batch", default="16,64,256")
    s.add_argument("--top", type=int, default=10)
    for p in (e, s):
        p.add_argument("--model", required=True)
        p.add_argument("--catalog", default=CATALOG)
    a = ap.parse_args(argv)
    catalog = _load(a.catalog)
    model = _load(os.path.join(HERE, "models.yaml"))[a.model]
    if a.cmd == "estimate":
        print(json.dumps(estimate(catalog, model, a.platform, a.weights, a.compute, a.kv, a.gpus, a.batch,
                                  a.context), indent=1))
        return 0
    rows = search(catalog, model, a.platforms.split(","), a.context, [int(b) for b in a.batch.split(",")])
    print(f"{'platform':22} {'gpus':>4} {'batch':>5} {'format':10} {'tok/s/GPU':>10} {'bound':7} a2a")
    for r in rows[:a.top]:
        print(f"{r['platform']:22} {r['gpus']:>4} {r['batch']:>5} {r['compute']:10} "
              f"{r['decode_tokens_per_s_per_gpu']:>10.0f} {r['decode_bound']:7} {r['all_to_all_link']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
