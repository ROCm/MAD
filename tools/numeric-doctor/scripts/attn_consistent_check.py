#!/usr/bin/env python3
"""Replay a flash-attention VARLEN backward capsule on CONSISTENT operands (debug-numeric step 7/8).

A shapes-only capsule replayed with --synthesize draws every float operand from its
own statistics, so `out` and `softmax_lse` do not belong to q/k/v -- a correct kernel
can overflow on them.  This script keeps the capsule's real layout (cu_seqlens,
max_seqlen, scale, causal, window) and q/k/v/dout statistics, recomputes out/lse with
the real forward, runs the backward, and compares dq/dk/dv with an fp32 reference
(per sequence, batched by length).  For a NaN it also reports which rows went
non-finite against their LSE (the overflow threshold of exp(-lse) in fp32 is -88.72).

    python3 attn_consistent_check.py <capsule.pt> [--seeds 3] [--layout capsule]
            [--save stock.pt | --compare stock.pt]

* --layout: capsule | nozero (drop empty sequences) | short (drop the longest) |
  long (the longest only) -- narrows a trigger by sequence layout.
* Fix validation: run once on stock (--save stock.pt), once with the patched library
  first on PYTHONPATH (--compare stock.pt): every element stock computed finite must be
  IDENTICAL (or report DIFFERS), and the patched run must be all-finite.
* Needs the model's environment (`<GPU_EXEC>`; aiter: AITER_TRITON_ONLY as the model set it).
Covers flash_attn._flash_attn_varlen_backward without dropout / softcap / alibi.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

LSE_OVERFLOW = -88.72          # exp(-lse) > fp32 max below this


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsule")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layout", default="capsule", choices=["capsule", "nozero", "short", "long"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--save")
    g.add_argument("--compare")
    a = ap.parse_args()

    import flash_attn  # noqa: F401  (registers torch.ops.flash_attn)
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    cap = torch.load(a.capsule, weights_only=False, map_location="cpu")
    if "varlen_backward" not in str(cap.get("op")):
        print(f"not a varlen backward capsule: {cap.get('op')}", file=sys.stderr)
        return 2
    A = cap["args"]
    if A[13] or A[18] or A[19] is not None:
        print("dropout / softcap / alibi set: not covered", file=sys.stderr)
        return 2
    lib = sys.modules.get("aiter") or flash_attn
    print(f"flash_attn backend from {os.path.dirname(getattr(lib, '__file__', '?'))}")
    cu0 = A[9]["base"].long()
    L = cu0[1:] - cu0[:-1]
    if a.layout == "nozero":
        L = L[L > 0]
    elif a.layout == "short":
        L = L[L < L.max()]
    elif a.layout == "long":
        L = L[L == L.max()][:1]
    cu = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
    cu_d = cu.int().cuda()
    T, H, D = int(cu[-1]), A[1]["size"][1], A[1]["size"][2]
    Hk = A[2]["size"][1]
    maxlen, scale, causal, (wl, wr) = int(L.max()), A[14], bool(A[15]), (A[16], A[17])
    print(f"layout {a.layout}: {L.numel()} seqs ({int((L == 0).sum())} empty), T={T}, "
          f"max_seqlen={maxlen}, heads q/k {H}/{Hk}, d={D}, scale={scale}, causal={causal}, window=({wl},{wr})")

    stock = torch.load(a.compare) if a.compare else None
    saved, bad_any = {}, False
    for seed in range(a.seeds):
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def draw(i, shape):
            s = A[i]["stats"]
            return (torch.randn(shape, generator=gen, device="cuda") * s["std"] + s["mean"]).clamp(s["min"], s["max"])

        dout = draw(0, (T, H, D)).to(A[0]["dtype"])
        q = draw(1, (T, H, D)).to(A[1]["dtype"])
        k, v = (draw(i, (T, Hk, D)).to(A[i]["dtype"]) for i in (2, 3))
        out, lse, _, _ = _flash_attn_varlen_forward(q, k, v, cu_d, cu_d, maxlen, maxlen, 0.0, scale,
                                                    causal=causal, window_size_left=wl, window_size_right=wr)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        torch.ops.flash_attn._flash_attn_varlen_backward(
            dout, q, k, v, out, lse, dq, dk, dv, cu_d, cu_d, maxlen, maxlen,
            0.0, scale, causal, wl, wr, 0.0, None, False, None)
        torch.cuda.synchronize()

        ref = {n: torch.zeros(T, x.shape[1], D, device="cuda") for n, x in (("dq", q), ("dk", k), ("dv", v))}
        rep = H // Hk
        for ln in torch.unique(L).tolist():
            if ln == 0:
                continue
            idx = (cu[:-1][L == ln][:, None] + torch.arange(ln)).cuda()               # (n, l)
            qq, kk, vv = (x.float()[idx].requires_grad_() for x in (q, k, v))
            ke, ve = (t.repeat_interleave(rep, dim=2) for t in (kk, vv))
            s = torch.einsum("nihd,njhd->nhij", qq, ke) * scale
            i_ = torch.arange(ln, device="cuda")
            allow = torch.ones(ln, ln, dtype=torch.bool, device="cuda")
            if causal:
                allow &= i_[None, :] <= i_[:, None]
            if wl >= 0:
                allow &= i_[None, :] >= i_[:, None] - wl
            if wr >= 0:
                allow &= i_[None, :] <= i_[:, None] + wr
            s = s.masked_fill(~allow, float("-inf"))
            o = torch.einsum("nhij,njhd->nihd", torch.nan_to_num(s.softmax(-1)), ve)
            o.backward(dout.float()[idx])
            for n, t in (("dq", qq), ("dk", kk), ("dv", vv)):
                ref[n][idx.flatten()] = t.grad.reshape(-1, t.shape[2], D)

        low = (lse.t() < LSE_OVERFLOW)                                             # (T, H)
        line = [f"seed {seed}: lse<{LSE_OVERFLOW} rows {int(low.sum())}"]
        for n, t in (("dq", dq), ("dk", dk), ("dv", dv)):
            fin = t.isfinite()
            nf = int((~fin).sum())
            bad_any |= nf > 0
            err = (t.float() - ref[n]).abs()[fin].max().item() if fin.any() else float("nan")
            part = f"{n} non-finite {nf} max|err| {err:.2g} (ref max {ref[n].abs().max().item():.2g})"
            key = f"{a.layout}/{seed}/{n}"
            saved[key] = t.cpu()
            if stock is not None and key in stock:
                sf = stock[key].isfinite()
                same = torch.equal(stock[key][sf].view(torch.int16), t.cpu()[sf].view(torch.int16))
                part += " IDENTICAL-on-stock-finite" if same else " DIFFERS-on-stock-finite"
            line.append(part)
        rows = (~dq.isfinite()).any(-1)
        if rows.any() and dq.shape[1] == H:
            line.append(f"dq NaN rows {int(rows.sum())}: {int((rows & low).sum())} have lse<{LSE_OVERFLOW}, "
                        f"{int((rows & ~low).sum())} do not; seq lens hit "
                        f"{torch.unique(L.repeat_interleave(L)[rows.any(-1).cpu()]).tolist()[:8]}")
        print("; ".join(line), flush=True)
    if a.save:
        torch.save(saved, a.save)
    print("ATTN_CHECK=" + ("NONFINITE" if bad_any else "CLEAN"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
