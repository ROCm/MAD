#!/usr/bin/env python3
"""Deterministic wrong/NaN GEMM: what does it depend on? (debug-numeric step 7, GEMM recipe)

    python3 gemm_shape_probe.py capsule <capsule.pt> [--call N] [--repeat 20] [--synthesize]
    python3 gemm_shape_probe.py sweep --k K --n N [--dtype bf16] [--m-to 20000] [--tile 128x112]
    python3 gemm_shape_probe.py sentinel --m M --k K --n N [--dtype bf16] [--tile 128x112]

capsule   replays the capsule's chain up to the GEMM-family call --call (default: the
          capsule's own op) with replay_nan_op's binder, then on that call's operands:
          per-group error vs fp64 (a 2-D x 3-D grouped GEMM with offs: which expert
          groups are wrong, with their row counts), repeatability, and one change at a
          time -- contiguous copies, fresh storage, random data at the recorded scale
          (real A / random B and the reverse), the bad group alone, the bad group as a
          plain torch.mm.  Saves the operands as <--out-dir, default $ND_ART/rootcause or the cwd>/gemm<call>_operands.pt
          (never beside the capsule: that may be a package dir, which would ship ~100 MB more).
          Fails on random data too = shape-dependent: go to `sweep`.
sweep     x(M,K) @ w(K,N) for every M in 1..--m-to on fresh random data vs fp32: the bad M
          as ranges, and per range the output tile count for --tile (hipBLASLt's MT<m>x<n>
          from the kernel name: m tiles the output COLUMNS (N), n the ROWS (M)).  Run it
          once per env knob in its own process (TENSILE_*, step7-rootcause.md): bands that
          vanish under a knob are the causal experiment.
sentinel  pre-fills out= with a sentinel (7, then -3) and reports how many wrong elements
          still hold it: all of them = the kernel never wrote those tiles (the model then
          reads leftover memory: NaN, Inf or finite garbage); none = computed wrong.

2026-11-09 (qwen3 MoE, hipBLASLt stream-K, gfx950): capsule -> only the 13803-row expert
group, also on random data and as plain mm; sweep -> 4 bands at 2*ceil(M/112) in
{224..248}; sentinel -> every wrong element unwritten.  Needs a GPU (run under <GPU_EXEC>).
"""
import argparse
import math
import os
import sys

import torch

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def rel_err(o, r):
    if not torch.isfinite(o).all():
        return float("inf")
    return ((o.double() - r).abs().max() / r.abs().max().clamp_min(1e-30)).item()


def tile_of(spec):
    m, n = spec.lower().split("x")
    return int(m), int(n)          # (cols per tile, rows per tile)


def bind_call(path, call, synthesize):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import replay_nan_op as R
    R.args = R.parse_args([path] + (["--synthesize"] if synthesize else []))
    R._import_torch()
    cap = R.upgrade_capsule(torch.load(path, map_location="cpu", weights_only=False))
    dev = torch.device("cuda")
    env, pool = R._chain_roots(cap, dev), cap["pool"]
    want = call if call is not None else cap.get("op_call_index")
    for n, rec in enumerate(cap["chain"]):
        R._restore_rng(rec.get("rng"))
        func = R.resolve_op(rec["op"])
        bound = R._bind(rec["args"], env, pool, dev)
        kwb = R._bind(rec["kwargs"], env, pool, dev)
        bound = R._chain_derive(cap, n, rec, func, bound, kwb, env)
        if rec["call_index"] == want and R._is_gemm(rec["op"]):
            return rec["op"], list(bound), dict(kwb)
        out = func(*bound, **kwb)
        live = [t for t in R.tree_leaves(out) if isinstance(t, torch.Tensor)]
        for key, t in zip(rec["produces"], live):
            if key:
                env[key] = t
    sys.exit("no GEMM-family call #%s in the chain (--call)" % want)


def capsule_mode(a):
    op, args_, kw = bind_call(a.capsule, a.call, a.synthesize)
    A, B = args_[0], args_[1]
    offs = args_[2] if len(args_) > 2 and isinstance(args_[2], torch.Tensor) else kw.get("offs")
    print("op %s  A %s %s %s  B %s %s %s  offs %s" % (op, tuple(A.shape), A.stride(), A.dtype, tuple(B.shape),
                                                    B.stride(), B.dtype, None if offs is None else offs.tolist()))
    out_dir = a.out_dir or (os.path.join(os.environ["ND_ART"], "rootcause") if os.environ.get("ND_ART") else os.getcwd())
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(out_dir, "gemm%s_operands.pt" % (a.call or "op"))
    torch.save({"op": op, "args": [x.cpu() if isinstance(x, torch.Tensor) else x for x in args_], "kwargs": kw},
               out_pt)
    print("operands saved: %s" % out_pt)
    if not (A.dim() == 2 and B.dim() == 3 and offs is not None):
        if A.dim() == 2 and B.dim() == 2:
            groups, call = [(0, A.shape[0])], lambda A_, B_, o_: A_ @ B_
            Bg = lambda B_, g: B_
        else:
            sys.exit("capsule mode handles mm (2-D x 2-D) and grouped mm (2-D x 3-D with offs)")
    else:
        ends = offs.tolist()
        groups = list(zip([0] + ends[:-1], ends))
        call = lambda A_, B_, o_: torch._grouped_mm(A_, B_, offs=o_, out_dtype=kw.get("out_dtype"))
        Bg = lambda B_, g: B_[g]

    def bad(out, A_, B_):
        res = []
        for g, (s, e) in enumerate(groups):
            if e > s:
                err = rel_err(out[s:e], A_[s:e].double() @ Bg(B_, g).double())
                if err > a.tol:
                    res.append((g, e - s, "%.3g" % err))
        return res

    print("groups (rows):", [e - s for s, e in groups])
    seen = {}
    for _ in range(a.repeat):
        k = tuple(bad(call(A, B, offs), A, B))
        seen[k] = seen.get(k, 0) + 1
    print("REPEAT %d runs -> bad (group, rows, rel_err) sets: %s" % (a.repeat, seen))
    badg = sorted({g for k in seen for g, _, _ in k})
    torch.manual_seed(0)
    Ar = (torch.randn(A.shape, device=A.device) * A.float().std()).to(A.dtype)
    Br = (torch.randn(B.shape, device=B.device) * B.float().std()).to(B.dtype)
    for name, A_, B_ in (("contiguous copies", A.contiguous(), B.contiguous()),
                         ("fresh storage, same strides", A.clone(memory_format=torch.preserve_format),
                          B.clone(memory_format=torch.preserve_format)),
                         ("random A and B, same layout", Ar, Br), ("real A, random B", A, Br),
                         ("random A, real B", Ar, B)):
        print("%-30s bad=%s" % (name, bad(call(A_, B_, offs), A_, B_)))
    for g in badg[:8]:
        s, e = groups[g]
        print("%-30s rel_err=%.3g" % ("group %d (%d rows) as torch.mm" % (g, e - s),
                                      rel_err(A[s:e] @ Bg(B, g), A[s:e].double() @ Bg(B, g).double())))
    print("PROBE_DONE")


def sweep_mode(a):
    dt, (tc, tr) = DT[a.dtype], tile_of(a.tile)
    torch.manual_seed(0)
    w = torch.randn(a.k, a.n, device="cuda").to(dt)
    bad = []
    for m in range(a.m_from, a.m_to + 1):
        x = torch.randn(m, a.k, device="cuda").to(dt)
        if rel_err(x @ w, x.double() @ w.double()) > a.tol:
            bad.append(m)
    rng = []
    for m in bad:
        if rng and m == rng[-1][1] + 1:
            rng[-1][1] = m
        else:
            rng.append([m, m])
    env = " ".join("%s=%s" % kv for kv in sorted(os.environ.items()) if kv[0].startswith(("TENSILE_", "HIPBLASLT_")))
    print("SWEEP [%s] %s K=%d N=%d M=%d..%d: %d bad M" % (env or "stock", a.dtype, a.k, a.n, a.m_from, a.m_to, len(bad)))
    print("BAD_M_RANGES %s" % [tuple(r) for r in rng])
    # Tile counts mean something only against this device's CU count (MI300X 304,
    # MI355X 256; 8 XCDs on both), so print it beside them.
    prop = torch.cuda.get_device_properties(0)
    print("DEVICE %s arch=%s CUs=%d" % (prop.name, getattr(prop, "gcnArchName", "?").split(":")[0],
                                        prop.multi_processor_count))
    for lo, hi in rng:
        print("  M %d..%d: row tiles %s, output tiles %s (tile %s)"
              % (lo, hi, sorted({math.ceil(m / tr) for m in (lo, hi)}),
                 sorted({math.ceil(m / tr) * math.ceil(a.n / tc) for m in (lo, hi)}), a.tile))
    print("PROBE_DONE")


def sentinel_mode(a):
    dt, (tc, tr) = DT[a.dtype], tile_of(a.tile)
    torch.manual_seed(0)
    w = torch.randn(a.k, a.n, device="cuda").to(dt)
    x = torch.randn(a.m, a.k, device="cuda").to(dt)
    ref = x.double() @ w.double()
    for sv in (7.0, -3.0):
        out = torch.full((a.m, a.n), sv, device="cuda", dtype=dt)
        torch.mm(x, w, out=out)
        badm = ~torch.isfinite(out) | ((out.double() - ref).abs() > a.tol * ref.abs().max())
        kept = (badm & (out == sv)).sum().item()
        tiles = sorted({(r // tr, c // tc) for r, c in badm.nonzero().tolist()})
        print("SENTINEL %g: wrong %d, still == sentinel %d (%s); bad tiles %d of %d (row_tile, col_tile) e.g. %s"
              % (sv, badm.sum().item(), kept, "correct" if not badm.any().item() else
                 "never written" if kept == badm.sum().item() else
                 ("partly unwritten" if kept else "computed wrong"), len(tiles),
                 math.ceil(a.m / tr) * math.ceil(a.n / tc), tiles[:6]))
    print("PROBE_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("capsule")
    c.add_argument("capsule")
    c.add_argument("--call", type=int, default=None)
    c.add_argument("--repeat", type=int, default=20)
    c.add_argument("--synthesize", action="store_true", help="shapes-only capsule")
    c.add_argument("--out-dir", default=None, help="where the operands .pt goes (default $ND_ART/rootcause, else cwd)")
    for p in (sub.add_parser("sweep"), sub.add_parser("sentinel")):
        p.add_argument("--k", type=int, required=True)
        p.add_argument("--n", type=int, required=True)
        p.add_argument("--dtype", choices=sorted(DT), default="bf16")
        p.add_argument("--tile", default="128x112",
                       help="MT<m>x<n> from the kernel name (kernel.*.txt); the default is "
                            "the 2026-11-09 gfx950 kernel's -- pass this kernel's")
        if p.prog.endswith("sweep"):
            p.add_argument("--m-from", type=int, default=1)
            p.add_argument("--m-to", type=int, default=20000)
        else:
            p.add_argument("--m", type=int, required=True)
    for p in sub.choices.values():
        p.add_argument("--tol", type=float, default=5e-2, help="max |err| / max |ref| counted as wrong")
    a = ap.parse_args()
    {"capsule": capsule_mode, "sweep": sweep_mode, "sentinel": sentinel_mode}[a.mode](a)


if __name__ == "__main__":
    main()
