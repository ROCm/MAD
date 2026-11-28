#!/usr/bin/env python3
"""debug-numeric step 8: the value and cost check of a library-overlay fix -- the capsule's
GEMM run with the stock library and with the overlay, each in a fresh process under
the capture's restored environment, compared against an fp64 CPU reference.

HIPBLASLT_TENSILE_LIBPATH is read when hipBLASLt initialises, so each mode is its own
child process. The overlay variables are set AFTER the capsule's environment is
restored: restore_env_dict() unsets HIPBLAS*/TENSILE_* names the capture did not have,
which is exactly the trap that makes a --restore-env replay silently run the stock
kernel. The child reports the libpath it actually saw.

Solo, so the stock run normally stays finite (a contention-gated NaN needs load): this
checks values and cost; the NaN rate under load is the sweep harness's job
(step8-fix.md).

    python3 fix_values.py <capsule.pt> --overlay <dir> [--iters 2000] [--json out]
                          [--rtol 1e-2] [--synthesize]

--synthesize: a shapes-only capsule (the capture default) has no float values; draw them
from its statistics (seed 0, as replay_nan_op.py --synthesize), the same operands in
both children and the fp64 reference.
    -> VALUES=OK|FAIL and a JSON report (finite, max_rel_err_vs_fp64 per mode,
       patched_bitwise_equal_stock, ms_per_launch, time_ratio_patched_over_stock_1run (one run
       per arm: not a cost measurement), libpath)
"""
import argparse
import json
import os
import subprocess
import sys

USER = os.environ.get("NUMERIC_DOCTOR_USER",
                      os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, USER)


def load(capsule, dev, synthesize=False):
    import torch
    from op_nan_capture import unpack, upgrade_capsule, synthesize_operands
    if synthesize:
        synthesize_operands(0)
    cap = upgrade_capsule(torch.load(capsule, map_location="cpu", weights_only=False))
    prec = cap["precision"]
    torch.backends.cuda.matmul.allow_tf32 = prec["allow_tf32"]
    torch.set_float32_matmul_precision(prec["float32_matmul_precision"])
    a = unpack(cap["args"], dev)
    kw = unpack(cap["kwargs"], dev)
    ns, name, overload = cap["op"].split(".")
    return cap, getattr(getattr(getattr(torch.ops, ns), name), overload), a, kw


def child(args):
    import torch
    _cap, op, a, kw = load(args.capsule, "cuda", args.synthesize)
    for _ in range(10):
        out = op(*a, **kw)
    torch.cuda.synchronize()
    bad = 0
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        out = op(*a, **kw)
        bad += int(not torch.isfinite(out).all())
    stop.record()
    torch.cuda.synchronize()
    torch.save(out.cpu(), args.child_out)
    print(json.dumps({"libpath": os.environ.get("HIPBLASLT_TENSILE_LIBPATH"),
                      "ms_per_launch": start.elapsed_time(stop) / args.iters,
                      "nonfinite_launches": bad, "iters": args.iters}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsule")
    ap.add_argument("--overlay", required=False)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rtol", type=float, default=1e-2,
                    help="max |patched - fp64| / max|fp64| to pass (default 1e-2: xf32)")
    ap.add_argument("--json")
    ap.add_argument("--synthesize", action="store_true",
                    help="shapes-only capsule: draw float operands from its statistics (seed 0)")
    ap.add_argument("--child-out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child_out:
        return child(args)
    if not args.overlay:
        sys.exit("--overlay is required")

    import torch
    import replay_nan_op as rg
    cap = torch.load(args.capsule, map_location="cpu", weights_only=False)
    base = rg.restore_env_dict(cap)
    for k in ("HIPBLASLT_TENSILE_LIBPATH", "TENSILE_LIBPATH"):
        base.pop(k, None)
    work = os.path.dirname(os.path.abspath(args.json or args.capsule))
    res = {}
    for mode in ("stock", "patched"):
        env = dict(base)
        if mode == "patched":
            import kernel_patch as kp      # per-arch layout: the arch dir, not the root
            env["HIPBLASLT_TENSILE_LIBPATH"] = env["TENSILE_LIBPATH"] = kp.tensile_libpath(args.overlay)
        out_pt = os.path.join(work, "out_%s.pt" % mode)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), args.capsule,
                            "--iters", str(args.iters), "--child-out", out_pt]
                           + (["--synthesize"] if args.synthesize else []),
                           env=env, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
        if p.returncode or line is None:
            sys.exit("%s child failed rc=%d\n%s" % (mode, p.returncode, p.stderr[-3000:]))
        res[mode] = json.loads(line)
        res[mode]["out"] = torch.load(out_pt)

    _c, op, a, kw = load(args.capsule, "cpu", args.synthesize)
    dbl = lambda x: x.double() if torch.is_tensor(x) and x.is_floating_point() else x  # noqa: E731
    ref = op(*[dbl(x) for x in a], **{k: dbl(v) for k, v in kw.items()})
    scale = ref.abs().max().item() or 1.0
    report = {"fp64_max_abs": scale}
    outs = {}
    for mode in ("stock", "patched"):
        o = res[mode].pop("out").double()
        outs[mode] = o
        res[mode]["finite"] = bool(torch.isfinite(o).all())
        res[mode]["max_rel_err_vs_fp64"] = ((o - ref).abs().max().item() / scale
                                            if res[mode]["finite"] else None)
    report.update(res)
    report["patched_bitwise_equal_stock"] = torch.equal(outs["stock"], outs["patched"])
    # one timed run per arm, run back to back: a ratio, not a cost -- 0.67 ("faster") read as a
    # measurement on 2026-11-20; quote a cost from a repeated A/B, not from this
    report["time_ratio_patched_over_stock_1run"] = (res["patched"]["ms_per_launch"]
                                                    / res["stock"]["ms_per_launch"])
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    ok = (res["patched"]["libpath"] == args.overlay and res["patched"]["finite"]
          and res["patched"]["max_rel_err_vs_fp64"] < args.rtol)
    print("VALUES=%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
