"""Twin adapters against the real kernels (T1) and the comparison against planted faults (T2a).

    python3 tests/twins/test_adapters.py <vllm|sglang|mla|inductor> [capsule.pt]    (in the matching image, on a GPU)

T1  each registry entry the image supports, on random inputs of several shapes and eps values: the twin
    agrees with the kernel within its floor (no false positive on a correct kernel).
T2a the kernel's output corrupted after the call -- rows scaled by 1.1, two rows swapped, a block zeroed, one
    constant written everywhere -- must exceed the floor; rows scaled by 1.003 (below the bf16 floor) must not.
Last line: TWIN_ADAPTERS=PASS|FAIL <n checks> <n failed>.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import torch  # noqa: E402

import twins  # noqa: E402

FAILS, N = [], [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def corruptions(t):
    """(label, should_flag, fn(tensor) -> corrupted copy) for an output tensor t (rows = dim 0 flattened)."""
    def rows(f):
        def g(x):
            y = x.clone()
            # rows = the leading dim, the rest flattened: an (8, 1024, 4, 1) kernel output has 8 distinct rows,
            # its last dim alone (1) would make every "row" one element
            v = y.view(y.shape[0], -1) if y.dim() > 1 and y.shape[0] > 1 else y.view(-1, y.shape[-1])
            f(v)
            return y
        return g
    out = [("scale rows x1.1", True, rows(lambda v: v[: max(1, v.shape[0] // 4)].mul_(1.1))),
           # rows 0 and 1 (adjacent, distinct whenever there are two); a one-row output cannot be swapped
           ("swap two rows", True, rows(lambda v: v.__setitem__([0, 1], v[[1, 0]].clone()))),
           ("zero a block", True, rows(lambda v: v[:, : max(1, v.shape[1] // 8)].zero_())),
           ("one constant everywhere", True, lambda x: torch.full_like(x, 0.25)),
           ("scale rows x1.003 (below the floor)", False, rows(lambda v: v.mul_(1.003)))]
    return out


def _rows_of(op_name, func, args, kwargs, pre, out):
    r = twins.check(op_name, func, args, kwargs, pre, out)
    g = next(iter(r["got"].values()))
    return g.shape[0] if g.dim() > 1 else 1


def judge(label, op_name, func, args, kwargs, pre, out, got_key=None, swap=None):
    """T1 on the call as made, then T2a on corrupted copies of the output named got_key."""
    r = twins.rel_err(op_name, func, args, kwargs, pre, out)
    check("T1 %s agrees" % label, r is not None and r[0] <= r[1],
          None if r is None else "rel err %.3g floor %.3g where %s source %s" % r[:4])
    if r is None or swap is None:
        return
    for c_label, flag, fn in corruptions(None):
        if c_label == "swap two rows" and _rows_of(op_name, func, args, kwargs, pre, out) < 2:
            continue
        a2, k2, out2 = swap(fn)
        r2 = twins.rel_err(op_name, func, a2, k2, pre, out2)
        hit = r2 is not None and r2[0] > r2[1]
        check("T2a %s: %s %s" % (label, c_label, "flagged" if flag else "not flagged"), hit == flag,
              None if r2 is None else "rel err %.3g floor %.3g" % (r2[0], r2[1]))


def vllm():
    import vllm._custom_ops  # noqa: F401  registers _C
    dev, dt = "cuda", torch.bfloat16
    for rows_, hid in ((7, 896), (128, 4096), (1, 1536)):
        for eps in (1e-6, 1e-5):
            x = torch.randn(rows_, hid, device=dev, dtype=dt)
            w = torch.randn(hid, device=dev, dtype=dt) * 0.1 + 1
            res = torch.empty_like(x)
            f = torch.ops._C.rms_norm.default
            f(res, x, w, eps)
            def swap(fn, res=res, x=x, w=w, eps=eps):
                return (fn(res), x, w, eps), {}, None
            judge("_C.rms_norm %dx%d eps %g" % (rows_, hid, eps), "_C.rms_norm.default", f, (res, x, w, eps), {},
                  None, None, swap=swap)
            # fused add: in place -- the pre-call copies come from `pre`, as the capture provides them
            xi, ri = x.clone(), torch.randn_like(x)
            pre = {"args": {0: xi.clone(), 1: ri.clone()}, "kwargs": {}}
            g = torch.ops._C.fused_add_rms_norm.default
            g(xi, ri, w, eps)
            def swap2(fn, xi=xi, ri=ri, w=w, eps=eps):
                return (fn(xi), ri, w, eps), {}, None
            judge("_C.fused_add_rms_norm %dx%d eps %g" % (rows_, hid, eps), "_C.fused_add_rms_norm.default", g,
                  (xi, ri, w, eps), {}, pre, None, swap=swap2)
        x2 = torch.randn(rows_, 2 * hid, device=dev, dtype=dt)
        o = torch.empty(rows_, hid, device=dev, dtype=dt)
        h = torch.ops._C.silu_and_mul.default
        h(o, x2)
        def swap3(fn, o=o, x2=x2):
            return (fn(o), x2), {}, None
        judge("_C.silu_and_mul %dx%d" % (rows_, hid), "_C.silu_and_mul.default", h, (o, x2), {}, None, None,
              swap=swap3)


def sglang():
    import aiter  # noqa: F401  registers torch.ops.aiter
    dev, dt = "cuda", torch.bfloat16
    for rows_, hid in ((7, 896), (128, 4096), (1, 1536)):
        for eps in (1e-6, 1e-5):
            x = torch.randn(rows_, hid, device=dev, dtype=dt)
            w = (torch.randn(hid, device=dev, dtype=dt) * 0.1 + 1).to(dt)
            f = torch.ops.aiter.rmsnorm2d_fwd.default
            out = f(x, w, eps, 0)
            def swap(fn, x=x, w=w, eps=eps, out=out):
                return (x, w, eps, 0), {}, fn(out)
            judge("aiter.rmsnorm2d_fwd %dx%d eps %g" % (rows_, hid, eps), "aiter.rmsnorm2d_fwd.default", f,
                  (x, w, eps, 0), {}, None, out, swap=swap)
            r = torch.randn_like(x)
            o, ro = torch.empty_like(x), torch.empty_like(x)
            g = torch.ops.aiter.rmsnorm2d_fwd_with_add.default
            g(o, x, r, ro, w, eps, False, 0)
            def swap2(fn, o=o, x=x, r=r, ro=ro, w=w, eps=eps):
                return (fn(o), x, r, ro, w, eps, False, 0), {}, None
            judge("aiter.rmsnorm2d_fwd_with_add %dx%d eps %g" % (rows_, hid, eps),
                  "aiter.rmsnorm2d_fwd_with_add.default", g, (o, x, r, ro, w, eps, False, 0), {}, None, None,
                  swap=swap2)


def mla(capsule):
    """The MLA example's capsule: its call replayed, judged by the Triton twin and by the hand-written value check;
    they must agree run by run (the kernel is wrong on some runs: a race)."""
    from capsule_operands import load
    import math_refs
    c = load(capsule, device="cuda")
    agree, hits_t, hits_v, runs = 0, 0, 0, 40
    for _ in range(runs):
        c.run()
        rt = twins.rel_err(c.op, c.func, c.args, c.kwargs)
        rv = math_refs.value_rel_err(c.op, c.func, c.args, c.kwargs)
        if rt is None or rv is None:
            check("T1 mla: both checks apply", False, "twin %r value %r" % (rt, rv))
            return
        ht, hv = rt[0] > rt[1], rv[0] > rv[1]
        hits_t += ht
        hits_v += hv
        agree += ht == hv
    print("mla: twin hits %d/%d, value-check hits %d/%d, agree on %d" % (hits_t, runs, hits_v, runs, agree))
    check("T1 mla: twin and hand-written check agree on every run", agree == runs)
    check("T2a mla: the race is seen by the twin (the capsule is a failing call)", hits_t > 0)


def inductor(capsule):
    """An Inductor op capsule (torchcompile example): finite outputs agree with the fp64 twin (T1), and corrupted
    copies are flagged (T2a)."""
    from capsule_operands import load
    c = load(capsule, device="cuda")
    c.run()
    written = [i for i in c.raw["triton_kernels"][c.op]["written"]]
    i = written[0]
    def swap(fn):
        a = list(c.args)
        a[i] = fn(a[i])
        return tuple(a), c.kwargs, None
    judge("inductor %s" % c.op.split(".")[1][:40], c.op, c.func, c.args, c.kwargs, None, None, swap=swap)


if __name__ == "__main__":
    which = sys.argv[1]
    {"vllm": vllm, "sglang": sglang}.get(which, lambda: None)()
    if which == "mla":
        mla(sys.argv[2])
    if which == "inductor":
        inductor(sys.argv[2])
    print("TWIN_ADAPTERS=%s %d %d" % ("FAIL" if FAILS else "PASS", N[0], len(FAILS)))
