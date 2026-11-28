#!/usr/bin/env python3
"""Replay one Inductor kernel from a kernel capsule, without the model (debug-numeric 5N compiled track).

    replay_inductor_kernel.py <inductor_*.pt> [--iters N] [--info] [--dump-source FILE]

The capsule (scripts/inductor_capsule/sitecustomize.py) holds the Triton kernel that Inductor's nan_asserts caught,
its launch line and every launch argument.  This recompiles the kernel from its source with Inductor's own
AsyncCompile (same torch/triton as the capture: run it in the model's image), rebuilds the inputs from their saved
values (strides kept), allocates fresh zeroed outputs, launches it --iters times and counts launches whose checked
output (the buffer the assert failed on) holds a NaN/Inf.  Last line:

    INDUCTOR_REPLAY_RESULT=<hits>/<iters> kernel=<name> checked=<buffer> nonfinite=<n> matches_model=<bool|n/a>

`matches_model`: every output of the last launch is bit-identical to the model's (the capsule's `expected`; n/a in
a capsule from before 2026-11-20, which stored the non-finite count only -- then `same_positions` compares counts).
A sparse input (`sparse_rows`: only the rows its integer inputs index, verified at capture) is rebuilt full-size
with those rows in place and NaN elsewhere -- a read of any other row would show.  --info prints the capsule
without a GPU.
"""
import argparse
import re
import sys

import torch


def describe(c):
    print("kernel   %s  (assert: %s)" % (c["kernel"], c["assert_line"]))
    print("launch   %s" % c["launch_line"])
    print("built    torch %s triton %s hip %s on %s %s" % (c["torch"], c["triton"], c["hip"], c["device"], c["arch"]))
    for a in c["args"]:
        if a.get("tensor"):
            print("arg      %-10s %-12s %-6s %s %s stride=%s%s%s" % (
                a["name"], a["role"], a["kind"], str(a["dtype"]).replace("torch.", ""), a["size"], a["stride"],
                "" if "nonfinite" not in a else " nonfinite=%d" % a["nonfinite"],
                " sparse: %d of %d rows" % (len(a["sparse_rows"]), a["size"][0]) if "sparse_rows" in a else
                "" if "data" in a or a["kind"] == "out" else " (values not saved: cap)"))
        else:
            print("arg      %-10s %-12s %-6s %s" % (a["name"], a["role"], a["kind"], a.get("value", a.get("error"))))
    for l in (c.get("kernel_source") or "").splitlines():   # Inductor's own note of what the kernel fuses
        if "Source Nodes" in l or "Original ATen" in l:
            print("source   %s" % l.strip().lstrip("#").strip())
    for f in c.get("frames", [])[:6]:
        print("frame    %s" % f)


def build_args(c, device="cuda"):
    """The launch arguments as tensors / ints: ({name: value}, {output name: tensor}).  Inputs from their saved
    values (a sparse input rebuilt full-size: its saved rows in place, NaN elsewhere), outputs fresh zeros.
    Import it for a reference computation on the capsule's inputs:
        from replay_inductor_kernel import build_args, by_role
        c = torch.load(p, weights_only=False); ns, _ = build_args(c, "cpu"); x = by_role(c, ns)["in"][0]
    The keys are the wrapper's buffer names (buf20), not in_ptr0: by_role() maps them."""
    ns, outs = {}, {}
    for x in c["args"]:
        if x.get("tensor"):
            if "sparse_rows" in x:
                t = torch.full(x["size"], float("nan"), dtype=x["dtype"], device=device)
                t[x["sparse_rows"].to(device)] = x["row_data"].to(device)
            elif x["kind"] == "out" or "data" not in x:
                t = torch.empty_strided(x["size"], x["stride"], dtype=x["dtype"], device=device).zero_()
                outs[x["name"]] = t
            else:
                t = torch.as_strided(x["data"].to(device), x["size"], x["stride"], 0)
            ns[x["name"]] = t
        elif "value" in x:
            ns[x["name"]] = x["value"]
    return ns, outs


def by_role(c, ns):
    """{"in": [...], "out": [...], "in_out": [...]} -- the launch's tensors by role, in launch order.  The names
    in `ns` are the WRAPPER's buffer names (buf20, buf25, primals_3), not the kernel's parameter names
    (in_ptr0, out_ptr0): look them up by role, not by in_ptr* (2026-11-23, a step-7 script's StopIteration)."""
    r = {"in": [], "out": [], "in_out": []}
    for x in c["args"]:
        if x.get("tensor") and x["name"] in ns:
            r.setdefault(x.get("kind") or "in", []).append(ns[x["name"]])
    return r


def build_kernel(c):
    from torch._inductor.async_compile import AsyncCompile
    k = AsyncCompile().triton(c["kernel"], c["kernel_source"], device_str="cuda")
    return k.result() if hasattr(k, "result") else k


def fp64_reference(c, call, ns, checked):
    """The checked output's status in the kernel's fp64 twin (inductor_ref.py) on the capsule's inputs:
    `finite` (a non-finite output is never correct here: the kernel), `NON-FINITE` (the computation itself
    is: NATIVE_MATH), or `unavailable:<why>` -- never read as either."""
    try:
        import inductor_ref
        ns_in, _ = build_args(c, "cuda")                # fresh inputs: an in_out argument is post-call in ns
        kinds = {x["name"]: x.get("kind") for x in c["args"]}
        if any(k == "in_out" for k in kinds.values()):
            return "unavailable:in_out_argument"
        names = re.findall(r"(\w+)\s*(?:,|$)", re.sub(r",?\s*stream=\w+", "", call))
        desc = {"kernel": c["kernel"], "src": c["kernel_source"],
                "written": [i for i, n in enumerate(names) if kinds.get(n) == "out"],
                "outbuf": [i for i, n in enumerate(names) if kinds.get(n) == "out"]}
        vals = [ns_in[n] if n in ns_in else eval(n, {}, ns) for n in names]
        outs = dict(zip((names[i] for i in desc["written"]), inductor_ref.reference(desc, vals)))
        if checked not in outs:
            return "unavailable:checked_buffer_not_written"
        return "finite" if bool(torch.isfinite(outs[checked]).all()) else "NON-FINITE"
    except Exception as e:                              # noqa: BLE001 - a reference never fails the replay
        return "unavailable:%s" % type(e).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsule")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--info", action="store_true", help="describe the capsule only (no GPU)")
    ap.add_argument("--dump-source", help="write the kernel's Triton source to this file")
    a = ap.parse_args()
    c = torch.load(a.capsule, map_location="cpu", weights_only=False)
    if c.get("kind") != "inductor_kernel":
        sys.exit("not an Inductor kernel capsule: %s" % a.capsule)
    describe(c)
    if a.dump_source:
        open(a.dump_source, "w").write(c["kernel_source"] or "")
        print("source   -> %s" % a.dump_source)
    if a.info:
        return 0
    if not c["kernel_source"]:
        sys.exit("the capsule has no kernel source (the wrapper format changed): cannot replay")
    if torch.__version__ != c["torch"]:
        print("!! torch %s here, %s at capture: replay in the model's image" % (torch.__version__, c["torch"]))
    kern = build_kernel(c)
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream

    ns, outs = build_args(c, "cuda")
    call = re.match(r"\s*\w+\.run\((.*)\)\s*$", c["launch_line"]).group(1)
    ns["__k"], ns["__stream"] = kern, get_raw_stream(torch.cuda.current_device())
    call = re.sub(r"stream=\w+", "stream=__stream", call)
    checked = c["checked_buffer"]
    want = next((x.get("nonfinite") for x in c["args"] if x["name"] == checked), None)
    hits, last = 0, 0
    for _ in range(a.iters):
        for t in outs.values():
            t.zero_()
        eval("__k.run(%s)" % call, {}, ns)
        torch.cuda.synchronize()
        if checked in ns:
            last = int((~torch.isfinite(ns[checked])).sum())
            hits += last > 0
    exp = c.get("expected") or {}
    if exp:
        # flat first: a size-1 dim with a large stride, e.g. (8,1024,4,1) stride (4096,4,1,32768),
        # counts as contiguous, so .contiguous() keeps it and the byte view raised "stride(-1)
        # must be 1" (2026-11-20, torchcompile 5N)
        bits = lambda t: t.contiguous().reshape(-1).view(torch.uint8)
        match = all(torch.equal(bits(ns[n].cpu()), bits(torch.as_strided(e, ns[n].size(), ns[n].stride(), 0)))
                    for n, e in exp.items() if n in ns)
        verdict = "matches_model=%s" % match
    else:
        verdict = "matches_model=n/a same_positions=%s" % (want is not None and last == want)
    print("INDUCTOR_REPLAY_RESULT=%d/%d kernel=%s checked=%s nonfinite=%d %s fp64_ref=%s"
          % (hits, a.iters, c["kernel"], checked, last, verdict, fp64_reference(c, call, ns, checked)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
