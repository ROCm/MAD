#!/usr/bin/env python3
"""Replay one graph piece saved by the vLLM graph check, without the model (a NaN only with graphs on).

    replay_graph_piece.py <graph_piece_*.pt> [--iters N] [--compile] [--import MOD ...] [--info]

The capsule (scripts/vllm_graphs/sitecustomize.py, ND_GRAPH_CHECK=1) holds the piece's FX code (when vLLM kept
the module: piecewise pieces; executed here as a plain function), the inputs of the replay that went non-finite, the graph's and the eager run's
outputs, and the batch descriptor.  This runs the module on those inputs two ways, --iters times each:

  eager   the module called directly
  graph   the module captured into a fresh torch.cuda.CUDAGraph (after a warm-up on a side stream), replayed

--compile runs torch.compile(the function) instead of the module (Inductor's kernels, closer to vLLM's compiled piece;
not vLLM's exact compile options).  The module's custom ops need their library imported: --import (default
vllm; repeatable).  Last line:

    GRAPH_PIECE_REPLAY=<GRAPH_ONLY|BOTH|NEITHER|EAGER_ONLY> eager=<hits>/<n> graph=<hits>/<n> piece=<i>
                       batch=<descriptor> compiled=<bool>

GRAPH_ONLY reproduces the model's finding without it.  NEITHER: the trigger is not in the piece alone -- the
shared graph pool, another piece's graph, vLLM's own capture (its pool, its warm-up) -- keep the model-level
A/B (graphs on vs ND_VLLM_CUDAGRAPH_MODE=NONE) as the reproducer.  --info describes the capsule without a GPU.
"""
import argparse
import importlib
import sys

import torch


def tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for x in obj for t in tensors(x)]
    if isinstance(obj, dict):
        return [t for x in obj.values() for t in tensors(x)]
    return []


def nonfinite(obj):
    return any(t.is_floating_point() and not bool(torch.isfinite(t.float()).all()) for t in tensors(obj))


def to(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return dict((k, to(v, dev)) for k, v in obj.items())
    return obj


def _by_path(rep):
    """The object a repr names by its dotted path, imported; None when it names none."""
    import re
    m = re.match(r"<(?:class|enum) '([\w.]+)'>|<function ([\w.]+) at |<module '([\w.]+)'", rep or "")
    path = m and next(g for g in m.groups() if g)
    if not path:
        return None
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:                               # noqa: BLE001
            continue
        try:
            for p in parts[i:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None
    return None


def build_forward(c):
    """The piece's forward(self, ...) from its FX code, in a namespace rebuilt from the names it read."""
    import builtins
    import math
    import operator
    ns = {"torch": torch, "operator": operator, "_operator": operator, "math": math, "builtins": builtins,
          "inf": float("inf"), "nan": float("nan"), "NoneType": type(None), "device": torch.device,
          "fx_pytree": torch.fx._pytree, "pytree": torch.utils._pytree}
    missing = [k for k in (c.get("fx_globals") or {}) if k not in ns]
    for k in missing:
        v = getattr(torch, k, None)         # torch.<name> spelled as a bare global
        if v is None:                       # "<class 'a.b.C'>" / "<function a.b.f at ..>" / "<module 'a.b'"
            v = _by_path(c["fx_globals"][k])
        if v is not None:
            ns[k] = v
    still = [k for k in missing if k not in ns]
    if still:
        print("note     FX globals not rebuilt (unused, or the replay fails on them): %s"
              % ", ".join("%s=%s" % (k, c["fx_globals"][k]) for k in still))
    exec(compile(c["fx_code"], "<graph piece %s>" % c["piece"], "exec"), ns)
    return ns["forward"]


def describe(c):
    print("piece    %s  mode=%s  batch=%s  runnable=%s" % (c["piece"], c["runtime_mode"], c["batch_descriptor"],
                                                           c.get("runnable")))
    print("model    verdict=%s  non-finite rows %s of %s  (graph output; eager output %s)"
          % (c["verdict"], c["bad_rows"][:16], c["rows"],
             "non-finite" if nonfinite(c["eager_out"]) else "finite"))
    for i, a in enumerate(c["args"]):
        if isinstance(a, torch.Tensor):
            print("arg %-4d %s %s%s" % (i, str(a.dtype).replace("torch.", ""), tuple(a.shape),
                                        "  NON-FINITE" if a.is_floating_point() and not torch.isfinite(a.float()).all()
                                        else ""))
        else:
            print("arg %-4d %r" % (i, a))
    print("built    torch %s vllm %s on %s" % (c.get("torch"), c.get("vllm"), c.get("device")))
    if not c.get("fx_code"):
        print("module   none saved (vLLM kept no FX module for this runnable: a full graph)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsule")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--import", dest="imports", action="append", default=None)
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args()
    for m in a.imports or ["vllm"]:
        try:
            importlib.import_module(m)
        except Exception as e:                          # noqa: BLE001
            print("!! import %s failed: %s" % (m, e))
    c = torch.load(a.capsule, map_location="cpu", weights_only=False)
    if c.get("kind") != "graph_piece":
        sys.exit("not a graph-piece capsule: %s" % a.capsule)
    describe(c)
    if a.info:
        print(c.get("fx_code") or "")
        return 0
    if not c.get("fx_code"):
        sys.exit("no FX code in the capsule: cannot replay the piece alone (keep the model-level A/B)")
    if c.get("fx_self_attrs"):
        sys.exit("the piece's code reads module attributes %s, which the capsule does not hold"
                 % c["fx_self_attrs"])
    fwd = build_forward(c)
    dev = "cuda"
    fn = (lambda *x, **k: fwd(None, *x, **k))
    fn = torch.compile(fn) if a.compile else fn
    args, kwargs = to(c["args"], dev), to(c["kwargs"], dev)
    with torch.no_grad():
        eager = sum(nonfinite(fn(*args, **kwargs)) for _ in range(a.iters))
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(*args, **kwargs)
        graph = 0
        for _ in range(a.iters):
            g.replay()
            torch.cuda.synchronize()
            graph += nonfinite(out)
    verdict = ("GRAPH_ONLY" if graph and not eager else "BOTH" if graph and eager
               else "EAGER_ONLY" if eager else "NEITHER")
    print("GRAPH_PIECE_REPLAY=%s eager=%d/%d graph=%d/%d piece=%s batch=%s compiled=%s"
          % (verdict, eager, a.iters, graph, a.iters, c["piece"], c["batch_descriptor"], a.compile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
