"""A capsule's operands as tensors, for step-7 experiments -- and a row-level view of what the failing op was fed.

A capsule stores every tensor flat (`args[i]["base"]`, `pool[<key>]["base"]`) with its size / stride / offset
beside it, and the chain's operands as references into the pool: an ad-hoc torch.load sees a (229376,) buffer
where the op saw (256, 896), and the embedding table only as a pool key (2026-11-20: the first step-7 script of
the SGLang investigation found neither). This rebuilds them as the op saw them.

Library (in the model's runtime, e.g. a step-7 experiment):

    from capsule_operands import load
    c = load("<capsule>.pt", device="cuda")
    c.op, c.func                        # 'aiter.rmsnorm2d_fwd.default', its OpOverload (None if unresolved)
    c.args, c.kwargs                    # the failing op's operands, tensors as the op saw them (recorded values)
    c.arg("epsilon"), c.arg("input")    # by schema name
    c.out                               # what the op returned / wrote, as recorded: shape and non-finite counts
    for n in c.chain:                   # the ops that fed it, oldest first: n.op, n.args, n.kwargs, n.call_index
        ...
    c.run()                             # the op once on its recorded operands (no env restore, no checks)

Command line -- one line per operand, and per 2-D float operand the rows that are all zero or non-finite, and
the integer operands' values at those rows (which token ids fed a zero row, say):

    nd run --gpus 0 -- 'python3 -m capsule_operands $ND_ART/capsules/t0/<capsule>.pt'

A shapes-only capsule (`data: shapes`, the capture default) has no values: operands are drawn from the
recorded statistics, as `replay_nan_op.py --synthesize` does, and the row view says so.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import op_nan_capture as onc  # noqa: E402


def _resolve(name):
    if name.startswith(("triton.", "inductor.")):
        # A Triton kernel launch: rebuilt from the capsule's triton_kernels descriptor,
        # registered by upgrade_capsule when the capsule was loaded.
        try:
            return onc.triton_op_for(name)
        except Exception:                               # noqa: BLE001
            return None
    ns, opname, ov = name.split(".")
    if ns != "aten":
        for mod in {"_C": ("vllm._custom_ops",)}.get(ns, (ns,)):
            try:
                __import__(mod)
            except ImportError:
                pass
    try:
        return getattr(getattr(getattr(torch.ops, ns), opname), ov)
    except (AttributeError, RuntimeError):
        return None


class Node:
    """One op of the capsule's chain, its operands bound."""

    def __init__(self, rec, args, kwargs):
        self.op, self.call_index = rec["op"], rec.get("call_index")
        self.args, self.kwargs = args, kwargs


class Capsule:
    def __init__(self, path, device="cpu", synth_seed=0):
        self.path = path
        self.raw = onc.upgrade_capsule(torch.load(path, map_location="cpu", weights_only=False))
        r = self.raw
        self.device = device
        self.shapes_only = r.get("data", "values") != "values"
        if self.shapes_only and not onc.synthesizing():
            onc.synthesize_operands(synth_seed)  # draw the float operands from the recorded statistics
        self.op = r["op"]
        self.func = _resolve(self.op)
        self.args = tuple(onc.unpack(r["args"], device))
        self.kwargs = dict(onc.unpack(r["kwargs"], device))
        self.out = {"size": r.get("out_size"), "nan": r.get("out_nan"), "inf": r.get("out_inf"),
                    "found_in": r.get("nan_found_in")}
        self.chain = self._bind_chain()

    def _bind_chain(self):
        r, env, nodes = self.raw, {}, []
        pool = r.get("pool") or {}

        def bind(o):
            if isinstance(o, dict) and onc._CHAIN_REF in o:
                k = o[onc._CHAIN_REF]
                if k not in env and k in pool:
                    env[k] = onc.unpack(pool[k], self.device)
                return env.get(k)
            if isinstance(o, dict) and (o.get(onc._TENSOR_TAG) or onc._MEMFMT_TAG in o):
                return onc.unpack(o, self.device)
            if isinstance(o, tuple):
                return tuple(bind(x) for x in o)
            if isinstance(o, list):
                return [bind(x) for x in o]
            if isinstance(o, dict):
                return dict((k, bind(v)) for k, v in o.items())
            return o

        for rec in r.get("chain") or []:
            nodes.append(Node(rec, bind(rec["args"]), bind(rec["kwargs"])))   # an operand no root or kept
                                                                                 # intermediate holds binds None
        return nodes

    def arg(self, name, default=None):
        """The failing op's operand `name`, by its schema."""
        if self.func is None:
            raise LookupError("%s did not resolve: positional args only" % self.op)
        names = [a.name for a in self.func._schema.arguments]
        if name not in names:
            return default
        i = names.index(name)
        return self.args[i] if i < len(self.args) else self.kwargs.get(name, default)

    def names(self):
        if self.func is None:
            return ["arg%d" % i for i in range(len(self.args))]
        names = [a.name for a in self.func._schema.arguments]
        return [names[i] if i < len(names) else "arg%d" % i for i in range(len(self.args))]

    def run(self):
        if self.func is None:
            raise LookupError("%s did not resolve in this process" % self.op)
        return self.func(*self.args, **self.kwargs)


def load(path, device="cpu", synth_seed=0):
    return Capsule(path, device, synth_seed)


def _describe(name, v, ints):
    if not isinstance(v, torch.Tensor):
        return "  %-14s %r" % (name, v)
    s = "  %-14s %s %s" % (name, tuple(v.shape), str(v.dtype).replace("torch.", ""))
    if not v.is_floating_point():
        if v.numel() <= 16:
            s += " = %s" % v.flatten().tolist()
        else:
            s += " min %d max %d" % (v.min().item(), v.max().item())
        return s
    f = v.float()
    fin = torch.isfinite(f)
    s += " nonfinite %d/%d" % ((~fin).sum().item(), f.numel())
    if fin.any():
        s += " finite |x| max %.3g" % f[fin].abs().max().item()
    if v.dim() == 2 and v.shape[0] > 1:
        zero = (f == 0).all(-1).nonzero().flatten().tolist()
        bad = (~fin).any(-1).nonzero().flatten().tolist()
        s += "\n  %-14s rows all zero: %s; rows non-finite: %s" % ("", _short(zero), _short(bad))
        for iname, it in ints:
            if it.dim() == 1 and it.numel() == v.shape[0] and (zero or bad):
                rows = sorted(set(zero + bad))[:8]
                s += "\n  %-14s %s at those rows: %s" % ("", iname, dict((r, it[r].item()) for r in rows))
    return s


def row_report(c):
    """Lines naming, for each 2-D float operand of the failing op, its all-zero and non-finite rows and the
    chain's integer operands (token ids, indices) at those rows -- usually the cause of a model NaN in one
    line (2026-11-20: `input rows all zero: [46] <- embedding indices 151819; epsilon 0.0`)."""
    if c.shapes_only:
        return ["(shapes only: no values to read)"]
    ints = [("%s %s" % (n.op.split(".")[1], "indices" if n.op == "aten.embedding.default" and i == 1 else "arg%d" % i), x)
            for n in c.chain for i, x in enumerate(n.args)
            if isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dim() == 1]
    lines, scalars = [], []
    for name, v in list(zip(c.names(), c.args)) + list(c.kwargs.items()):
        if isinstance(v, float) and ("eps" in name or "epsilon" in name):
            scalars.append("%s %g" % (name, v))
        if not (isinstance(v, torch.Tensor) and v.is_floating_point() and v.dim() == 2 and v.shape[0] > 1):
            continue
        f = v.float()
        zero = (f == 0).all(-1).nonzero().flatten().tolist()
        bad = (~torch.isfinite(f)).any(-1).nonzero().flatten().tolist()
        if not zero and not bad:
            continue
        s = "%s rows all zero: %s, non-finite: %s" % (name, _short(zero), _short(bad))
        rows = sorted(set(zero + bad))[:8]
        for iname, it in ints:
            if it.numel() == v.shape[0]:
                s += " <- %s %s" % (iname, dict((r, it[r].item()) for r in rows))
        lines.append(s)
    if lines and scalars:
        lines[-1] += "; " + ", ".join(scalars)
    return lines


def _short(xs, n=12):
    return "%s%s" % (xs[:n], " ... (%d)" % len(xs) if len(xs) > n else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capsule")
    a = ap.parse_args()
    c = load(a.capsule)
    print("op %s%s  out %s nan=%s inf=%s in %s" % (c.op, "" if c.func else " (unresolved here)", c.out["size"],
                                                   c.out["nan"], c.out["inf"], c.out["found_in"]))
    if c.shapes_only:
        print("  (shapes-only capsule: no recorded values -- rows below are not the model's)")
    nodes = c.chain or []
    ints = []
    for n in nodes:   # integer operands anywhere in the chain (token ids, indices): name them by op
        for i, x in enumerate(n.args):
            if isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dim() == 1:
                ints.append(("%s arg%d" % (n.op.split(".")[1], i), x))
    for name, v in zip(c.names(), c.args):
        print(_describe(name, v, ints))
    for k, v in c.kwargs.items():
        print(_describe(k, v, ints))
    if nodes:
        print("chain: %s" % " -> ".join(n.op for n in nodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
