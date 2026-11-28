#!/usr/bin/env python3
"""What a capsule says, in ~20 lines, without a replay (debug-numeric steps 4.5 / 5a).

    capsule_info.py <capsule.pt> [...] [--frames N]

Per capsule: op, call index, output (shape, dtype, nan/inf, where), inputs_finite
and which operands were dirty, the chain's ops and why it stopped, nan_origin
and settle (settle-mode capsules), and the last N MODEL frames of the stack
(torch, stdlib and numeric-doctor frames dropped -- in backward the stack is the
autograd engine's, so the op's autograd node is the location instead).

For a pointwise op it also re-runs the op on CPU from the capsule's operands and
in fp64: "math" there (fp64 non-finite at the same positions: 0/0, sqrt(-1))
means the arithmetic is undefined, not a kernel fault -- the usual false
positive (e.g. torch's own sinc_backward at x == 0).  Any other op: replay it on
the GPU with replay_nan_op.py.  Runs on CPU; needs torch -- a login node without it:
`apptainer exec <the model's .sif> python3 capsule_info.py ...` (CPU only, no job).

`args` lists the op's operands (dtype, shape, stride) and the values of small integer
operands (group offsets, cu_seqlens: a grouped GEMM's per-group sizes are there).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import op_nan_capture as g  # noqa: E402


# A custom op's namespace -> the module whose import registers it (vLLM registers its ops on
# import; without it the CPU re-run below reported "not registered" for every vllm.* capsule).
_OP_LIBS = {"vllm": ("vllm", "vllm.attention.layer", "vllm.model_executor.layers.attention"),
            "_C": ("vllm._C",), "_rocm_C": ("vllm._rocm_C",), "aiter": ("aiter",),
            "fbgemm": ("fbgemm_gpu",)}


def resolve(name):
    ns, op, overload = name.split(".")
    try:
        return getattr(getattr(getattr(torch.ops, ns), op), overload)
    except (AttributeError, RuntimeError):
        import importlib
        for mod in _OP_LIBS.get(ns, (ns,)):
            try:
                importlib.import_module(mod)
            except Exception:                           # noqa: BLE001 -- best effort
                continue
            try:
                return getattr(getattr(getattr(torch.ops, ns), op), overload)
            except (AttributeError, RuntimeError):
                pass
        raise


def model_frames(stack, n):
    keep = []
    for line in stack.splitlines():
        line = line.strip()
        if not line.startswith('File "'):
            continue
        path = line.split('"')[1]
        if any(s in path for s in ("/torch/", "/python3.", "numeric-doctor/scripts", "<frozen")):
            continue
        keep.append(line.replace('File "', "").replace('", line ', ":").replace(", in ", "  "))
    return keep[-n:]


FP32_MAX = 3.4028234663852886e38
REDUCTIONS = ("_foreach_norm", "norm", "linalg_vector_norm", "frobenius_norm", "sum", "mean",
              "nansum", "_local_scalar_dense", "var", "std", "dot", "vdot")


def _float_stats(x, out):
    """Every recorded float tensor's stats under x (args may nest: a foreach op's list)."""
    if isinstance(x, dict):
        st = x.get("stats")
        if isinstance(st, dict) and st.get("kind") == "float":
            out.append(st)
        else:
            for v in x.values():
                _float_stats(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _float_stats(v, out)
    return out


def overflow_line(c):
    """A reduction that returned inf (no NaN) from finite inputs large enough to overflow fp32
    is a correct op: the cause is the exploding values upstream, not this kernel."""
    name = c["op"].split(".")[1] if c["op"].count(".") >= 2 else c["op"]
    if name not in REDUCTIONS or not c.get("inputs_finite") or c.get("out_nan") or not c.get("out_inf"):
        return None
    stats = _float_stats(c.get("args"), [])
    if not stats:
        return None
    mx = max(max(abs(s.get("min") or 0.0), abs(s.get("max") or 0.0)) for s in stats)
    numel = max(s.get("numel") or 1 for s in stats)
    norm = "norm" in name or name in ("var", "std", "dot", "vdot")
    certain = mx * mx > FP32_MAX if norm else mx > FP32_MAX
    possible = (mx * mx * numel if norm else mx * numel) > FP32_MAX
    if not possible:
        return None
    return ("overflow %s: finite inputs up to %.3g (%d tensor(s), up to %d elements) overflow fp32 "
            "in %s -- a correct op; the cause is upstream (exploding values). Log per-step "
            "gradient norms: scripts/grad_trajectory.py"
            % ("expected" if certain else "likely", mx, len(stats), numel, name))


INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool)


def describe_arg(x, pool, nvals=64):
    """One operand as text: 'dtype (shape) stride=(..)', plus the values of an integer
    tensor of <= nvals elements; chain references are looked up in the pool."""
    if isinstance(x, dict) and g._CHAIN_REF in x:
        x = (pool or {}).get(x[g._CHAIN_REF], x)
        if g._CHAIN_REF in x:
            return "chain tensor %s" % x[g._CHAIN_REF]
    if isinstance(x, dict) and x.get(g._TENSOR_TAG):
        s = "%s %s stride=%s" % (str(x["dtype"]).replace("torch.", ""), tuple(x["size"]), tuple(x["stride"]))
        n = 1
        for d in x["size"]:
            n *= d
        if x["dtype"] in INT_DTYPES and x.get("base") is not None and n <= nvals:
            s += " = %s" % g.unpack(x, "cpu").flatten().tolist()
        return s
    if isinstance(x, (list, tuple)):
        return "[%s]" % ", ".join(describe_arg(v, pool, nvals) for v in x)
    return repr(x)[:40]


def info(path, nframes):
    c = g.upgrade_capsule(torch.load(path, map_location="cpu", weights_only=False))
    p = print
    p("== %s" % path)
    p("op       %s  call #%s" % (c["op"], c.get("op_call_index")))
    p("data     %s" % c.get("data", "values"))
    p("out      %s %s  nan=%s inf=%s of %s  found in %s"
      % (tuple(c["out_size"]), str(c["out_dtype"]).replace("torch.", ""), c["out_nan"],
         c["out_inf"], c["out_numel"], c.get("nan_found_in")))
    nfo = c.get("nonfinite_operands") or {}
    dirty = ["%s %s nan=%d inf=%d" % (d["operand"], tuple(d["shape"]), d["nan"], d["inf"])
             for k in ("inputs", "written_ambiguous") for d in nfo.get(k, [])]
    p("inputs   inputs_finite=%s%s" % (c.get("inputs_finite"),
                                       ("  dirty: " + "; ".join(dirty)) if dirty else ""))
    tk = (c.get("triton_kernels") or {}).get(c["op"]) if c["op"].startswith(("triton.", "inductor.")) else None
    if tk:
        if tk.get("kind") == "inductor":
            src = [l.strip().lstrip("#").strip() for l in (tk.get("src") or "").splitlines()
                   if "Source Nodes" in l or "Original ATen" in l]
            p("inductor %s  (%s)  -- a torch.compile kernel; written: %s%s"
              % (tk.get("kernel"), tk.get("file"),
                 ", ".join(tk["arg_names"][i] for i in tk.get("written", [])) or "none",
                 "".join("\n         " + l for l in src)))
        else:
            p("triton   %s.%s  (%s)  -- a Triton kernel launched from Python; tensors taken as written: %s"
              % (tk.get("module"), tk.get("kernel"), tk.get("file"),
                 ", ".join(tk["arg_names"][i] for i in tk.get("written", [])) or "none"))
    names = tk["arg_names"] if tk else []
    try:
        ops = ["[%d%s] %s" % (i, (" " + names[i]) if i < len(names) else "",
                              describe_arg(x, c.get("pool"))) for i, x in enumerate(c.get("args") or [])]
        ops += ["%s=%s" % (k, describe_arg(v, c.get("pool"))) for k, v in (c.get("kwargs") or {}).items()]
        p("args     %s" % "\n         ".join(ops or ["-"]))
    except Exception as e:                              # noqa: BLE001 -- informational only
        p("args     (not readable: %s)" % e)
    try:                # which rows of the operands are zero / non-finite, and the ids that fed them
        import capsule_operands
        # always a line: an absent one read as a missing feature, not a clean answer (2026-11-20)
        for ln in capsule_operands.row_report(capsule_operands.load(path)) or ["no input row all zero or non-finite"]:
            p("rows     %s" % ln)
    except Exception as e:                              # noqa: BLE001 -- informational only
        p("rows     (not readable: %s: %s)" % (type(e).__name__, str(e)[:100]))
    if c.get("chain"):
        ops = [r["op"].replace("aten.", "") for r in c["chain"] if not r.get("is_view")]
        stops = {}
        for r in (c.get("truncation") or {}).values():
            stops[r] = stops.get(r, 0) + 1
        p("chain    %s  (stopped: %s)" % (" -> ".join(ops[-8:]),
                                          ", ".join("%s x%d" % kv for kv in stops.items()) or "-"))
    if c.get("nan_origin"):
        ot = g._origin_text(c["nan_origin"])
        if c["op"].startswith("inductor.") and (c["nan_origin"] or {}).get("class") == "no_reference":
            # the capture judges origins in the model's process (pointwise ATen re-runs only); an Inductor
            # kernel's reference is its fp64 twin, run by the replay -- "no_reference" read as a
            # contradiction of the replay's finite fp64 check (2026-11-23 validation run)
            ot += "  [at capture; the fp64 twin judges it -- the fp64 line below, or the replay's [2]]"
        p("origin   %s" % ot)
    ovf = overflow_line(c)
    if ovf:
        p(ovf)
    if c.get("collective"):
        k = c["collective"]
        p("collective %s #%s of group %s (ranks %s, step %s): a synchronous collective -- every member rank "
          "captured this call; judge it on all of them: collective_r3.py check <capsule dir> "
          "(no GPU), replay under torchrun" % (k.get("op"), k.get("seq"), k.get("group"), k.get("group_ranks"),
                                                k.get("step")))
    if c.get("settle"):
        s = c["settle"]
        p("settle   %s: %s" % (s.get("result"), s.get("reason", "")))
    if c.get("masked_sites"):
        p("sites    %s" % "; ".join("%s %s/%s x%d" % (m["site"], m["status"], m["origin"], m["hits"])
                                    for m in c["masked_sites"]))
    fr = model_frames(c.get("stack", ""), nframes)
    p("frames   %s" % ("\n         ".join(fr) if fr else
                       "(none outside torch: a backward op -- see the chain / autograd node)"))
    if tk:
        p("cpu      a %s kernel: no CPU re-run; replay on the GPU (replay_nan_op.py)"
          % ("torch.compile" if tk.get("kind") == "inductor" else "Triton"))
        if tk.get("kind") == "inductor":
            p("fp64     %s" % _inductor_fp64(path, c))
        return
    try:
        func = resolve(c["op"])
    except Exception:                                   # noqa: BLE001
        func = None
        p("cpu      %s is not registered here (its library %s did not import -- run in the model's "
          "image): no CPU re-run; replay on the GPU (replay_nan_op.py --chain). The lines above "
          "stand." % (c["op"], "/".join(_OP_LIBS.get(c["op"].split(".")[0], (c["op"].split(".")[0],)))))
    if not g.has_data(c):
        p("cpu      shapes-only capsule (no float values; integer tensors kept): no re-run; replay with --synthesize")
    elif func is not None and "pointwise" in g._tag_names(func):
        try:
            args, kwargs = g.unpack(c["args"], "cpu"), g.unpack(c["kwargs"], "cpu")
            out = func(*args, **kwargs)
            ws = c.get("nan_found_in") or "return"
            bad = out if ws == "return" else (args[int(ws[3:])] if ws.startswith("arg") else None)
            if isinstance(bad, (tuple, list)):
                bad = bad[0]
            o = g._nan_origin(func, args, kwargs, bad, "return" if ws == "return" else ws, True)
            p("cpu      re-run on CPU: %d non-finite (capsule: %d); %s"
              % (o["nonfinite"], c["out_nan"] + c["out_inf"], g._origin_text(o)))
        except Exception as e:                          # noqa: BLE001
            p("cpu      re-run failed: %s: %s" % (type(e).__name__, str(e)[:120]))
    elif func is not None:
        p("cpu      not pointwise: replay on the GPU (replay_nan_op.py --chain)")


def _inductor_fp64(path, c):
    """The written output's status in the kernel's fp64 twin (inductor_ref.py), on a GPU with real values."""
    if not g.has_data(c):
        return "shapes-only capsule: no fp64 twin (replay with --synthesize)"
    if not torch.cuda.is_available():
        return "fp64 twin (inductor_ref.py) needs a GPU: run capsule_info there, or read the replay's [2]"
    try:
        import capsule_operands
        import inductor_ref
        co = capsule_operands.load(path, device="cuda")
        d = g._TRITON_DESCS[co.op]
        if not set(d["written"]) <= set(d["outbuf"]):
            return "unavailable: an in_out_ptr kernel (pre-call values gone)"
        outs = inductor_ref.reference(d, list(co.args))
        n = sum(int((~torch.isfinite(t)).sum()) for t in outs)
        return ("fp64 twin (inductor_ref.py): %s -- %s" % (
            "finite" if n == 0 else "NON-FINITE (%d)" % n,
            "a non-finite output is never correct here: the kernel" if n == 0
            else "the computation itself is non-finite: math, not the kernel"))
    except Exception as e:                              # noqa: BLE001
        return "fp64 twin unavailable: %s: %s" % (type(e).__name__, str(e).splitlines()[0][:100] if str(e) else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsules", nargs="+")
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    for path in a.capsules:
        info(path, a.frames)


if __name__ == "__main__":
    main()
