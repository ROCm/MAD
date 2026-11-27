# debug-numeric 5N compiled track (C1): capture the Inductor kernel whose output Inductor's own nan_asserts caught --
# a capsule for compiled code, which the per-op capture cannot see (step5n-compiled.md *Kernel capsule*).
#
#   PYTHONPATH=<REPO_ROOT>/scripts/inductor_capsule INDUCTOR_CAPSULE_DIR=<dir> TORCHINDUCTOR_NAN_ASSERTS=1 \
#     python3 serve.py ...          (vLLM: HIP graphs off, compilation on -- step5n-compiled.md *vLLM*)
#
# With TORCHINDUCTOR_NAN_ASSERTS=1 the generated wrapper checks every kernel output with
# `assert not bufN.isnan().any().item()`.  At the first such AssertionError (Python 3.12 sys.monitoring RAISE
# event, so it is seen where it is raised, whatever catches it later) this saves, from the raising frame:
# the kernel's name and Triton source, the launch line `<kernel>.run(<args>, stream=...)` just above the
# assert, every launch argument (tensors with their values: the inputs' real data; sizes as ints), which
# parameters are in / out / in_out pointers (the kernel's def line), the assert, the model frames, and the
# torch / triton versions and device.  One capsule per process: <dir>/inductor_<pid>_<kernel>.pt.
# Replay without the model: scripts/replay_inductor_kernel.py <capsule>.
#
# Loaded at interpreter start, so neither the model nor torch is edited; prints `[op-nan-inductor] ...` lines.
# Nothing happens without INDUCTOR_CAPSULE_DIR.  Private Inductor output format (torch 2.13): re-check after a
# torch upgrade.  INDUCTOR_CAPSULE_MAX_BYTES (4 GiB) caps the tensor data saved.
import os
import sys

_DIR = os.environ.get("INDUCTOR_CAPSULE_DIR", "")
_TOOL = 3                       # sys.monitoring tool id (0 debugger, 1 coverage, 2 profiler, 5 optimizer)
_state = {"done": False}


def _say(msg):
    print("[op-nan-inductor] " + msg, file=sys.stderr, flush=True)


def _span_flat(t):
    """The elements a strided tensor touches, as a flat CPU copy (offset rebased to 0)."""
    import torch
    if t.numel() == 0:
        return torch.empty(0, dtype=t.dtype)
    span = 1 + sum((s - 1) * st for s, st in zip(t.size(), t.stride()))
    return torch.as_strided(t, (span,), (1,), t.storage_offset()).detach().cpu().clone()


def _bits_equal(a, b):
    """Bitwise equality (NaN payloads included), any dtype and stride."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _launch(kern, launch_line, ns):
    """Re-run the kernel's launch line with the tensors in ns (name -> value)."""
    import re
    call = re.match(r"\s*\w+\.run\((.*)\)\s*$", launch_line).group(1)
    eval("__k.run(%s)" % call, {}, dict(ns, __k=kern))


def _sparsify(args, live, kern, launch_line, loc):
    """{input name: (row indices, those rows on CPU)} for the inputs that verified as sparse-safe."""
    import os
    import torch
    min_bytes = int(os.environ.get("INDUCTOR_CAPSULE_SPARSE_MIN_BYTES", str(16 << 20)))
    if os.environ.get("INDUCTOR_CAPSULE_SPARSE", "1") == "0":
        return {}
    ints = [live[d["name"]] for d in args if d.get("tensor") and d["kind"] == "in"
            and not d["dtype"].is_floating_point and live[d["name"]].numel() > 0]
    outs = [d["name"] for d in args if d.get("tensor") and d["kind"] in ("out", "in_out")]
    if any(d.get("kind") == "in_out" for d in args):
        return {}                                   # post-call values: the relaunch would not be the model's
    result = {}
    for d in args:
        if not (d.get("tensor") and d["kind"] == "in" and d["dtype"].is_floating_point):
            continue
        x = live[d["name"]]
        if x.numel() * x.element_size() < min_bytes or x.dim() < 2 or not x.is_contiguous():
            continue
        rows = None
        for ix in ints:                            # an integer input whose values all index x's rows
            v = ix.long().flatten()
            if int(v.min()) >= 0 and int(v.max()) < x.size(0):
                rows = v.unique() if rows is None else torch.cat([rows, v]).unique()
        if rows is None or rows.numel() * 2 > x.size(0):
            continue
        drop = os.environ.get("INDUCTOR_CAPSULE_SPARSE_SELFTEST") == "drop"   # test: lose one row
        keep = rows[1:] if drop and rows.numel() > 1 else rows
        rebuilt = torch.full_like(x, float("nan"))
        rebuilt[keep] = x[keep]
        ok = True
        for tensors in (None, rebuilt):
            ns = dict(loc)
            for n in outs:
                ns[n] = torch.empty_strided(live[n].size(), live[n].stride(), dtype=live[n].dtype,
                                            device=live[n].device).zero_()
            if tensors is not None:
                ns[d["name"]] = tensors
            _launch(kern, launch_line, ns)
            torch.cuda.synchronize()
            got = [ns[n] for n in outs]
            if tensors is None:
                ref = [g.clone() for g in got]
            else:
                ok = all(_bits_equal(a, b) for a, b in zip(ref, got))
        del rebuilt
        _say("sparse %s: %d of %d rows -- %s" % (d["name"], keep.numel(), x.size(0),
             "verified (outputs bit-identical to the full input)" if ok else
             "NOT identical -> saved whole"))
        if ok:
            result[d["name"]] = (keep.cpu(), x[keep].detach().cpu().clone())
    return result


def _save(frame):
    import ast
    import re
    import torch

    path = frame.f_code.co_filename
    try:
        src = open(path).read()
    except OSError as e:
        _say("wrapper source unreadable (%s): no capsule" % e)
        return
    lines = src.splitlines()
    lno = frame.f_lineno
    assert_line = lines[lno - 1].strip() if 0 < lno <= len(lines) else ""
    m = re.search(r"assert not (\w+)\.is(nan|inf)\(", assert_line)
    checked = m.group(1) if m else None
    launch = None
    for i in range(lno - 2, max(lno - 400, -1), -1):         # the kernel launch just above the asserts
        mm = re.match(r"\s*(\w+)\.run\((.*)\)\s*$", lines[i])
        if mm:
            launch = (i + 1, mm.group(1), lines[i].strip())
            break
    if launch is None:
        _say("no `<kernel>.run(...)` above %s:%d -- the assert is not on a kernel output (a graph input or "
             "output check): no capsule" % (path, lno))
        return
    kname = launch[1]
    call = ast.parse(launch[2]).body[0].value
    ksrc = None
    mk = re.search(r"^%s = async_compile\.triton\('%s', '''(.*?)''', device_str='(\w+)'\)" % (kname, kname),
                   src, re.S | re.M)
    if mk:
        ksrc = mk.group(1)
    params = []
    if ksrc:
        md = re.search(r"def %s\((.*?)\):" % kname, ksrc, re.S)
        if md:
            params = [p.split(":")[0].strip() for p in md.group(1).split(",") if p.strip()]
    loc = dict(frame.f_globals)
    loc.update(frame.f_locals)
    cap = int(os.environ.get("INDUCTOR_CAPSULE_MAX_BYTES", str(4 << 30)))
    args, live = [], {}
    for i, a in enumerate(call.args):
        role = params[i] if i < len(params) else "?"
        kind = ("in_out" if role.startswith("in_out_ptr") else "out" if role.startswith("out_ptr")
                else "in" if role.startswith("in_ptr") else "scalar")
        if isinstance(a, ast.Constant):
            args.append({"name": repr(a.value), "role": role, "kind": "scalar", "value": a.value})
            continue
        name = ast.unparse(a)
        try:
            v = eval(name, {}, loc)
        except Exception as e:                                 # noqa: BLE001
            args.append({"name": name, "role": role, "kind": kind, "error": repr(e)})
            continue
        if isinstance(v, torch.Tensor):
            d = {"name": name, "role": role, "kind": kind, "tensor": True, "dtype": v.dtype,
                 "size": tuple(v.size()), "stride": tuple(v.stride())}
            if v.is_floating_point():
                d["nonfinite"] = int((~torch.isfinite(v)).sum())
            live[name] = v
            args.append(d)
        else:
            try:
                args.append({"name": name, "role": role, "kind": "scalar", "value": int(v)})
            except Exception:                                  # noqa: BLE001
                args.append({"name": name, "role": role, "kind": kind, "value": repr(v)})

    # Sparse inputs: a big input indexed row-wise by an integer input (an embedding table by token ids)
    # is saved as the indexed rows only -- the whole table made a 216 MB capsule of model weights
    # (2026-11-19).  Proven, not assumed: the kernel is relaunched here on the full inputs and on the
    # rebuilt one (saved rows in place, every other row NaN); any output differing bit for bit -> full save.
    kern = frame.f_globals.get(kname)
    sparse = _sparsify(args, live, kern, launch[2], loc) if kern is not None else {}
    used = 0
    for d in args:
        if not d.get("tensor") or d["kind"] == "out":
            continue
        if d["name"] in sparse:
            rows, data = sparse[d["name"]]
            d.update(sparse_rows=rows, row_data=data, fill="nan")
            used += data.numel() * data.element_size()
            continue
        v = live[d["name"]]
        nb = v.numel() * v.element_size()
        if used + nb <= cap:
            d["data"] = _span_flat(v)
            used += nb
    # the model's outputs of this launch, for the replay's bitwise check (outputs are small next to inputs)
    expected = {}
    for d in args:
        if d.get("kind") in ("out", "in_out") and d.get("tensor"):
            v = live[d["name"]]
            if used + v.numel() * v.element_size() <= cap:
                expected[d["name"]] = _span_flat(v)
                used += v.numel() * v.element_size()
    frames, f = [], frame.f_back
    while f is not None and len(frames) < 12:
        fn = f.f_code.co_filename
        if not any(s in fn for s in ("/torch/", "/python3.", "<frozen", "/triton/")):
            frames.append("%s:%d %s" % (fn, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    dev = torch.cuda.get_device_properties(torch.cuda.current_device()) if torch.cuda.is_available() else None
    try:
        import triton
        tv = triton.__version__
    except Exception:                                          # noqa: BLE001
        tv = None
    capsule = {
        "kind": "inductor_kernel", "kernel": kname, "kernel_source": ksrc, "params": params,
        "launch_line": launch[2], "assert_line": assert_line, "checked_buffer": checked,
        "wrapper_path": path, "wrapper_line": lno, "args": args, "frames": frames, "expected": expected,
        "torch": torch.__version__, "triton": tv, "hip": torch.version.hip,
        "device": dev.name if dev else None, "arch": getattr(dev, "gcnArchName", None) if dev else None,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TORCHINDUCTOR_", "VLLM_", "HIP_"))},
    }
    os.makedirs(_DIR, exist_ok=True)
    out = os.path.join(_DIR, "inductor_%d_%s.pt" % (os.getpid(), kname))
    torch.save(capsule, out + ".tmp")
    os.replace(out + ".tmp", out)
    ins = [a for a in args if a.get("kind") in ("in", "in_out") and a.get("tensor")]
    _say("captured kernel %s (assert on %s) -> %s; inputs %s; data %.1f MB%s%s" % (
        kname, checked, out,
        "finite" if all(not a.get("nonfinite") for a in ins) else
        "NON-FINITE: " + ", ".join("%s=%d" % (a["name"], a["nonfinite"]) for a in ins if a.get("nonfinite")),
        used / 1e6, "".join("; %s: %d of %d rows (verified)" % (n, len(r), live[n].size(0))
                            for n, (r, _) in sparse.items()),
        "; in_out pointers hold post-call values: replay may differ"
        if any(a.get("kind") == "in_out" for a in args) else ""))


def _on_raise(code, offset, exc):
    if _state["done"] or not isinstance(exc, AssertionError):
        return
    fn = code.co_filename
    if "inductor" not in fn and "torch_compile_cache" not in fn:
        return
    f = sys._getframe(1)
    while f is not None and f.f_code is not code:
        f = f.f_back
    if f is None:
        return
    _state["done"] = True
    try:
        _save(f)
    except Exception as e:                                     # noqa: BLE001 -- never break the model
        _say("capture failed: %r" % (e,))


if _DIR:
    if hasattr(sys, "monitoring"):
        try:
            sys.monitoring.use_tool_id(_TOOL, "numeric-doctor-inductor-capsule")
            sys.monitoring.register_callback(_TOOL, sys.monitoring.events.RAISE, _on_raise)
            sys.monitoring.set_events(_TOOL, sys.monitoring.events.RAISE)
            _say("armed: the first Inductor nan_asserts failure is saved to %s (TORCHINDUCTOR_NAN_ASSERTS=%s)"
                 % (_DIR, os.environ.get("TORCHINDUCTOR_NAN_ASSERTS", "unset")))
        except Exception as e:                                 # noqa: BLE001
            _say("NOT armed: %r" % (e,))
    else:
        _say("NOT armed: Python %s has no sys.monitoring (3.12+)" % sys.version.split()[0])
