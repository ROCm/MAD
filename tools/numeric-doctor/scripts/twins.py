"""Second implementations ("twins") of fused ops, as wrong-value references.

A fused op gets a wrong-FINITE-value check only from a reference; writing one by hand per op
(math_refs.VALUE_CHECKS) does not scale, and the stack usually ships a second implementation of the
same op already.  Each entry here runs that one on the op's own (pre-call) inputs and returns what the
op should have written next to what it did write; op_nan_capture's twin check (NAN_CAPTURE_TWIN_EVERY)
and the replay judge the difference like any value check.

    check(op_name, func, args, kwargs, pre=None, out=None)
        -> {"got": {name: tensor}, "ref": {name: tensor}, "floor": float, "source": str, "kind": str}
    rel_err(op_name, func, args, kwargs, pre=None, out=None) -> (rel err, floor, where, source, kind) | None
    has(op_name) -> bool

`kind` says how independent the twin is of the kernel it checks:
    native      a framework's pure-PyTorch implementation (vLLM IR `native`, SGLang `forward_native`)
    alt_kernel  another kernel of the same library (aiter Triton beside aiter asm): shares its conventions
    fp64_twin   the kernel's own source promoted to fp64 (inductor_ref.py): the same algorithm, so it
                catches lowering / precision faults (2026-11-23: tl.sqrt_rn), not a fusion computing the
                wrong thing
A native twin is preferred when an op has several.

Adapters only map arguments: the math is the twin's own.  A variant an adapter does not handle raises
NotImplementedError and the op runs unchecked, never misjudged (the math_refs rule).  Framework imports
are lazy: an entry whose framework is absent from the image is skipped.

Inputs: an op that writes into its inputs (fused_add_rms_norm: input and residual) is fed their
pre-call values from `pre` (the capture's snapshot of what the op writes: {"args": {i: tensor},
"kwargs": {name: tensor}}); without it the variant is not modelled.  `got` is read from the post-call
arguments, or `out` (the op's return value).

Seeded from the ops the 2026-11 examples hit; add one by its dispatcher name.
"""
import types

import torch

_FLOOR = {torch.bfloat16: 3e-2, torch.float16: 5e-3, torch.float32: 1e-4, torch.float64: 1e-9}
# fp64 twin of an fp32 Inductor kernel: libdevice exp/log/rsqrt are not correctly rounded, and the
# kernel's own reductions accumulate in fp32 -- a correct kernel sits well above fp32's ulp
_FLOOR_FP64_TWIN = {torch.bfloat16: 3e-2, torch.float16: 5e-3, torch.float32: 2e-3}


class _Skip(NotImplementedError):
    """This twin does not apply here (framework absent, variant not modelled)."""


def _arg(func, args, kwargs, name, default=None):
    names = [a.name for a in func._schema.arguments]
    if name not in names:
        return default
    i = names.index(name)
    return args[i] if i < len(args) else kwargs.get(name, default)


def _pre_arg(func, args, kwargs, pre, name):
    """The pre-call value of a written argument (an in-place op's input)."""
    names = [a.name for a in func._schema.arguments]
    i = names.index(name)
    pa, pk = (pre or {}).get("args") or {}, (pre or {}).get("kwargs") or {}
    v = pa.get(i) if i < len(args) else pk.get(name)
    if v is None:
        raise _Skip("no pre-call copy of %s (in-place input)" % name)
    return v


# -- vLLM: its IR ops' `native` implementation ---------------------------------------------------
def _vllm_ir_native(name):
    try:
        import vllm.ir.ops as irops
    except Exception as e:                              # noqa: BLE001
        raise _Skip("vllm.ir not importable: %s" % e)
    op = getattr(irops, name, None)
    impl = op.impls.get("native") if op is not None else None
    if impl is None:
        raise _Skip("vllm.ir.%s has no native implementation" % name)
    for attr in ("impl_fn", "fn", "impl", "func"):
        fn = getattr(impl, attr, None)
        if callable(fn):
            return fn
    raise _Skip("vllm.ir.%s native implementation not callable" % name)


def _c_rms_norm(func, args, kwargs, pre, out):
    """_C::rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon)"""
    x, w, eps = _arg(func, args, kwargs, "input"), _arg(func, args, kwargs, "weight"), \
        float(_arg(func, args, kwargs, "epsilon"))
    ref = _vllm_ir_native("rms_norm")(x, w, eps, None)
    return {"got": {"result": _arg(func, args, kwargs, "result")}, "ref": {"result": ref}}


def _c_fused_add_rms_norm(func, args, kwargs, pre, out):
    """_C::fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon): input <- norm(input +
    residual), residual <- input + residual"""
    x = _pre_arg(func, args, kwargs, pre, "input").clone()
    r = _pre_arg(func, args, kwargs, pre, "residual").clone()
    w, eps = _arg(func, args, kwargs, "weight"), float(_arg(func, args, kwargs, "epsilon"))
    res = _vllm_ir_native("fused_add_rms_norm")(x, r, w, eps, None)
    if not isinstance(res, (tuple, list)) or len(res) != 2:
        raise _Skip("fused_add_rms_norm native returned %r" % type(res))
    return {"got": {"input": _arg(func, args, kwargs, "input"), "residual": _arg(func, args, kwargs, "residual")},
            "ref": {"input": res[0], "residual": res[1]}}


def _c_silu_and_mul(func, args, kwargs, pre, out):
    """_C::silu_and_mul(Tensor! result, Tensor input)"""
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception as e:                              # noqa: BLE001
        raise _Skip("vllm SiluAndMul not importable: %s" % e)
    ref = SiluAndMul.forward_native(_arg(func, args, kwargs, "input"))
    return {"got": {"result": _arg(func, args, kwargs, "result")}, "ref": {"result": ref}}


def _vllm_ir_op(name):
    """vllm_ir.<name>: the IR op called un-lowered -- its own native implementation is the twin."""
    def twin(func, args, kwargs, pre, out):
        if kwargs:
            raise _Skip("vllm_ir op with keyword arguments")
        if func._schema.arguments and any(a.alias_info is not None and a.alias_info.is_write
                                          for a in func._schema.arguments):
            raise _Skip("in-place vllm_ir op")
        ref = _vllm_ir_native(name)(*args)
        got, refs = {}, {}
        for i, (g, r) in enumerate(zip(_as_list(out), _as_list(ref))):
            got["return%d" % i], refs["return%d" % i] = g, r
        return {"got": got, "ref": refs}
    return twin


# -- SGLang: RMSNorm.forward_native with a stand-in for `self` -------------------------------------
def _sglang_rmsnorm(x, w, eps, residual=None):
    try:
        from sglang.srt.layers.layernorm import RMSNorm
    except Exception as e:                              # noqa: BLE001
        raise _Skip("sglang RMSNorm not importable: %s" % e)
    me = types.SimpleNamespace(weight=w, variance_epsilon=eps, hidden_size=x.shape[-1], variance_size_override=None,
                               override_orig_dtype=None, fp32_residual=False, cast_x_before_out_mul=False)
    return RMSNorm.forward_native(me, x, residual)


def _aiter_rmsnorm2d_fwd(func, args, kwargs, pre, out):
    """aiter::rmsnorm2d_fwd(Tensor input, Tensor weight, float epsilon, SymInt use_model_sensitive_rmsnorm=0) -> Tensor"""
    if int(_arg(func, args, kwargs, "use_model_sensitive_rmsnorm", 0) or 0):
        raise _Skip("use_model_sensitive_rmsnorm not modelled")
    x, w, eps = _arg(func, args, kwargs, "input"), _arg(func, args, kwargs, "weight"), \
        float(_arg(func, args, kwargs, "epsilon"))
    try:
        ref, src = _sglang_rmsnorm(x, w, eps), "sglang RMSNorm.forward_native"
    except _Skip:
        ref, src = _vllm_ir_native("rms_norm")(x, w, eps, None), "vllm.ir rms_norm native"
    return {"got": {"return": out}, "ref": {"return": ref}, "source": src}


def _aiter_rmsnorm2d_fwd_with_add(func, args, kwargs, pre, out):
    """aiter::rmsnorm2d_fwd_with_add(Tensor! out, Tensor input, Tensor residual_in, Tensor! residual_out, Tensor weight,
    float epsilon, bool gemma_norm=False, SymInt use_model_sensitive_rmsnorm=0)"""
    if _arg(func, args, kwargs, "gemma_norm", False) or int(_arg(func, args, kwargs, "use_model_sensitive_rmsnorm", 0) or 0):
        raise _Skip("gemma_norm / use_model_sensitive_rmsnorm not modelled")
    x, r = _arg(func, args, kwargs, "input"), _arg(func, args, kwargs, "residual_in")
    w, eps = _arg(func, args, kwargs, "weight"), float(_arg(func, args, kwargs, "epsilon"))
    try:
        y, r2 = _sglang_rmsnorm(x.clone(), w, eps, residual=r.clone())
        src = "sglang RMSNorm.forward_native"
    except _Skip:
        y, r2 = _vllm_ir_native("fused_add_rms_norm")(x.clone(), r.clone(), w, eps, None)
        src = "vllm.ir fused_add_rms_norm native"
    return {"got": {"out": _arg(func, args, kwargs, "out"), "residual_out": _arg(func, args, kwargs, "residual_out")},
            "ref": {"out": y, "residual_out": r2}, "source": src}


# -- aiter MLA decode: the asm kernel against aiter's Triton decode ---------------------------------
def _aiter_mla_stage1(func, args, kwargs, pre, out):
    """aiter::mla_decode_stage1_asm_fwd: what aiter's stage 2 merges from the asm kernel's splits (math_refs.
    mla_stage1_got, the variants it models) against aiter's Triton MLA decode on the same Q / KV / page table."""
    import math_refs
    try:
        got, m = math_refs.mla_stage1_got(func, args, kwargs)
    except NotImplementedError as e:
        raise _Skip(str(e))
    try:
        from aiter.ops.triton.attention.mla_decode import decode_attention_fwd
    except Exception as e:                              # noqa: BLE001
        raise _Skip("aiter Triton MLA decode not importable: %s" % e)
    q, rows, lens, kv_indptr, idx = m["q"], m["rows"], m["lens"], m["kv_indptr"], m["idx"]
    B, H, Lv, D, dev = m["B"], m["H"], m["Lv"], m["D"], q.device
    if B == 0:
        raise _Skip("empty batch")
    lmax = int(lens.max())
    # page size 1: one token per page; the Triton kernel reads KV through a per-request token table
    table = torch.zeros(B, max(lmax, 1), dtype=torch.int32, device=dev)
    for b in range(B):
        n = int(lens[b])
        if n:
            s0 = int(kv_indptr[b])
            table[b, :n] = idx[s0:s0 + n].to(torch.int32)
    k_buf = rows.reshape(-1, 1, D)
    v_buf = k_buf[..., :Lv]
    splits = 4
    o = torch.empty(B, H, Lv, dtype=q.dtype, device=dev)
    lse = torch.empty(B, H, dtype=torch.float32, device=dev)
    logits = torch.empty(B, H, splits, Lv + 1, dtype=torch.float32, device=dev)
    decode_attention_fwd(q, k_buf, v_buf, o, lse, table, lens.to(torch.int32), logits, splits, m["scale"])
    # floor from the kernel's dtype: `got` is stage 2's merge already in fp32, and fp32's floor (1e-4) flagged
    # every correct call (2026-11-24: 40/40; good calls sit at ~1e-2 against this twin, bad ones at 0.14)
    return {"got": {"splitData": got}, "ref": {"splitData": o.float()}, "floor": _FLOOR.get(q.dtype, 3e-2)}


# -- Inductor kernels: the fp64 twin -----------------------------------------------------------------
def _inductor(func, args, kwargs, pre, out):
    import inductor_ref
    d = getattr(func, "desc", None)
    if d is None:
        raise _Skip("no Inductor descriptor")
    if kwargs:
        raise _Skip("Inductor launch with keyword operands")
    outbuf = set(d.get("outbuf") or ())
    a = list(args)
    pa = (pre or {}).get("args") or {}
    for i in d["written"]:
        if i not in outbuf:               # in_out_ptr: fed its pre-call value
            if i not in pa:
                raise _Skip("in_out_ptr without a pre-call copy")
            a[i] = pa[i]
    try:
        refs = inductor_ref.reference(d, a, keep_casts=True, on_gpu=True)
    except NotImplementedError as e:
        clear_device_error()            # a failed fp64 launch must not surface in the model's next op
        raise _Skip(str(e))
    got, ref = {}, {}
    for i, r in zip(d["written"], refs):
        n = d["arg_names"][i]
        got[n], ref[n] = args[i], r
    return {"got": got, "ref": ref}


# op name (as captured: str(OpOverload) / InductorOp.name) -> [(source, kind, adapter)], tried in order
REGISTRY = {
    "_C.rms_norm.default": [("vllm.ir rms_norm native", "native", _c_rms_norm)],
    "_C.fused_add_rms_norm.default": [("vllm.ir fused_add_rms_norm native", "native", _c_fused_add_rms_norm)],
    "_C.silu_and_mul.default": [("vllm SiluAndMul.forward_native", "native", _c_silu_and_mul)],
    "aiter.rmsnorm2d_fwd.default": [("sglang / vllm native RMSNorm", "native", _aiter_rmsnorm2d_fwd)],
    "aiter.rms_norm.default": [("sglang / vllm native RMSNorm", "native", _aiter_rmsnorm2d_fwd)],
    "aiter.rmsnorm2d_fwd_with_add.default": [("sglang / vllm native fused add + RMSNorm", "native",
                                              _aiter_rmsnorm2d_fwd_with_add)],
    "aiter.mla_decode_stage1_asm_fwd.default": [("aiter Triton MLA decode (decode_attention_fwd)", "alt_kernel",
                                                 _aiter_mla_stage1)],
}
_PREFIX = [("inductor.", ("Inductor kernel source in fp64 (inductor_ref.py)", "fp64_twin", _inductor))]
# ops whose twin judges what the consumer reads, not the raw outputs: their scratch outputs keep slots the kernel
# never writes (a split-KV op's unused splits hold allocator garbage, NaN included), so the capture's NaN scan of
# them would flag correct calls.  Every other twin-checked op still gets the NaN scan: a twin compares finite
# positions only.
OWNS_OUTPUTS = {"aiter.mla_decode_stage1_asm_fwd.default"}


def _entries(op_name):
    if op_name in REGISTRY:
        return REGISTRY[op_name]
    if op_name.startswith("vllm_ir."):
        name = op_name.split(".")[1]
        return [("vllm.ir %s native" % name, "native", _vllm_ir_op(name))]
    return [e for p, e in _PREFIX if op_name.startswith(p)]


def clear_device_error():
    """Reset the HIP / CUDA runtime's pending error after a twin failed to launch: left set, the model's next PyTorch
    op raised it as its own (2026-11-24: an fp64 twin kernel with no code object for the GPU, `no kernel image is
    available`, surfaced in the model's forward and killed the trial)."""
    import ctypes
    for lib, fn in (("libamdhip64.so", "hipGetLastError"), ("libcudart.so", "cudaGetLastError")):
        try:
            getattr(ctypes.CDLL(lib), fn)()
            return
        except (OSError, AttributeError):
            continue


def has(op_name):
    return bool(_entries(op_name))


def owns_outputs(op_name):
    return op_name in OWNS_OUTPUTS


def _as_list(x):
    if isinstance(x, torch.Tensor):
        return [x]
    return [t for t in (x or ()) if isinstance(t, torch.Tensor)]


def check(op_name, func, args, kwargs, pre=None, out=None):
    """The first applicable twin's {"got", "ref", "floor", "source", "kind"}; NotImplementedError if none
    applies (framework absent, variant not modelled)."""
    why = []
    for source, kind, fn in _entries(op_name):
        try:
            with torch.no_grad():
                r = fn(func, tuple(args), dict(kwargs), pre, out)
        except NotImplementedError as e:
            why.append("%s: %s" % (source, e))
            continue
        r.setdefault("source", source)
        r["kind"] = kind
        # a floor per output, from its own dtype unless the adapter set one: a kernel writing an fp32 and a bf16
        # output judged both at fp32's floor, and the bf16 one's plain rounding (3e-3) read as a wrong value
        # (2026-11-24, an Inductor RMSNorm kernel)
        table = _FLOOR_FP64_TWIN if kind == "fp64_twin" else _FLOOR
        fixed = r.get("floor")
        r["floors"] = dict((n, fixed if fixed is not None else table.get(g.dtype, 3e-2))
                           for n, g in r["got"].items() if g is not None)
        r.setdefault("floor", max(r["floors"].values()) if r["floors"] else 3e-2)
        return r
    raise NotImplementedError("; ".join(why) or "no twin for %s" % op_name)


def rel_err(op_name, func, args, kwargs, pre=None, out=None):
    """(max over outputs of max|got - ref| / max|ref|, floor, where, source, kind), or None when no twin applies
    or a twin's result is not finite where the kernel's is (a twin failure, never a verdict).  `where` names the
    worst output."""
    try:
        r = check(op_name, func, args, kwargs, pre, out)
    except NotImplementedError:
        return None
    worst, where, ratio = -1.0, None, -1.0
    for name, g in r["got"].items():
        f = r["ref"].get(name)
        if g is None or f is None or tuple(g.shape) != tuple(f.shape):
            return None
        g32, f32 = g.detach().float(), f.detach().float().to(g.device)
        fin = torch.isfinite(g32)
        if not bool(fin.any()):
            continue                      # all non-finite: the NaN path owns it
        if not bool(torch.isfinite(f32[fin]).all()):
            return None                   # the twin is non-finite where the kernel is finite
        scale = float(f32[fin].abs().amax())
        err = float((g32[fin] - f32[fin]).abs().amax()) / max(scale, 1e-30)
        fl = r["floors"].get(name, r["floor"])
        if err / fl > ratio:            # the output furthest past ITS floor
            worst, where, ratio = err, name, err / fl
    if where is None:
        return None
    return worst, r["floors"].get(where, r["floor"]), where, r["source"], r["kind"]
