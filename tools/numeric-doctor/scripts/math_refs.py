"""fp64 math references for fused ops that have no fp64 kernel of their own.

A fused norm or activation from aiter, sgl_kernel or vLLM's _C refuses fp64, so neither the replay's walk
(replay_nan_op.py: is a non-finite output ever correct on these operands?) nor the capture's nan_origin
(op_nan_capture.py: math / overflow / unexplained) could judge it: the verdict stopped at `unavailable` /
`no_reference`, and a `0 * rsqrt(0)` in an eps-0 RMSNorm read as KERNEL_BUG_UNVERIFIED (2026-11-19, SGLang's
aiter.rmsnorm2d_fwd). Each reference computes what the op writes, from its inputs only, in the dtype it is
given (the callers pass fp64 copies).

    reference(op_name, func, args, kwargs) -> {name: tensor}

`name` is the schema name of each tensor the op writes (`out`, `residual_out`, `result`), or "return" for its
return value. Keyed by the op string as captured (str(OpOverload): "aiter.rmsnorm2d_fwd.default"). A variant a
reference does not model raises NotImplementedError: callers report `unavailable`, never a value. Ops that
overwrite their own inputs (rotary embedding, fused_add_rms_norm) have no entry: once the op ran, its inputs
are gone (the replay's walk skips them; the capture judges them from its pre-call snapshot only).

Schemas: aiter / sgl_kernel of lmsysorg/sglang:v0.5.23-rocm10-mi35x, _C of vllm/vllm-openai-rocm (2026-11-20).
Add an op: one entry in REFERENCES, from its schema (torch._C._jit_get_all_schemas() in the image).
"""
import math
import os

import torch


def _get(func, args, kwargs, name, default=None):
    """Argument `name` of a call, positional or keyword, by the op's schema."""
    names = [a.name for a in func._schema.arguments]
    if name not in names:           # an optional argument this op variant does not have
        return default
    i = names.index(name)
    if i < len(args):
        return args[i]
    return kwargs.get(name, default)


def _rms(x, w, eps, gemma=False):
    r = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if w is None:
        return r
    return r * (1 + w) if gemma else r * w


def _plain(func, a, kw):
    if _get(func, a, kw, "use_model_sensitive_rmsnorm", 0):
        raise NotImplementedError("use_model_sensitive_rmsnorm not modelled")


def _rmsnorm_return(func, a, kw):
    """aiter.rmsnorm2d_fwd / rms_norm(input, weight, epsilon, use_model_sensitive_rmsnorm=0) -> Tensor"""
    _plain(func, a, kw)
    return {"return": _rms(_get(func, a, kw, "input"), _get(func, a, kw, "weight"),
                           float(_get(func, a, kw, "epsilon")))}


def _rmsnorm_out(out_name):
    """rmsnorm into an out-param: aiter.rmsnorm(out, input, weight, epsilon, gemma_norm=False),
    _C.rms_norm(result, input, weight?, epsilon)."""
    def ref(func, a, kw):
        return {out_name: _rms(_get(func, a, kw, "input"), _get(func, a, kw, "weight"),
                               float(_get(func, a, kw, "epsilon")),
                               bool(_get(func, a, kw, "gemma_norm", False)) if out_name == "out" else False)}
    return ref


def _add_rmsnorm(func, a, kw):
    """aiter.rmsnorm2d_fwd_with_add / add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon,
    gemma_norm=False[, use_model_sensitive_rmsnorm=0]): residual_out = input + residual_in, out = rms(that)."""
    _plain(func, a, kw)
    s = _get(func, a, kw, "input") + _get(func, a, kw, "residual_in")
    return {"out": _rms(s, _get(func, a, kw, "weight"), float(_get(func, a, kw, "epsilon")),
                        bool(_get(func, a, kw, "gemma_norm", False))),
            "residual_out": s}


def _gate(act, out_name, act_first=True):
    """act(x[..., :d]) * x[..., d:] (act_first) or x[..., :d] * act(x[..., d:]), d = last dim / 2."""
    def ref(func, a, kw):
        if float(_get(func, a, kw, "limit", 0.0) or 0.0):
            raise NotImplementedError("silu_and_mul limit (clamp) not modelled")
        x = _get(func, a, kw, "input")
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return {out_name: act(x1) * x2 if act_first else x1 * act(x2)}
    return ref


def _silu(x):
    return x * torch.sigmoid(x)


def _gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2.0)))


def _gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def _layernorm(func, a, kw):
    """aiter.layernorm2d_fwd / layer_norm(input, weight, bias, epsilon, x_bias=None) -> Tensor"""
    if _get(func, a, kw, "x_bias") is not None:
        raise NotImplementedError("x_bias not modelled")
    x, w, b = _get(func, a, kw, "input"), _get(func, a, kw, "weight"), _get(func, a, kw, "bias")
    eps = float(_get(func, a, kw, "epsilon", 1e-5))
    mu = x.mean(-1, keepdim=True)
    r = (x - mu) * torch.rsqrt((x - mu).pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        r = r * w
    if b is not None:
        r = r + b
    return {"return": r}


REFERENCES = {
    # RMSNorm (SGLang's default on ROCm: aiter.rmsnorm2d_fwd; vLLM eager: _C.rms_norm)
    "aiter.rmsnorm2d_fwd.default": _rmsnorm_return,
    "aiter.rms_norm.default": _rmsnorm_return,
    "aiter.rmsnorm.default": _rmsnorm_out("out"),
    "_C.rms_norm.default": _rmsnorm_out("result"),
    "aiter.rmsnorm2d_fwd_with_add.default": _add_rmsnorm,
    "aiter.add_rmsnorm.default": _add_rmsnorm,
    # gated activations
    "aiter.silu_and_mul.default": _gate(_silu, "out"),
    "sgl_kernel.silu_and_mul.default": _gate(_silu, "out"),
    "_C.silu_and_mul.default": _gate(_silu, "result"),
    "_C.mul_and_silu.default": _gate(_silu, "out", act_first=False),
    "sgl_kernel.gelu_and_mul.default": _gate(_gelu, "out"),
    "_C.gelu_and_mul.default": _gate(_gelu, "out"),
    "sgl_kernel.gelu_tanh_and_mul.default": _gate(_gelu_tanh, "out"),
    "_C.gelu_tanh_and_mul.default": _gate(_gelu_tanh, "out"),
    # LayerNorm
    "aiter.layernorm2d_fwd.default": _layernorm,
    "aiter.layer_norm.default": _layernorm,
}


def reference(op_name, func, args, kwargs):
    """{name: tensor} of what `op_name` writes, computed from its inputs."""
    return REFERENCES[op_name](func, tuple(args), dict(kwargs))


# --------------------------------------------------------------------------------------
# Value checks: wrong FINITE output of a fused op (the capture's NAN_CAPTURE_VERIFY_EVERY,
# the replay's wrong-value scoring).
#
#     value_check(op_name, func, args, kwargs) -> {"got": t, "ref": t, "floor": rel, "where": name}
#
# Called AFTER the op ran, on its post-call arguments: `got` is what the op's outputs say
# (combined the way its consumer combines them), `ref` the same quantity recomputed in
# fp32 from its inputs, `floor` the relative error a correct call stays under, `where`
# the schema name of the written tensor a capsule reports.  The REFERENCES above answer
# "is a non-finite output ever correct here?"; these answer "is this finite output right?"
# -- a kernel race that reads the wrong tile writes plausible numbers, never a NaN
# Same NotImplementedError contract for variants not modelled.
# --------------------------------------------------------------------------------------

def _mla_mgc(nhead, q_dtype, kv_dtype):
    """Tokens per split granule, as aiter.mla.mla_decode_fwd picks it for qseqlen 1."""
    fp8 = getattr(torch, "float8_e4m3fnuz", None), getattr(torch, "float8_e4m3fn", None)
    if nhead == 128 and q_dtype in fp8 and kv_dtype in fp8:
        return 32
    if nhead == 64 and q_dtype == torch.bfloat16 and kv_dtype == torch.bfloat16:
        return 32
    return 64 if nhead in (8, 16) else 16


def mla_stage1_got(func, a, kw):
    """aiter.mla_decode_stage1_asm_fwd, the non-persistent MLA decode (absorbed weights):
    per request b and split z, partial attention of Q[b] over its KV rows into
    splitData[b, z] with its log-sum-exp in splitLse[b, z]; aiter's Triton stage 2 merges
    the splits into `output`.  `got` is that merge, done here the way stage 2 does it
    (aiter/mla.py _fwd_kernel_stage2_asm, amd-aiter 0.1.26): one split per request --
    stage 1 wrote the final output itself, in the output's dtype, at the front of
    splitData (FINAL_OUT) -- else exp(lse - max)-weighted over the valid splits.  `ref` is
    softmax(Q K^T * scale) V in fp32 over the same rows, V = the first Lv columns of the
    latent row.  Independent of how the kernel deals tiles to splits (a check that
    shared the kernel's partitioning would share its mistakes).  Modelled: qseqlen 1, page_size 1, one KV head, no work_meta_data
    (persistent mode), no q/kv scales, no context parallelism, unshuffled KV."""
    g = lambda n, d=None: _get(func, a, kw, n, d)                      # noqa: E731
    if g("work_meta_data") is not None:
        raise NotImplementedError("persistent mode (work_meta_data) not modelled")
    if int(g("max_seqlen_q")) != 1 or int(g("page_size")) != 1 or int(g("nhead_kv")) != 1:
        raise NotImplementedError("only qseqlen 1, page_size 1, one KV head modelled")
    if g("q_scale") is not None or g("kv_scale") is not None:
        raise NotImplementedError("q/kv scales (fp8) not modelled")
    if int(g("cp_world_size", 1)) != 1 or int(g("kv_shuffled", 0)):
        raise NotImplementedError("context parallelism / shuffled KV not modelled")
    q, kv, out = g("Q"), g("KV"), g("output")
    data, lse = g("splitData"), g("splitLse")
    qo, kv_indptr, idx = g("qo_indptr"), g("kv_indptr"), g("kv_page_indices")
    split_indptr = g("num_kv_splits_indptr")
    if split_indptr is None:
        raise NotImplementedError("no num_kv_splits_indptr")
    B, H = qo.numel() - 1, q.shape[1]
    if q.shape[0] != B or not torch.equal(qo.long().cpu(), torch.arange(B + 1)):
        raise NotImplementedError("qo_indptr is not one query per request")
    Lv, D = data.shape[-1], q.shape[-1]
    dev = q.device
    lens = (kv_indptr[1:] - kv_indptr[:-1]).long()
    rows = kv.reshape(-1, D)
    scale = float(g("softmax_scale"))

    # got: stage 2's merge
    nsplit = (split_indptr[1:] - split_indptr[:-1]).long()
    if int(split_indptr[-1]) == B and out is not None and out.dtype != data.dtype:
        got = data.view(out.dtype).reshape(-1)[: B * H * Lv].view(B, H, Lv).float()
    else:
        S = data.shape[1]
        mgc = _mla_mgc(H, q.dtype, kv.dtype)
        nvalid = torch.minimum(nsplit, (lens + mgc - 1) // mgc)
        vsc = g("valid_split_count")
        if S > 1 and vsc is not None:
            nvalid = torch.minimum(nvalid, vsc.long())
        ok = torch.arange(S, device=dev)[None, :] < nvalid[:, None]  # (B, S)
        l = lse[..., 0].float().masked_fill(~ok[:, :, None], float("-inf"))  # (B, S, H)
        m = l.amax(1, keepdim=True)
        w = torch.exp(l - torch.where(torch.isfinite(m), m, torch.zeros_like(m)))
        w = torch.where(ok[:, :, None], w, torch.zeros_like(w))
        num = (w[..., None] * torch.where(ok[:, :, None, None], data.float(),
                                          torch.zeros((), device=dev))).sum(1)
        den = w.sum(1)[..., None]
        got = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
    return got, {"q": q, "kv": kv, "lens": lens, "kv_indptr": kv_indptr, "idx": idx, "rows": rows,
                 "scale": scale, "B": B, "H": H, "Lv": Lv, "D": D}


def _mla_decode_stage1(func, a, kw):
    """The value check of aiter.mla_decode_stage1_asm_fwd: `got` (mla_stage1_got: what aiter's stage 2
    merges from the kernel's splits) against softmax(Q K^T * scale) V in fp32 over the same rows."""
    got, m = mla_stage1_got(func, a, kw)
    q, kv_indptr, idx, rows, lens = m["q"], m["kv_indptr"], m["idx"], m["rows"], m["lens"]
    B, H, Lv, scale, dev = m["B"], m["H"], m["Lv"], m["scale"], q.device
    # ref: padded batch, chunked so the score tensor stays ~0.5 GB
    lmax = int(lens.max()) if B else 0
    ref = torch.zeros(B, H, Lv, dtype=torch.float32, device=dev)
    ar = torch.arange(lmax, device=dev)
    step = max(1, (1 << 27) // max(1, H * lmax))
    for s in range(0, B, step):
        e = min(B, s + step)
        valid = ar[None, :] < lens[s:e, None]                         # (b, L)
        pos = (kv_indptr[s:e, None].long() + ar[None, :]).clamp_max(max(idx.numel() - 1, 0))
        k = rows[idx.long()[pos]].float()                             # (b, L, D)
        sc = torch.einsum("bhd,bld->bhl", q[s:e].float(), k) * scale
        sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.nan_to_num(sc.softmax(-1))                          # empty request -> 0
        ref[s:e] = torch.einsum("bhl,blc->bhc", p, k[..., :Lv])

    return {"got": got, "ref": ref, "floor": 3e-2 if q.dtype != torch.float32 else 1e-3,
            "where": "splitData"}


VALUE_CHECKS = {
    "aiter.mla_decode_stage1_asm_fwd.default": _mla_decode_stage1,
}
# NUMERIC_DOCTOR_VALUE_CHECKS=0: none of them -- what the generic checks (the rerun check) find on
# their own, e.g. a blind validation on an op whose value check was written for it.
if os.environ.get("NUMERIC_DOCTOR_VALUE_CHECKS", "1") == "0":
    VALUE_CHECKS = {}


def value_check(op_name, func, args, kwargs):
    """{"got", "ref", "floor", "where"} for `op_name`'s finite output (see VALUE_CHECKS)."""
    with torch.no_grad():
        return VALUE_CHECKS[op_name](func, tuple(args), dict(kwargs))


def value_rel_err(op_name, func, args, kwargs):
    """(max|got - ref| / max|ref|, floor, where), or None when the variant is not modelled
    or the reference itself is not finite."""
    try:
        r = value_check(op_name, func, args, kwargs)
    except NotImplementedError:
        return None
    got, ref = r["got"], r["ref"]
    scale = float(ref.abs().amax()) if ref.numel() else 0.0
    if not math.isfinite(scale) or got.shape != ref.shape:
        return None
    return float((got - ref).abs().amax()) / max(scale, 1e-30), r["floor"], r["where"]


def pick(ref, func, where):
    """The tensor of `ref` matching a capture's `where` ("return", "arg<i>", "kw:<name>"), or None."""
    if where == "return":
        return ref.get("return")
    names = [a.name for a in func._schema.arguments]
    if where.startswith("arg"):
        i = int(where[3:])
        return ref.get(names[i]) if i < len(names) else None
    return ref.get(where.split(":", 1)[1])
