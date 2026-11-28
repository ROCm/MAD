"""Unit tests of the twin machinery that need no framework and no GPU (T8).

    python3 tests/twins/test_unit.py        (any Python with torch; CPU tensors only)

  registry   framework twins are skipped cleanly when vLLM / SGLang / aiter are absent; has() / owns_outputs()
  rel_err    the comparison: per-output floors, the output furthest past ITS floor named, kernel non-finite
             positions left to the NaN path, a twin non-finite where the kernel is finite never judged
  fp64 src   inductor_ref.fp64_source: signature pointers to fp64, tl.sqrt_rn -> libdevice.sqrt, 16-bit casts
             removed (finiteness reference) or kept (value reference), tl.dot refused
Last line: TWIN_UNIT=PASS|FAIL <n> <failed>.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import torch  # noqa: E402

import inductor_ref  # noqa: E402
import twins  # noqa: E402

FAILS, N = [], [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


class _Arg:
    def __init__(self, name):
        self.name, self.alias_info = name, None


class _Schema:
    def __init__(self, names):
        self.arguments = [_Arg(n) for n in names]


class FakeOp:
    """Enough of an OpOverload for the adapters' argument lookup."""
    def __init__(self, names):
        self._schema = _Schema(names)


def registry():
    have = {}
    for mod in ("vllm", "sglang", "aiter"):
        try:
            __import__(mod)
            have[mod] = True
        except Exception:                                   # noqa: BLE001
            have[mod] = False
    check("registry: has() for seeded ops", all(twins.has(o) for o in (
        "_C.rms_norm.default", "aiter.rmsnorm2d_fwd.default", "aiter.mla_decode_stage1_asm_fwd.default",
        "inductor.triton_poi_fused_add_0.s0123456789", "vllm_ir.rms_norm.default")))
    check("registry: no twin for an unknown op", not twins.has("aten.mm.default"))
    check("registry: MLA owns its outputs, RMSNorm does not",
          twins.owns_outputs("aiter.mla_decode_stage1_asm_fwd.default")
          and not twins.owns_outputs("aiter.rmsnorm2d_fwd.default"))
    if not have["vllm"]:
        f = FakeOp(["result", "input", "weight", "epsilon"])
        x = torch.randn(4, 8)
        try:
            twins.check("_C.rms_norm.default", f, (torch.empty_like(x), x, torch.ones(8), 1e-6), {})
            ok = False
        except NotImplementedError as e:
            ok = "vllm" in str(e)
        check("registry: vLLM twin skipped (NotImplementedError) without vLLM", ok)
        check("registry: rel_err is None without the framework",
              twins.rel_err("_C.rms_norm.default", f, (torch.empty_like(x), x, torch.ones(8), 1e-6), {}) is None)


def _with_entry(name, fn):
    twins.REGISTRY[name] = [("unit", "native", fn)]


def rel_err():
    f = FakeOp(["x"])
    a32, a16 = torch.randn(64, 32), torch.randn(64, 32).to(torch.bfloat16)
    # two outputs, fp32 exact and bf16 with plain rounding vs an fp64 ref
    _with_entry("unit.two.default", lambda func, args, kw, pre, out: {
        "got": {"o32": a32, "o16": a16}, "ref": {"o32": a32.double(), "o16": a16.double() * (1 + 3e-3)}})
    r = twins.rel_err("unit.two.default", f, (), {})
    check("rel_err: bf16 output judged at bf16's floor, not fp32's", r is not None and r[0] <= r[1] and r[2] == "o16",
          r)
    _with_entry("unit.bad32.default", lambda func, args, kw, pre, out: {
        "got": {"o32": a32 * 1.01, "o16": a16}, "ref": {"o32": a32.double(), "o16": a16.double()}})
    r = twins.rel_err("unit.bad32.default", f, (), {})
    check("rel_err: the fp32 output 1% off is the one named, past its floor", r is not None and r[2] == "o32"
          and r[0] > r[1], r)
    g = a32.clone()
    g[0, 0] = float("nan")
    _with_entry("unit.nan.default", lambda func, args, kw, pre, out: {"got": {"o": g}, "ref": {"o": a32.double()}})
    r = twins.rel_err("unit.nan.default", f, (), {})
    check("rel_err: a kernel NaN is skipped (the NaN path's), the rest compared", r is not None and r[0] < 1e-6, r)
    ref = a32.double().clone()
    ref[1, 1] = float("nan")
    _with_entry("unit.twinnan.default", lambda func, args, kw, pre, out: {"got": {"o": a32}, "ref": {"o": ref}})
    check("rel_err: a twin non-finite where the kernel is finite is not judged",
          twins.rel_err("unit.twinnan.default", f, (), {}) is None)
    _with_entry("unit.shape.default", lambda func, args, kw, pre, out: {"got": {"o": a32}, "ref": {"o": a32[:3]}})
    check("rel_err: shape mismatch is not judged", twins.rel_err("unit.shape.default", f, (), {}) is None)
    _with_entry("unit.floor.default", lambda func, args, kw, pre, out: {
        "got": {"o": a32 * 1.02}, "ref": {"o": a32.double()}, "floor": 3e-2})
    r = twins.rel_err("unit.floor.default", f, (), {})
    check("rel_err: an adapter's floor overrides the dtype's", r is not None and r[1] == 3e-2 and r[0] <= r[1], r)


SRC = """
@triton_heuristics.pointwise(size_hints={'x': 64}, filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'scale': 'fp32', 'XBLOCK': 'constexpr'}, 'device': 0},
    inductor_meta={'kernel_name': 'k'})
@triton.jit
def k(in_ptr0, in_ptr1, out_ptr0, xnumel, scale, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + x0).to(tl.float32)
    tmp1 = tl.sqrt_rn(tmp0)
    tmp2 = tmp1.to(tl.bfloat16)
    tmp3 = tmp2.to(tl.float32) * scale
    tl.store(out_ptr0 + x0, tmp3, None)
"""


def fp64_src():
    s = inductor_ref.fp64_source(SRC)
    check("fp64 src: pointer signature promoted", "'in_ptr0': '*fp64'" in s and "'in_ptr1': '*fp64'" in s
          and "'out_ptr0': '*fp64'" in s and "'xnumel': 'i32'" in s, s.split("triton_meta=")[1][:160])
    check("fp64 src: fp32 scalar promoted", "'scale': 'fp64'" in s)
    check("fp64 src: tl.sqrt_rn -> libdevice.sqrt", "libdevice.sqrt(" in s and "tl.sqrt_rn(" not in s)
    check("fp64 src: casts removed for the finiteness reference", "tl.bfloat16" not in s and "tl.float32" not in s)
    k = inductor_ref.fp64_source(SRC, keep_casts=True)
    check("fp64 src: keep_casts keeps the 16-bit rounding, widens fp32", "to(tl.bfloat16)" in k
          and "tl.float32" not in k and "to(tl.float64)" in k)
    try:
        inductor_ref.fp64_source(SRC.replace("tmp1 = tl.sqrt_rn(tmp0)", "tmp1 = tl.dot(tmp0, tmp0)"))
        ok = False
    except NotImplementedError:
        ok = True
    check("fp64 src: tl.dot refused", ok)


if __name__ == "__main__":
    registry()
    rel_err()
    fp64_src()
    print("TWIN_UNIT=%s %d %d" % ("FAIL" if FAILS else "PASS", N[0], len(FAILS)))
