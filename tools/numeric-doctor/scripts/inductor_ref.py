"""fp64 reference of an Inductor-generated Triton kernel: its own source, every float type promoted to fp64.

An `inductor.<kernel>.s<hash>` op (a torch.compile kernel the capture watched) has no CPU or fp64 kernel, so
the replay's reference read `unavailable` and a deterministic kernel NaN stopped at KERNEL_BUG_UNVERIFIED: step 6
then needed a hand-written eager / fp64 run of the kernel's source nodes (2026-11-20 and 2026-11-23, the router
`sqrt(softmax(z))` kernel, both times by hand).  This builds the reference from the kernel itself:

* the recorded module source with `tl.float32` / `tl.bfloat16` / `tl.float16` -> `tl.float64` in the body, and
  the float pointer and scalar types of `triton_meta`'s signature -> fp64 (the launcher is compiled for that
  signature: a fp64 tensor behind a `*fp32` pointer would be read as fp32 bits);
* every float tensor argument upcast to fp64 (exact), strides kept; out-buffers (out_ptr*) zero-filled;
* `tl.sqrt_rn` (fp32-only) -> `libdevice.sqrt` (ocml's correctly rounded fp64 sqrt);
* compiled with Inductor's own AsyncCompile and launched once on the GPU.

It answers the reference's question -- given exactly these inputs, is a non-finite output ever correct for this
computation? -- the way replay_nan_op's `_cpu64` does for ATen ops: intermediate low-precision casts become
fp64 too, so a `.to(tl.bfloat16)` mid-kernel does not overflow in the reference.  It is the same algorithm in
a wider type, not an independent implementation: a defect of the fp64 lowering of the same `tl.*` call would
be shared (step 7 checks the instruction, `step7-rootcause.md`).  A kernel whose fp64 twin does not compile
(fp64 `tl.dot`, inline asm, a libdevice call with no fp64 variant) raises: callers report `unavailable`.

    reference(desc, args) -> tuple of the written arguments' fp64 values (desc["written"] order), on the CPU
    python3 -m inductor_ref <capsule.pt>   # the twin on a capsule's operands, finite or not per written arg
"""
import re
import sys

import torch

_FLOAT_TL = re.compile(r"\btl\.(?:float32|bfloat16|float16)\b")
_FP32_TL = re.compile(r"\btl\.float32\b")
_SIG_PTR = re.compile(r"'\*(?:fp32|bf16|fp16)'")
_SIG_SCALAR = re.compile(r"(:\s*)'(?:fp32|bf16|fp16)'")


def fp64_source(src, keep_casts=False):
    """The kernel module's source with every float type promoted to fp64 (body and signature)."""
    m = re.search(r"'signature'\s*:\s*\{[^}]*\}", src)
    if m is None:
        raise NotImplementedError("no triton_meta signature in the kernel source")
    sig = _SIG_SCALAR.sub(r"\1'fp64'", _SIG_PTR.sub("'*fp64'", m.group(0)))
    src = src[:m.start()] + sig + src[m.end():]
    if re.search(r"\btl\.dot\b|inline_asm", src):
        raise NotImplementedError("tl.dot / inline asm: no fp64 twin")
    # tl.sqrt_rn is fp32-only; ocml's fp64 sqrt (libdevice.sqrt) is correctly rounded -- and the reference then
    # does not share tl.sqrt_rn's lowering, the one a kernel NaN has implicated before (step7-rootcause.md)
    src = re.sub(r"\btl\.sqrt_rn\(", "libdevice.sqrt(", src)
    # keep_casts: the kernel's explicit 16-bit roundings stay (`.to(tl.bfloat16)` of an intermediate is the model's
    # own arithmetic) and only its fp32 math widens -- the value reference (twins.py).  Without them the twin computes
    # something more exact than the model specifies: an RMSNorm kernel read 3e-3 off a correct fp32 output, a false
    # wrong value (2026-11-24).  Off for the finiteness reference: there a bf16 cast would overflow spuriously.
    return (_FP32_TL if keep_casts else _FLOAT_TL).sub("tl.float64", src)


def _gpu64(t, zero):
    out = torch.empty_strided(t.size(), t.stride(), device="cuda",
                              dtype=torch.float64 if t.is_floating_point() else t.dtype)
    if zero:
        return out.zero_()
    return out.copy_(t.to("cuda").to(out.dtype))


# compiled fp64 twins by (kernel, source, keep_casts) -- or the exception building / launching one raised: the
# capture's twin check calls this on every sampled launch, and looking the twin up through AsyncCompile each time
# made a 100-step run 17x slower at TWIN_EVERY=1 (2026-11-24); a twin that cannot be built is not retried
_TWINS = {}


def _twin_kernel(desc, keep_casts):
    from torch._inductor.async_compile import AsyncCompile
    key = (desc["kernel"], desc.get("src_sha") or hash(desc["src"]), bool(keep_casts))
    k = _TWINS.get(key)
    if k is None:
        try:
            k = AsyncCompile().triton(desc["kernel"], fp64_source(desc["src"], keep_casts), device_str="cuda")
            k = k.result() if hasattr(k, "result") else k
        except Exception as e:                          # noqa: BLE001 - remembered, re-raised below
            k = e
        _TWINS[key] = k
    if isinstance(k, Exception):
        raise NotImplementedError("fp64 twin unavailable (%s: %s)" % (type(k).__name__, str(k)[:80]))
    return k, key


def reference(desc, args, keep_casts=False, on_gpu=False):
    """Launch the fp64 twin of the kernel `desc` (an Inductor descriptor: kernel, src, arg_names, written,
    outbuf) on `args` (the op's positional operands, any device / float dtype); return the written arguments'
    fp64 values, in desc["written"] order: on the CPU (synchronised), or on_gpu=True left on the device in stream
    order (the twin check compares them there)."""
    from torch._C import _cuda_getCurrentRawStream
    k, key = _twin_kernel(desc, keep_casts)
    outbuf = set(desc.get("outbuf") or ())
    a64 = [_gpu64(a, i in outbuf) if isinstance(a, torch.Tensor) else a for i, a in enumerate(args)]
    try:
        k.run(*a64, stream=_cuda_getCurrentRawStream(torch.cuda.current_device()))
    except Exception as e:                              # noqa: BLE001 - a launch failure is not retried either
        _TWINS[key] = e
        raise NotImplementedError("fp64 twin launch failed (%s: %s)" % (type(e).__name__, str(e)[:80]))
    if on_gpu:
        return tuple(a64[i] for i in desc["written"])
    torch.cuda.synchronize()
    return tuple(a64[i].cpu() for i in desc["written"])


def main():
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from capsule_operands import load
    import op_nan_capture as onc
    c = load(sys.argv[1], device="cuda")
    if not c.op.startswith("inductor."):
        sys.exit("not an inductor.* op: %s" % c.op)
    d = onc._TRITON_DESCS[c.op]
    for i, t in zip(d["written"], reference(d, list(c.args))):
        n = int((~torch.isfinite(t)).sum())
        print("INDUCTOR_REF %s %s fp64 non-finite %d of %d, max|.| %.3g"
              % (d["arg_names"][i], tuple(t.shape), n, t.numel(),
                 t[torch.isfinite(t)].abs().max().item() if n < t.numel() else float("nan")))


if __name__ == "__main__":
    main()
