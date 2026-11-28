# debug-numeric step 8: a workaround for a broken Triton op that Inductor emits -- rewrite the text Inductor
# generates for one aten-level op, e.g. torch.sqrt -> libdevice.sqrt instead of tl.sqrt_rn (E52).
#
#   PYTHONPATH=<REPO_ROOT>/scripts/inductor_override \
#   NAN_FIX_INDUCTOR_OVERRIDES='sqrt:tl.sqrt_rn(=libdevice.sqrt(' python3 train.py ...
#
# NAN_FIX_INDUCTOR_OVERRIDES: comma-separated <TritonOverrides method>:<old text>=<new text>; each method's
# generated string has its first <old> replaced by <new>.  Loaded at interpreter start (sitecustomize), so
# neither the model nor torch is edited; prints one `[fix] ...` marker line per override (grep it from every
# trial: a dropped PYTHONPATH reads as a clean pass).  Private Inductor API (torch 2.10
# torch._inductor.codegen.triton.TritonOverrides): re-check after a torch upgrade.  Check the generated code
# too: TORCH_LOGS=output_code shows <new> and no <old>.  An override that fails to apply stops the process
# (NAN_FIX_INDUCTOR_OVERRIDES_STRICT=0: warn and continue stock).  The value is taken verbatim: no quotes
# around it unless a shell (or wave.sh's env field) strips them.
import os
import sys

_spec = os.environ.get("NAN_FIX_INDUCTOR_OVERRIDES", "")
if _spec:
    try:
        from torch._inductor.codegen import triton as _t

        def _make(orig, old, new):
            def op(*args, **kwargs):
                return orig(*args, **kwargs).replace(old, new, 1)
            return op

        for item in filter(None, (x.strip() for x in _spec.split(","))):
            name, sub = item.split(":", 1)
            old, new = sub.split("=", 1)
            orig = getattr(_t.TritonOverrides, name)
            setattr(_t.TritonOverrides, name, staticmethod(_make(orig, old, new)))
            print(f"[fix] inductor {name}: {old} -> {new}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[fix] inductor override NOT applied ({_spec}): {e!r}", file=sys.stderr, flush=True)
        # A fix that silently does not load runs the model stock and reads as "the fix fails" (2026-11-23: a
        # quoted value, five A/B trials void).  Stop the process instead -- the trial scores ERROR, not a
        # verdict.  NAN_FIX_INDUCTOR_OVERRIDES_STRICT=0 keeps the old warn-and-continue.
        if os.environ.get("NAN_FIX_INDUCTOR_OVERRIDES_STRICT", "1") != "0":
            raise SystemExit(f"[fix] stopping: the requested inductor override did not apply ({e!r})")
