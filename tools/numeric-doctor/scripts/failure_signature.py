#!/usr/bin/env python3
"""How an output is wrong, not only that it is (debug-numeric: replays, native candidates, step 7).

    from failure_signature import signature
    sig = signature(out, ref, prev=None)     # out: the op's output; ref: a higher-precision reference
    print(sig["line"])                       # SIGNATURE <values> | <where> | <reference> [| <reading>]

Different mechanisms leave different marks, and a verdict that only says "non-finite" or
"wrong" throws them away:

  values     NaN / +-Inf / saturated at a dtype's max (a guard that clamps fired) / zero where the
             reference is not (unwritten, dropped) / garbage (|out| far beyond the reference:
             uninitialised or stale memory) / the previous run's value (a race, a stale buffer) /
             finite but off by a small relative error (precision)
  where      how many rows and columns carry it, the longest contiguous run along the last dim and
             the alignment it starts on (a 32-wide aligned run: a block of a blocked kernel or codec;
             whole rows: a tile or a token)
  reference  at the bad positions: finite or not, within the output dtype's range or not, and above
             which narrower float's max (fp16, fp8): a value the output dtype holds but a narrower
             intermediate cannot -- a cast or packing inside the op
  reading    the mechanism the marks point to, as a hypothesis to test (step 7), never a verdict

Model- and op-independent: tensors and dtypes only.  CPU or GPU tensors; nothing is moved
back to the device.  `python3 failure_signature.py <pt>` reads {"out", "ref"[, "prev"]} from a file.
"""
import sys

try:
    import torch
except ImportError:                     # --help where there is no torch (a login node)
    if __name__ == "__main__":
        print(__doc__)
        sys.exit(0 if "-h" in sys.argv or "--help" in sys.argv else 2)
    raise

# narrower float formats an op may convert through: name, largest finite value
NARROW = (("fp16", 65504.0), ("fp8_e5m2", 57344.0), ("fp8_e4m3fn", 448.0), ("fp8_e4m3fnuz", 240.0))
GARBAGE = 1e6           # |out| this many times the reference's scale: not arithmetic on these inputs


def _maxes():
    m = [(str(d).replace("torch.", ""), torch.finfo(d).max)
         for d in (torch.float16, torch.bfloat16, torch.float32)]
    return m + [n for n in NARROW if n[0] != "fp16"]


def _runs(mask2d):
    """(longest contiguous run along the last dim, the alignments its runs start on)."""
    best, starts = 0, []
    m = mask2d.to(torch.int8).cpu()
    for r in m.nonzero()[:, 0].unique().tolist()[:256]:
        row = m[r].tolist()
        i = 0
        while i < len(row):
            if row[i]:
                j = i
                while j < len(row) and row[j]:
                    j += 1
                best = max(best, j - i)
                starts.append(i)
                i = j
            else:
                i += 1
    align = 1
    for a in (128, 64, 32, 16, 8):
        if starts and all(s % a == 0 for s in starts):
            align = a
            break
    return best, align


def signature(out, ref=None, prev=None, rtol=None):
    """dict with the counts and a one-line `line`.  `ref` (same shape, any float dtype) and `prev`
    (the same call's output from an earlier run) are optional; without `ref` only the value and
    position marks of non-finite outputs are given."""
    o = out.detach()
    of = o.double()
    r = ref.detach().double().to(of.device) if ref is not None else None
    fin = torch.isfinite(of)
    s = {"numel": o.numel(), "nan": int(torch.isnan(of).sum()), "posinf": int((of == float("inf")).sum()),
         "neginf": int((of == float("-inf")).sum())}
    bad = ~fin
    tol = rtol if rtol is not None else {torch.float16: 1e-2, torch.bfloat16: 3e-2}.get(o.dtype, 1e-4)
    if r is not None and r.shape == of.shape:
        rfin = torch.isfinite(r)
        scale = float(r[rfin].abs().max()) if rfin.any() else 1.0
        err = (of - r).abs()
        wrong = fin & rfin & (err > tol * torch.clamp(r.abs(), min=scale * 1e-3))
        bad = bad | wrong
        s["wrong_finite"] = int(wrong.sum())
        sat = []
        for name, mx in _maxes():
            hit = fin & ((of.abs() - mx).abs() <= mx * 2 ** -10) & (r.abs() > mx * (1 + 2 ** -9))
            if int(hit.sum()):
                sat.append("%s max x%d" % (name, int(hit.sum())))
        s["saturated"] = sat
        s["zero"] = int((fin & (of == 0) & (r.abs() > scale * 1e-3) & wrong).sum())
        s["garbage"] = int((fin & (of.abs() > GARBAGE * max(scale, 1e-30))).sum())
        rb = r[bad]
        s["ref_at_bad"] = {"n": int(bad.sum()), "finite": bool(torch.isfinite(rb).all()) if rb.numel() else None,
                           "max_abs": float(rb[torch.isfinite(rb)].abs().max()) if torch.isfinite(rb).any() else None}
        dmax = torch.finfo(o.dtype).max if o.is_floating_point() else None
        s["ref_beyond_dtype"] = int((rfin & (r.abs() > dmax)).sum()) if dmax else 0
        narrow = [(n, mx) for n, mx in NARROW if dmax and mx < dmax and torch.isfinite(rb).any()
                  and float(rb[torch.isfinite(rb)].abs().max()) > mx]
        s["narrower_than_ref"] = [n for n, _ in narrow]
    if prev is not None and prev.shape == o.shape:
        p = prev.detach().double().to(of.device)
        s["stale"] = int((bad & ((of == p) | (torch.isnan(of) & torch.isnan(p)))).sum())
    # where
    nb = int(bad.sum())
    s["bad"] = nb
    if nb and o.dim() >= 2:
        m2 = bad.reshape(-1, o.shape[-1])
        rows = int(m2.any(1).sum())
        cols = int(m2.any(0).sum())
        run, align = _runs(m2)
        rrun, ralign = _runs(m2.any(1).reshape(1, -1))          # whole bad rows: blocks of rows (tiles)
        s["where"] = {"rows": rows, "of_rows": m2.shape[0], "cols": cols, "of_cols": m2.shape[1],
                      "longest_run": run, "run_align": align, "row_run": rrun, "row_align": ralign}
    # the line
    vals = []
    if s["nan"]:
        vals.append("nan=%d" % s["nan"])
    if s["posinf"] or s["neginf"]:
        vals.append("inf=%d (+%d/-%d)" % (s["posinf"] + s["neginf"], s["posinf"], s["neginf"]))
    for k in ("wrong_finite", "zero", "garbage", "stale"):
        if s.get(k):
            vals.append("%s=%d" % (k, s[k]))
    if s.get("saturated"):
        vals.append("saturated at " + ", ".join(s["saturated"]))
    parts = ["SIGNATURE " + (" ".join(vals) or "no bad element") + " of %d" % s["numel"]]
    w = s.get("where")
    if w:
        parts.append("where: %d/%d rows, %d/%d cols, runs up to %d along the last dim%s; rows in runs up to %d%s"
                     % (w["rows"], w["of_rows"], w["cols"], w["of_cols"], w["longest_run"],
                        ", %d-aligned" % w["run_align"] if w["run_align"] > 1 else "", w["row_run"],
                        ", %d-aligned" % w["row_align"] if w["row_align"] > 1 else ""))
    rab = s.get("ref_at_bad")
    if rab and rab["n"]:
        txt = "reference at them: %s" % ("finite" if rab["finite"] else "NON-FINITE too")
        if rab["max_abs"] is not None:
            txt += ", max |ref| %.4g" % rab["max_abs"]
        if s.get("ref_beyond_dtype"):
            txt += ", beyond %s's range" % str(o.dtype).replace("torch.", "")
        elif s.get("narrower_than_ref"):
            txt += ", within %s but above %s max" % (str(o.dtype).replace("torch.", ""),
                                                    "/".join(s["narrower_than_ref"]))
        parts.append(txt)
    reading = reading_of(s, o.dtype)
    if reading:
        parts.append("reading: " + reading)
    s["line"] = " | ".join(parts)
    return s


def reading_of(s, dtype):
    """The mechanism the marks suggest -- a hypothesis for step 7, not a verdict."""
    rab = s.get("ref_at_bad") or {}
    out = []
    if rab.get("finite") is False:
        return "the true result is non-finite too: the arithmetic (inputs / model), not the kernel"
    if s.get("ref_beyond_dtype"):
        return "the true result does not fit the output dtype: overflow by design, not a kernel fault"
    if (s["posinf"] + s["neginf"]) and s.get("narrower_than_ref"):
        out.append("Inf where the true value fits %s but not %s: a conversion to %s inside the op with no "
                   "saturation on this path (is a clamp/saturation mode present in the source, and in THIS "
                   "arch's binary?)" % (str(dtype).replace("torch.", ""), s["narrower_than_ref"][0],
                                        s["narrower_than_ref"][0]))
    if s.get("saturated"):
        out.append("values pinned at a dtype max: a saturation guard fired -- finite but wrong")
    if s.get("stale"):
        out.append("bad elements equal the previous run's: stale data (a race or a missing write)")
    if s.get("garbage"):
        out.append("magnitudes no arithmetic on these inputs gives: uninitialised or foreign memory")
    if s.get("zero"):
        out.append("zeros where the result is not: unwritten / dropped stores")
    w = s.get("where") or {}
    if w.get("run_align", 1) >= 8 and w.get("longest_run", 0) >= 8 and w.get("longest_run") < w.get("of_cols", 0):
        out.append("bad elements come in %d-aligned runs of up to %d: one block of a blocked kernel or "
                   "codec goes bad together (a per-block scale, a tile)" % (w["run_align"], w["longest_run"]))
    if w.get("row_align", 1) >= 8 and w.get("row_run", 0) >= 8 and w.get("rows", 0) < w.get("of_rows", 0):
        out.append("whole rows go bad in %d-aligned blocks of up to %d: a tile / work unit along that dim "
                   "(which ones? a work-distribution or bounds condition)" % (w["row_align"], w["row_run"]))
    if not out and s.get("wrong_finite") and not (s["nan"] or s["posinf"] or s["neginf"]):
        out.append("finite, off beyond the dtype's tolerance: precision / accumulation order, or wrong data")
    return "; ".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 2)
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(signature(d["out"], d.get("ref"), d.get("prev"))["line"])
