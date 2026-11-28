#!/usr/bin/env python3
"""Is a capsule the same bug as a reference capsule?  (debug-numeric step 4, node-count ladder)

    same_site.py <reference capsule.pt> <capsule.pt> [--frames N]

The node-count ladder captures the NaN again on fewer nodes and keeps the smaller count only if
the NaN is born at the same place.  Same place = the same ATen op AND either the same
stack digest (the capture's `identity.stack_digest`: the whole stack, byte for byte) or the
same last N MODEL frames (`capsule_info.model_frames`: torch, stdlib and numeric-doctor frames
dropped -- a different data-parallel degree can route the same model line through other
torch internals, which changes the digest but not the site).  Operand dtypes and shapes are
reported, never required: per-rank shapes stay the same when only the data-parallel degree
shrinks, and a difference is worth a line in the ledger, not a rejection.

Last line: SITE=SAME (stack|frames) | SITE=DIFFERENT, then ref=<op @ frame> new=<op @ frame>.
Exit 0 on SAME, 1 on DIFFERENT, 2 on a usage or load error.  Needs torch (CPU only).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import op_nan_capture as g  # noqa: E402
from capsule_info import model_frames  # noqa: E402


def site(path, nframes):
    try:
        c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:          # an old-format (non-zip) file cannot be mmapped
        c = torch.load(path, map_location="cpu", weights_only=False)
    c = g.upgrade_capsule(c)
    ident = c.get("identity") or {}
    return {
        "op": str(ident.get("op", c.get("op", "?"))),
        "digest": ident.get("stack_digest"),
        "frames": model_frames(c.get("stack", ""), nframes),
        "dtypes": list(ident.get("dtypes") or ()),
        "geometry": ident.get("geometry_key"),
        "rank": (c.get("dist") or {}).get("rank"),
        "world": (c.get("dist") or {}).get("world_size"),
    }


def label(s):
    return "%s @ %s (rank %s/%s)" % (s["op"], s["frames"][-1] if s["frames"] else "?",
                                     s["rank"], s["world"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference")
    ap.add_argument("capsule")
    ap.add_argument("--frames", type=int, default=4, help="model frames compared (default 4)")
    a = ap.parse_args()
    try:
        ref, new = site(a.reference, a.frames), site(a.capsule, a.frames)
    except Exception as e:
        print("SITE=ERROR %s: %s" % (type(e).__name__, e))
        return 2
    for k in ("dtypes", "geometry"):
        if ref[k] != new[k]:
            print("note: %s differ: ref %s, new %s" % (k, ref[k], new[k]))
    how = None
    if ref["op"] == new["op"]:
        if ref["digest"] and ref["digest"] == new["digest"]:
            how = "stack"
        elif ref["frames"] and ref["frames"] == new["frames"]:
            how = "frames"
    if how is None and ref["op"] == new["op"]:
        print("ref frames:\n  " + "\n  ".join(ref["frames"] or ["(none)"]))
        print("new frames:\n  " + "\n  ".join(new["frames"] or ["(none)"]))
    print("SITE=%s ref=%s new=%s" % ("SAME (%s)" % how if how else "DIFFERENT", label(ref), label(new)))
    return 0 if how else 1


if __name__ == "__main__":
    sys.exit(main())
