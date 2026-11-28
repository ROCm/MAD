#!/usr/bin/env python3
"""The op-only copy of a chain capsule: the failing op's own operands, the chain dropped.

    python3 -m capsule_slim <capsule.pt> <out.pt>

A chain capsule (format 3) carries every value of the chain that fed the failing op -- with real
values that is most of its size (2026-11-21, aiter MLA decode: 1.56 GB of a 1.99 GB capsule,
428 MB the op's own operands).  When the op alone reproduces, a vendor needs only those operands:
capture_to_repro.sh packages this copy, verifies it with CAPSULE_MODE=op from an extracted copy,
and ships it when it reproduces, the full capsule kept beside it (repro-<G>-chain).

The copy is a format-2 capsule: every key a single-op replay reads is unchanged; `slim_of` names
the capsule it came from.  Prints SLIM=<out> bytes=<n> from=<bytes>.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import op_nan_capture as onc  # noqa: E402

# the keys _chain_payload adds (rng stays: a few bytes, and a single-op reader ignores it)
CHAIN_KEYS = ("chain", "pool", "roots", "truncation", "config")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    src, dst = sys.argv[1], sys.argv[2]
    cap = onc.upgrade_capsule(torch.load(src, map_location="cpu", weights_only=False))
    if not cap.get("chain"):
        sys.exit("capsule_slim: %s has no chain (format %s): it is already op-only" % (src, cap.get("format")))
    n_ops = len(cap["chain"])
    for k in CHAIN_KEYS:
        cap.pop(k, None)
    cap["format"] = 2
    cap["slim_of"] = {"capsule": os.path.basename(src), "sha256": sha256(src), "chain_ops": n_ops}
    onc._share_bases(cap)
    tmp = dst + ".tmp"
    torch.save(cap, tmp)
    os.replace(tmp, dst)
    print("SLIM=%s bytes=%d from=%d (chain of %d ops dropped)"
          % (dst, os.path.getsize(dst), os.path.getsize(src), n_ops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
