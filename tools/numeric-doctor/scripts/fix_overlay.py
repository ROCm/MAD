#!/usr/bin/env python3
"""debug-numeric step 8: build a PERSISTENT hipBLASLt library overlay carrying a binary fix
to one Tensile kernel, and read it back.

The mutation sweep (step 7) builds its overlays in temp dirs and throws them away;
a fix has to outlive the job and ship. This applies one selector + payload from
kernel_patch.py (the same registries the sweep uses, so a CLEAN sweep row becomes a
fix with the same bytes), builds the overlay under --out, re-disassembles what the
overlay actually holds, and fails unless exactly the selected sites changed.

Run INSIDE the image the NaN reproduces in (<GPU_EXEC>): the overlay symlinks every
other library file into that image's /opt/rocm tree. Use it with
    HIPBLASLT_TENSILE_LIBPATH=<overlay> TENSILE_LIBPATH=<overlay>
set AFTER any capsule env restore (restore_env_dict() unsets HIPBLAS*/TENSILE_* names
the capture did not have; see step8-fix.md).

    python3 fix_overlay.py --out <ARTIFACT_DIR>/fix-overlay-build \\
        --kernel-file <ARTIFACT_DIR>/replay/G1/kernel.<job>.<host>.txt \\
        [--selector lgkm_wait] [--payload lgkmcnt0] [--addr 0x... ...] [--expect-sites 6]
    -> READBACK=OK|FAIL, OVERLAY=<path>, <out>/overlay.json, and the two files step 8 ships
       in fix/patch/: <out>/kernel.diff (the disassembly before -> after, read back from the
       overlay) and <out>/kernel_name.txt
"""
import argparse
import difflib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("NUMERIC_DOCTOR_USER",
                                  os.path.dirname(os.path.abspath(__file__))))
import kernel_patch as kp                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="work dir; the overlay is built inside it")
    k = ap.add_mutually_exclusive_group(required=True)
    k.add_argument("--kernel", help="full Tensile kernel name")
    k.add_argument("--kernel-file", help="the replay ladder's kernel.*.txt (its `kernel:` line)")
    ap.add_argument("--selector", default="lgkm_wait", help="kernel_patch SELECTORS name")
    ap.add_argument("--payload", default="lgkmcnt0", help="kernel_patch PAYLOADS name")
    ap.add_argument("--addr", action="append", default=[],
                    help="keep only these vaddrs of the selector's sites (repeatable)")
    ap.add_argument("--expect-sites", type=int, help="fail unless this many sites are patched")
    args = ap.parse_args()

    kernel = args.kernel
    if args.kernel_file:
        with open(args.kernel_file) as f:
            kernel = next(l.split()[1] for l in f if l.startswith("kernel:"))
    work = os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)
    # rnan.build_overlay() puts the overlay in tempfile's dir: keep it beside `work` (same
    # filesystem, so OverlayCache.apply()'s os.replace works) and out of node-local /tmp,
    # so the overlay outlives the job.
    tempfile.tempdir = os.path.abspath(args.out)

    t = kp.PatchTarget(kernel, None, None, work)
    sites = t.select(args.selector)
    if args.addr:
        want = {int(a, 16) for a in args.addr}
        sites = [s for s in sites if s.addr in want]
        if {s.addr for s in sites} != want:
            sys.exit("--addr not all among the %s sites: %s"
                     % (args.selector, sorted(hex(a) for a in want - {s.addr for s in sites})))
    if not sites or (args.expect_sites and len(sites) != args.expect_sites):
        sys.exit("selector %s found %d site(s)%s" % (args.selector, len(sites),
                 ", expected %d" % args.expect_sites if args.expect_sites else ""))
    offsets = t.offsets_for(sites, args.payload)
    cache = kp.OverlayCache(t, args.out)
    overlay = cache.apply(offsets)

    # Read back what the overlay actually holds: exactly the selected sites changed.
    want = sorted(s.addr for s in sites)
    before = {i.addr: i.text for i in t.insns}
    after = {i.addr: i.text for i in cache.disassemble()}
    changed = sorted(a for a in before if after.get(a) != before[a])
    ok = changed == want
    rec = {"overlay": overlay, "patched_co": cache.dst, "kernel": kernel,
           "selector": args.selector, "payload": args.payload,
           "changed": ["0x%X %s -> %s" % (a, before[a], after.get(a)) for a in changed],
           "readback_ok": ok}
    with open(os.path.join(args.out, "overlay.json"), "w") as f:
        json.dump(rec, f, indent=1)
    # The .diff step8-fix.md ships in fix/patch/ -- made from the read-back, so it shows what
    # the overlay holds, not what was asked (2026-11-20: none was written; made by hand)
    fmt = lambda d: ["0x%X  %s\n" % (a, d[a]) for a in sorted(d)]
    with open(os.path.join(args.out, "kernel.diff"), "w") as f:
        f.writelines(difflib.unified_diff(fmt(before), fmt(after), "%s (stock)" % kernel,
                                          "%s (%s:%s)" % (kernel, args.selector, args.payload), n=3))
    with open(os.path.join(args.out, "kernel_name.txt"), "w") as f:
        f.write(kernel + "\n")
    for line in rec["changed"]:
        print("  " + line)
    print("READBACK=%s (%d instruction(s) differ, %d selected)"
          % ("OK" if ok else "FAIL", len(changed), len(want)))
    print("OVERLAY=%s" % overlay)
    print("LIBPATH=%s   (HIPBLASLT_TENSILE_LIBPATH / TENSILE_LIBPATH: the arch dir for a per-arch library)"
          % kp.tensile_libpath(overlay, t.arch))
    print("DIFF=%s" % os.path.join(args.out, "kernel.diff"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
