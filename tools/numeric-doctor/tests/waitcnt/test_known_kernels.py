"""waitcnt_scan.py against kernels whose minimal fix was measured on hardware (no GPU needed).

    python3 tests/waitcnt/test_known_kernels.py [<artifacts root>]

The listings are investigation artifacts (not in git): a missing one is SKIP, not FAIL.
  hipBLASLt solution 483266 (2026-11-23): lgkmcnt0_subset@set{0x2A8D9D0 0x2A8F008 0x2A90574} CLEAN, @rest NAN
  aiter MLA stage 1 (2026-11-23): vmcnt_minus1_each@0x2848 CLEAN, the 13 other vmcnt waits NAN
  hipBLASLt solution 344952 (MT64x128x64): the scan's sites predicted BLIND, then measured 2026-11-25
    (4 GPUs x 300k chain runs, 12 co-tenants): @set CLEAN x4 (143 expected hits), @rest NAN x4
and their fixed kernels (the .after.s of those rows), which must scan clean.
Last line: WAITCNT_KNOWN=PASS|FAIL <checks> <failed> skipped=<n>.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import waitcnt_scan as ws  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "artifacts")
CASES = [
    ("hipBLASLt 483266 stock", "2026-11-23-qwen_tf32_hipblaslt_nan-val/asm/lgkmcnt0.Cijk_*.before.s",
     {"0x2A8D9D0", "0x2A8F008", "0x2A90574"}, set()),
    ("hipBLASLt 483266 fixed", "2026-11-23-qwen_tf32_hipblaslt_nan-val/asm/lgkmcnt0_subset@set.Cijk_*.after.s",
     set(), set()),
    ("aiter MLA stage 1 stock", "2026-11-23-deepseek_mla_aiter_decode_wrong-val/asm/vmcnt0@1572.*.before.s",
     {"0x2848"}, {"0x2848"}),
    ("hipBLASLt 344952 stock (blind prediction, confirmed)",
     "../../artifacts/2026-10-22-debug-nan/asm/lgkmcnt0@344952.Cijk_*.before.s",
     {"0x2516D30", "0x25177A0", "0x2518144"}, set()),
    ("aiter MLA stage 1 fixed",
     "2026-11-23-deepseek_mla_aiter_decode_wrong-val/asm/vmcnt_minus1_each@0x2848@1572.*.after.s", set(), set()),
]
N, FAILS, SKIP = [0], [], [0]
for name, pat, want_h1, want_h3 in CASES:
    files = glob.glob(os.path.join(ROOT, pat))
    if not files:
        print("SKIP  %s  (no %s)" % (name, pat))
        SKIP[0] += 1
        continue
    insns = ws.parse_listing(open(files[0]).read())
    res = ws.scan(insns)
    h1 = {"0x%X" % insns[w].addr for w in res["h1"]}
    h3 = {"0x%X" % insns[w].addr for w in res["h3"]}
    ok = h1 == want_h1 and h3 == want_h3
    N[0] += 1
    print("%s  %s  H1=%s H3=%s" % ("PASS" if ok else "FAIL", name, sorted(h1), sorted(h3)))
    if not ok:
        FAILS.append(name)
print("WAITCNT_KNOWN=%s %d %d skipped=%d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS), SKIP[0]))
sys.exit(1 if FAILS else 0)
