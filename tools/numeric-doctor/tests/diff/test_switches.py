"""T1: did each switch of a diff wave change what ran, and did the model still run? Reads a finished diff_wave.sh
directory (no GPU, no torch).

    python3 tests/diff/test_switches.py <ART>/diff/<job> [--expect-inert name,...]

Per flip arm: rc 0 and as many records as A (the model still runs); the shim's `[diff-tap] switch` line in its log
for a switch applied through diff_tap; and its outputs bitwise different from A's in at least one record while A
and A2 are bitwise identical (the switch changed the computation) -- CHANGES; all records identical -- INERT (the
switch did nothing here: wrong name, or the component is not used by this model; --expect-inert for the latter);
A and A2 themselves differ -- UNKNOWN (no bitwise baseline).
Last line: DIFF_SWITCHES=PASS|FAIL <n checks> <n failed>.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import output_diff as od  # noqa: E402

FAILS, N = [], [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def bitwise_diff(ra, rb):
    keys = set(ra) & set(rb)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("kind", "i")}  # noqa: E731
    return sum(1 for k in keys if strip(ra[k]) != strip(rb[k])), len(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--expect-inert", default="")
    args = p.parse_args()
    inert_ok = set(x for x in args.expect_inert.split(",") if x)
    d = args.dir.rstrip("/")
    _, ra = od.load(os.path.join(d, "A", "outputs.jsonl"))
    _, ra2 = od.load(os.path.join(d, "A2", "outputs.jsonl"))
    base, n = bitwise_diff(ra, ra2)
    print("A vs A2: %d of %d records differ bitwise" % (base, n))
    for arm_dir in sorted(x for x in glob.glob(os.path.join(d, "flip_*")) if os.path.isdir(x)):
        arm = os.path.basename(arm_dir)
        name = arm[len("flip_"):]
        log = open(arm_dir + ".log", errors="replace").read()
        out = os.path.join(arm_dir, "outputs.jsonl")
        _, rx = od.load(out) if os.path.isfile(out) else ({}, {})
        check("%s: the model ran (records = A's)" % arm, len(rx) == len(ra), (len(rx), len(ra)))
        envw = re.search(r"DIFF_ARM %s env=\[(.*)\]" % re.escape(arm), log)
        via_tap = envw and re.search(r"\bND_(VLLM|SGLANG|DIFF)_", envw.group(1))
        if via_tap:
            check("%s: diff_tap applied the switch" % arm, "[diff-tap] switch" in log)
        nd, nk = bitwise_diff(rx, ra)
        state = "UNKNOWN" if base else ("CHANGES" if nd else "INERT")
        print("     %s: %s (%d of %d records differ bitwise from A)" % (arm, state, nd, nk))
        if state == "INERT":
            check("%s: the switch changed the computation" % arm, name in inert_ok,
                  "INERT -- expected" if name in inert_ok else "INERT: no record differs from A")
        else:
            check("%s: the switch changed the computation" % arm, True, state)
    print("DIFF_SWITCHES=%s %d %d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
