#!/usr/bin/env python3
"""Did the replay dispatch the kernel the model did?

    kernel_match.py <replay kernel.<job>.<host>.txt> <model hipBLASLt log> [<log> ...]

The kernel file is the 5b ladder's (replay_verdict.py kernel): its `solution_index:`
and its `bench:` line, whose geometry is the capsule's.  The model logs are
hipBLASLt bench-format call logs (HIPBLASLT_LOG_MASK=32, HIPBLASLT_LOG_FILE=...)
from a run of the model -- one capture trial with the log on is enough: the
solution for a given problem is fixed at the first call.

Prints one line per solution the model used for that geometry, then exactly one of
  KERNEL_MATCH=<idx>          the model used the replay's solution
  KERNEL_MISMATCH=<idx>/<idx> replay/model(s): a replay on another kernel cannot
                              reproduce -- pin the model's (--solution-index) first
  KERNEL_UNKNOWN=<why>        no geometry, no solution index, or no model call of it
Exit status 0 for MATCH, 1 otherwise.  No torch, no GPU: runs on a login node.
"""
import re
import sys
from collections import Counter

# The bench flags that pick the problem; alpha/beta/strides/scales do not.
KEY_FLAGS = ("-m", "-n", "-k", "--lda", "--ldb", "--ldd", "--transA", "--transB",
             "--batch_count", "--a_type", "--b_type", "--d_type", "--compute_type")


def flags(line):
    toks = line.split()
    out = {}
    for i, t in enumerate(toks):
        if t.startswith("-") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[t] = toks[i + 1]
        elif t.startswith("--"):
            out.setdefault(t, "")
    return out


def key(f):
    return tuple(f.get(k) for k in KEY_FLAGS) + ("--bias_vector" in f,)


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    txt = open(argv[1]).read()
    idx = re.search(r"^solution_index:\s*(\d+)", txt, re.M)
    bench = re.search(r"^bench:\s*(hipblaslt-bench .+)$", txt, re.M)
    if not (idx and bench):
        print("KERNEL_UNKNOWN=%s has no solution_index or bench line" % argv[1])
        return 1
    rep_idx, rep_f = idx.group(1), flags(bench.group(1))
    # The replay's bench line pins its solution; the problem key ignores that, and
    # ignores bias: the capsule may be the mm or the addmm form of the same GEMM.
    want = key(rep_f)[:-1]
    print("replay: solution %s for %s" % (rep_idx, " ".join(
        "%s %s" % (k, rep_f.get(k)) for k in KEY_FLAGS if rep_f.get(k) is not None)))
    used = Counter()
    for path in argv[2:]:
        for line in open(path, errors="replace"):
            if not line.startswith("hipblaslt-bench"):
                continue
            f = flags(line)
            if key(f)[:-1] == want:
                used[(f.get("--solution_index", "?"), "--bias_vector" in f)] += 1
    if not used:
        print("KERNEL_UNKNOWN=no call of this geometry in the model log(s)")
        return 1
    for (s, bias), n in used.most_common():
        print("model:  solution %s%s  x%d" % (s, " (bias)" if bias else "", n))
    model_idx = sorted({s for s, _ in used})
    if rep_idx in model_idx and len(model_idx) == 1:
        print("KERNEL_MATCH=%s" % rep_idx)
        return 0
    if rep_idx in model_idx:
        print("KERNEL_MATCH=%s  (the model also used %s for this geometry)"
              % (rep_idx, ",".join(s for s in model_idx if s != rep_idx)))
        return 0
    print("KERNEL_MISMATCH=%s/%s" % (rep_idx, ",".join(model_idx)))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
