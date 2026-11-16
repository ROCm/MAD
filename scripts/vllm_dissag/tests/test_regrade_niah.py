#!/usr/bin/env python3
"""regrade_niah.py on saved log lines: parse, skip, and agree with the live scorer.

Re-grading old logs is how rows scored before a scorer fix get corrected, so it
must read exactly the line benchmark_niah.py prints and grade it with the same
score_answer. No cluster.
"""
import contextlib
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

import regrade_niah  # noqa: E402
from niah_score import score_answer  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def live_line(n_words, seed, text):
    """The exact line benchmark_niah.run() prints."""
    return "words=%6d  seed=%d  answer=%r" % (n_words, seed, text)


CLEAN = " elephant, giraffe, kangaroo, penguin, dolphin, tiger, rhinoceros, octopus, crocodile, panda."
PRIOR = " cat, 2. dog, 3. fish, 4. bird, 5. horse"
DENIAL = " kangaroo, penguin, **crocodile** (not found)"


def main():
    log = [
        "[niah] 2026-11-16 header line, not an answer",
        live_line(2000, 0, CLEAN),
        live_line(32000, 0, PRIOR),
        "words=  8000  seed=0  answer=<unterminated",
        live_line(8000, 1, DENIAL),
    ]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rows = regrade_niah.regrade(log, "sample")
    text = out.getvalue()

    check("reads only answer lines", len(rows) == 3, f"rows={len(rows)}")
    check("keeps words and seed", [(w, s) for w, s, _ in rows] == [(2000, 0), (32000, 0), (8000, 1)])
    check("skips an unparsable answer", all(w != 8000 or s != 0 for w, s, _ in rows))
    check("same score as the live scorer", rows[0][2] == score_answer(CLEAN))
    check("clean walk is r@10 1.00", abs(rows[0][2]["r_at_k"] - 1.0) < 1e-9)
    check("farm prior is r@10 0.00", rows[1][2]["r_at_k"] == 0.0)
    check("prints the label", "=== sample ===" in text)
    check("prints a header", "prefix_verdict" in text and "lprec" in text)
    check("reports negated needles", "negated (excluded): crocodile" in text, text)

    with contextlib.redirect_stdout(io.StringIO()) as out:
        rows = regrade_niah.regrade(["no answers here"], "empty")
    check("no rows prints nothing", rows == [] and out.getvalue() == "")

    r = subprocess.run([sys.executable, "regrade_niah.py", "-"], cwd=SRC,
                       input="\n".join(log), capture_output=True, text=True)
    check("CLI reads stdin", r.returncode == 0 and "=== stdin ===" in r.stdout, r.stderr)
    r = subprocess.run([sys.executable, "regrade_niah.py"], cwd=SRC,
                       capture_output=True, text=True)
    check("CLI with no args exits 2", r.returncode == 2)
    r = subprocess.run([sys.executable, "regrade_niah.py", "/nonexistent/niah.log"], cwd=SRC,
                       capture_output=True, text=True)
    check("CLI skips a missing file", r.returncode == 0 and "skip /nonexistent/niah.log" in r.stderr)

    if FAILED:
        print(f"\n{len(FAILED)} regrade check(s) failed")
        return 1
    print("\nall regrade checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
