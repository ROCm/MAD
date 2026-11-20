#!/usr/bin/env python3
"""Re-grade saved NIAH ``answer=`` lines with order-aware metrics.

Does not re-run the model. Reads ``answer=%r`` (first 400 chars) from niah
logs. Live bag ``found=`` / ``decoys=`` / ``verdict=`` stay the product row;
this prints streak / p@10 / product_pass / junk on the same text.

Usage: regrade_niah.py <niah_log> [more logs ...]
       cat niah_*.log | regrade_niah.py -
"""
import ast
import re
import sys

from niah_score import score_answer

ANSWER_RE = re.compile(r"words=\s*(\d+)\s+seed=(\d+)\s+answer=(.+)$")


def regrade(lines, label=""):
    rows = []
    for line in lines:
        m = ANSWER_RE.search(line.rstrip("\n"))
        if not m:
            continue
        n_words, seed, raw = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        try:
            text = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        rows.append((n_words, seed, score_answer(text)))

    if not rows:
        return rows

    if label:
        print("=== %s ===" % label)
    print(
        "%7s %5s  %6s  %5s  %5s  %5s  %5s  %16s  %8s  %s"
        % ("words", "seed", "streak", "r@10", "p@10", "extra", "lprec",
           "prefix_verdict", "product", "leading")
    )
    for n_words, seed, s in rows:
        lead = ", ".join(s["order"][:8])
        note = ""
        if s["junk"]:
            note = "  junk=%s" % s["junk"]
        print(
            "%7d %5d  %6d  %5.2f  %5.2f  %5d  %5.2f  %16s  %8s  %s%s"
            % (
                n_words,
                seed,
                s["streak"],
                s["r_at_k"],
                s["p_at_k"],
                s["extra"],
                s["list_precision"],
                s["prefix_verdict"],
                s["product_pass"],
                lead,
                note,
            )
        )
        if s["negated"]:
            print("%7s %5s  negated (excluded): %s"
                  % ("", "", ", ".join(s["negated"])))
    return rows


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    if args == ["-"]:
        regrade(sys.stdin.read().splitlines(), "stdin")
        return 0
    for path in args:
        try:
            with open(path) as f:
                regrade(f.read().splitlines(), path)
        except OSError as e:
            print("skip %s: %s" % (path, e), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
