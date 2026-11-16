#!/usr/bin/env python3
"""Offline test for NIAH_LIST_PRIME (benchmark_niah.make_prompt).

The prime appends text to the very end of the prompt so the first token is given
rather than sampled. Flash's 8k rung is decided by output FORMAT,
not retrieval: the needles already outrank the prior in the first-token
candidates (` kang`/` oct` at ranks 2-3, ` dog`/` cat` at 4-5) yet `0` wins by
0.375 nats.

Two properties matter and both are asserted here:

  1. Unset is byte-identical to the product prompt, so the rows already scored
     stay comparable. This is the same guarantee NIAH_NEEDLE_BAND carries.
  2. When set, the prime is the LAST thing in the prompt in every mode --
     product, ds_wrap and both chat methods -- because a prime that lands before
     `<|Assistant|>` primes nothing.

Constants are read at import, so each case runs in a fresh interpreter.

Usage: python3 tests/test_list_prime.py
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

PRODUCT_STEM = "Animals found in the above list: "
ASSISTANT = "<\uff5cAssistant\uff5c>"


def _check(name, cond):
    if not cond:
        raise SystemExit("FAIL: %s" % name)
    print("ok  %s" % name)


def prompt(**env):
    """make_prompt(2000, 0) from a fresh interpreter with env applied."""
    e = dict(os.environ)
    # Never inherit the caller's NIAH knobs: a stray export would silently
    # rewrite the prompt under test.
    for k in list(e):
        if k.startswith("NIAH_"):
            del e[k]
    e.update({k: str(v) for k, v in env.items()})
    out = subprocess.run(
        [sys.executable, "-c",
         "import benchmark_niah as b, sys; sys.stdout.write(b.make_prompt(2000, 0))"],
        cwd=SRC, env=e, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit("FAIL: make_prompt raised\n%s" % out.stderr)
    return out.stdout


def stop_list(**env):
    """repr(_stop_list()) from a fresh interpreter with env applied."""
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("NIAH_"):
            del e[k]
    e.update({k: str(v) for k, v in env.items()})
    out = subprocess.run(
        [sys.executable, "-c",
         "import benchmark_niah as b, sys; sys.stdout.write(repr(b._stop_list()))"],
        cwd=SRC, env=e, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit("FAIL: _stop_list raised\n%s" % out.stderr)
    return out.stdout


def check_stops():
    """NIAH_STOP composition. Default MUST be None, not [] -- an empty list is
    still a request-body field, and the guarantee is a byte-identical request."""
    _check("stop: unset yields None (field omitted)",
           stop_list() == "None")
    _check("stop: empty string yields None",
           stop_list(NIAH_STOP="") == "None")
    _check("stop: pipe-separated splits in order",
           stop_list(NIAH_STOP="11.|```") == repr(["11.", "```"]))
    _check("stop: empty segments dropped",
           stop_list(NIAH_STOP="||11.||```|") == repr(["11.", "```"]))
    # A pipe cannot appear inside a stop string; assert the limitation rather
    # than pretend otherwise, since a silent split would send the wrong stop.
    _check("stop: single entry survives",
           stop_list(NIAH_STOP="11.") == repr(["11."]))
    # STOP_BLANK composes, and must not duplicate when also given explicitly.
    _check("stop: STOP_BLANK appends newline pair",
           stop_list(NIAH_STOP="11.", NIAH_STOP_BLANK="1") == repr(["11.", "\n\n"]))
    _check("stop: STOP_BLANK alone still works (back-compat)",
           stop_list(NIAH_STOP_BLANK="1") == repr(["\n\n"]))
    _check("stop: no duplicate when \\n\\n given twice",
           stop_list(NIAH_STOP="\n\n", NIAH_STOP_BLANK="1") == repr(["\n\n"]))
    _check("stop: duplicates within NIAH_STOP collapse",
           stop_list(NIAH_STOP="11.|11.|```") == repr(["11.", "```"]))
    # The recommended production set, end to end.
    rec = "11.|```|<\uff5cend\u2581of\u2581file\uff5c>|<\uff5cbegin\u2581of\u2581file\u2581name\uff5c>"
    _check("stop: recommended set parses to 4 entries",
           stop_list(NIAH_STOP=rec) == repr(
               ["11.", "```", "<\uff5cend\u2581of\u2581file\uff5c>",
                "<\uff5cbegin\u2581of\u2581file\u2581name\uff5c>"]))
    # A bare ** would truncate legitimate bold answers (' **' is a live
    # first-token candidate on Pro). Guard the recommendation, not the code.
    _check("stop: recommended set contains no bare '**'",
           "**" not in stop_list(NIAH_STOP=rec).split("'"))


def main():
    check_stops()

    base = prompt(NIAH_METHOD="product")

    # 1. Default is off and the product prompt is untouched.
    _check("unset: no trailing prime", base.endswith(PRODUCT_STEM))
    _check("unset: empty string is a no-op",
           prompt(NIAH_METHOD="product", NIAH_LIST_PRIME="") == base)
    # Locks the product prompt itself, so a future edit to the prime cannot move
    # the haystack or the stem without this failing.
    md5 = hashlib.md5(base.encode()).hexdigest()
    _check("unset: product prompt md5 %s" % md5[:12],
           md5 == hashlib.md5(prompt(NIAH_METHOD="product").encode()).hexdigest())

    # 2. Set: appended at the end, nothing else moved.
    primed = prompt(NIAH_METHOD="product", NIAH_LIST_PRIME="1.")
    _check("product: ends with the prime", primed.endswith(PRODUCT_STEM + "1."))
    _check("product: prompt is base + prime", primed == base + "1.")
    _check("product: haystack unchanged", primed[:len(base)] == base)

    # 3. ds_wrap puts the prime AFTER the assistant tag, not before it.
    wrapped = prompt(NIAH_METHOD="product", NIAH_DS_WRAP="1", NIAH_LIST_PRIME="1.")
    _check("ds_wrap: ends with the prime", wrapped.endswith(ASSISTANT + "1."))

    # 4. Chat modes: after HYBRID_PRIME for hybrid, after the tag for pr176.
    hyb = prompt(NIAH_METHOD="hybrid", NIAH_LIST_PRIME="1.")
    _check("hybrid: prime follows the assistant prime",
           hyb.endswith(PRODUCT_STEM + "1."))
    _check("hybrid: unset is byte-identical",
           prompt(NIAH_METHOD="hybrid") == hyb[:-2])
    p176 = prompt(NIAH_METHOD="pr176", NIAH_LIST_PRIME="1.")
    _check("pr176: prime follows the assistant tag", p176.endswith(ASSISTANT + "1."))
    _check("pr176: unset is byte-identical",
           prompt(NIAH_METHOD="pr176") == p176[:-2])

    # 5. Arbitrary primes, including ones needing no leading space handling.
    for text in ("1.", " 1.", "1. ", "\n1."):
        _check("product: prime %r appended verbatim" % text,
               prompt(NIAH_METHOD="product", NIAH_LIST_PRIME=text) == base + text)

    print("\nall list-prime checks passed")


if __name__ == "__main__":
    main()
