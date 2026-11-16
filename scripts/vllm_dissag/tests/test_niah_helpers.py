#!/usr/bin/env python3
"""benchmark_niah.py helpers that only matter when a run goes wrong.

_fail_side decides which leg a failed request is blamed on, and
_collect_logprobs feeds the first-token margin diagnostic. Neither runs on a
clean ladder, so a regression would only show up mid-incident. No cluster.
"""
import os
import sys

for _k in [k for k in os.environ if k.startswith("NIAH_")]:
    del os.environ[_k]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import benchmark_niah as b  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def side(err):
    return b._fail_side(err).split(" ")[0]


def main():
    print("_fail_side")
    check("no error is ok", b._fail_side(None) == "ok" and b._fail_side("") == "ok")
    check("proxy JSON wins over keywords",
          side('HTTP 502 {"fail_side": "decode", "detail": "prefill timed out"}') == "decode")
    check("proxy JSON prefill", side('{"fail_side":"prefill"}') == "prefill")
    check("unreachable upstream is named",
          "upstream unreachable" in b._fail_side('{"fail_side": "prefill", "error": "upstream_unreachable"}'))
    check("prefill keyword", side("prefill 400: min_tokens must be <= max_tokens") == "prefill")
    check("timeout is client-side", side("Read timed out. (read timeout=1800)") == "client-timeout")
    check("EngineCore 500 is decode", side("InternalServerError: EngineCore died") == "decode")
    check("SSE error is decode", side("SSE error object in stream") == "decode")
    check("refused is proxy", side("Connection refused") == "proxy")
    check("unknown falls through", side("something odd") == "proxy-or-unknown")

    print("_collect_logprobs")
    out = []
    b._collect_logprobs({"choices": [{"logprobs": {
        "tokens": [" kang", "aroo"],
        "top_logprobs": [{" cat": -2.88, " kang": -2.50, " G": -2.88}, {"aroo": -0.01}],
    }}]}, out)
    check("one entry per token", [t for t, _ in out] == [" kang", "aroo"], repr(out))
    check("candidates ranked best first", [t for t, _ in out[0][1]][0] == " kang", repr(out[0][1]))
    out = []
    for _ in range(b.LOGPROBS_STEPS + 3):
        b._collect_logprobs({"choices": [{"logprobs": {"tokens": ["x"], "top_logprobs": [{"x": -1.0}]}}]}, out)
    check("capped at NIAH_LOGPROBS_STEPS", len(out) == b.LOGPROBS_STEPS, f"len={len(out)}")
    out = []
    b._collect_logprobs({"choices": [{"logprobs": {"tokens": ["a", "b"], "top_logprobs": [None]}}]}, out)
    check("missing candidates are empty, not an error", out == [("a", []), ("b", [])], repr(out))
    for bad in ({}, {"choices": []}, {"choices": [{}]}, {"choices": [{"logprobs": None}]}, None):
        out = []
        b._collect_logprobs(bad, out)
        check(f"malformed chunk {bad!r} is ignored", out == [])

    if FAILED:
        print(f"\n{len(FAILED)} helper check(s) failed")
        return 1
    print("\nall NIAH helper checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
