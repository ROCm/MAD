#!/usr/bin/env python3
"""Offline checks for benchmark_accuracy.py: prompt encoding and grading."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

import benchmark_accuracy as acc  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


check(len(acc.CASES) == 20, "20 cases")
check(len({c[0] for c in acc.CASES}) == 20, "case ids unique")

p = acc.encode_prompt("Q?", "chat")
check(p == acc.BOS + acc.USER + "Q?" + acc.ASSISTANT + acc.THINK_END, "chat ends Assistant+</think>")
p = acc.encode_prompt("Q?", "thinking")
check(p.endswith(acc.ASSISTANT + "<think>"), "thinking ends Assistant+<think>")

check(acc.split_answer("Paris", "chat") == ("Paris", ""), "chat keeps whole text")
check(acc.split_answer("hmm 42</think> 42", "thinking") == (" 42", "hmm 42"), "thinking grades after </think>")
check(acc.split_answer("still thinking", "thinking") == ("", "still thinking"), "unclosed think -> empty answer")

expect = dict((c[0], c[3]) for c in acc.CASES)
check(acc.grade(" Paris.", expect["geography1"], "stop")[0] == "PASS", "Paris passes")
check(acc.grade("The answer is 0.05 dollars", expect["wordprob1"], "stop")[0] == "PASS", "bat-ball passes")
check(acc.grade("0.10", expect["wordprob1"], "stop")[0] == "FAIL", "bat-ball 0.10 fails")
check(acc.grade("SELECT * FROM users WHERE age > 30;", expect["code3"], "stop")[0] == "PASS", "SQL passes case-insensitive")
check(acc.grade("", expect["arith1"], "length")[0] == "TRUNCATED", "empty + length -> TRUNCATED")
check(acc.grade("", expect["arith1"], "stop")[0] == "FAIL", "empty + stop -> FAIL")
check(acc.grade("yes", expect["logic1"], "stop")[0] == "FAIL", "logic yes fails")

print("%d failed" % fails)
sys.exit(1 if fails else 0)
