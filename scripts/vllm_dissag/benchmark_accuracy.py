#!/usr/bin/env python3
"""20-prompt correctness suite against a live disagg server.

Same prompts and regex grades as explore/kimiK2 correctness_kimi_k2.sh
(factual, math, reasoning, code, language, instruction following), sent the
way MoRIIO routes them: DeepSeek-V4 chat template rendered client-side,
streamed from /v1/completions with an x-request-id.

ACC_THINKING=chat (default) ends the prompt with Assistant + </think>, so the
generation is the answer. ACC_THINKING=thinking ends it with Assistant + <think>
and grades only the text after </think>; give it ACC_MAXTOK=4096 or more,
because an unclosed reasoning block grades as TRUNCATED, not FAIL.

Usage (inside the container, via benchmark_accuracy.sh):
  ACC_URL=http://127.0.0.1:10001 ACC_MODEL=... python3 benchmark_accuracy.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
USER = "<\uff5cUser\uff5c>"
ASSISTANT = "<\uff5cAssistant\uff5c>"
THINK_START = "<think>"
THINK_END = "</think>"

URL = os.environ.get("ACC_URL") or "http://127.0.0.1:10001"
MODEL = os.environ.get("ACC_MODEL") or os.environ.get("MODEL_PATH", "unknown")
THINKING = os.environ.get("ACC_THINKING", "chat").strip().lower()
MAXTOK = int(os.environ.get("ACC_MAXTOK") or (4096 if THINKING == "thinking" else 1024))
TIMEOUT = float(os.environ.get("ACC_TIMEOUT", "900"))
OUT_JSONL = os.environ.get("ACC_JSONL", "")

# (id, category, question, expect regex) -- verbatim from correctness_kimi_k2.sh.
CASES = [
    ("geography1", "factual", "What is the capital of France? Answer in one word.", r"Paris"),
    ("geography2", "factual", "What is the longest river in Africa? Answer with the name only.", r"Nile"),
    ("science1", "factual", "What is the chemical symbol for gold? Answer with the symbol only.", r"\bAu\b"),
    ("science2", "factual", "How many planets are in our solar system? Answer with a number.", r"\b8\b|eight"),
    ("company1", "factual", "Who is the CEO of AMD? Answer with the name only.", r"Lisa Su"),
    ("arith1", "math", "Compute 17 * 24. Give only the number.", r"408"),
    ("arith2", "math", "Compute 144 divided by 12. Give only the number.", r"\b12\b"),
    ("arith3", "math", "What is 15% of 240? Give only the number.", r"\b36\b"),
    ("wordprob1", "reasoning",
     "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost?",
     r"0\.05|5 cents|five cents"),
    ("wordprob2", "reasoning",
     "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?", r"\b150\b"),
    ("logic1", "reasoning",
     "All roses are flowers. Some flowers fade quickly. Does it follow logically that some roses fade quickly? Answer yes or no.",
     r"\bno\b|does not follow|doesn.t follow"),
    ("count1", "reasoning",
     "How many times does the letter r appear in the word strawberry? Answer with a number.", r"\b3\b|three"),
    ("units1", "knowledge", "How many minutes are there in a full day? Answer with a number.", r"1440|1,440"),
    ("code1", "code", "Write a single line of Python that reverses the string s. Output only code.", r"s\[::-1\]"),
    ("code2", "code",
     "Write a Python one-liner that returns the sum of a list called nums. Output only code.", r"sum\(nums\)"),
    ("code3", "code",
     "Write a SQL query returning all columns from the table users where age is greater than 30.",
     r"select \*.*from users.*where.*age.*>.*30"),
    ("translate1", "language", "Translate 'thank you' into Spanish. Answer with the phrase only.", r"gracias"),
    ("instruct1", "following", "Reply with exactly the single word: banana", r"^ *banana *$|\bbanana\b"),
    ("json1", "following",
     'Output only a JSON object with keys "city" set to "Paris" and "country" set to "France". No other text.',
     r'"city".*"Paris".*"country".*"France"'),
    ("common1", "knowledge",
     "If you leave an ice cube on a hot sidewalk, what happens to it? Answer in one short sentence.", r"melt|water"),
]


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def encode_prompt(question, thinking=THINKING):
    tail = THINK_START if thinking == "thinking" else THINK_END
    return BOS + USER + question + ASSISTANT + tail


def split_answer(text, thinking=THINKING):
    """Return (answer, reasoning). Thinking mode keeps only the text after </think>."""
    if thinking != "thinking":
        return text, ""
    if THINK_END in text:
        reasoning, answer = text.split(THINK_END, 1)
        return answer, reasoning
    return "", text


def grade(answer, expect, finish):
    flat = re.sub(r" +", " ", answer.replace("\n", " "))
    if re.search(expect, flat, re.IGNORECASE):
        return "PASS", flat
    if finish == "length" and not flat.strip():
        return "TRUNCATED", flat
    return "FAIL", flat


def _request(prompt, rid):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": MAXTOK,
        "stream": True,
    }
    req = urllib.request.Request(
        URL.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-request-id": rid},
    )
    t0 = time.time()
    text, finish, ctok = "", "none", 0
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    return text, "error", ctok, time.time() - t0, "sse error: %s" % msg
                ch = (obj.get("choices") or [{}])[0]
                text += ch.get("text") or ""
                ctok += 1 if ch.get("text") else 0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                usage = obj.get("usage") or {}
                if usage.get("completion_tokens"):
                    ctok = int(usage["completion_tokens"])
        return text, finish, ctok, time.time() - t0, None
    except Exception as exc:
        return text, "error", ctok, time.time() - t0, str(exc)


def main():
    if THINKING not in ("chat", "thinking"):
        print("Error: ACC_THINKING=%r (chat|thinking)" % THINKING, file=sys.stderr)
        sys.exit(2)
    print("=== accuracy suite (%d prompts) ===" % len(CASES), flush=True)
    print("url=%s  model=%s  thinking=%s  maxtok=%d  timeout=%.0fs  temperature=0"
          % (URL, MODEL, THINKING, MAXTOK, TIMEOUT), flush=True)
    out = open(OUT_JSONL, "w") if OUT_JSONL else None
    counts = {"PASS": 0, "FAIL": 0, "TRUNCATED": 0, "ERROR": 0}
    by_cat = {}
    for i, (cid, cat, q, expect) in enumerate(CASES, 1):
        rid = "acc-%03d-%s" % (i, cid)
        text, finish, ctok, dt, err = _request(encode_prompt(q), rid)
        answer, reasoning = split_answer(text)
        if err:
            verdict, flat = "ERROR", re.sub(r"\s+", " ", answer)
        else:
            verdict, flat = grade(answer, expect, finish)
        counts[verdict] += 1
        p, n = by_cat.get(cat, (0, 0))
        by_cat[cat] = (p + (verdict == "PASS"), n + 1)
        print("[acc] %s %-11s %-10s %-9s %6.1fs ctok=%-5d fin=%-6s x-request-id=%s answer=%r%s"
              % (_utc(), cid, cat, verdict, dt, ctok, finish, rid, flat[:120],
                 "  err=%s" % err if err else ""), flush=True)
        if out:
            out.write(json.dumps({
                "id": cid, "category": cat, "question": q, "expect": expect,
                "answer": flat, "verdict": verdict, "finish": finish, "secs": round(dt, 1),
                "completion_tokens": ctok, "reasoning_chars": len(reasoning), "error": err,
            }) + "\n")
            out.flush()
    if out:
        out.close()
    print("=== accuracy summary ===", flush=True)
    for cat in sorted(by_cat):
        print("  %-10s %d/%d" % (cat, by_cat[cat][0], by_cat[cat][1]), flush=True)
    print("ACCURACY PASS=%d/%d FAIL=%d TRUNCATED=%d ERROR=%d"
          % (counts["PASS"], len(CASES), counts["FAIL"], counts["TRUNCATED"], counts["ERROR"]), flush=True)
    sys.exit(0 if counts["PASS"] > 0 else 1)


if __name__ == "__main__":
    main()
