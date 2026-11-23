#!/usr/bin/env python3
"""Accuracy of a served model on task sets, through its OpenAI-compatible endpoint.

A task set is a JSONL file, one item per line: {"id", "prompt", "answer", "match"} with match
"exact" (case-insensitive, trimmed), "contains" or "regex" (answer is the pattern). Each set is
asked at temperature 0 and summarised in one line per set, which parse_to_csv.py --accuracy reads:

    ACCURACY task=<set> score=<correct/n> correct=<c> n=<n> errors=<requests that failed>

Env: ACC_URL (chat completions URL), ACC_MODEL, ACC_TASKS (comma-separated files), ACC_CONCURRENCY
(8), ACC_MAX_TOKENS (64), ACC_TIMEOUT (300 s per request). Standard library only, as the
containers have it.
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.request


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def correct(item, reply):
    want, got, how = str(item["answer"]), reply.strip(), item.get("match", "exact")
    if how == "contains":
        return want.lower() in got.lower()
    if how == "regex":
        return re.search(want, got) is not None
    return got.lower() == want.strip().lower()


def run_set(path, url, model, conc, max_tokens, timeout):
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    ok = err = 0
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(ask, url, model, it["prompt"], max_tokens, timeout): it for it in items}
        for fut in cf.as_completed(futs):
            it = futs[fut]
            try:
                reply = fut.result()
            except Exception as e:  # noqa: BLE001 -- a failed request is counted, never scored
                err += 1
                print(f"[accuracy] {it.get('id')}: request failed: {e}")
                continue
            good = correct(it, reply)
            ok += good
            print(f"[accuracy] {it.get('id')}: {'ok' if good else 'wrong'} ({reply.strip()[:60]!r})")
    name = os.path.splitext(os.path.basename(path))[0]
    n = len(items)
    print(f"ACCURACY task={name} score={ok / n if n else 0:.4f} correct={ok} n={n} errors={err}")
    return err == 0 and n > 0


def main():
    url, model = os.environ["ACC_URL"], os.environ["ACC_MODEL"]
    sets = [p for p in os.environ["ACC_TASKS"].split(",") if p]
    ok = [run_set(p, url, model, int(os.environ.get("ACC_CONCURRENCY", "8")),
                  int(os.environ.get("ACC_MAX_TOKENS", "64")), float(os.environ.get("ACC_TIMEOUT", "300")))
          for p in sets]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
