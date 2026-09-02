#!/usr/bin/env python3
"""Filter/trim a downloaded weka_trace corpus to fit a model, then re-emit it as
per-session JSON files aiperf can replay via `--input-file <dir>`.

The HuggingFace weka corpora ship as a single `traces.jsonl` (one session per
line) whose per-session schema matches what gen_agentx_profile.py emits:
    {id, models, block_size, hash_id_scope, requests:[{t, in, out, hash_ids, ...}]}
This reads that download (a directory holding traces.jsonl and/or per-session
*.json files, or a single .jsonl/.json file) and writes filtered
session_XXXXX.json files into --out-dir.

Filters, applied IN ORDER:
  1. --max-turns N: TRUNCATE each session to its first N requests (keeps the
                    growing-prefix structure intact).
  2. --max-isl N  : PER-TURN drop -- discard a session if ANY of its (already
                    truncated) turns' `in` exceeds N (it won't fit the window).
                    Applied AFTER truncation so "trim to first N turns, then keep
                    what fits" works for growing-prefix corpora whose late turns
                    always exceed a small window. No token trim.
  3. --sample N   : RANDOM subset of N sessions with a fixed seed=42
                    (reproducible) when more than N remain; else keep all.

An empty result is an error (exit 1) so an over-aggressive filter fails loudly.

Pure Python stdlib only (json/os/sys/random/glob).
"""
import json, os, sys, random, glob


def _argval(it, flag):
    try:
        return next(it)
    except StopIteration:
        sys.stderr.write(f"[filter_weka_corpus] {flag} requires a value\n")
        raise SystemExit(2)


def _iter_file(path):
    if path.endswith(".jsonl"):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path) as fh:
            obj = json.load(fh)
        if isinstance(obj, list):
            for s in obj:
                yield s
        else:
            yield obj


def _iter_sessions(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) + \
                sorted(glob.glob(os.path.join(path, "*.json")))
        for f in files:
            yield from _iter_file(f)
    else:
        yield from _iter_file(path)


def _max_turn_isl(session):
    m = 0
    for req in session.get("requests", []) or []:
        v = req.get("in")
        if isinstance(v, int) and v > m:
            m = v
    return m


def filter_corpus(sessions, max_isl=None, max_turns=None, sample=None):
    out = []
    for s in sessions:
        if max_turns is not None:
            s = dict(s)
            s["requests"] = list(s.get("requests", []) or [])[:max_turns]
        if max_isl is not None and _max_turn_isl(s) > max_isl:
            continue
        out.append(s)
    if sample is not None and len(out) > sample:
        out = random.Random(42).sample(out, sample)
    return out


def main(argv):
    inp = out_dir = None
    max_isl = max_turns = sample = None
    it = iter(argv)
    for a in it:
        if a == "--input":
            inp = _argval(it, a)
        elif a == "--out-dir":
            out_dir = _argval(it, a)
        elif a == "--max-isl":
            max_isl = int(_argval(it, a))
        elif a == "--max-turns":
            max_turns = int(_argval(it, a))
        elif a == "--sample":
            sample = int(_argval(it, a))
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            sys.stderr.write(f"[filter_weka_corpus] unknown arg: {a}\n")
            return 2
    if not inp or not out_dir:
        sys.stderr.write("usage: filter_weka_corpus.py --input SRC --out-dir DIR "
                         "[--max-isl N] [--max-turns N] [--sample N]\n")
        return 2

    sessions = list(_iter_sessions(inp))
    kept = filter_corpus(sessions, max_isl, max_turns, sample)
    if not kept:
        sys.stderr.write(
            f"[filter_weka_corpus] filter too aggressive: 0 sessions from {len(sessions)} "
            f"(max_isl={max_isl}, max_turns={max_turns}, sample={sample})\n")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    for i, s in enumerate(kept):
        with open(os.path.join(out_dir, f"session_{i:05d}.json"), "w") as fh:
            json.dump(s, fh)
    print(f"wrote {len(kept)}/{len(sessions)} sessions -> {out_dir} "
          f"(max_isl={max_isl}, max_turns={max_turns}, sample={sample})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
