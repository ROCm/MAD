#!/usr/bin/env python3
"""Scan files or directories for content that must not be published.

    python3 public_content_scan.py PATH [PATH ...] [--patterns FILE] [--quiet]

Built-in patterns cover credentials and markers that never belong in public code.
--patterns (or $PUBLIC_CONTENT_DENYLIST) names a PRIVATE file of extra regexes, one
per line, kept outside the repository so the list does not publish what it protects.
Exit 0 = clean, 1 = matches, 2 = usage error.
"""
import argparse
import os
import re
import sys

BUILTIN = [
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}", "API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY", "private key"),
    (r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", "credential assignment"),
    (r"/home/[a-z][a-z0-9_-]{2,}/", "home directory of a person"),
    (r"\[E\d+(?:[,;] ?E?\d+)*\]", "evidence citation"),
    (r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9-]+\.(?:com|org|net|edu)\b", "e-mail address"),
]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", "venv", ".venv"}
SKIP_EXT = {".pyc", ".so", ".pt", ".png", ".jpg", ".gz", ".zst", ".tar", ".whl"}


def load_patterns(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() and not line.lstrip().startswith("#"):
                out.append((line, "denylist"))
    return out


def files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        else:
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for n in names:
                    if os.path.splitext(n)[1] not in SKIP_EXT:
                        yield os.path.join(root, n)


def scan(paths, extra=(), allow=()):
    pats = [(re.compile(p), why) for p, why in list(BUILTIN) + list(extra)]
    hits = []
    for fp in files(paths):
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for no, line in enumerate(text.splitlines(), 1):
            if any(a in line for a in allow):
                continue
            for rx, why in pats:
                m = rx.search(line)
                if m:
                    hits.append((fp, no, why, m.group(0), line.strip()[:140]))
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--patterns", default=os.environ.get("PUBLIC_CONTENT_DENYLIST"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    extra = []
    if a.patterns:
        if not os.path.isfile(a.patterns):
            print("denylist not found: %s" % a.patterns, file=sys.stderr)
            return 2
        extra = load_patterns(a.patterns)
    hits = scan(a.paths, extra)
    if not a.quiet:
        for fp, no, why, m, line in hits:
            print("%s:%d: [%s] %r  | %s" % (fp, no, why, m, line))
    print("PUBLIC_CONTENT=%s (%d match(es), %d private pattern(s))" % ("CLEAN" if not hits else "FOUND", len(hits), len(extra)))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
