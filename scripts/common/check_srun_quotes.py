#!/usr/bin/env python3
"""Find apostrophes that silently truncate a single-quoted srun/bash -c body.

The launchers hand a whole script to a compute node as one single-quoted string:

    srun --nodelist="$LIST" bash -c '
    ...many lines...
    '

An apostrophe inside that -- including one in a comment -- closes the string. The
text after it escapes as shell words and `bash -c` receives a TRUNCATED script,
silently. `bash -n` accepts the result: it is valid, just not what was written.

Two runs died this way. One on the word cluster.sh<apostrophe>s, which
surfaced as "unbound variable" naming a variable assigned a few lines above,
because the assignment had run in the other shell. The other on a comment quoting
an error message that itself contained quoted words:

    #   docker: 'docker run' requires at least 1 argument

Parity is NOT the test. That line has two apostrophes and balances, so the earlier
version of this checker passed it -- while the body still ended mid-comment,
`docker run` was never in the executed script, and the job failed after 66 seconds
with no container output at all.

The real question is what sits BETWEEN a closing apostrophe and the reopening one.
Exactly one form is deliberate: the splice idiom '"$VAR"' (or '"${VAR}"'), which
closes the string to interpolate a value from the submitting shell and reopens it.
Anything else between a pair is text that has escaped the string.

Usage: python3 scripts/common/check_srun_quotes.py [paths...]
"""
import glob
import re
import sys

OPEN = re.compile(r"bash\s+-c\s+'\s*$")
SPLICE = re.compile(r'^"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"$')
# What actually truncates the body is UNQUOTED WHITESPACE in the escaped text: it
# splits the shell word, so everything after becomes a separate argument and
# `bash -c` gets a shorter script. Escaped text with no whitespace merges back into
# the same word and the body survives -- which is why `srun bash -c '...'` inside a
# comment is harmless while 'docker run' is fatal. Metacharacters are flagged too:
# unquoted, the batch shell would expand or glob them.
DANGEROUS = re.compile(r'[\s$`*?\[\]|&;<>()]')


def check(path):
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    problems = []
    i = 0
    while i < len(lines):
        if not OPEN.search(lines[i]):
            i += 1
            continue
        start, inq, i = i, True, i + 1
        while i < len(lines) and inq:
            line = lines[i]
            if "'" in line:
                # A line whose first non-blank character is the apostrophe is the
                # deliberate close, with or without trailing args:
                #     ' _ "${MODEL_PATH}" "${PREWARM_JOBS:-4}"; then
                if line.lstrip().startswith("'"):
                    inq = False
                    i += 1
                    continue
                cols = [c for c, ch in enumerate(line) if ch == "'"]
                if len(cols) % 2 == 1:
                    problems.append((i + 1, start + 1, "odd apostrophe", line.strip()[:64]))
                    inq = False
                else:
                    for a, b in zip(cols[0::2], cols[1::2]):
                        inner = line[a + 1:b]
                        if not SPLICE.match(inner) and DANGEROUS.search(inner):
                            problems.append(
                                (i + 1, start + 1,
                                 "escaped text splits the word: %r" % inner[:32],
                                 line.strip()[:64]))
            i += 1
    return problems


def main(argv):
    paths = argv[1:] or sorted(
        set(glob.glob("scripts/**/*.slurm", recursive=True)
            + glob.glob("scripts/**/*.sh", recursive=True)))
    rc = 0
    total = 0
    for path in paths:
        for ln, st, why, txt in check(path):
            print("%s:%d  %s" % (path, ln, why))
            print("    body opened at line %d" % st)
            print("    %s" % txt)
            print("    fix: move the prose above the srun, or drop the apostrophes.")
            total += 1
            rc = 1
    if not rc:
        print("checked %d file(s): no apostrophe truncates an srun body" % len(paths))
    else:
        print("\n%d problem(s)." % total)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
