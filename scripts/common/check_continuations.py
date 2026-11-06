#!/usr/bin/env python3
"""Reject a comment placed inside a backslash continued command.

Bash splices a trailing backslash with the next line. When that next line is a
comment, the ``#`` ends the command there and every following line runs on its
own. The result is valid shell, so ``bash -n`` accepts it -- which is why this
reached a cluster:

    -e MODEL_NAME=$MODEL_NAME \\
    # Provenance for the perf CSV...
    -e DOCKER_IMAGE_NAME=... \\

became ``docker run ... -e MODEL_NAME=...`` with no image, and produced

    docker: 'docker run' requires at least 1 argument
    /usr/bin/bash: line 148: -e: command not found

after a 56-second job that had already staged 1.5T of weights and pulled the
image. Two launchers carried it; only the one that happened to be re-run showed
it. This is the same shape as an apostrophe inside an srun body closing the
string early -- a comment that changes what executes.

A continued line that is ITSELF a comment is fine: the backslash is already
inside a comment and splices nothing.

Usage: python3 scripts/common/check_continuations.py [paths...]
Exits 1 and names every offender.
"""
import glob
import sys


def scan(paths):
    bad = []
    for path in paths:
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        for i in range(len(lines) - 1):
            cur = lines[i].rstrip()
            if not cur.endswith("\\") or cur.lstrip().startswith("#"):
                continue
            nxt = lines[i + 1].lstrip()
            if nxt.startswith("#"):
                bad.append((path, i + 2, cur.strip(), nxt))
    return bad


def main(argv):
    paths = argv[1:] or sorted(
        set(
            glob.glob("scripts/**/*.slurm", recursive=True)
            + glob.glob("scripts/**/*.sh", recursive=True)
        )
    )
    bad = scan(paths)
    for path, line, cont, comment in bad:
        print("%s:%d  comment inside a continuation" % (path, line))
        print("    continued: %s" % cont[:72])
        print("    comment:   %s" % comment[:72])
        print("    fix: move the comment above the command.")
    if bad:
        print("\n%d broken continuation(s)." % len(bad))
        return 1
    print("checked %d file(s): no comment breaks a continuation" % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
