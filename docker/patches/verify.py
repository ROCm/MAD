#!/usr/bin/env python3
"""Prove each patch in docker/patches/ rebuilds the fork commit it replaces.

For every patched component of every pin set: fetch the upstream commit and the
fork commit (no file contents beyond what the patch touches), apply the patch to
the upstream tree in a scratch index, and require the result to be the fork
commit's tree. Needs git and network access to the upstream repository; GitHub
serves a fork's commits through the repository it was forked from.

  docker/patches/verify.py [--work DIR]
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pins"))
import pins  # noqa: E402



def git(repo, *args, env=None, check=True):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, env=env, check=check).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix="mad-patch-verify.")
    bad = 0
    for n in pins.names():
        for comp, c in pins.load(n)["components"].items():
            for p in c.get("patches") or []:
                repo = os.path.join(work, comp)
                if not os.path.isdir(repo):
                    subprocess.run(["git", "init", "-q", "--bare", repo], check=True)
                git(repo, "fetch", "-q", "--filter=blob:none", c["repo"], c["commit"], c["fork_commit"])
                idx = os.path.join(work, f"index-{os.getpid()}")
                env = dict(os.environ, GIT_INDEX_FILE=idx)
                git(repo, "read-tree", c["commit"], env=env)
                git(repo, "apply", "--cached", "--binary", os.path.join(HERE, p["file"]), env=env)
                tree = git(repo, "write-tree", env=env)
                os.unlink(idx)
                want = git(repo, "rev-parse", f"{c['fork_commit']}^{{tree}}")
                ok = tree == want
                bad += not ok
                print(f"{'ok  ' if ok else 'BAD '} {n} {comp} {p['file']}: {c['commit'][:10]} + patch "
                      f"{'==' if ok else '!='} fork {c['fork_commit'][:10]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
