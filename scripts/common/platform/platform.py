#!/usr/bin/env python3
"""The node platform a run uses, from the capability catalog (compat/rocm.yaml `platforms`).

    platform.py shell [--catalog PATH]

prints shell assignments for scripts/common/cluster.sh: the platform (MAD_PLATFORM, else the
catalog's default_platform), its GPUs per node, and -- for a platform the run named -- the
accelerator and GPU count the probes must find (MAD_PLATFORM_ACCELERATOR, MAD_PLATFORM_GPUS).
An unknown MAD_PLATFORM is an error naming the known ones.
"""
import argparse
import os
import shlex
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "..", "..", "..", "compat", "rocm.yaml")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["shell"])
    ap.add_argument("--catalog", default=os.environ.get("MAD_CATALOG") or CATALOG)
    a = ap.parse_args(argv)
    with open(a.catalog) as f:
        catalog = yaml.safe_load(f) or {}
    platforms = catalog.get("platforms") or {}
    named = os.environ.get("MAD_PLATFORM", "")
    name = named or catalog.get("default_platform", "")
    p = platforms.get(name)
    if p is None:
        print(f"unknown platform {name!r}; the catalog has: {', '.join(sorted(platforms))}", file=sys.stderr)
        return 1
    q = shlex.quote
    print(f"_platform_name={q(name)}; _platform_gpus={q(str(p['gpus_per_node']))}")
    if named:
        print(f"export MAD_PLATFORM_ACCELERATOR={q(p['accelerator'])} MAD_PLATFORM_GPUS={q(str(p['gpus_per_node']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
