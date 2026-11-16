#!/usr/bin/env python3
"""The node platform a run uses, from the capability catalog (compat/rocm.yaml `platforms`).

    platform.py shell [--catalog PATH]
    platform.py cluster [--catalog PATH]      the site profile (MAD_CLUSTER / MAD_CLUSTER_FILE)

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


def cluster_shell(catalog):
    """The site profile (MAD_CLUSTER: a catalog entry; MAD_CLUSTER_FILE: a file of that shape) as
    shell lines that set only what the run did not: MAD_PLATFORM, CLUSTER_ARCHETYPE, its env."""
    name, path = os.environ.get("MAD_CLUSTER", ""), os.environ.get("MAD_CLUSTER_FILE", "")
    if not (name or path):
        return 0
    if path:
        with open(path) as f:
            c = yaml.safe_load(f) or {}
        where = path
    else:
        c = (catalog.get("clusters") or {}).get(name)
        where = f"compat/rocm.yaml clusters.{name}"
        if c is None:
            print(f"unknown cluster {name!r}; the catalog has: {', '.join(sorted(catalog.get('clusters') or {}))}",
                  file=sys.stderr)
            return 1
    if c.get("platform") and c["platform"] not in (catalog.get("platforms") or {}):
        print(f"{where}: unknown platform {c['platform']!r}", file=sys.stderr)
        return 1
    if c.get("fabric") and c["fabric"] not in (catalog.get("fabrics") or {}):
        print(f"{where}: unknown fabric {c['fabric']!r}", file=sys.stderr)
        return 1
    q = shlex.quote
    sets = {"MAD_PLATFORM": c.get("platform"), "CLUSTER_ARCHETYPE": c.get("fabric"), **(c.get("env") or {})}
    for k, v in sets.items():
        if v is not None:
            print(f'[ -n "${{{k}:-}}" ] || export {k}={q(str(v))}')
    print(f"echo {q('[cluster] site profile: ' + where)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["shell", "cluster"])
    ap.add_argument("--catalog", default=os.environ.get("MAD_CATALOG") or CATALOG)
    a = ap.parse_args(argv)
    with open(a.catalog) as f:
        catalog = yaml.safe_load(f) or {}
    if a.cmd == "cluster":
        return cluster_shell(catalog)
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
