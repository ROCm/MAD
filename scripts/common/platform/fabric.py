#!/usr/bin/env python3
"""The scale-out fabric of this node, from the capability catalog (compat/rocm.yaml `fabrics`).

scripts/common/cluster.sh evals `fabric.py shell`: it names the fabric family (detected from the
node's RDMA devices and their drivers, unless CLUSTER_ARCHETYPE names it) and the values the run
gets when it passes none -- GID index, control interface, RDMA drivers, KV NIC, the GPU rails --
as shell assignments to cluster.sh's _arch_* variables. Detection reads CLUSTER_SYSFS_IB (default
/sys/class/infiniband), or `ibv_devices` when there is no sysfs tree.

    fabric.py shell [--catalog PATH]
    fabric.py detect [--catalog PATH]       the family name
"""
import argparse
import os
import shlex
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "..", "..", "..", "compat", "rocm.yaml")


def load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _driver(ib_root, dev):
    try:
        return os.path.basename(os.readlink(os.path.join(ib_root, dev, "device", "driver")))
    except OSError:
        return ""


def devices(ib_root):
    """[(device, driver)] on this node, in natural order."""
    if os.path.isdir(ib_root):
        import re
        key = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]  # noqa: E731
        return [(d, _driver(ib_root, d)) for d in sorted(os.listdir(ib_root), key=key)]
    try:
        out = subprocess.run(["ibv_devices"], capture_output=True, text=True, timeout=10).stdout
        return [(line.split()[0], "") for line in out.splitlines()[2:] if line.split()]
    except (OSError, subprocess.SubprocessError):
        return []


def detect(catalog, devs):
    fabrics = catalog.get("fabrics") or {}
    order = (catalog.get("fabric_detection") or {}).get("order") or list(fabrics)
    names = " ".join(f"{d} {drv}" for d, drv in devs)
    for name in order:
        if any(m in names for m in (fabrics.get(name) or {}).get("match") or []):
            return name
    return "unknown"


def settle(catalog, family, devs):
    """The fallback values for `family` on a node with `devs`."""
    fabrics = catalog.get("fabrics") or {}
    f = fabrics.get(family) or fabrics.get((catalog.get("fabric_detection") or {}).get("unknown")) or {}
    rails_spec = f.get("rails") or {}
    found = [d for d, drv in devs if rails_spec.get("driver") and drv == rails_spec["driver"]]
    rails = found or list(rails_spec.get("default") or [])
    kv = found[0] if found else f.get("kv_nic", "")
    return {"gid": f.get("gid_index", ""), "iface": f.get("socket_ifname", ""),
            "drivers": f.get("rdma_drivers", ""), "kv_nic": kv, "rails": ",".join(rails),
            "env": f.get("env") or {}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["shell", "detect"])
    ap.add_argument("--catalog", default=os.environ.get("MAD_CATALOG") or CATALOG)
    a = ap.parse_args(argv)
    catalog = load(a.catalog)
    devs = devices(os.environ.get("CLUSTER_SYSFS_IB", "/sys/class/infiniband"))
    family = os.environ.get("CLUSTER_ARCHETYPE") or detect(catalog, devs)
    if a.cmd == "detect":
        print(family)
        return 0
    v = settle(catalog, family, devs)
    q = shlex.quote
    print(f"CLUSTER_ARCHETYPE={q(family)}")
    print(f"_arch_gid={q(str(v['gid']))}; _arch_iface={q(v['iface'])}; _arch_drivers={q(v['drivers'])}")
    print(f"_arch_kv_nic={q(v['kv_nic'])}; _arch_rails={q(v['rails'])}")
    print("_arch_env=(" + " ".join(q(f"{k}={val}") for k, val in v["env"].items()) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
