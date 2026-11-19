#!/usr/bin/env python3
"""The capability catalog, as every MAD tool reads it: one dict from the files in compat/.

    compat/rocm.yaml       software: ROCm releases, indexes and authorized component builds (the
                           image-spec matrix madengine reads), each build's capabilities; engines
                           (how a format is asked for), kernels (engines an image's base provides)
    compat/hardware.yaml   formats, accelerators, fabrics, scale-up fabrics, platforms
    compat/sites/*.yaml    example site profiles (one per file: `clusters.<file name>`)
    compat/images/*.yaml   named image specs (under IMAGES_KEY)

MAD_CATALOG naming a file reads that one file as the whole catalog (tests point it at a fixture).
Standard library and PyYAML only.

    catalog.py dump        the merged catalog as YAML
"""
import os
import sys

import yaml

COMPAT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "compat"))
IMAGES_KEY = "__images__"
HARDWARE_KEYS = ("formats", "format_families", "accelerators", "fabric_detection", "fabrics",
                 "scale_up_fabrics", "default_platform", "platforms")


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dir(path):
    return {f[:-5]: _read(os.path.join(path, f)) for f in sorted(os.listdir(path))
            if f.endswith(".yaml")} if os.path.isdir(path) else {}


def load(compat=None):
    """The merged catalog. `compat`: a compat/ directory, or a file read as the whole catalog
    (default: MAD_CATALOG, else this checkout's compat/)."""
    compat = compat or os.environ.get("MAD_CATALOG") or COMPAT
    if not os.path.exists(compat) or (os.path.isdir(compat) and not any(
            os.path.isfile(os.path.join(compat, f)) for f in ("rocm.yaml", "hardware.yaml"))):
        raise FileNotFoundError(f"no capability catalog at {compat}")
    if os.path.isfile(compat):
        out = _read(compat)
        base = os.path.dirname(compat)
    else:
        out = {}
        for name in ("rocm.yaml", "hardware.yaml"):
            p = os.path.join(compat, name)
            if os.path.isfile(p):
                for k, v in _read(p).items():
                    if k in out:
                        raise ValueError(f"{p}: {k} is also in another compat file")
                    out[k] = v
        sites = _dir(os.path.join(compat, "sites"))
        if sites:
            out.setdefault("clusters", {}).update(sites)
        base = compat
    images = _dir(os.path.join(base, "images"))
    if images:
        out[IMAGES_KEY] = images
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["dump"]:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.write(yaml.safe_dump({k: v for k, v in load().items() if k != IMAGES_KEY}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
