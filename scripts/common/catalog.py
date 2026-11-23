#!/usr/bin/env python3
"""The capability catalog, as every MAD tool reads it: one dict from the files in compat/.

    compat/rocm.yaml       software: ROCm releases, indexes and authorized component builds (the
                           image-spec matrix madengine reads), each build's capabilities; engines
                           (how a format is asked for), kernels (engines an image's base provides)
    compat/hardware.yaml   formats, accelerators, fabrics, scale-up fabrics, platforms
    compat/sites/*.yaml    example site profiles (one per file: `clusters.<file name>`)
    compat/images/*.yaml   named image specs (under IMAGES_KEY), each with its `extends` chain merged
                           in, as madengine composes them

MAD_CATALOG naming a file reads that one file as the whole catalog (tests point it at a fixture).
Standard library and PyYAML only.

    catalog.py dump        the merged catalog as YAML
"""
import os
import re
import sys

import yaml

COMPAT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "compat"))
IMAGES_KEY = "__images__"
REF = re.compile(r"\$\{images\.([A-Za-z0-9_.-]+)\}")
HARDWARE_KEYS = ("formats", "format_families", "accelerators", "fabric_detection", "fabrics",
                 "scale_up_fabrics", "default_platform", "platforms")


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dir(path):
    return {f[:-5]: _read(os.path.join(path, f)) for f in sorted(os.listdir(path))
            if f.endswith(".yaml")} if os.path.isdir(path) else {}


def _merge(base, over):
    """madengine's merge of a spec over the one it extends: mappings merged, anything else replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _flatten(images, name, seen=()):
    """Named spec *name* with the spec it extends (`extends: ${images.<other>}`) merged under it."""
    if name in seen:
        raise ValueError(f"compat/images: extends loop: {' -> '.join(seen + (name,))}")
    spec = images.get(name) or {}
    m = REF.fullmatch(str(spec.get("extends") or ""))
    if not m:
        return spec
    if m.group(1) not in images:
        raise ValueError(f"compat/images/{name}.yaml extends {m.group(1)!r}, which does not exist")
    return _merge(_flatten(images, m.group(1), seen + (name,)), {k: v for k, v in spec.items() if k != "extends"})


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
        out[IMAGES_KEY] = {n: _flatten(images, n) for n in images}
    return out


def spec_components(spec, arch=None):
    """{component: build key} a named spec selects (its variants.<arch> laid over it when given)."""
    comps = dict((spec or {}).get("components") or {})
    if arch:
        comps.update(((spec or {}).get("variants") or {}).get(arch, {}).get("components") or {})
    return {str(k): str(v) for k, v in comps.items()}


def spec_archs(spec):
    """The GPUs a named spec has its own selection for (its variants), besides its default one."""
    return sorted(((spec or {}).get("variants") or {}))


def selections(spec):
    """Every {component: build key} the spec selects: the default one, then each variant's."""
    return [spec_components(spec)] + [spec_components(spec, a) for a in spec_archs(spec)]


def image_of_card(card, catalog=None):
    """The named spec a card builds from ("image_spec": "${images.<name>}", or extending one), or None."""
    value = card.get("image_spec")
    if isinstance(value, dict):
        value = value.get("extends")
    m = REF.fullmatch(str(value or ""))
    return m.group(1) if m else None


def components_of_card(card, arch=None, catalog=None):
    """MAD_IMAGE_COMPONENTS for a card built from a named spec (name=key,...), as madengine passes an
    image_spec image's builds to the run; None for a card with a hand-written Dockerfile."""
    catalog = catalog if catalog is not None else load(COMPAT)
    name = image_of_card(card)
    spec = (catalog.get(IMAGES_KEY) or {}).get(name) if name else None
    if not spec:
        return None
    return ",".join(f"{k}={v}" for k, v in sorted(spec_components(spec, arch).items())) or None   # by name


def image_of_components(catalog, components):
    """The named spec whose selected builds an image holds (MAD_IMAGE_COMPONENTS), or None. Specs
    that select the same builds (one composed from others) name the same image: the first is given."""
    have = dict(p.split("=", 1) for p in (components or "").split(",") if "=" in p)
    for name, spec in (catalog.get(IMAGES_KEY) or {}).items():
        for sel in selections(spec):
            if sel and all(have.get(k) == v for k, v in sel.items()):
                return name
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["dump"]:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.write(yaml.safe_dump({k: v for k, v in load().items() if k != IMAGES_KEY}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
