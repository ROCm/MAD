#!/usr/bin/env python3
"""Read a recipe from a models.yaml: the one reader every launcher and tool uses.

A recipe may be written once for several accelerators:

    Kimi-K3-MXFP4:                 # the base: what holds on every accelerator
      default_arch: gfx942         # the variant used when nothing names an accelerator
      env: {...}
      prefill: {dp: "..."}
      variants:
        gfx950:                    # over the base on gfx950: env values (null removes one),
          env: {EP_TP_SIZE: "1", AITER_SITUV2_A8W4: null}
          prefill: {dp: "..."}     # flags replace the base's, key by key

    Kimi-K3-MXFP4-MI355X:          # a name for one variant, so a card can ask for it
      alias: Kimi-K3-MXFP4
      arch: gfx950

effective(models, name, arch) is the recipe as a launcher uses it: the alias followed, the
accelerator's variant (else the recipe's `default_arch` one) applied over the base (mappings merged, a null value removing the key,
anything else replaced), without the `variants`, `alias` and `arch` keys. A recipe that uses
neither is returned as written.

A recipe may size memory from the GPU it lands on, instead of in bytes for one GPU:

    memory:
      KV_CACHE_MEMORY_BYTES: {fraction: 0.104}    # of one GPU's memory (MAD_GPU_MEMORY_BYTES)
      MORI_SHMEM_HEAP_SIZE: {gib: 16}
An explicit env value of the same name wins (an absolute override).

CLI:  models.py YAML NAME [--arch ARCH]          the effective recipe, as JSON
      models.py YAML NAME [--arch ARCH] --write FILE
                                               FILE: a models.yaml holding just NAME's effective
                                               recipe, for readers that know no aliases (the
                                               role scripts in the containers)
      models.py YAML NAME --uses-variants      exit 0 when NAME needs resolving (alias, variants,
                                               memory sizes)
      models.py YAML NAME --needs-memory       exit 0 when NAME sizes memory from the GPU
"""
import argparse
import copy
import json
import os
import sys

import yaml

META = ("variants", "alias", "arch", "default_arch")


def _overlay(base, over):
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    for k, v in (over or {}).items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _overlay(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def effective(models, name, arch=None, _seen=()):
    """NAME's recipe as used on ARCH (None: the base, or the alias's own arch)."""
    recipe = (models or {}).get(name)
    if not isinstance(recipe, dict):
        return recipe
    if "alias" in recipe:
        if name in _seen:
            raise ValueError(f"recipe alias loop: {' -> '.join(_seen + (name,))}")
        target = recipe["alias"]
        out = effective(models, target, recipe.get("arch") or arch, _seen + (name,))
        if out is None:
            raise ValueError(f"recipe {name!r} is an alias of {target!r}, which does not exist")
        extra = {k: v for k, v in recipe.items() if k not in META}
        return _overlay(out, extra)
    arch = arch or recipe.get("default_arch")
    variant = ((recipe.get("variants") or {}).get(arch) or {}) if arch else {}
    out = {k: v for k, v in recipe.items() if k not in META}
    return _overlay(out, variant)


def memory_needed(recipe):
    """The `memory:` entries the recipe's env does not set explicitly (an explicit value wins)."""
    env = (recipe or {}).get("env") or {}
    return {k: v for k, v in ((recipe or {}).get("memory") or {}).items() if k not in env}


def derive_memory(recipe, hbm_bytes):
    """{NAME: bytes as str} for each needed `memory:` entry, from one GPU's memory (bytes):
    {fraction: F} of it, {gib: N}, or {bytes: N}."""
    out = {}
    for name, spec in memory_needed(recipe).items():
        spec = spec or {}
        if "fraction" in spec:
            if not hbm_bytes:
                raise ValueError(f"{name}: a fraction of GPU memory, but GPU memory is unknown "
                                 "(MAD_GPU_MEMORY_BYTES)")
            v = int(float(spec["fraction"]) * int(hbm_bytes))
        elif "gib" in spec:
            v = int(float(spec["gib"]) * 2**30)
        elif "bytes" in spec:
            v = int(spec["bytes"])
        else:
            raise ValueError(f"{name}: memory entry needs fraction, gib or bytes, got {spec!r}")
        out[name] = str(v)
    return out


def with_memory(recipe, hbm_bytes):
    """The recipe with its `memory:` entries resolved into env (explicit env values win)."""
    if not isinstance(recipe, dict) or "memory" not in recipe:
        return recipe
    out = copy.deepcopy(recipe)
    out["env"] = dict(out.get("env") or {}, **derive_memory(recipe, hbm_bytes))
    out.pop("memory")
    return out


def uses_variants(models, name):
    """Whether NAME's recipe needs resolving before a plain reader can use it: an alias,
    per-accelerator variants, or memory sizes derived from the GPU."""
    r = (models or {}).get(name)
    if not isinstance(r, dict):
        return False
    return "alias" in r or bool(r.get("variants")) or bool(memory_needed(effective(models, name)))


def hbm_from_env():
    v = os.environ.get("MAD_GPU_MEMORY_BYTES")
    return int(v) if v else None


def load(path, name, arch=None):
    with open(path) as f:
        return effective(yaml.safe_load(f) or {}, name, arch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml")
    ap.add_argument("name")
    ap.add_argument("--arch")
    ap.add_argument("--write")
    ap.add_argument("--uses-variants", action="store_true")
    ap.add_argument("--needs-memory", action="store_true",
                    help="exit 0 when NAME's recipe sizes memory from the GPU (memory: fraction)")
    ap.add_argument("--hbm-bytes", type=int, default=hbm_from_env(),
                    help="one GPU's memory, for memory: fractions (default: MAD_GPU_MEMORY_BYTES)")
    a = ap.parse_args(argv)
    with open(a.yaml) as f:
        models = yaml.safe_load(f) or {}
    if a.uses_variants:
        return 0 if uses_variants(models, a.name) else 1
    recipe = effective(models, a.name, a.arch)
    if recipe is None:
        print(f"no recipe {a.name!r} in {a.yaml}", file=sys.stderr)
        return 2
    if a.needs_memory:
        return 0 if any("fraction" in (v or {}) for v in memory_needed(recipe).values()) else 1
    try:
        recipe = with_memory(recipe, a.hbm_bytes)
    except ValueError as e:
        print(f"{a.name}: {e}", file=sys.stderr)
        return 1
    if a.write:
        with open(a.write, "w") as f:
            yaml.safe_dump({a.name: recipe}, f, sort_keys=False)
    else:
        print(json.dumps(recipe, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
