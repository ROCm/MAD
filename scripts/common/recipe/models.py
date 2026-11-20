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
    variants = recipe.get("variants") or {}
    # The GPU's variant; a GPU the recipe has none for gets the default one, whose GPU_ARCHS then
    # refuses it -- never the bare base, which would run with none of the GPU-specific settings.
    variant = variants.get(arch) or variants.get(recipe.get("default_arch")) or {}
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


BEST_KNOWN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf", "best_known.jsonl")
SHAPE_KEYS = ("xP", "yD", "NNODES", "TP_SIZE", "PP_SIZE", "EP_TP_SIZE", "WIDE_EP", "DP_MODE")


def _registry_path(env):
    """MAD_BEST_REGISTRY (a relative path is taken from scripts/, as topology specs are), else the
    repository's perf/best_known.jsonl."""
    p = env.get("MAD_BEST_REGISTRY")
    if not p:
        return BEST_KNOWN
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", p)


def best_record(name, env=None):
    """The best-known configuration for this run of NAME, or None: a record of the registry
    (MAD_BEST_REGISTRY, default perf/best_known.jsonl) for this model and launcher (MAD_LAUNCHER),
    on this platform (MAD_PLATFORM; else a platform with the detected accelerator, MAD_GPU_ARCH,
    and GPUS_PER_NODE), with the card's shape and, when the record names one, image (the named
    spec whose builds the run's image holds, MAD_IMAGE_COMPONENTS); the highest throughput among
    those. MAD_BEST_CONFIG=0: none."""
    env = os.environ if env is None else env
    if env.get("MAD_BEST_CONFIG", "1") == "0" or not name:
        return None
    path = _registry_path(env)
    try:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("#")]
    except FileNotFoundError:
        return None
    rows = [r for r in rows if r.get("model") == name and r.get("launcher") == env.get("MAD_LAUNCHER")]
    if not rows:
        return None
    platforms = _catalog_platforms()
    def platform_ok(r):
        if env.get("MAD_PLATFORM"):
            return r.get("platform") == env["MAD_PLATFORM"]
        p = platforms.get(r.get("platform")) or {}
        return (bool(env.get("MAD_GPU_ARCH")) and p.get("accelerator") == env["MAD_GPU_ARCH"]
                and str(p.get("gpus_per_node")) == str(env.get("GPUS_PER_NODE", p.get("gpus_per_node"))))
    def shape_ok(r):
        return all(str(env.get(k, "")) == str(v) for k, v in (r.get("shape") or {}).items())
    def image_ok(r):
        if not r.get("image"):
            return True
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import catalog
        return r["image"] == catalog.image_of_components(catalog.load(catalog.COMPAT),
                                                         env.get("MAD_IMAGE_COMPONENTS", ""))
    rows = [r for r in rows if platform_ok(r) and shape_ok(r) and image_ok(r)]
    return max(rows, key=lambda r: r.get("throughput", 0)) if rows else None


def has_best_records(name, env=None):
    """Whether the registry holds any record for NAME and this launcher (so the GPU is needed)."""
    env = os.environ if env is None else env
    if env.get("MAD_BEST_CONFIG", "1") == "0":
        return False
    try:
        with open(_registry_path(env)) as f:
            return any(json.loads(l).get("model") == name and json.loads(l).get("launcher") == env.get("MAD_LAUNCHER")
                       for l in f if l.strip() and not l.lstrip().startswith("#"))
    except FileNotFoundError:
        return False


def _catalog_platforms():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import catalog
    try:
        return catalog.load(catalog.COMPAT).get("platforms") or {}
    except FileNotFoundError:
        return {}


def for_job(models, name, arch=None, env=None):
    """The recipe this run uses: effective(name, arch) with the best-known configuration for the
    run laid over it (best_record). Returns (recipe, record or None)."""
    recipe = effective(models, name, arch)
    rec = best_record(name, env) if recipe is not None else None
    if rec:
        recipe = _overlay(recipe, rec.get("apply") or {})
    return recipe, rec


def uses_variants(models, name):
    """Whether NAME's recipe needs resolving before a plain reader can use it: an alias,
    per-accelerator variants, or memory sizes derived from the GPU."""
    r = (models or {}).get(name)
    if not isinstance(r, dict):
        return False
    return ("alias" in r or bool(r.get("variants")) or bool(memory_needed(effective(models, name)))
            or best_record(name) is not None)


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
    ap.add_argument("--arch", default=os.environ.get("MAD_GPU_ARCH") or None,
                    help="the GPU whose variant applies (default: MAD_GPU_ARCH; an alias pins its own)")
    ap.add_argument("--write")
    ap.add_argument("--uses-variants", action="store_true")
    ap.add_argument("--base", action="store_true",
                    help="print the recipe NAME is an alias of, followed to the end (else NAME)")
    ap.add_argument("--picks-variant", action="store_true",
                    help="exit 0 when what NAME runs depends on the GPU (variants, or best-known records)")
    ap.add_argument("--needs-memory", action="store_true",
                    help="exit 0 when NAME's recipe sizes memory from the GPU (memory: fraction)")
    ap.add_argument("--hbm-bytes", type=int, default=hbm_from_env(),
                    help="one GPU's memory, for memory: fractions (default: MAD_GPU_MEMORY_BYTES)")
    a = ap.parse_args(argv)
    with open(a.yaml) as f:
        models = yaml.safe_load(f) or {}
    if a.uses_variants:
        return 0 if uses_variants(models, a.name) else 1
    if a.base:
        name, seen = a.name, set()
        while isinstance(models.get(name), dict) and "alias" in models[name] and name not in seen:
            seen.add(name)
            name = models[name]["alias"]
        print(name)
        return 0
    if a.picks_variant:
        r = models.get(a.name)
        return 0 if (isinstance(r, dict) and "alias" not in r and r.get("variants")) or has_best_records(a.name) else 1
    recipe, best = for_job(models, a.name, a.arch)
    if best and a.write:
        print(f"[best-known] {a.name}: the {best.get('platform')} record of {best.get('date', 'undated')} "
              "over the recipe", file=sys.stderr)
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
