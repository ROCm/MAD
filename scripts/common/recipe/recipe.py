#!/usr/bin/env python3
"""Resolve a recipe's serving roles against the hardware and the engine build.

A recipe (scripts/<dir>/models.yaml) may describe what each serving role of a model
needs, in vendor-neutral terms, under `roles:`:

    Kimi-K3-MXFP4:
      roles:
        default:                       # every role, unless the role says otherwise
          quantization: {weights: mxfp4, activations: fp8}
          kv_cache: fp8
        decode:                        # e.g. weight-only decode (disaggregated quantization)
          quantization: {activations: bf16}
        variants:                      # per accelerator, over the above
          gfx942: {default: {...}}

Per role: `artifact` (a checkpoint other than the model's own), `quantization.weights` and
`quantization.activations`, `kv_cache`, `placement` (hbm, host or nvme_stream). A format is
an exact name from compat/rocm.yaml `formats`, or a family (`fp8`, `mx`, ...) meaning the
accelerator's native member of it.

Resolving a role for an accelerator and an engine build (compat/rocm.yaml `accelerators`,
`kernels`) gives the formats it will really use -- converted where the build converts one
the hardware cannot compute on -- and the engine arguments that say so; or a refusal that
names every need nothing meets. Nothing is guessed: a capability the catalog does not list
is not assumed.

CLI (from the MAD root):
  recipe.py check   --recipes M.yaml --model NAME --arch ARCH [--engine E] [--build ENGINE@BUILD | --pin-set NAME]
                    [--roles R,..]
  recipe.py explain ... (same arguments): the resolution of every role, as JSON
  recipe.py validate                      : the catalog's own references
  recipe.py bringup --arch ARCH            : what stands between an accelerator and a run
"""
import argparse
import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAD_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CATALOG = os.path.join(MAD_ROOT, "compat", "rocm.yaml")
ROLES = ("prefill", "decode", "server")
NEEDS = ("weights", "activations", "kv_cache")


def load_catalog(path=CATALOG):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge(base, over):
    out = copy.deepcopy(base or {})
    for k, v in (over or {}).items():
        out[k] = _merge(out.get(k), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def role_spec(recipe, role, arch, platform=None):
    """A role's needs: `default`, then the role, then the accelerator's variant of both, then the
    platform's (a variant key is an accelerator or a platform name; the platform is more specific)."""
    roles = (recipe or {}).get("roles") or {}
    spec = _merge(roles.get("default"), roles.get(role))
    for key in (arch, platform):
        variant = (roles.get("variants") or {}).get(key) or {} if key else {}
        spec = _merge(spec, _merge(variant.get("default"), variant.get(role)))
    return spec


def _family(catalog, name):
    f = (catalog.get("formats") or {}).get(name)
    return f.get("family") if f else (name if name in (catalog.get("format_families") or []) else None)


def _pick(catalog, asked, offered, prefer):
    """The format to use for `asked` (an exact format or a family) among `offered`,
    preferring members of `prefer` (what the hardware computes on natively)."""
    if asked in (catalog.get("formats") or {}):
        return asked if asked in offered else None
    members = [f for f in offered if _family(catalog, f) == asked]
    native = [f for f in members if f in prefer]
    return (native or members or [None])[0]


def resolve_role(catalog, recipe, role, arch, build=None, engine="vllm", platform=None):
    """{'role', 'accelerator', 'build', 'artifact', 'weights', 'activations', 'kv_cache',
    'placement', 'engine_args', 'kv_cache_setting', 'notes', 'problems'}; problems empty when it resolves."""
    out = {"role": role, "accelerator": arch, "build": build, "engine_args": [], "notes": [], "problems": []}
    acc = (catalog.get("accelerators") or {}).get(arch)
    if acc is None:
        out["problems"].append(f"accelerator {arch!r} is not in the catalog (compat/rocm.yaml accelerators)")
        return out
    spec = role_spec(recipe, role, arch, platform)
    q = spec.get("quantization") or {}
    out["artifact"] = spec.get("artifact")
    matmul = acc.get("matmul") or []
    kernels = None
    if build:
        kernels = ((catalog.get("kernels") or {}).get(build) or {}).get(arch)
        if kernels is None:
            out["problems"].append(f"engine build {build!r} lists no support for {arch} (compat/rocm.yaml kernels)")
            return out
    else:
        out["notes"].append("no engine build given: checked against the accelerator only")

    def offered(kind):
        if kernels is not None:
            return kernels.get(kind) or []
        return list(matmul) if kind != "weights" else [
            f for f, d in (catalog.get("formats") or {}).items() if f in matmul or d.get("weight_only")]

    # Activations: what the matrix cores compute on.
    act = q.get("activations")
    if act:
        used = _pick(catalog, act, offered("activations"), matmul)
        if used is None:
            out["problems"].append(f"{role}: activations {act!r} -- {arch} with {build or 'any engine'} offers "
                                   f"{offered('activations')}")
        elif used not in matmul and not (catalog["formats"][used].get("weight_only")):
            out["problems"].append(f"{role}: activations {used} are not native on {arch} (matmul {matmul})")
        out["activations"] = {"asked": act, "used": used}

    # Weights: stored as asked, or converted by the build to one the target can use.
    w = q.get("weights")
    if w:
        used = _pick(catalog, w, offered("weights"), matmul)
        conv = None
        if used is None and kernels is not None:
            exact = w if w in (catalog.get("formats") or {}) else _pick(catalog, w, list(catalog["formats"]), matmul)
            conv = (kernels.get("convert") or {}).get(exact)
            if conv:
                used = conv
                out["notes"].append(f"{role}: {exact} weights are converted to {conv} by {build} on {arch}")
                arg = ((catalog.get("engines") or {}).get(engine, {}).get("convert_args") or {}).get(conv)
                if arg:
                    out["engine_args"].append(arg)
        if used is None:
            out["problems"].append(f"{role}: weights {w!r} -- {arch} with {build or 'any engine'} offers "
                                   f"{offered('weights')} and no conversion")
        out["weights"] = {"asked": w, "used": used, **({"converted": True} if conv else {})}

    # KV cache: one format both sides of a transfer must agree on (topology checks the pair).
    kv = spec.get("kv_cache")
    if kv:
        used = _pick(catalog, kv, offered("kv_cache"), matmul)
        if used is None:
            out["problems"].append(f"{role}: kv_cache {kv!r} -- {arch} with {build or 'any engine'} offers "
                                   f"{offered('kv_cache')}")
        else:
            eng = (catalog.get("engines") or {}).get(engine) or {}
            vals = eng.get("kv_cache_values") or {}
            v = vals.get(used) or vals.get(_family(catalog, used))
            if v and v != "auto":
                # How the engine is told: a variable or an argument; a recipe may use either.
                out["kv_cache_setting"] = {"value": v, "env": eng.get("kv_cache_env"), "arg": eng.get("kv_cache_arg")}
        out["kv_cache"] = {"asked": kv, "used": used}

    # Where the weights live.
    place = spec.get("placement", "hbm")
    allowed = (kernels or {}).get("placement") or ["hbm"]
    if place not in allowed:
        out["problems"].append(f"{role}: placement {place!r} -- {build or 'the engine'} supports {allowed}")
    out["placement"] = place
    return out


def build_of_pin_set(catalog, pin_set):
    """The catalog build (<engine>@<pin set>) of a pin set, or None if it has no entry."""
    found = [b for b in (catalog.get("kernels") or {}) if b.split("@", 1)[-1] == pin_set]
    return found[0] if len(found) == 1 else None


def builds_for(catalog, arch, engine):
    """The engine builds the catalog lists for an accelerator."""
    return [b for b, per in (catalog.get("kernels") or {}).items()
            if b.split("@", 1)[0] == engine and arch in (per or {})]


def resolve(catalog, recipe, arch, build=None, roles=None, engine="vllm", platform=None):
    """Every role's resolution; also checks the KV-cache format agrees across roles that
    hand a KV cache over (prefill -> decode).

    With no build named, the first catalog build for `arch` that resolves every role is
    used (and said so); when none does, the accelerator alone is checked, which names
    what the hardware lacks."""
    if build is None:
        for b in builds_for(catalog, arch, engine):
            res = resolve(catalog, recipe, arch, b, roles, engine, platform)
            if not any(r["problems"] for r in res) and not disagreements(recipe, res, catalog, engine):
                for r in res:
                    r["notes"].insert(0, f"no engine build given: resolves with {b} (the first catalog build "
                                         f"for {arch} that does; MAD_ENGINE_BUILD names the image's)")
                return res
    declared = [r for r in ROLES if r in ((recipe or {}).get("roles") or {})]
    roles = roles or declared or ["prefill", "decode"]
    out = [resolve_role(catalog, recipe, r, arch, build, engine, platform) for r in roles]
    by = {r["role"]: r for r in out}
    p, d = by.get("prefill"), by.get("decode")
    if p and d and (p.get("kv_cache") or {}).get("used") != (d.get("kv_cache") or {}).get("used"):
        d["problems"].append(f"KV cache handed from prefill ({(p.get('kv_cache') or {}).get('used')}) to decode "
                             f"({(d.get('kv_cache') or {}).get('used')}) differs, and no conversion is declared")
    return out


def disagreements(recipe, resolved, catalog=None, engine="vllm", flags=None):
    """Where the recipe's own flags contradict what its roles resolve to; fills each role's
    `inject` with the resolved engine arguments its flags do not already carry.

    Flags and roles must not disagree: flags that convert weights to a format the target does
    not need are refused (the recipe is for another accelerator). A resolved argument the flags
    lack -- a conversion, or the KV-cache format when neither the variable nor the argument sets
    it -- is added for that role (MAD_ROLE_ENGINE_ARGS_<ROLE>), so a recipe can leave what the
    roles decide out of its flags and one recipe serves several accelerators. `flags` are the
    engine flags of roles whose flags are not in the recipe (colocated serving takes them from
    the card); without them such roles are neither compared nor given arguments."""
    out = []
    env = (recipe or {}).get("env") or {}
    shared = " ".join(str(v) for k, v in (recipe or {}).items() if k.endswith("_flags") and isinstance(v, str))
    for r in resolved:
        r.setdefault("inject", [])
        block = (recipe or {}).get(r["role"]) or {}
        if block:
            role_flags = " ".join(str(v) for v in block.values()) if isinstance(block, dict) else str(block)
            role_flags = f"{shared} {role_flags}"
        elif flags is not None:
            role_flags = flags
        else:
            continue  # the role's flags are elsewhere and were not given
        kv = r.get("kv_cache_setting")
        if kv:
            by_env = kv["env"] and str(env.get(kv["env"])) == kv["value"]
            by_arg = kv["arg"] and f"{kv['arg']} {kv['value']}" in role_flags
            if not (by_env or by_arg):
                if kv["arg"]:
                    r["inject"].append(f"{kv['arg']} {kv['value']}")
                else:
                    out.append(f"{r['role']}: resolves to a {r['kv_cache']['used']} KV cache, but the recipe "
                               f"does not set {kv['env']}={kv['value']}")
        r["inject"] += [arg for arg in r["engine_args"] if arg not in role_flags]
        convs = ((catalog or {}).get("engines") or {}).get(engine, {}).get("convert_args") or {}
        for fmt, arg in convs.items():
            if arg in role_flags and arg not in r["engine_args"]:
                out.append(f"{r['role']}: its flags convert weights to {fmt} ({arg!r}), which "
                           f"{r['accelerator']} does not need -- the recipe is for another accelerator")
    return out


def supported_archs(recipe):
    """The accelerators a recipe declares it is written for (GPU_ARCHS); empty: any."""
    return str(((recipe or {}).get("env") or {}).get("GPU_ARCHS", "")).replace(",", " ").split()


def requirement_problems(catalog, recipe, arch, platform=None):
    """The recipe's `requires:` against the hardware. min_gpu_memory_gb: against the platform's
    product when the run names one, else the accelerator's largest product (unknown: no refusal)."""
    req = ((recipe or {}).get("roles") or {}).get("requires") or (recipe or {}).get("requires") or {}
    need = req.get("min_gpu_memory_gb")
    if not need:
        return []
    acc = (catalog.get("accelerators") or {}).get(arch) or {}
    products = acc.get("products") or {}
    plat = (catalog.get("platforms") or {}).get(platform) if platform else None
    if plat and plat.get("product") in products:
        have, what = products[plat["product"]].get("hbm_gb"), f"{plat['product']} ({platform})"
    else:
        sizes = {p: d.get("hbm_gb") for p, d in products.items() if d.get("hbm_gb")}
        if not sizes:
            return []
        what = max(sizes, key=sizes.get)
        have = sizes[what]
    if have is not None and have < need:
        return [f"needs {need} GB per GPU; {what} has {have} GB"]
    return []


def accuracy_problems(recipe):
    """`accuracy:` records (a variant's quality validation) must say what was measured and when."""
    out = []
    for i, a in enumerate((recipe or {}).get("accuracy") or []):
        for key in ("task", "score", "date"):
            if key not in (a or {}):
                out.append(f"accuracy[{i}]: missing {key}")
    return out


def check(catalog, recipe, arch, build=None, roles=None, engine="vllm", flags=None, platform=None):
    """(resolution, problems): the roles' resolution and everything that stops the recipe
    running on `arch` -- its declared accelerators, requirements, unmet needs, contradicting flags."""
    only = supported_archs(recipe)
    if only and arch not in only:
        return [], [f"the recipe is written for {', '.join(only)} (GPU_ARCHS), not {arch}"]
    pre = requirement_problems(catalog, recipe, arch, platform) + accuracy_problems(recipe)
    if pre:
        return [], pre
    res = resolve(catalog, recipe, arch, build, roles, engine, platform)
    problems = list(dict.fromkeys(p for r in res for p in r["problems"]))
    return res, problems or disagreements(recipe, res, catalog, engine, flags)


def shell_exports(resolved):
    """export lines for the launcher: each role's artifact and the engine arguments to add.
    The arguments are shell-quoted words (shlex), safe for shlex.split and for eval."""
    import shlex
    lines = []
    for r in resolved:
        role = r["role"].upper()
        if r.get("artifact"):
            lines.append(f"export MAD_ROLE_ARTIFACT_{role}={shlex.quote(str(r['artifact']))}")
        if r.get("inject"):
            words = shlex.join(w for a in r["inject"] for w in shlex.split(a))
            lines.append(f"export MAD_ROLE_ENGINE_ARGS_{role}={shlex.quote(words)}")
    return "".join(x + "\n" for x in lines)


def validate(catalog):
    """The catalog's own references: every format a kernel or accelerator names exists."""
    fmts = set(catalog.get("formats") or {})
    bad = []
    for name, acc in (catalog.get("accelerators") or {}).items():
        bad += [f"accelerators.{name}.matmul: unknown format {f!r}" for f in acc.get("matmul") or [] if f not in fmts]
    for build, per in (catalog.get("kernels") or {}).items():
        for arch, k in per.items():
            if arch not in (catalog.get("accelerators") or {}):
                bad.append(f"kernels.{build}: unknown accelerator {arch!r}")
            for kind in NEEDS:
                bad += [f"kernels.{build}.{arch}.{kind}: unknown format {f!r}" for f in k.get(kind) or [] if f not in fmts]
            for src, dst in (k.get("convert") or {}).items():
                bad += [f"kernels.{build}.{arch}.convert: unknown format {x!r}" for x in (src, dst) if x not in fmts]
    for fam in {d.get("family") for d in (catalog.get("formats") or {}).values()}:
        if fam not in (catalog.get("format_families") or []):
            bad.append(f"formats: family {fam!r} not in format_families")
    fabrics = catalog.get("fabrics") or {}
    det = catalog.get("fabric_detection") or {}
    for name in list(det.get("order") or []) + ([det["unknown"]] if det.get("unknown") else []):
        if name not in fabrics:
            bad.append(f"fabric_detection: unknown fabric {name!r}")
    for name, f in fabrics.items():
        for key in ("match", "gid_index", "socket_ifname", "rdma_drivers", "kv_nic"):
            if key not in (f or {}):
                bad.append(f"fabrics.{name}: missing {key}")
    accs, ups = catalog.get("accelerators") or {}, catalog.get("scale_up_fabrics") or {}
    for name, p in (catalog.get("platforms") or {}).items():
        if p.get("accelerator") not in accs:
            bad.append(f"platforms.{name}: unknown accelerator {p.get('accelerator')!r}")
        elif p.get("product") and p["product"] not in ((accs[p["accelerator"]] or {}).get("products") or {}):
            bad.append(f"platforms.{name}: {p['product']!r} is not a product of {p['accelerator']}")
        if (p.get("scale_up") or {}).get("fabric") not in ups:
            bad.append(f"platforms.{name}: unknown scale-up fabric {(p.get('scale_up') or {}).get('fabric')!r}")
        if not isinstance(p.get("gpus_per_node"), int) or p["gpus_per_node"] < 1:
            bad.append(f"platforms.{name}: gpus_per_node must be a positive integer")
    if catalog.get("default_platform") and catalog["default_platform"] not in (catalog.get("platforms") or {}):
        bad.append(f"default_platform: unknown platform {catalog['default_platform']!r}")
    for name, c in (catalog.get("clusters") or {}).items():
        if (c or {}).get("platform") not in (catalog.get("platforms") or {}):
            bad.append(f"clusters.{name}: unknown platform {(c or {}).get('platform')!r}")
        if (c or {}).get("fabric") and c["fabric"] not in fabrics:
            bad.append(f"clusters.{name}: unknown fabric {c['fabric']!r}")
    return bad


def bringup(catalog, arch, pins_dir=None):
    """What stands between an accelerator and a run: its compiler target, the pin sets that build
    for it (and the components they pin there), the engine builds the catalog lists for it.
    {"arch", "target", "pin_sets": {name: [components]}, "builds": [...], "missing": [...]}"""
    import glob as _glob
    pins_dir = pins_dir or os.path.join(MAD_ROOT, "docker", "pins")
    acc = (catalog.get("accelerators") or {}).get(arch)
    out = {"arch": arch, "target": None, "pin_sets": {}, "builds": [], "missing": []}
    if acc is None:
        out["missing"].append(f"{arch} is not an accelerator in compat/rocm.yaml")
        return out
    out["target"] = acc.get("target", arch)
    if out["target"] == "tbd":
        out["missing"].append("compiler target not yet published (target: tbd)")
    for f in sorted(_glob.glob(os.path.join(pins_dir, "*.json"))):
        with open(f) as fh:
            ps = json.load(fh)
        if arch in (ps.get("gpu_archs") or []):
            comps = [c for c, d in (ps.get("components") or {}).items()
                     if isinstance(d, dict) and d.get("enabled") is not False
                     and (not d.get("archs") or arch in d["archs"])]
            out["pin_sets"][ps["name"]] = comps
    out["builds"] = [b for b, per in (catalog.get("kernels") or {}).items() if arch in (per or {})]
    if not out["pin_sets"]:
        out["missing"].append("no pin set builds an image for it (docker/pins gpu_archs)")
    if not out["builds"]:
        out["missing"].append("no engine build lists what it runs there (compat/rocm.yaml kernels)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("check", "explain"):
        p = sub.add_parser(cmd)
        p.add_argument("--recipes", required=True, help="models.yaml")
        p.add_argument("--model", required=True)
        p.add_argument("--arch", required=True)
        p.add_argument("--engine", default="vllm", help="the engine the launcher runs (vllm, sglang)")
        p.add_argument("--platform", default=os.environ.get("MAD_PLATFORM") or None,
                       help="the run's platform (MAD_PLATFORM): platform variants and memory requirements")
        p.add_argument("--build", help="ENGINE@BUILD, e.g. vllm@vllm-kimi-k3")
        p.add_argument("--pin-set", help="the image's pin set (docker/pins/<name>.json): its catalog build")
        p.add_argument("--roles", help="comma-separated (default: the roles the recipe declares)")
        p.add_argument("--flags", help="engine flags of roles whose flags are not in the recipe "
                                       "(colocated serving: the card's COLOCATED_EXTRA_ARGS)")
        p.add_argument("--catalog", default=CATALOG)
        p.add_argument("--exports", help="write the launcher's export lines (role artifacts and the "
                                         "engine arguments to add) to this file when it resolves")
    v = sub.add_parser("validate")
    v.add_argument("--catalog", default=CATALOG)
    b = sub.add_parser("bringup", help="what stands between an accelerator and a run")
    b.add_argument("--arch", required=True)
    b.add_argument("--catalog", default=CATALOG)
    a = ap.parse_args(argv)
    catalog = load_catalog(a.catalog)
    if a.cmd == "bringup":
        r = bringup(catalog, a.arch)
        print(f"{a.arch}: target {r['target']}")
        for name, comps in r["pin_sets"].items():
            print(f"  pin set {name}: {', '.join(comps)}")
        for b in r["builds"]:
            print(f"  engine build {b}")
        for m in r["missing"]:
            print(f"  missing: {m}")
        return 1 if r["missing"] else 0
    if a.cmd == "validate":
        bad = validate(catalog)
        for b in bad:
            print("ERROR " + b)
        print("catalog ok" if not bad else f"{len(bad)} problem(s)")
        return 1 if bad else 0
    with open(a.recipes) as f:
        recipes = yaml.safe_load(f) or {}
    # The recipe as the launcher applies it: the alias followed (it pins its variant), else the
    # variant of the GPU checked (the detected one, as config_apply_recipe), else the default.
    recipe = models.effective(recipes, a.model, a.arch)
    if recipe is None:
        print(f"ERROR: no recipe {a.model!r} in {a.recipes}", file=sys.stderr)
        return 2
    if not recipe.get("roles"):
        print(f"{a.model}: declares no roles; nothing to resolve")
        return 0
    if a.pin_set and not a.build:
        a.build = build_of_pin_set(catalog, a.pin_set)
        if a.build is None:
            print(f"ERROR: {a.model}: the image's pin set {a.pin_set!r} has no entry in the capability "
                  f"catalog (compat/rocm.yaml kernels), so what it can run is unknown", file=sys.stderr)
            return 1
    engine = a.build.split("@", 1)[0] if a.build else a.engine
    if a.build and engine != a.engine:
        print(f"ERROR: {a.model}: the image's build {a.build} is not a {a.engine} build", file=sys.stderr)
        return 1
    res, problems = check(catalog, recipe, a.arch, a.build, a.roles.split(",") if a.roles else None, engine,
                          a.flags, a.platform)
    if a.exports and not problems:
        with open(a.exports, "w") as f:
            f.write(shell_exports(res))
    if a.cmd == "explain":
        print(json.dumps(res, indent=1))
    else:
        for n in dict.fromkeys(n for r in res for n in r["notes"]):
            print(f"[recipe] {a.model} {n}")
        for r in res:
            if r.get("inject"):
                print(f"[recipe] {a.model} {r['role']}: adds {' '.join(r['inject'])} (resolved; not in its flags)")
            if r.get("artifact"):
                print(f"[recipe] {a.model} {r['role']}: serves its own artifact {r['artifact']}")
        for p in problems:
            print(f"ERROR: {a.model} on {a.arch}{' with ' + a.build if a.build else ''}: {p}", file=sys.stderr)
        if not problems:
            print(f"[recipe] {a.model} resolves on {a.arch}" + (f" with {a.build}" if a.build else "")
                  + ": " + "; ".join(f"{r['role']} " + ",".join(
                      f"{k}={(r.get(k) or {}).get('used')}" for k in NEEDS if r.get(k)) for r in res))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
