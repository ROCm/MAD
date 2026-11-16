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
  recipe.py check   --recipes M.yaml --model NAME --arch ARCH [--build ENGINE@BUILD] [--roles R,..]
  recipe.py explain ... (same arguments): the resolution of every role, as JSON
  recipe.py validate                      : the catalog's own references
"""
import argparse
import copy
import json
import os
import sys

import yaml

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


def role_spec(recipe, role, arch):
    """A role's needs: `default`, then the role, then the accelerator's variant of both."""
    roles = (recipe or {}).get("roles") or {}
    variant = (roles.get("variants") or {}).get(arch) or {}
    spec = _merge(roles.get("default"), roles.get(role))
    return _merge(spec, _merge(variant.get("default"), variant.get(role)))


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


def resolve_role(catalog, recipe, role, arch, build=None, engine="vllm"):
    """{'role', 'accelerator', 'build', 'artifact', 'weights', 'activations', 'kv_cache',
    'placement', 'engine_args', 'engine_env', 'notes', 'problems'}; problems empty when it resolves."""
    out = {"role": role, "accelerator": arch, "build": build, "engine_args": [], "engine_env": {},
           "notes": [], "problems": []}
    acc = (catalog.get("accelerators") or {}).get(arch)
    if acc is None:
        out["problems"].append(f"accelerator {arch!r} is not in the catalog (compat/rocm.yaml accelerators)")
        return out
    spec = role_spec(recipe, role, arch)
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
            if v and eng.get("kv_cache_env"):
                out["engine_env"][eng["kv_cache_env"]] = v
        out["kv_cache"] = {"asked": kv, "used": used}

    # Where the weights live.
    place = spec.get("placement", "hbm")
    allowed = (kernels or {}).get("placement") or ["hbm"]
    if place not in allowed:
        out["problems"].append(f"{role}: placement {place!r} -- {build or 'the engine'} supports {allowed}")
    out["placement"] = place
    return out


def builds_for(catalog, arch, engine):
    """The engine builds the catalog lists for an accelerator."""
    return [b for b, per in (catalog.get("kernels") or {}).items()
            if b.split("@", 1)[0] == engine and arch in (per or {})]


def resolve(catalog, recipe, arch, build=None, roles=None, engine="vllm"):
    """Every role's resolution; also checks the KV-cache format agrees across roles that
    hand a KV cache over (prefill -> decode).

    With no build named, the first catalog build for `arch` that resolves every role is
    used (and said so); when none does, the accelerator alone is checked, which names
    what the hardware lacks."""
    if build is None:
        for b in builds_for(catalog, arch, engine):
            res = resolve(catalog, recipe, arch, b, roles, engine)
            if not any(r["problems"] for r in res) and not disagreements(recipe, res, catalog, engine):
                for r in res:
                    r["notes"].insert(0, f"no engine build given: resolves with {b} (the first catalog build "
                                         f"for {arch} that does; MAD_ENGINE_BUILD names the image's)")
                return res
    declared = [r for r in ROLES if r in ((recipe or {}).get("roles") or {})]
    roles = roles or declared or ["prefill", "decode"]
    out = [resolve_role(catalog, recipe, r, arch, build, engine) for r in roles]
    by = {r["role"]: r for r in out}
    p, d = by.get("prefill"), by.get("decode")
    if p and d and (p.get("kv_cache") or {}).get("used") != (d.get("kv_cache") or {}).get("used"):
        d["problems"].append(f"KV cache handed from prefill ({(p.get('kv_cache') or {}).get('used')}) to decode "
                             f"({(d.get('kv_cache') or {}).get('used')}) differs, and no conversion is declared")
    return out


def disagreements(recipe, resolved, catalog=None, engine="vllm"):
    """Where the recipe's own flags and env say something other than its roles resolve to.

    While roles are declared alongside hand-written flags (before they generate them), this
    keeps the two from drifting: a conversion's engine argument must be in the role's flags,
    and a KV-cache variable must have the resolved value."""
    out = []
    env = (recipe or {}).get("env") or {}
    for r in resolved:
        block = (recipe or {}).get(r["role"]) or {}
        if not block:
            continue  # the role's flags are not in the recipe (e.g. colocated: the card's)
        flags = " ".join(str(v) for v in block.values()) if isinstance(block, dict) else str(block)
        for arg in r["engine_args"]:
            if arg not in flags:
                out.append(f"{r['role']}: resolves to engine argument {arg!r}, which its flags lack")
        convs = ((catalog or {}).get("engines") or {}).get(engine, {}).get("convert_args") or {}
        for fmt, arg in convs.items():
            if arg in flags and arg not in r["engine_args"]:
                out.append(f"{r['role']}: its flags convert weights to {fmt} ({arg!r}), which "
                           f"{r['accelerator']} does not need -- the recipe is for another accelerator")
        for k, v in r["engine_env"].items():
            if str(env.get(k)) != str(v):
                out.append(f"{r['role']}: resolves to {k}={v}, but the recipe's env sets {k}={env.get(k)}")
    return out


def check(catalog, recipe, arch, build=None, roles=None, engine="vllm"):
    """(resolution, problems): the roles' resolution and everything that stops the recipe
    running on `arch` -- unmet needs first, else disagreements with its own flags."""
    res = resolve(catalog, recipe, arch, build, roles, engine)
    problems = list(dict.fromkeys(p for r in res for p in r["problems"]))
    return res, problems or disagreements(recipe, res, catalog, engine)


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
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("check", "explain"):
        p = sub.add_parser(cmd)
        p.add_argument("--recipes", required=True, help="models.yaml")
        p.add_argument("--model", required=True)
        p.add_argument("--arch", required=True)
        p.add_argument("--build", help="ENGINE@BUILD, e.g. vllm@vllm-kimi-k3")
        p.add_argument("--roles", help="comma-separated (default: the roles the recipe declares)")
        p.add_argument("--catalog", default=CATALOG)
    v = sub.add_parser("validate")
    v.add_argument("--catalog", default=CATALOG)
    a = ap.parse_args(argv)
    catalog = load_catalog(a.catalog)
    if a.cmd == "validate":
        bad = validate(catalog)
        for b in bad:
            print("ERROR " + b)
        print("catalog ok" if not bad else f"{len(bad)} problem(s)")
        return 1 if bad else 0
    with open(a.recipes) as f:
        recipes = yaml.safe_load(f) or {}
    recipe = recipes.get(a.model)
    if recipe is None:
        print(f"ERROR: no recipe {a.model!r} in {a.recipes}", file=sys.stderr)
        return 2
    if not recipe.get("roles"):
        print(f"{a.model}: declares no roles; nothing to resolve")
        return 0
    engine = (a.build or "vllm@").split("@", 1)[0] or "vllm"
    res, problems = check(catalog, recipe, a.arch, a.build, a.roles.split(",") if a.roles else None, engine)
    if a.cmd == "explain":
        print(json.dumps(res, indent=1))
    else:
        for n in dict.fromkeys(n for r in res for n in r["notes"]):
            print(f"[recipe] {a.model} {n}")
        for p in problems:
            print(f"ERROR: {a.model} on {a.arch}{' with ' + a.build if a.build else ''}: {p}", file=sys.stderr)
        if not problems:
            print(f"[recipe] {a.model} resolves on {a.arch}" + (f" with {a.build}" if a.build else "")
                  + ": " + "; ".join(f"{r['role']} " + ",".join(
                      f"{k}={(r.get(k) or {}).get('used')}" for k in NEEDS if r.get(k)) for r in res))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
