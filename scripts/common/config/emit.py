#!/usr/bin/env python3
"""A card's complete configuration as one workload file, generated from the files it uses today.

For a multinode card this composes, from the same places its launcher reads:

    card       name, launcher, tags                       (models.json; informational)
    image      dockerfile / docker_build_arg / image_spec (models.json; for the build)
    recipes    its recipe (MODEL_NAME) and every recipe it is an alias of, as declared in the
               launcher's models.yaml: per-GPU variants, memory fractions and roles kept, so the
               file runs on any GPU as the card does
    connectors every connector env file of the launcher (connectors/*.env)
    topology   the card's MAD_TOPOLOGY spec (a path), when it has one
    model.env  the card's env_vars over its workload file's (MAD_CONFIG), as the launcher applies them
    benchmark  its workload file's benchmark entries (less what the card's env_vars override)

Pointing the card's MAD_CONFIG at the result runs the same thing (snapshot.py --from-emitted
proves it per card and GPU); nothing the card uses today has to change. Used by `madengine build
--emit-config` and the pipeline's STANDALONE mode alike.

    emit.py CATALOG CARD [-o FILE]       one card (CATALOG: vllm_dissag, vllm_multinode, sglang_disagg)
    emit.py --all -o DIR                 every card of the three catalogs, DIR/<catalog>/<card>.yaml
    emit.py --card-json FILE [-o OUT]    a card given as JSON (as madengine has it, the run's
                                         context merged); its catalog from its scripts path
Standard library and PyYAML only.
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import workload  # noqa: E402

CATALOGS = ("vllm_dissag", "vllm_multinode", "sglang_disagg")
# Where each launcher reads its recipes and connector env (run_multinode.slurm: vllm_dissag's).
SOURCES = {"vllm_dissag": "vllm_dissag", "vllm_multinode": "vllm_dissag", "sglang_disagg": "sglang_disagg"}
VERSION = 2


class EmitError(ValueError):
    pass


def _escape(value):
    """Strings as the reader takes them literally: `${` written `\\${`."""
    if isinstance(value, str):
        return value.replace("${", "\\${")
    if isinstance(value, dict):
        return {k: _escape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_escape(v) for v in value]
    return value


def read_env_file(path):
    """KEY=VALUE lines, as the launchers read a connector .env (comments and blanks skipped)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k] = v
    return out


def recipe_closure(recipes, name):
    """NAME's recipe and every recipe it is an alias of, in models.yaml form."""
    out, seen = {}, set()
    while name in recipes and name not in seen:
        seen.add(name)
        out[name] = recipes[name]
        r = recipes[name]
        name = r.get("alias") if isinstance(r, dict) else None
    return out


def catalog_of(card):
    """The catalog a card belongs to, from its scripts path (scripts/<catalog>/<launcher>)."""
    parts = os.path.normpath(card.get("scripts") or "").split(os.sep)
    for c in CATALOGS:
        if c in parts:
            return c
    raise EmitError(f"{card.get('name')}: scripts {card.get('scripts')!r} is not a multinode launcher "
                    f"({', '.join(CATALOGS)})")


def card_of(catalog, name):
    for c in json.loads(open(os.path.join(SCRIPTS, catalog, "models.json")).read()):
        if c["name"] == name:
            return c
    raise EmitError(f"{catalog}: no card {name!r}")


def compose(catalog, card):
    """The card's complete configuration (a dict in workload-file form)."""
    if catalog not in CATALOGS:
        raise EmitError(f"{catalog}: not a multinode catalog ({', '.join(CATALOGS)})")
    src = os.path.join(SCRIPTS, SOURCES[catalog])
    card_env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
    base = {}
    path = workload.find(card_env.pop("MAD_CONFIG", ""), os.path.join(SCRIPTS, catalog))
    if path:
        base = workload.load(path)

    model_env = dict(((base.get("model") or {}).get("env")) or {})
    topology = card_env.pop("MAD_TOPOLOGY", None) or base.get("topology")
    model_env.update(card_env)
    # The card's env_vars are submitted, so they win over the workload's benchmark entries too.
    bench = []
    for e in base.get("benchmark") or []:
        e = dict(e)
        e["env"] = {k: v for k, v in (e.get("env") or {}).items() if k not in card_env}
        bench.append(e)

    recipes = base.get("recipes")
    if recipes is None:
        rpath = os.path.join(src, "models.yaml")
        recipes = (yaml.safe_load(open(rpath)) or {}) if os.path.isfile(rpath) else {}
    name = model_env.get("MODEL_NAME")
    closure = recipe_closure(recipes, name) if name else {}

    connectors = base.get("connectors")
    if connectors is None:
        cdir = os.path.join(src, "connectors")
        connectors = {f[:-4]: read_env_file(os.path.join(cdir, f))
                      for f in sorted(os.listdir(cdir)) if f.endswith(".env")} if os.path.isdir(cdir) else {}

    out = {"version": VERSION,
           "card": {k: card[k] for k in ("name", "scripts", "tags") if k in card}}
    out["card"]["catalog"] = catalog
    image = {k: card[k] for k in ("dockerfile", "docker_build_arg", "image_spec") if card.get(k)}
    if image:
        out["image"] = image
    if closure:
        out["recipes"] = _escape(closure)
    if connectors:
        out["connectors"] = _escape(connectors)
    if topology:
        out["topology"] = topology
    out["model"] = {"env": _escape(model_env)}
    if any(e.get("env") for e in bench):
        out["benchmark"] = _escape([e for e in bench if e.get("env") or e.get("kind")])
    return out


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):      # write shared values out in full: no &id001 / *id001
        return True


def dump(data):
    head = (f"# Generated by scripts/common/config/emit.py from card {data['card']['name']} "
            f"({data['card']['catalog']}).\n"
            "# Point the card's MAD_CONFIG here to run it from this file; edit freely.\n")
    return head + yaml.dump(data, Dumper=_Dumper, sort_keys=False, width=1 << 16, allow_unicode=True)


def emit(catalog, card_name, out=None):
    text = dump(compose(catalog, card_of(catalog, card_name)))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalog", nargs="?")
    ap.add_argument("card", nargs="?")
    ap.add_argument("--all", action="store_true", help="every card of the three catalogs")
    ap.add_argument("--card-json", help="the card as JSON (a file), instead of CATALOG CARD")
    ap.add_argument("--catalog", dest="catalog_opt", help="with --card-json: its catalog (default: from its scripts path)")
    ap.add_argument("-o", "--out", help="file (one card) or directory (--all)")
    a = ap.parse_args(argv)
    try:
        if a.all:
            if not a.out:
                ap.error("--all needs -o DIR")
            n = 0
            for catalog in CATALOGS:
                for c in json.loads(open(os.path.join(SCRIPTS, catalog, "models.json")).read()):
                    emit(catalog, c["name"], os.path.join(a.out, catalog, f"{c['name']}.yaml"))
                    n += 1
            print(f"emit: {n} card(s) -> {a.out}")
            return 0
        if a.card_json:
            card = json.loads(open(a.card_json).read())
            catalog = a.catalog_opt or a.catalog or catalog_of(card)
            text = dump(compose(catalog, card))
            if a.out:
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
                with open(a.out, "w") as f:
                    f.write(text)
            else:
                sys.stdout.write(text)
            return 0
        if not (a.catalog and a.card):
            ap.error("CATALOG CARD, --card-json FILE, or --all")
        text = emit(a.catalog, a.card, a.out)
        if not a.out:
            sys.stdout.write(text)
        return 0
    except (EmitError, workload.WorkloadError, OSError, yaml.YAMLError) as e:
        print(f"emit: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
