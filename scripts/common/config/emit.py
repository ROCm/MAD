#!/usr/bin/env python3
"""A card's configuration as a workload file that fits the shared layers.

A card's run configuration is placed where each part belongs, not copied:

    recipe, connector env  stay where they live (the launcher's models.yaml, connectors/); the file
                           names neither, so a fix there reaches the card
    shared layers          `defaults:` -- every layer of scripts/common/workloads (serving shapes,
                           benchmarks, model families, base) whose values the card has, all of them
    the card's own part    model.env -- only what no chosen layer gives (and BENCHMARK_SCRIPT, which
                           selects the benchmark entries)
    topology               the card's MAD_TOPOLOGY spec (a path), when it has one
    card, image            what the file was made from; when the build recorded the image's stack,
                           `image.spec` is that image generalized over compat (madengine) and
                           `image.unresolved` what compat cannot express yet

A card that already has a workload file (MAD_CONFIG) keeps its own sections (recipes:,
connectors:, topology:). The file resolves to exactly the environment the card runs with:
snapshot.py --from-emitted runs every card from it, on every GPU its recipe covers, and requires
the same engine commands and container environment. `madengine build --emit-config` and the
pipeline's STANDALONE mode call this, so every run records its configuration in this form.

    emit.py CATALOG CARD [-o FILE]       one card (CATALOG: vllm_dissag, vllm_multinode, sglang_disagg)
    emit.py --card-json FILE [-o FILE]   a card as madengine has it; its catalog from its scripts path
    emit.py --all -o DIR [--suggest]     every card, DIR/<catalog>/<card>.yaml; --suggest lists
                                         values many cards keep that no shared layer holds yet
Standard library and PyYAML only.
"""
import argparse
import collections
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", ".."))
LAYERS = os.path.join(SCRIPTS, "common", "workloads")
sys.path.insert(0, HERE)
import workload  # noqa: E402

CATALOGS = ("vllm_dissag", "vllm_multinode", "sglang_disagg")
VERSION = 2
OWN = ("BENCHMARK_SCRIPT",)        # always in the card's own part: it selects the benchmark entries


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


def layers():
    """{path: data} of every shared layer (scripts/common/workloads), merged over its own defaults."""
    out = {}
    for root, _, files in os.walk(LAYERS):
        for f in sorted(files):
            if f.endswith(".yaml"):
                p = os.path.join(root, f)
                out[p] = workload.load(p)
    return out


def card_env(catalog, card):
    """(the environment the card runs with, its workload file's data or {}): its workload file
    (MAD_CONFIG) under its env_vars, as the launcher applies them."""
    env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
    path = workload.find(env.pop("MAD_CONFIG", ""), os.path.join(SCRIPTS, catalog))
    data = workload.load(path) if path else {}
    if data:
        env = {**workload.env_of(data, env.get("BENCHMARK_SCRIPT")), **env}
    return env, data


def choose(env, candidates):
    """The layers whose every value the card has, largest first, each adding something; and the
    names they give."""
    kind = env.get("BENCHMARK_SCRIPT")
    given, chosen = set(), []
    fits = []
    for path, data in candidates.items():
        values = workload.env_of(data, kind)
        if values and all(env.get(k) == v for k, v in values.items()):
            fits.append((len(values), path, values))
    for _, path, values in sorted(fits, key=lambda x: (-x[0], x[1])):
        if set(values) - given:
            chosen.append(path)
            given |= set(values)
    # Written base, serving shape, model family, benchmark (their values never conflict: each is the card's).
    rank = {"base.yaml": 0, "serve": 1, "family": 2, "bench": 3}
    chosen.sort(key=lambda p: (rank.get(os.path.basename(p), rank.get(os.path.basename(os.path.dirname(p)), 4)), p))
    return chosen, given


def compose(catalog, card, at=None):
    """The card's workload file (a dict). `at`: the directory the file is written to, which its
    `defaults:` paths are relative to (default: the catalog's workloads/)."""
    if catalog not in CATALOGS:
        raise EmitError(f"{catalog}: not a multinode catalog ({', '.join(CATALOGS)})")
    at = os.path.abspath(at or os.path.join(SCRIPTS, catalog, "workloads"))
    env, data = card_env(catalog, card)
    topology = env.pop("MAD_TOPOLOGY", None) or data.get("topology")
    chosen, given = choose(env, layers())
    own = {k: v for k, v in env.items() if k not in given or k in OWN}

    out = {"version": VERSION, "card": {k: card[k] for k in ("name", "scripts", "tags") if k in card}}
    out["card"]["catalog"] = catalog
    image = {k: card[k] for k in ("dockerfile", "docker_build_arg", "image_spec") if card.get(k)}
    # The image generalized from its build (madengine: the stack it recorded, over compat).
    if card.get("_image_spec"):
        image["spec"] = card["_image_spec"]
    if card.get("_image_unresolved"):
        image["unresolved"] = card["_image_unresolved"]
    if image:
        out["image"] = _escape(image)        # written as text: ${images.<stack>} is madengine's reference
    if chosen:
        out["defaults"] = [os.path.relpath(p, at) for p in chosen]
    for section in ("recipes", "connectors"):          # the card's own, when its file has them
        if data.get(section):
            out[section] = _escape(data[section])
    if topology:
        out["topology"] = topology
    out["model"] = {"env": _escape(own)}
    return out


def suggest(composed, min_cards=3):
    """Values several cards of a catalog keep in their own part, which no shared layer holds:
    candidates for a new layer. [(catalog, {name: value}, [cards])]."""
    by = collections.defaultdict(list)
    for catalog, d in composed:
        for k, v in (d["model"]["env"] or {}).items():
            if k not in OWN and k not in ("MODEL_NAME", "DOCKER_IMAGE_NAME"):
                by[(catalog, k, v)].append(d["card"]["name"])
    groups = collections.defaultdict(dict)
    for (catalog, k, v), cards in by.items():
        if len(cards) >= min_cards:
            groups[(catalog, tuple(sorted(cards)))][k] = v
    return [(c, values, list(cards)) for (c, cards), values in sorted(groups.items()) if len(values) > 1]


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):      # write shared values out in full: no &id001 / *id001
        return True


def dump(data):
    head = (f"# Generated by scripts/common/config/emit.py from card {data['card']['name']} "
            f"({data['card']['catalog']}).\n"
            "# Point the card's MAD_CONFIG here to run it from this file; edit freely.\n")
    return head + yaml.dump(data, Dumper=_Dumper, sort_keys=False, width=1 << 16, allow_unicode=True)


def _write(out, text):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalog", nargs="?")
    ap.add_argument("card", nargs="?")
    ap.add_argument("--all", action="store_true", help="every card of the three catalogs")
    ap.add_argument("--card-json", help="the card as JSON (a file), instead of CATALOG CARD")
    ap.add_argument("--catalog", dest="catalog_opt", help="with --card-json: its catalog (default: from its scripts path)")
    ap.add_argument("--suggest", action="store_true", help="with --all: candidates for new shared layers")
    ap.add_argument("-o", "--out", help="file (one card) or directory (--all)")
    a = ap.parse_args(argv)
    try:
        if a.all:
            if not a.out:
                ap.error("--all needs -o DIR")
            composed = []
            for catalog in CATALOGS:
                for c in json.loads(open(os.path.join(SCRIPTS, catalog, "models.json")).read()):
                    out = os.path.join(a.out, catalog, f"{c['name']}.yaml")
                    d = compose(catalog, c, os.path.dirname(out))
                    _write(out, dump(d))
                    composed.append((catalog, d))
            print(f"emit: {len(composed)} card(s) -> {a.out}")
            if a.suggest:
                for catalog, values, cards in suggest(composed):
                    print(f"suggest {catalog}: a layer with {values} -- kept by {len(cards)} cards: "
                          f"{', '.join(cards)}")
            return 0
        if a.card_json:
            card = json.loads(open(a.card_json).read())
            catalog = a.catalog_opt or a.catalog or catalog_of(card)
        elif a.catalog and a.card:
            catalog, card = a.catalog, card_of(a.catalog, a.card)
        else:
            ap.error("CATALOG CARD, --card-json FILE, or --all")
        text = dump(compose(catalog, card, os.path.dirname(os.path.abspath(a.out)) if a.out else None))
        if a.out:
            _write(a.out, text)
        else:
            sys.stdout.write(text)
        return 0
    except (EmitError, workload.WorkloadError, OSError, yaml.YAMLError) as e:
        print(f"emit: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
