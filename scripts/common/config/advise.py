#!/usr/bin/env python3
"""Advice, never a failure: what multinode cards and recipes could gain from the newer ways.

    advise.py            notes per catalog; always exits 0

- A recipe without `roles:` gets no check against the GPU and the image before a run, and no
  arguments added per GPU (docs/configuring-a-model.md, "Refused before queueing").
- Cards that differ only by GPU (skip_gpu_arch, GPU_ARCHS, a GPU-named recipe alias, the build's
  GPU) could be one card, with the recipe's per-GPU `variants:` ("One card for every GPU").
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", ".."))

CATALOGS = {"vllm_dissag": "vllm_dissag", "vllm_multinode": "vllm_dissag", "sglang_disagg": "sglang_disagg"}
GPU_ENV = {"GPU_ARCHS", "EP_TP_SIZE", "DOCKER_IMAGE_NAME", "MAD_GPU_ARCH"}


def _base(recipes, name):
    seen = set()
    while isinstance(recipes.get(name), dict) and "alias" in recipes[name] and name not in seen:
        seen.add(name)
        name = recipes[name]["alias"]
    return name


def advice():
    notes = []
    for src in sorted(set(CATALOGS.values())):
        path = os.path.join(SCRIPTS, src, "models.yaml")
        recipes = yaml.safe_load(open(path)) or {}
        for name, r in recipes.items():
            if name.startswith("_"):            # a YAML anchor block, not a recipe
                continue
            if isinstance(r, dict) and "alias" not in r and not r.get("roles"):
                notes.append(f"{src}/models.yaml {name}: no roles: -- add them to be refused before a run "
                             "on a GPU or image that cannot serve it, with the reason")
    for catalog, src in CATALOGS.items():
        recipes = yaml.safe_load(open(os.path.join(SCRIPTS, src, "models.yaml"))) or {}
        groups = {}
        for c in json.load(open(os.path.join(SCRIPTS, catalog, "models.json"))):
            env = {k: str(v) for k, v in (c.get("env_vars") or {}).items() if k not in GPU_ENV}
            if env.get("MAD_CONFIG"):
                continue
            env["MODEL_NAME"] = _base(recipes, env.get("MODEL_NAME", ""))
            build = {k: v for k, v in (c.get("docker_build_arg") or {}).items() if k != "MAD_SYSTEM_GPU_ARCHITECTURE"}
            key = json.dumps([c.get("scripts"), env, build, (c.get("slurm") or {}).get("nodes")], sort_keys=True)
            groups.setdefault(key, []).append(c["name"])
        for names in groups.values():
            if len(names) < 2:
                continue
            agnostic = [x for x in names if x.endswith("_auto")]
            if agnostic:
                for x in names:
                    if x not in agnostic:
                        notes.append(f"{catalog}: {x} has a GPU-agnostic twin, {agnostic[0]}, that runs the same "
                                     "on every GPU it supports")
            else:
                notes.append(f"{catalog}: {', '.join(names)} differ only by GPU -- one card, with the recipe's "
                             "per-GPU variants:, would serve every GPU")
    return notes


def main():
    notes = advice()
    for n in notes:
        print(f"note  {n}")
    print(f"advise: {len(notes)} note(s) (advice only; nothing fails)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
