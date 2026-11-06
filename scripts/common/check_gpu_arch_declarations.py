#!/usr/bin/env python3
"""A multinode card must say the same thing about GPUs in both places it says it.

  skip_gpu_arch  on the card: read by orchestrators (madengine, or a CI
                 preflight) BEFORE an allocation, to skip the card.
  GPU_ARCHS      on the recipe: read by the launcher (cluster_require_gpu_arch in
                 cluster.sh) ON the allocation, to refuse the wrong nodes. For
                 scripts/vllm_dissag it lives in models.yaml under the card's
                 MODEL_NAME; elsewhere, in the card's env_vars. A card's own
                 env_vars value wins, as it does at run time.

If they disagree, one path skips a card the other runs, or the launcher refuses
nodes the orchestrator chose for it -- the two CI paths stop being equivalent.
Rules, per card that declares GPU_ARCHS:
  1. no arch it supports is also in skip_gpu_arch;
  2. every KNOWN arch it does not support is in skip_gpu_arch, so an orchestrator
     skips it without spending an allocation to find out.
A card with no GPU_ARCHS is unrestricted and is not checked.

Usage: check_gpu_arch_declarations.py [MAD_ROOT]   (default: this checkout)
"""
import glob
import json
import os
import sys

import yaml

# The architectures multinode cards are run on. A new one goes here, and rule 2
# then makes every restricted card say whether it supports it.
KNOWN_ARCHS = ("gfx942", "gfx950")


def split(v):
    return {a for a in str(v or "").replace(",", " ").split() if a}


def main(root):
    recipes = {}
    ypath = os.path.join(root, "scripts/vllm_dissag/models.yaml")
    if os.path.exists(ypath):
        recipes = yaml.safe_load(open(ypath)) or {}

    problems, checked = [], 0
    for path in sorted(glob.glob(os.path.join(root, "scripts/*/models.json"))):
        group = os.path.basename(os.path.dirname(path))
        for card in json.load(open(path)):
            if not ((card.get("distributed") or {}).get("launcher") or card.get("slurm")):
                continue
            env = card.get("env_vars") or {}
            allowed = split(env.get("GPU_ARCHS"))
            source = "card env_vars"
            if not allowed and group == "vllm_dissag":
                rec = recipes.get(env.get("MODEL_NAME", "")) or {}
                allowed = split((rec.get("env") or {}).get("GPU_ARCHS"))
                source = "models.yaml %s" % env.get("MODEL_NAME")
            if not allowed:
                continue
            checked += 1
            skip = split(card.get("skip_gpu_arch"))
            name = "%s/%s" % (group, card["name"])
            both = allowed & skip
            if both:
                problems.append("%s: supports %s (%s) but skip_gpu_arch also lists it"
                                % (name, ",".join(sorted(both)), source))
            missing = set(KNOWN_ARCHS) - allowed - skip
            if missing:
                problems.append("%s: supports only %s (%s); add %s to skip_gpu_arch"
                                % (name, ",".join(sorted(allowed)), source,
                                   ",".join(sorted(missing))))

    for p in problems:
        print("  FAIL  " + p)
    print("checked %d arch-restricted multinode card(s): %s"
          % (checked, "consistent" if not problems else "%d problem(s)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(here))))
