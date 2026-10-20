#!/usr/bin/env python3
"""Global batch size for a Primus Megatron config at the current world size.

Primus configs tune global_batch_size for a single 8-GPU node. Megatron requires
global_batch_size % (micro_batch_size * data_parallel_size) == 0, so on any other
world size the run aborts before the first step. This prints the smallest valid
global batch size at or above the configured one, and nothing when the configured
value is already valid (single-node runs stay unchanged) or cannot be resolved.

With --micro-batch it prints the micro batch size the run uses instead (YAML value
after CLI overrides), for the perf report.

Usage: scaleout_batch.py <config.yaml> --world-size N [-- <primus cli overrides>]
       scaleout_batch.py <config.yaml> --micro-batch [-- <primus cli overrides>]
"""

import argparse
import os
import re
import sys

import yaml

_ENV_REF = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")
_KEYS = (
    "micro_batch_size",
    "global_batch_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
)


def resolve_int(value):
    """Return value as int, expanding a ${VAR:default} reference; None if not an int."""
    if isinstance(value, str):
        match = _ENV_REF.fullmatch(value.strip())
        if match:
            value = os.environ.get(match.group(1), match.group(2) or "")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def trainer_overrides(cfg):
    modules = (cfg or {}).get("modules") or {}
    for key in ("pre_trainer", "post_trainer"):
        module = modules.get(key) or {}
        if module:
            return module.get("overrides") or {}
    return {}


def cli_overrides(args):
    """Collect `--<key> <value>` pairs for the batch/parallel keys from Primus CLI args."""
    found = {}
    for i, arg in enumerate(args[:-1]):
        key = arg[2:] if arg.startswith("--") else None
        if key in _KEYS:
            found[key] = args[i + 1]
    return found


def adjusted_global_batch_size(overrides, world_size):
    """Return the renormalized global batch size, or None when no change is needed."""
    mbs = resolve_int(overrides.get("micro_batch_size"))
    gbs = resolve_int(overrides.get("global_batch_size"))
    if not mbs or not gbs or mbs <= 0 or world_size <= 0:
        return None
    model_parallel = 1
    for key in _KEYS[2:]:
        model_parallel *= resolve_int(overrides.get(key)) or 1
    if world_size % model_parallel:
        return None
    unit = mbs * (world_size // model_parallel)
    if gbs % unit == 0:
        return None
    return -(-gbs // unit) * unit


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--world-size", type=int)
    mode.add_argument("--micro-batch", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        overrides = dict(trainer_overrides(yaml.safe_load(f)))
    overrides.update(cli_overrides(extra))
    if args.micro_batch:
        mbs = resolve_int(overrides.get("micro_batch_size"))
        if mbs is not None:
            print(mbs)
        return 0
    adjusted = adjusted_global_batch_size(overrides, args.world_size)
    if adjusted is not None:
        print(adjusted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
