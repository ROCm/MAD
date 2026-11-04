#!/usr/bin/env python3
"""Global batch size for a Primus Megatron config at the current world size.

Primus configs tune global_batch_size for a single 8-GPU node. Megatron requires
global_batch_size % (micro_batch_size * data_parallel_size) == 0, so on any other
world size the run aborts before the first step. With an interleaved pipeline
(virtual pipeline stages, set explicitly or implied by a pipeline_model_parallel_layout)
the number of microbatches must also suit the schedule:
Megatron's interleaved scheduler takes microbatches in groups of
microbatch_group_size_per_vp_stage (default: the pipeline size) and rejects a final
group smaller than the pipeline size, which with the default group means the
microbatch count must be a multiple of the pipeline size. This prints the smallest
valid global batch size at or above the configured one, and nothing when the
configured value is already valid (single-node runs stay unchanged) or cannot be
resolved.

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
# Keys of Megatron's interleaved pipeline schedule (virtual pipeline stages). A pipeline
# layout implies them (stages // PP) and wins over the explicit counts; either explicit
# count makes the schedule interleaved; the group key overrides its microbatch group size.
_VPP_KEYS = ("virtual_pipeline_model_parallel_size", "num_virtual_stages_per_pipeline_rank")
_LAYOUT_KEY = "pipeline_model_parallel_layout"
_GROUP_KEY = "microbatch_group_size_per_vp_stage"
_CLI_KEYS = _KEYS + _VPP_KEYS + (_LAYOUT_KEY, _GROUP_KEY)


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
    """Collect the batch/parallel keys from Primus CLI args; the last occurrence wins.

    Primus (parse_cli_overrides) takes `--key value`, `--key=value` and `key=value`, so
    all three forms are read here: a global batch size passed as `--global_batch_size=64`
    must not be missed and renormalized behind the caller's back.
    """
    found = {}
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key[2:] if key.startswith("--") else key
            if key in _CLI_KEYS:
                found[key] = value
        elif arg.startswith("--") and arg[2:] in _CLI_KEYS and i < len(args):
            found[arg[2:]] = args[i]
            i += 1
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
    pipeline = resolve_int(overrides.get("pipeline_model_parallel_size")) or 1
    group = interleaved_group_size(overrides, pipeline)
    microbatches = -(-gbs // unit)
    # Terminates within `group` steps: a multiple of the group size always qualifies.
    while not _schedule_accepts(microbatches, group, pipeline):
        microbatches += 1
    adjusted = microbatches * unit
    return None if adjusted == gbs else adjusted


def layout_stage_count(layout):
    """Pipeline stages in a layout, counted as Megatron does (PipelineParallelLayerLayout).

    Commas are cosmetic, `(ab|cd)*n` and `x*n` unroll, stages are split on `|`; a list
    layout has one entry per stage.
    """
    if isinstance(layout, (list, tuple)):
        return len(layout)
    text = layout.replace(",", "")
    for pattern in (r"\(([^)]+)\)\*(\d+)", r"(.)\*(\d+)"):
        text = re.sub(pattern, lambda m: m.group(1) * int(m.group(2)), text)
    return len(text.split("|"))


def _layout_of(overrides):
    """The pipeline layout in effect (env references expanded), or None when there is none."""
    value = overrides.get(_LAYOUT_KEY)
    if isinstance(value, str):
        match = _ENV_REF.fullmatch(value.strip())
        if match:
            value = os.environ.get(match.group(1), match.group(2) or "")
        value = value.strip()
        if value.lower() in ("", "null", "none", "~"):
            return None
    return value if isinstance(value, (str, list, tuple)) and value else None


def virtual_stages_per_rank(overrides, pipeline_size):
    """Virtual pipeline stages per rank as Megatron's validate_args derives them (1 = none).

    A layout decides on its own: stages // PP. Otherwise the explicit counts do.
    """
    layout = _layout_of(overrides)
    if layout is not None:
        stages = layout_stage_count(layout)
        # A layout that does not divide into the pipeline is rejected whatever the batch.
        return stages // pipeline_size if stages % pipeline_size == 0 else 1
    return max((resolve_int(overrides.get(key)) or 0) for key in _VPP_KEYS) or 1


def interleaved_group_size(overrides, pipeline_size):
    """Microbatch group size of Megatron's interleaved schedule, or None when it is not used.

    The schedule is interleaved when there are virtual pipeline stages (more than one per
    rank) and a real pipeline. The group size defaults to the pipeline size; one smaller
    than that is invalid in Megatron regardless of the batch, so it is not our call.
    """
    if pipeline_size <= 1 or virtual_stages_per_rank(overrides, pipeline_size) <= 1:
        return None
    group = resolve_int(overrides.get(_GROUP_KEY)) or pipeline_size
    return group if group >= pipeline_size else None


def _schedule_accepts(microbatches, group, pipeline_size):
    """Megatron's interleaved scheduler wants M >= group and a last group of 0 or >= PP."""
    if group is None:
        return True
    remainder = microbatches % group
    return microbatches >= group and (remainder == 0 or remainder >= pipeline_size)


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
