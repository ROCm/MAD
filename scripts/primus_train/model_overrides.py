#!/usr/bin/env python3
"""Primus CLI overrides for Megatron configs whose YAML alone does not run as benchmarked.

The table below is what scripts/primus_megatron-lm used to hardcode per model. A rule
applies to one config (by basename) and, optionally, to one GPU model family or world
size; a rule with a variant applies only to the madengine model of that variant
(get_models_json.py emits one, e.g. ``..._deepseek_v3-BF16-pretrain-proxy``) and the
base model ignores it.

run.sh prints the matching arguments before the caller's own, so anything the
deployment passes (model args, PRIMUS_EXTRA_ARGS) still wins. Knobs the old script
read from the environment are expressed through PRIMUS_EXTRA_ARGS instead:
  SCOUT_HF_TOKENIZER=X      -> --tokenizer_model X
  SCOUT_MOE_DISPATCHER=ag   -> --moe_token_dispatcher_type allgather --moe_shared_expert_overlap false
  PRIMUS_USE_DEEPEP=1       -> --use_turbo_deepep true --moe_shared_expert_overlap false
  PRIMUS_SANITY_TRAIN_ITERS -> the kimi_k2_thinking "proxy" variant

Usage: model_overrides.py <config.yaml> --device MI355X --world-size 16 [--variant proxy]
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

GFX942 = frozenset({"MI300X", "MI325X"})
GFX950 = frozenset({"MI355X", "MI350X"})


@dataclass(frozen=True)
class Rule:
    config: str
    args: tuple
    variant: str = ""
    devices: Optional[frozenset] = None
    when: Optional[Callable[[str, int], bool]] = field(default=None, compare=False)

    def matches(self, config, variant, device, world_size):
        if self.config != config or self.variant != variant:
            return False
        if self.devices is not None and device not in self.devices:
            return False
        return self.when is None or self.when(device, world_size)


def _gpt_oss_needs_memory_headroom(device, world_size):
    # The MI355X config (MBS 8, recompute 4 layers) is tuned for 288 GB HBM at 32+ GPUs.
    # It OOMs on 192 GB gfx942, and on any HBM size at 16 GPUs: TP1 x PP2 fixes model
    # parallelism at 2, so DP is 8 and the distributed optimizer costs ~6 GB/GPU more.
    return device in GFX942 or world_size <= 16


_KIMI = "kimi_k2_thinking-BF16-pretrain"

RULES = (
    # Llama-4-Scout: the model YAML points at the gated meta-llama tokenizer; the mirror
    # needs no per-account license grant. The dispatcher is pinned to the one the
    # shipped moe_shared_expert_overlap requires.
    Rule("llama4_17B16E-BF16-pretrain", (
        "--tokenizer_model", "unsloth/Llama-4-Scout-17B-16E",
        "--moe_token_dispatcher_type", "alltoall",
    )),
    # Kimi-K2-Thinking (1T total): MBS 1 and full recompute are required to fit, not tuning.
    Rule(_KIMI, (
        "--micro_batch_size", "1", "--global_batch_size", "128",
        "--recompute_granularity", "full", "--recompute_method", "uniform",
        "--recompute_num_layers", "1",
    ), devices=GFX950),
    Rule(_KIMI, ("--micro_batch_size", "2", "--global_batch_size", "32"), devices=GFX942),
    Rule("gpt_oss_120B-BF16-pretrain", ("--micro_batch_size", "2", "--recompute_num_layers", "9"),
         when=_gpt_oss_needs_memory_headroom),
    Rule("gpt_oss_120B-FP8-pretrain", ("--micro_batch_size", "1", "--recompute_num_layers", "9"),
         when=_gpt_oss_needs_memory_headroom),
    # Proxies: the full model shape cut down to a few layers, for fast single-node checks.
    Rule("llama3.1_70B-FP8-pretrain", (
        "--num_layers", "40", "--fp8", "hybrid", "--no_fp8_weight_transpose_cache", "true",
    ), variant="proxy", devices=GFX942),
    Rule("deepseek_v3-BF16-pretrain", (
        "--num_layers", "3", "--moe_layer_freq", "1", "--train_iters", "50",
        "--micro_batch_size", "8", "--global_batch_size", "64",
        "--moe_use_fused_router_with_aux_score", "True", "--moe_permute_fusion", "True",
        "--pipeline_model_parallel_size", "1", "--pipeline_model_parallel_layout", "null",
        "--recompute_granularity", "null", "--recompute_layer_ids", "null",
        "--overlap_grad_reduce", "false", "--overlap_param_gather", "false",
        "--gradient_accumulation_fusion", "false",
    ), variant="proxy", devices=GFX950),
    Rule("deepseek_v3-BF16-pretrain", ("--num_layers", "3", "--moe_layer_freq", "1"),
         variant="proxy", devices=GFX942),
    Rule("mixtral_8x22B_v0.1-BF16-pretrain", (
        "--num_layers", "4", "--pipeline_model_parallel_size", "1",
    ), variant="proxy"),
    Rule(_KIMI, (
        "--num_layers", "3", "--moe_layer_freq", "1",
        "--micro_batch_size", "1", "--global_batch_size", "32",
    ), variant="proxy"),
)

# Models discovery cannot find in the Primus submodule. Kimi-K2-Thinking's YAMLs live in
# docker/kimi_k2_configs/ and only the RCCL overlay image carries them. GPT-OSS-120B
# ships an MI355X config only; gfx942 runs it with the memory overrides above.
OVERLAY_DOCKERFILE = "../../docker/primus_megatron_train_rccl_overlay"
EXTRA_MODELS = (
    *(("megatron", arch, f"examples/megatron/configs/{arch}/{_KIMI}.yaml", OVERLAY_DOCKERFILE)
      for arch in ("MI355X", "MI300X", "MI325X")),
    *(("megatron", arch, f"examples/megatron/configs/MI355X/gpt_oss_120B-{p}-pretrain.yaml", None)
      for arch in ("MI300X", "MI325X") for p in ("BF16", "FP8")),
)


def config_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def override_args(config, device, world_size, variant=""):
    """CLI arguments for *config* (path or basename) on *device* at *world_size*."""
    name = config_name(config)
    out = []
    for rule in RULES:
        if rule.matches(name, variant, device, world_size):
            out.extend(rule.args)
    return out


def variants_for(config, arch):
    """Variants with at least one rule for *config* on GPU model *arch*, in table order."""
    name = config_name(config)
    seen = []
    for rule in RULES:
        if rule.config != name or not rule.variant or rule.variant in seen:
            continue
        if rule.devices is None or arch in rule.devices:
            seen.append(rule.variant)
    return seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--device", default="")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--variant", default="")
    args = parser.parse_args(argv)
    for arg in override_args(args.config, args.device, args.world_size, args.variant):
        print(arg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
