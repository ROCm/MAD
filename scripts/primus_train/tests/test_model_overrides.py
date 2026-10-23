#!/usr/bin/env python3
"""model_overrides.py rules and the models get_models_json.py derives from them."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_overrides as mo  # noqa: E402

CFG = "examples/megatron/configs/{arch}/{name}.yaml"


def args_for(name, device, world_size=8, variant=""):
    return mo.override_args(CFG.format(arch=device, name=name), device, world_size, variant)


def test_unlisted_config_has_no_overrides():
    assert args_for("llama3.1_8B-BF16-pretrain", "MI355X") == []


def test_device_rules_pick_the_gpu_family():
    assert args_for("kimi_k2_thinking-BF16-pretrain", "MI300X") == [
        "--micro_batch_size", "2", "--global_batch_size", "32",
    ]
    assert args_for("kimi_k2_thinking-BF16-pretrain", "MI355X")[:4] == [
        "--micro_batch_size", "1", "--global_batch_size", "128",
    ]


def test_gpt_oss_memory_rule_follows_device_and_world_size():
    name = "gpt_oss_120B-FP8-pretrain"
    expected = ["--micro_batch_size", "1", "--recompute_num_layers", "9"]
    assert args_for(name, "MI355X", world_size=16) == expected
    assert args_for(name, "MI355X", world_size=32) == []
    assert args_for(name, "MI325X", world_size=64) == expected


def test_variant_rules_replace_base_rules():
    proxy = args_for("kimi_k2_thinking-BF16-pretrain", "MI355X", variant="proxy")
    assert "--recompute_granularity" not in proxy
    assert proxy[:2] == ["--num_layers", "3"]
    assert args_for("llama3.1_70B-FP8-pretrain", "MI300X", variant="proxy")[:2] == [
        "--num_layers", "40",
    ]
    assert args_for("llama3.1_70B-FP8-pretrain", "MI355X", variant="proxy") == []


def test_variants_for_respects_devices():
    assert mo.variants_for("x/deepseek_v3-BF16-pretrain.yaml", "MI355X") == ["proxy"]
    assert mo.variants_for("x/llama3.1_70B-FP8-pretrain.yaml", "MI355X") == []
    assert mo.variants_for("x/mixtral_8x22B_v0.1-BF16-pretrain.yaml", "MI455X") == ["proxy"]


def test_cli_prints_one_argument_per_line(capsys):
    assert mo.main(["a/llama4_17B16E-BF16-pretrain.yaml", "--device", "MI355X"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "--tokenizer_model", "unsloth/Llama-4-Scout-17B-16E",
        "--moe_token_dispatcher_type", "alltoall",
    ]


def _discover(monkeypatch, tmp_path, configs):
    class CustomModel:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    utils = types.ModuleType("madengine.utils.discover_models")
    utils.CustomModel = CustomModel
    for mod in ("madengine", "madengine.utils"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "madengine.utils.discover_models", utils)
    sys.modules.pop("get_models_json", None)
    import get_models_json as gmj

    for rel in configs:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    monkeypatch.setattr(gmj, "PRIMUS_ROOT", str(tmp_path))
    monkeypatch.setattr(
        gmj, "CONFIGS_GLOB", str(tmp_path / "examples" / "*" / "configs" / "**" / "*.yaml")
    )
    return {m.name: m for m in gmj.list_models()}


def test_discovery_adds_variants_and_extra_models(monkeypatch, tmp_path):
    models = _discover(monkeypatch, tmp_path, [
        CFG.format(arch="MI355X", name="deepseek_v3-BF16-pretrain"),
        CFG.format(arch="MI355X", name="gpt_oss_120B-BF16-pretrain"),
        CFG.format(arch="MI300X", name="llama3.1_70B-FP8-pretrain"),
    ])
    proxy = models["megatron_MI355X_deepseek_v3-BF16-pretrain-proxy"]
    assert proxy.args.endswith(" --variant proxy")
    assert "proxy" in proxy.tags
    assert "megatron_MI300X_llama3.1_70B-FP8-pretrain-proxy" in models

    kimi = models["megatron_MI300X_kimi_k2_thinking-BF16-pretrain"]
    assert kimi.dockerfile == mo.OVERLAY_DOCKERFILE
    assert kimi.training_precision == "bf16"
    assert "megatron_MI355X_kimi_k2_thinking-BF16-pretrain-proxy" in models

    gpt = models["megatron_MI325X_gpt_oss_120B-FP8-pretrain"]
    assert gpt.args == "--config_path " + CFG.format(arch="MI355X", name="gpt_oss_120B-FP8-pretrain")
    assert gpt.dockerfile == "../../docker/primus"
