#!/usr/bin/env python3
"""scaleout_batch must renormalize global_batch_size only when Megatron would reject it."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scaleout_batch.py"

_spec = importlib.util.spec_from_file_location("scaleout_batch", HELPER)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

adjusted = _mod.adjusted_global_batch_size


def test_single_node_config_is_unchanged():
    assert adjusted({"micro_batch_size": 16, "global_batch_size": 128}, 8) is None


def test_two_nodes_round_up_to_mbs_times_dp():
    # qwen3_32B MI355X: MBS 16, GBS 128 -> 16 GPUs need a multiple of 256.
    assert adjusted({"micro_batch_size": 16, "global_batch_size": 128}, 16) == 256
    # llama3.3_70B MI355X BF16: MBS 6, GBS 48 -> multiple of 96.
    assert adjusted({"micro_batch_size": 6, "global_batch_size": 48}, 16) == 96


def test_already_divisible_is_unchanged():
    assert adjusted({"micro_batch_size": 1, "global_batch_size": 512}, 32) is None


def test_model_parallel_shrinks_data_parallel():
    # TP 8 x PP 2 on 32 GPUs leaves DP 2: unit is 2 * 2 = 4, GBS 6 -> 8.
    overrides = {
        "micro_batch_size": 2,
        "global_batch_size": 6,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 2,
    }
    assert adjusted(overrides, 32) == 8


def test_env_reference_resolves_to_default(monkeypatch):
    monkeypatch.delenv("PRIMUS_GBS", raising=False)
    overrides = {"micro_batch_size": 4, "global_batch_size": "${PRIMUS_GBS:32}"}
    assert adjusted(overrides, 16) == 64


def test_unresolvable_values_are_left_alone():
    assert adjusted({"micro_batch_size": "${UNSET_MBS:}", "global_batch_size": 32}, 16) is None
    assert adjusted({"global_batch_size": 32}, 16) is None


def test_micro_batch_prints_the_effective_value(capsys):
    cfg = "modules:\n  pre_trainer:\n    overrides:\n      micro_batch_size: 16\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
    args = ["--micro_batch_size", "4", "--micro_batch_size", "2"]
    assert _mod.main([f.name, "--micro-batch", "--", *args]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert _mod.main([f.name, "--micro-batch"]) == 0
    assert capsys.readouterr().out.strip() == "16"


def test_cli_overrides_take_precedence(capsys):
    cfg = "modules:\n  pre_trainer:\n    overrides:\n      micro_batch_size: 16\n      global_batch_size: 128\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
    assert _mod.main([f.name, "--world-size", "16", "--", "--micro_batch_size", "2"]) == 0
    assert capsys.readouterr().out == ""
    assert _mod.main([f.name, "--world-size", "16"]) == 0
    assert capsys.readouterr().out.strip() == "256"


def test_cli_override_forms_all_parse():
    # Primus accepts `--key value`, `--key=value` and `key=value`.
    for args in (
        ["--micro_batch_size", "3"], ["--micro_batch_size=3"], ["micro_batch_size=3"],
    ):
        assert _mod.cli_overrides(args) == {"micro_batch_size": "3"}


def test_cli_override_last_occurrence_wins_across_forms():
    args = ["--micro_batch_size", "3", "--seq_length=4096", "micro_batch_size=5", "--use_x"]
    assert _mod.cli_overrides(args) == {"micro_batch_size": "5"}


def test_cli_override_ignores_values_of_other_flags():
    # A value that merely contains a key name, or a flag without a value, is not a key.
    args = ["--log_file", "/tmp/micro_batch_size=2.log", "--flag", "--global_batch_size"]
    assert _mod.cli_overrides(args) == {}


# v26.8 gpt_oss_120B-BF16 on MI355X: MBS 8, GBS 2048, TP 1, PP 2, VPP 2 (interleaved schedule).
GPT_OSS = {
    "micro_batch_size": 8,
    "global_batch_size": 2048,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 2,
    "virtual_pipeline_model_parallel_size": 2,
}


def test_interleaved_pipeline_needs_an_accepted_microbatch_count():
    # 48 GPUs: DP 24, unit 192. 2048 rounds to 11 microbatches, whose final group of one is
    # smaller than PP 2, so Megatron's interleaved scheduler raises; 12 microbatches work.
    assert adjusted(GPT_OSS, 48) == 2304


def test_interleaved_pipeline_keeps_valid_batches():
    # 8 and 16 GPUs give 64 and 32 microbatches, both multiples of PP 2.
    assert adjusted(GPT_OSS, 8) is None
    assert adjusted(GPT_OSS, 16) is None


def test_plain_pipeline_keeps_the_data_parallel_rounding():
    plain = {k: v for k, v in GPT_OSS.items() if k != "virtual_pipeline_model_parallel_size"}
    assert adjusted(plain, 48) == 2112


def test_virtual_stages_per_rank_key_also_means_interleaved():
    overrides = {k: v for k, v in GPT_OSS.items() if k != "virtual_pipeline_model_parallel_size"}
    overrides["num_virtual_stages_per_pipeline_rank"] = 4
    assert adjusted(overrides, 48) == 2304


def test_virtual_pipeline_env_reference_resolves(monkeypatch):
    overrides = {**GPT_OSS, "virtual_pipeline_model_parallel_size": "${PRIMUS_VP:2}"}
    monkeypatch.delenv("PRIMUS_VP", raising=False)
    assert adjusted(overrides, 48) == 2304
    monkeypatch.setenv("PRIMUS_VP", "1")
    assert adjusted(overrides, 48) == 2112


def test_microbatch_group_size_widens_the_rule():
    # Group of 4 with PP 2: a final group of 2 or 3 is fine, of 1 is not.
    group4 = {**GPT_OSS, "microbatch_group_size_per_vp_stage": 4}
    assert adjusted(group4, 48) == 2112                                   # 11 microbatches: 11 % 4 = 3
    assert adjusted({**group4, "global_batch_size": 1728}, 48) == 1920    # 9 -> 10 (9 % 4 = 1)


def test_group_smaller_than_pipeline_is_not_ours_to_fix():
    assert adjusted({**GPT_OSS, "microbatch_group_size_per_vp_stage": 1}, 48) == 2112


def test_cli_can_turn_the_interleaved_rule_on_and_off(capsys):
    cfg = (
        "modules:\n  pre_trainer:\n    overrides:\n"
        "      micro_batch_size: 8\n      global_batch_size: 2048\n"
        "      tensor_model_parallel_size: 1\n      pipeline_model_parallel_size: 2\n"
        "      virtual_pipeline_model_parallel_size: 2\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
    assert _mod.main([f.name, "--world-size", "48"]) == 0
    assert capsys.readouterr().out.strip() == "2304"
    assert _mod.main([f.name, "--world-size", "48", "--", "--virtual_pipeline_model_parallel_size=1"]) == 0
    assert capsys.readouterr().out.strip() == "2112"
    assert _mod.main([f.name, "--world-size", "48", "--", "microbatch_group_size_per_vp_stage=4"]) == 0
    assert capsys.readouterr().out.strip() == "2112"


# qwen3_235B_A22B (MI355X): MBS 4, GBS 8192, TP 1, PP 4; the layout has 16 stages, so VPP 4.
QWEN3_LAYOUT = "Et*5|" + "t*6|" * 14 + "t*5,L"
QWEN3 = {
    "micro_batch_size": 4,
    "global_batch_size": 8192,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 4,
    "pipeline_model_parallel_layout": QWEN3_LAYOUT,
}


def test_layout_stage_count_follows_megatron_parsing():
    assert _mod.layout_stage_count(QWEN3_LAYOUT) == 16
    # Megatron's own docstring example: Ettt | 29 x tt | m | L.
    assert _mod.layout_stage_count("Ettt|(tt|)*29,m|L") == 32
    assert _mod.layout_stage_count("Et|tL") == 2
    assert _mod.layout_stage_count([["E", "t"], ["t", "L"]]) == 2


def test_pipeline_layout_implies_the_interleaved_schedule():
    # 48 GPUs: DP 12, unit 48. 8192 is 171 microbatches (171 % 4 = 3 < PP 4, rejected);
    # 172 is a multiple of PP.
    assert adjusted(QWEN3, 48) == 172 * 48
    # Where the plain rounding already gives a multiple of PP nothing changes.
    for world in (8, 16, 32, 64):
        assert adjusted(QWEN3, world) is None


def test_deepseek_style_layout_with_two_virtual_stages():
    # PP 16, 32 stages -> VPP 2. 48 GPUs: DP 3, unit 6; 4096 is 683 microbatches, the
    # next multiple of PP 16 is 688.
    overrides = {
        "micro_batch_size": 2, "global_batch_size": 4096, "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 16,
        "pipeline_model_parallel_layout": "Et|" + "t|" * 30 + "tL",
    }
    assert adjusted(overrides, 48) == 688 * 6


def test_one_stage_per_rank_layout_is_not_interleaved():
    # 4 stages on PP 4: VPP 1, Megatron uses the plain schedule, so only DP rounding applies.
    plain = {**QWEN3, "pipeline_model_parallel_layout": "Et|t|t|tL"}
    assert adjusted(plain, 48) == 8208


def test_layout_wins_over_explicit_virtual_stage_counts():
    # Megatron derives VPP from the layout when there is one: 4 stages on PP 4 is VPP 1,
    # whatever virtual_pipeline_model_parallel_size says.
    overrides = {
        **QWEN3, "pipeline_model_parallel_layout": "Et|t|t|tL",
        "virtual_pipeline_model_parallel_size": 4,
    }
    assert adjusted(overrides, 48) == 8208


def test_layout_that_does_not_divide_into_the_pipeline_is_left_alone():
    # 5 stages on PP 4: Megatron rejects it regardless of the batch, so no extra rule.
    assert adjusted({**QWEN3, "pipeline_model_parallel_layout": "Et|t|t|t|tL"}, 48) == 8208


@pytest.mark.parametrize("layout", [None, "", "null", "None"])
def test_absent_layout_values_do_not_interleave(layout):
    assert adjusted({**QWEN3, "pipeline_model_parallel_layout": layout}, 48) == 8208


def test_layout_env_reference_resolves(monkeypatch):
    overrides = {**QWEN3, "pipeline_model_parallel_layout": "${PRIMUS_LAYOUT:" + QWEN3_LAYOUT + "}"}
    monkeypatch.delenv("PRIMUS_LAYOUT", raising=False)
    assert adjusted(overrides, 48) == 172 * 48
    monkeypatch.setenv("PRIMUS_LAYOUT", "null")
    assert adjusted(overrides, 48) == 8208


def test_cli_layout_override_is_honoured(capsys):
    cfg = (
        "modules:\n  pre_trainer:\n    overrides:\n"
        "      micro_batch_size: 4\n      global_batch_size: 8192\n"
        "      tensor_model_parallel_size: 1\n      pipeline_model_parallel_size: 4\n"
        f"      pipeline_model_parallel_layout: \"{QWEN3_LAYOUT}\"\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
    assert _mod.main([f.name, "--world-size", "48"]) == 0
    assert capsys.readouterr().out.strip() == str(172 * 48)
    # model_overrides.py clears the layout for the DeepSeek proxy this way.
    assert _mod.main(
        [f.name, "--world-size", "48", "--", "--pipeline_model_parallel_layout", "null"]
    ) == 0
    assert capsys.readouterr().out.strip() == "8208"
    # Without a pipeline there is no interleaving even though the layout is still set: the
    # plain rule on DP 48 (unit 192) gives 43 x 192.
    assert _mod.main(
        [f.name, "--world-size", "48", "--", "--pipeline_model_parallel_size=1"]
    ) == 0
    assert capsys.readouterr().out.strip() == str(43 * 192)
