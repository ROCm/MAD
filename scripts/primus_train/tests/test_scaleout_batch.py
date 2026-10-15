#!/usr/bin/env python3
"""scaleout_batch must renormalize global_batch_size only when Megatron would reject it."""

import importlib.util
import sys
import tempfile
from pathlib import Path

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


def test_cli_overrides_take_precedence(capsys):
    cfg = "modules:\n  pre_trainer:\n    overrides:\n      micro_batch_size: 16\n      global_batch_size: 128\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
    assert _mod.main([f.name, "--world-size", "16", "--", "--micro_batch_size", "2"]) == 0
    assert capsys.readouterr().out == ""
    assert _mod.main([f.name, "--world-size", "16"]) == 0
    assert capsys.readouterr().out.strip() == "256"
