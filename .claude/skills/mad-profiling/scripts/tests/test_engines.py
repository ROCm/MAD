"""Engine detection: never a guess, and never silent."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import coll_line, write

from collprof import engines


def test_a_serving_run_is_recognised(sglang_run: Path):
    spec, reason = engines.detect(sglang_run)
    assert spec.name == "sglang-disagg"
    assert "4 log(s)" in reason


def test_a_training_run_is_recognised(primus_run: Path):
    spec, reason = engines.detect(primus_run)
    assert spec.name == "primus"
    assert "2 log(s)" in reason


def test_an_unrecognised_run_lists_what_was_looked_for(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        engines.detect(tmp_path)
    message = str(exc.value)
    assert "no known engine" in message
    for name in engines.REGISTRY:
        assert name in message
    assert "--engine" in message


def test_an_ambiguous_run_refuses_to_pick(tmp_path: Path):
    """Two layouts in one directory means the answer is unknowable, not that the first wins."""
    write(tmp_path / "prefill_NODE0.log", [coll_line()])
    write(tmp_path / "node_0" / "stdout.out", [coll_line()])
    with pytest.raises(SystemExit, match="more than one engine"):
        engines.detect(tmp_path)


def test_an_unknown_engine_name_lists_the_known_ones():
    with pytest.raises(SystemExit, match="unknown engine"):
        engines.get("vllm-disagg")


def test_every_registered_engine_declares_what_a_report_needs():
    """The registry is the contract; an engine missing a piece produces a misleading report."""
    for name, spec in engines.REGISTRY.items():
        assert spec.name == name
        assert spec.summary, f"{name} has no summary for the report header"
        assert spec.logs.globs, f"{name} declares no log globs"
        assert spec.limits.max_msg_bytes > 0
        if spec.iteration_metric:
            keys = {m.key for m in spec.metrics}
            assert spec.iteration_metric in keys, f"{name} counts iterations with an absent metric"


def test_the_moe_parallelism_knobs_the_kimi_entries_set_are_classified():
    """`models.yaml` passes these three beside `--moe-a2a-backend` on every Kimi-K2 entry, so a
    difference the verdict lists must not then be called unable to move throughput."""
    from collprof.engines.sglang_disagg import SPEC

    perf = SPEC.run_config.perf_relevant
    for setting in ("moe_dense_tp_size", "enable_dp_lm_head",
                    "enable_dp_attention_local_control_broadcast"):
        assert setting in perf, setting
        assert setting not in SPEC.run_config.noise, setting


def test_the_ab_catalog_entries_pin_the_kernel_variant():
    """The pair only isolates the backend if MoRI is held to its throughput kernel.

    The launcher defaults MoRI decode to low-latency by token count and DeepEP has no such path,
    so uncontrolled the pair measures the mode too: 10.6 ms against the matched 14.7 ms.
    """
    import json
    from pathlib import Path

    catalog = json.loads((Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
                          / "models.json").read_text())
    ab = {m["name"]: m["env_vars"] for m in catalog if m["name"].endswith("-ab")}

    assert len(ab) == 2, "the A/B pair"
    for name, env in ab.items():
        assert env["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] == "0", name
