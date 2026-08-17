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
