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
    """The MoE parallelism knobs the Kimi-K2 entries set are perf-relevant, not noise."""
    from collprof.engines.sglang_disagg import SPEC

    perf = SPEC.run_config.perf_relevant
    for setting in ("moe_dense_tp_size", "enable_dp_lm_head",
                    "enable_dp_attention_local_control_broadcast"):
        assert setting in perf, setting
        assert setting not in SPEC.run_config.noise, setting


def test_the_ab_catalog_entries_pin_the_kernel_variant():
    """Both A/B catalog entries pin MoRI to its throughput kernel, so the pair isolates the
    backend rather than the dispatch mode."""
    import json
    from pathlib import Path

    catalog = json.loads((Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
                          / "models.json").read_text())
    ab = {m["name"]: m["env_vars"] for m in catalog if m["name"].endswith("-ab")}

    assert len(ab) == 2, "the A/B pair"
    for name, env in ab.items():
        assert env["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] == "0", name


def test_the_ab_pair_differs_only_by_the_backend_flag():
    """The two `-AB` entries' `dp_flags` differ only by the backend name, keeping the pair
    one-factor."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
            / "models.yaml").read_text()
    flags = {}
    name = None
    for line in text.splitlines():
        hit = re.match(r"^(\S+):\s*$", line)
        if hit:
            name = hit.group(1)
        elif name and "dp_flags:" in line:
            flags.setdefault(name, line.split("dp_flags:", 1)[1].strip().strip('"'))

    pair = {n: f for n, f in flags.items() if n.endswith("-AB")}
    assert len(pair) == 2, f"the A/B pair, got {sorted(pair)}"

    # Positionally. As sets, swapping two option values passes: the same words are present.
    left, right = (f.split() for _n, f in sorted(pair.items()))
    assert len(left) == len(right), f"different flag counts: {left} against {right}"
    differ = [(a, b) for a, b in zip(left, right) if a != b]
    assert differ == [("deepep", "mori")] or differ == [("mori", "deepep")], \
        f"only the backend may differ, got {differ}"
