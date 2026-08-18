"""Aggregations on a phase, and the cache that lets a 2 GB parse be reused."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import coll_line, write

from collprof.core.cache import PARSE_VERSION, ParseCache, file_signature
from collprof.core.phase import Phase
from collprof.core.rccl_log import parse_run
from collprof.engines import sglang_disagg


def test_totals_ignore_single_rank_communicators():
    phase = Phase("decode")
    phase.sizes[("AllReduce", 8, 1024, "bf16")] = 3
    phase.sizes[("AllReduce", 1, 4096, "bf16")] = 5
    totals = phase.collective_totals()
    assert totals["AllReduce"] == {"calls": 3, "bytes": 3072, "sizes": {1024}}
    assert phase.collective_totals(multi_rank_only=False)["AllReduce"]["calls"] == 8


def test_the_metric_series_comes_from_the_node_that_logged_it():
    """Megatron prints iteration timings from one rank, so the fullest series is the real one."""
    phase = Phase("BF16")
    phase.add_metric("iter_ms", "node_0", 100.0)
    for value in (200.0, 300.0):
        phase.add_metric("iter_ms", "node_1", value)
    assert phase.metric("iter_ms") == [200.0, 300.0]
    assert phase.metric("absent") == []


def test_an_all_idle_phase_falls_back_to_every_rank():
    """Dividing by zero representative ranks would be worse than dividing by all of them."""
    phase = Phase("decode")
    phase.per_rank[("n", "0")] = [0, 0]
    assert phase.active_ranks(0.05) == {("n", "0"): [0, 0]}


def test_a_phase_survives_the_round_trip_through_the_cache(sglang_run: Path):
    original = parse_run(sglang_run, sglang_disagg.SPEC)["decode"]
    restored = Phase.from_state(original.to_state())
    assert restored.name == original.name
    assert restored.engine == original.engine
    assert restored.sizes == original.sizes
    assert restored.per_rank == original.per_rank
    assert restored.edges == original.edges
    assert restored.damage == original.damage
    # The restored phase must still behave like one, not like a bag of plain dicts.
    assert restored.collective_totals() == original.collective_totals()
    restored.add_metric("iter_ms", "n", 1.0)


def test_the_cache_is_reused_only_for_the_same_inputs(tmp_path: Path):
    log = write(tmp_path / "prefill_NODE0.log", [coll_line()])
    cache_path = tmp_path / "cache.pkl"
    calls = []

    def parse():
        calls.append(1)
        return "parsed"

    for _ in range(2):
        cache = ParseCache(cache_path)
        assert cache.get("logs", file_signature([log]), parse) == "parsed"
        cache.flush()
    assert len(calls) == 1, "the second run should have reused the cache"

    write(tmp_path / "prefill_NODE0.log", [coll_line(), coll_line()])
    cache = ParseCache(cache_path)
    cache.get("logs", file_signature([log]), parse)
    assert len(calls) == 2, "a changed log must be reparsed"


def test_the_parser_version_invalidates_a_cache_built_by_older_logic(tmp_path: Path):
    log = write(tmp_path / "prefill_NODE0.log", [coll_line()])
    signature = file_signature([log])
    assert signature[0] == PARSE_VERSION
    assert signature != [PARSE_VERSION + 1] + signature[1:]


def test_caching_off_still_parses(tmp_path: Path):
    cache = ParseCache(None)
    assert cache.get("k", [1], lambda: "value") == "value"
    cache.flush()


def test_a_truncated_cache_costs_a_reparse_and_not_the_run(tmp_path: Path, capsys):
    """A disk that filled up mid-write left a 0-byte pickle, and every later run died on it."""
    cache_path = tmp_path / "cache.pkl"
    cache_path.write_bytes(b"")

    cache = ParseCache(cache_path)
    assert cache.get("logs", [1], lambda: "parsed") == "parsed"
    assert "ignoring unreadable parse cache" in capsys.readouterr().out
    cache.flush()
    assert ParseCache(cache_path).store["logs"]["data"] == "parsed"


def test_a_failed_write_leaves_the_previous_cache_intact(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "cache.pkl"
    cache = ParseCache(cache_path)
    cache.get("logs", [1], lambda: "first")
    cache.flush()
    good = cache_path.read_bytes()

    cache = ParseCache(cache_path)
    cache.get("logs", [2], lambda: "second")
    monkeypatch.setattr("collprof.core.cache.pickle.dump",
                        lambda *_, **__: (_ for _ in ()).throw(OSError("Disk quota exceeded")))
    with pytest.raises(OSError):
        cache.flush()
    assert cache_path.read_bytes() == good
    assert not list(tmp_path.glob("*.tmp")), "the temporary file must not be left behind"
