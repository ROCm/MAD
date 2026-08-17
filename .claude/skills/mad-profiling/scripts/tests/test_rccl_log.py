"""Discovery, phase attribution and the damage checks -- the parts that cost the most to get
right."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import coll_line, topo_line, write

from collprof.core.phase import (DAMAGE_MSG_CAP, DAMAGE_NO_TAIL, DAMAGE_NRANKS_RANGE,
                                 DAMAGE_RANK_RANGE, DAMAGE_TWO_RECORDS, DAMAGE_UNKNOWN_COLL)
from collprof.core.rccl_log import discover_logs, log_stem, parse_run
from collprof.engines import primus, sglang_disagg


def test_discovery_takes_node_and_role_from_the_file_name(sglang_run: Path):
    found = discover_logs(sglang_run, sglang_disagg.SPEC)
    assert {(node, phase) for _log, node, phase in found} == {
        ("prefill_NODE0", "prefill"), ("prefill_NODE1", "prefill"),
        ("decode_NODE2", "decode"), ("decode_NODE3", "decode")}


def test_discovery_takes_node_from_the_parent_directory(primus_run: Path):
    found = discover_logs(primus_run, primus.SPEC)
    assert {node for _log, node, _phase in found} == {"node_0", "node_1"}
    # Primus announces phases in the log, so discovery leaves the phase open.
    assert {phase for _log, _node, phase in found} == {None}


def test_a_gzipped_log_gives_the_same_node_and_role(tmp_path: Path):
    write(tmp_path / "decode_NODE2.log.gz", [coll_line()], compress=True)
    (log, node, phase), = discover_logs(tmp_path, sglang_disagg.SPEC)
    assert (node, phase) == ("decode_NODE2", "decode")
    assert log_stem(log) == "decode_NODE2"


def test_a_gzipped_log_is_parsed_like_a_plain_one(tmp_path: Path):
    write(tmp_path / "decode_NODE2.log.gz", [coll_line(count=64)] * 5, compress=True)
    phases = parse_run(tmp_path, sglang_disagg.SPEC)
    assert phases["decode"].collective_totals()["AllReduce"]["calls"] == 5


def test_discovery_says_what_it_looked_for(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prefill_NODE"):
        discover_logs(tmp_path, sglang_disagg.SPEC)


def test_message_size_is_the_element_count_times_the_datatype_width(tmp_path: Path):
    write(tmp_path / "prefill_NODE0.log", [coll_line(count=1000, dtype=9)])  # bf16, 2 bytes
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    (coll, nranks, msg_bytes, dtype), calls = next(iter(phase.sizes.items()))
    assert (coll, nranks, msg_bytes, dtype, calls) == ("AllReduce", 8, 2000, "bf16", 1)


def test_phases_come_from_the_markers_in_a_shared_log(primus_run: Path):
    phases = parse_run(primus_run, primus.SPEC)
    assert sorted(phases) == ["BF16", "FP8"]
    # Two nodes x two ranks x three calls per phase.
    assert phases["BF16"].collective_totals()["AllReduce"]["calls"] == 12
    assert phases["BF16"].nodes == {"node_0", "node_1"}


def test_declared_metrics_are_harvested_without_the_core_knowing_them(primus_run: Path):
    phase = parse_run(primus_run, primus.SPEC)["FP8"]
    assert phase.metric("iter_ms") == [250.5]
    assert phase.metric("tokens") == [1200.5]
    assert phase.metric("tflops") == [410.2]


def test_topology_lines_become_edges(tmp_path: Path):
    write(tmp_path / "prefill_NODE0.log",
          [coll_line(), topo_line(0, 1, 0), topo_line(0, 1, 1), topo_line(0, 7, 0, "IB/0")])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert phase.edges[(0, 1, "P2P/IPC")] == {0, 1}
    assert phase.edges[(0, 7, "IB/0")] == {0}


def test_the_rank_comes_from_globalrank_not_from_the_pid(tmp_path: Path):
    """Tearing corrupts the host:pid prefix more often than the tail, and used to invent ranks."""
    write(tmp_path / "prefill_NODE0.log",
          [coll_line(grank=3, pid=111), coll_line(grank=3, pid=222)])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert list(phase.per_rank) == [("prefill_NODE0", "3")]


@pytest.mark.parametrize("line, reason", [
    (coll_line(coll="prllReduce"), DAMAGE_UNKNOWN_COLL),
    (coll_line(nranks=55688946, grank=0), DAMAGE_NRANKS_RANGE),
    (coll_line(nranks=8, grank=22), DAMAGE_RANK_RANGE),
    # A write cut off mid-field, with the next record's header landing inside it: the regex would
    # otherwise take the collective from the first record and the count from the second.
    (coll_line()[:60] + coll_line(coll="AllGather", count=99), DAMAGE_TWO_RECORDS),
])
def test_a_torn_record_is_counted_by_reason_and_not_by_volume(tmp_path: Path, line: str,
                                                             reason: str):
    write(tmp_path / "prefill_NODE0.log", [coll_line()] * 20 + [line])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert phase.damage[reason] == 1
    assert phase.damaged == 1
    assert phase.collective_totals()["AllReduce"]["calls"] == 20


def test_a_record_above_the_size_cap_is_rejected_as_a_splice(tmp_path: Path):
    """97920 and 854624 arrived concatenated, as a 91 GiB AllReduce carrying 16% of a report."""
    write(tmp_path / "prefill_NODE0.log", [coll_line()] * 20 + [coll_line(count=97920854624)])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert phase.damage[DAMAGE_MSG_CAP] == 1


def test_a_larger_cap_lets_a_genuinely_large_message_through(tmp_path: Path):
    """The bound is a property of the run's scale, so it has to be raisable."""
    import dataclasses
    big = 600 * 1024 ** 2 // 2  # 600 MiB of bf16, above the default 512 MiB cap
    write(tmp_path / "prefill_NODE0.log", [coll_line(count=big)])
    spec = sglang_disagg.SPEC
    assert parse_run(tmp_path, spec)["prefill"].damage[DAMAGE_MSG_CAP] == 1

    raised = dataclasses.replace(spec, limits=dataclasses.replace(spec.limits,
                                                                  max_msg_bytes=1024 ** 3))
    phase = parse_run(tmp_path, raised)["prefill"]
    assert not phase.damage
    assert phase.collective_totals()["AllReduce"]["bytes"] == big * 2


def test_a_rank_local_communicator_is_kept_rather_than_discarded(tmp_path: Path):
    """nranks=1 is a real value: a local no-op, printed with `stream (nil)` and no stream address.

    Requiring nranks>=2 and a stream address discarded every one of them -- 17200 records per node
    on a training run -- and made the report's single-rank section unreachable.
    """
    write(tmp_path / "prefill_NODE0.log",
          [coll_line(grank=r) for r in range(8)]
          + [coll_line(coll="Broadcast", count=8192, dtype=4, nranks=1, stream="(nil)")] * 3)
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert not phase.damage
    assert phase.sizes[("Broadcast", 1, 8192 * 8, "int64")] == 3
    # Single-rank records move no data between ranks, so no total may include them.
    assert "Broadcast" not in phase.collective_totals()
    assert phase.collective_totals(multi_rank_only=False)["Broadcast"]["calls"] == 3


def test_a_missing_tail_is_damage_only_where_the_log_has_tails(tmp_path: Path):
    write(tmp_path / "prefill_NODE0.log", [coll_line()] * 20 + [coll_line(tail=False)])
    assert parse_run(tmp_path, sglang_disagg.SPEC)["prefill"].damage[DAMAGE_NO_TAIL] == 1


def test_an_older_rccl_without_tails_is_not_treated_as_damaged(tmp_path: Path):
    """Older builds print no tail at all; holding them to it would discard every record."""
    write(tmp_path / "prefill_NODE0.log", [coll_line(tail=False)] * 20)
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert not phase.damage
    assert phase.collective_totals()["AllReduce"]["calls"] == 20
