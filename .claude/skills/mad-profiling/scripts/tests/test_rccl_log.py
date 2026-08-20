"""Discovery, phase attribution and the damage checks -- the parts that cost the most to get
right."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import coll_line, topo_line, write

from collprof.core.phase import (DAMAGE_MSG_CAP, DAMAGE_NO_TAIL, DAMAGE_NRANKS_RANGE,
                                 DAMAGE_RANK_RANGE, DAMAGE_TOPO_TRANSPORT, DAMAGE_TWO_RECORDS,
                                 DAMAGE_UNKNOWN_COLL)
from collprof.core.rccl_log import discover_logs, log_stem, parse_run, transport_scope
from collprof.core.spec import LOG_PER_RANK, LOG_SHARED
from collprof.engines import primus, sglang_disagg


def test_discovery_takes_node_and_role_from_the_file_name(sglang_run: Path):
    found = discover_logs(sglang_run, sglang_disagg.SPEC)
    assert {(node, phase) for _log, node, phase, _layout in found} == {
        ("prefill_NODE0", "prefill"), ("prefill_NODE1", "prefill"),
        ("decode_NODE2", "decode"), ("decode_NODE3", "decode")}


def test_discovery_takes_node_from_the_parent_directory(primus_run: Path):
    found = discover_logs(primus_run, primus.SPEC)
    assert {node for _log, node, _phase, _layout in found} == {"node_0", "node_1"}
    # Primus announces phases in the log, so discovery leaves the phase open.
    assert {phase for _log, _node, phase, _layout in found} == {None}


def test_a_gzipped_log_gives_the_same_node_and_role(tmp_path: Path):
    write(tmp_path / "decode_NODE2.log.gz", [coll_line()], compress=True)
    (log, node, phase, _layout), = discover_logs(tmp_path, sglang_disagg.SPEC)
    assert (node, phase) == ("decode_NODE2", "decode")
    assert log_stem(log) == "decode_NODE2"


def test_per_rank_files_are_read_beside_the_shared_log(sglang_run: Path):
    """`NCCL_DEBUG_FILE` gives each server process its own file; the role and node stay the same."""
    for rank in range(8):
        write(sglang_run / "rccl" / f"prefill_NODE0.worker0.{2000 + rank}.log",
              [coll_line(count=4096, grank=rank, pid=2000 + rank)] * 5)
    phase = parse_run(sglang_run, sglang_disagg.SPEC)["prefill"]
    assert phase.nodes == {"prefill_NODE0", "prefill_NODE1"}
    assert phase.per_rank[("prefill_NODE0", "0")] == [7, 5 * 4096 * 2 + 2 * 1024 * 2]
    assert phase.writers == {LOG_SHARED, LOG_PER_RANK}


def test_per_rank_files_alone_carry_the_whole_phase(tmp_path: Path):
    """A run measured only through NCCL_DEBUG_FILE has nothing in the shared log to fall back on."""
    write(tmp_path / "decode_NODE2.log", ["starting decode server"])
    for rank in range(8):
        write(tmp_path / "rccl" / f"decode_NODE2.worker1.{3000 + rank}.log",
              [coll_line(count=256, grank=rank, pid=3000 + rank)] * 4)
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["decode"]
    assert phase.collective_totals()["AllReduce"]["calls"] == 32
    assert len(phase.per_rank) == 8
    assert phase.writers == {LOG_PER_RANK}


def test_a_training_phase_comes_from_the_file_name_and_its_metrics_from_stdout(primus_run: Path,
                                                                               tmp_path: Path):
    """The two sources carry different halves: RCCL in the files, markers and timings in stdout."""
    rccl = tmp_path / "rccl"
    for rank in range(8):
        write(rccl / f"BF16.node_0.host0.{4000 + rank}.log",
              [coll_line(count=512, grank=rank, pid=4000 + rank)] * 2)
    phases = parse_run(primus_run, primus.SPEC, rccl_dir=rccl)
    assert phases["BF16"].nodes == {"node_0", "node_1"}
    assert phases["BF16"].metric("iter_ms") == [250.5]
    # Eight ranks from the files plus the two the fixture's stdout logs on each of two nodes.
    assert len(phases["BF16"].per_rank) == 10
    assert phases["FP8"].writers == {LOG_SHARED}, "the files named only BF16"


def test_a_gzipped_log_is_parsed_like_a_plain_one(tmp_path: Path):
    write(tmp_path / "decode_NODE2.log.gz", [coll_line(count=64)] * 5, compress=True)
    phases = parse_run(tmp_path, sglang_disagg.SPEC)
    assert phases["decode"].collective_totals()["AllReduce"]["calls"] == 5


def test_a_log_left_in_both_forms_is_counted_once(tmp_path: Path):
    """`gzip -k`, or a gzip that died halfway, must not double the numbers of that rank."""
    write(tmp_path / "decode_NODE2.log", [coll_line(count=64)] * 5)
    write(tmp_path / "decode_NODE2.log.gz", [coll_line(count=64)] * 5, compress=True)
    found = discover_logs(tmp_path, sglang_disagg.SPEC)
    assert [log.name for log, _n, _p, _l in found] == ["decode_NODE2.log"]
    assert parse_run(tmp_path, sglang_disagg.SPEC)["decode"].collective_totals(
    )["AllReduce"]["calls"] == 5


def test_a_doubled_log_is_reported_rather_than_dropped_quietly(tmp_path: Path, capsys):
    write(tmp_path / "decode_NODE2.log", [coll_line()])
    write(tmp_path / "decode_NODE2.log.gz", [coll_line()], compress=True)
    discover_logs(tmp_path, sglang_disagg.SPEC)
    warning = capsys.readouterr().out
    assert "decode_NODE2.log" in warning and "compressed and not" in warning


def test_the_same_name_on_two_nodes_is_not_a_duplicate(primus_run: Path):
    """Both nodes write `stdout.out`; only a `.gz` beside its own plain file is a doubled log."""
    found = discover_logs(primus_run, primus.SPEC)
    assert len(found) == 2


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
          [coll_line(), topo_line(0, 1, 0), topo_line(0, 1, 1),
           topo_line(0, 7, 0, "NET/IB/0/GDRDMA/Shared")])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert phase.edges[(0, 1, "P2P/IPC")] == {0, 1}
    assert phase.edges[(0, 7, "NET/IB/0/GDRDMA/Shared")] == {0}


@pytest.mark.parametrize("transport, scope", [
    ("P2P/IPC", "intra-node"),
    ("P2P/direct pointer/read", "intra-node"),
    ("SHM/direct/direct", "intra-node"),
    ("LOC", "intra-node"),
    ("NET/IB/3/GDRDMA/Shared", "inter-node"),
    ("NET/Socket/0", "inter-node"),
    # Every one of these reached a report as a transport before the whitelist, and each is a
    # spliced prefix of a real name, which is why the match has to be exact.
    ("P2P/IPCrank", None),
    ("P2P/Iproxy", None),
    ("PCCL", None),
    ("P50", None),
    ("localRank", None),
])
def test_only_transports_rccl_can_print_are_recognised(transport: str, scope: str | None):
    assert transport_scope(transport) == scope


def test_a_torn_topology_line_is_counted_instead_of_becoming_an_edge(tmp_path: Path):
    """20 such lines once made a prefill report claim inter-node links it could not have."""
    write(tmp_path / "prefill_NODE0.log",
          [coll_line(), topo_line(0, 1, 0), topo_line(0, 3, 1, "P2P/IPCrank"),
           topo_line(0, 5, 2, "PCCL")])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert list(phase.edges) == [(0, 1, "P2P/IPC")]
    assert phase.topo_damage[DAMAGE_TOPO_TRANSPORT] == 2
    # Topology damage says nothing about the volume, so it stays out of the record share.
    assert not phase.damage


def test_a_topology_line_torn_mid_transport_is_still_counted(tmp_path: Path):
    """The overwriting write brings punctuation with it, which must not end the match early.

    A transport pattern that gave up at the first `:` or `[` left these lines matching nothing at
    all -- neither an edge nor a rejection -- which is the one outcome the damage counters exist to
    prevent. Twelve lines of one real prefill log disappeared that way.
    """
    torn = topo_line(0, 3, 1).split(" comm ")[0] + "rank 3 worker:1:2 [4] NCCL INFO Channel 07/0"
    write(tmp_path / "prefill_NODE0.log", [coll_line(), topo_line(0, 1, 0), torn])
    phase = parse_run(tmp_path, sglang_disagg.SPEC)["prefill"]
    assert list(phase.edges) == [(0, 1, "P2P/IPC")]
    assert phase.topo_damage[DAMAGE_TOPO_TRANSPORT] == 1


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
