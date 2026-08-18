"""End to end: a synthetic run in, a report out -- and no engine's claims leaking into another's."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from conftest import coll_line, topo_line, trace_event, write

from collprof import engines
from collprof.cli import main
from collprof.core.rccl_log import parse_run
from collprof.core.report import ReportContext, emit_phase
from collprof.core.spec import (NODE_FROM_STEM, PHASE_FROM_FILENAME, EngineSpec, LogLayout,
                                ReportNotes)

#: A sentence that belongs to exactly one engine, used to prove nothing else inherits it.
SGLANG_ONLY = "mooncake"
PRIMUS_ONLY = "local-ranks-filter"


def build(run: Path, out: Path, extra: list | None = None) -> dict:
    main(["--run-dir", str(run), "--out-dir", str(out)] + (extra or []))
    return {p.parent.name.rsplit("_", 1)[-1]: p.read_text()
            for p in out.parent.glob(f"{out.name}_*/report.md")}


def test_a_serving_run_produces_one_report_per_role(sglang_run: Path, tmp_path: Path):
    reports = build(sglang_run, tmp_path / "out")
    assert sorted(reports) == ["decode", "prefill"]
    for text in reports.values():
        assert "## Traffic by collective (per rank)" in text
        assert "## Rank-to-rank connectivity" in text


def test_the_report_records_the_engine_and_how_it_was_produced(sglang_run: Path, tmp_path: Path):
    """A number a reader cannot trace back to a command is a number they have to take on faith."""
    text = build(sglang_run, tmp_path / "out")["prefill"]
    assert "Engine: **sglang-disagg**" in text
    assert "--run-dir" in text and "parser version" in text


def test_per_rank_figures_divide_by_the_ranks_that_carried_traffic(sglang_run: Path,
                                                                   tmp_path: Path):
    # Each prefill node in the fixture has 8 ranks x 2 calls of 2048 B: 2 calls, 4 KiB per rank.
    text = build(sglang_run, tmp_path / "out")["prefill"]
    assert "over 16 ranks = 2 per rank" in text
    assert "Traffic per rank: **4.00 KiB**" in text


def test_an_idle_replica_does_not_halve_every_per_rank_figure(tmp_path: Path):
    run = tmp_path / "run"
    write(run / "decode_NODE2.log", [coll_line(grank=r) for r in range(8) for _ in range(100)])
    # The second replica served one request: present in the log, but not a peer to average over.
    # A real 2P2D proxy did exactly this, 8 batches against 3874 on the other node.
    write(run / "decode_NODE3.log", [coll_line(grank=0)])
    text = build(run, tmp_path / "out")["decode"]
    assert "of which 8 carried traffic" in text
    assert "leaving out 1 rank(s)" in text


def test_discarded_records_are_broken_down_by_reason(sglang_run: Path, tmp_path: Path):
    log = sglang_run / "prefill_NODE0.log"
    log.write_text(log.read_text() + coll_line(coll="prllReduce") + "\n")
    text = build(sglang_run, tmp_path / "out")["prefill"]
    assert "Discarded as unusable: 1 of" in text
    assert "unknown collective name: 1" in text
    assert "share one stdout" in text, "the cause of the damage comes from the engine's notes"


def test_a_dropped_topology_line_is_admitted_where_the_connectivity_table_is(sglang_run: Path,
                                                                             tmp_path: Path):
    """The reader of that table has to know it is short, or a missing edge reads as no edge."""
    log = sglang_run / "prefill_NODE0.log"
    log.write_text(log.read_text() + topo_line(0, 5, 3, "PCCL") + "\n")
    text = build(sglang_run, tmp_path / "out")["prefill"]
    assert "1 topology line(s) named no transport RCCL can print" in text
    # The share of discarded collective records is about collective records only.
    assert "Discarded as unusable" not in text


def test_an_engine_that_claims_no_cause_of_damage_has_none_stated(sglang_run: Path, tmp_path: Path):
    """Why records tore is a claim about one engine's logging, not something the core may assume."""
    spec = dataclasses.replace(
        engines.REGISTRY["sglang-disagg"],
        notes=dataclasses.replace(engines.REGISTRY["sglang-disagg"].notes, damage_cause=""))
    log = sglang_run / "prefill_NODE0.log"
    log.write_text(log.read_text() + coll_line(coll="prllReduce") + "\n")
    out = tmp_path / "out_prefill"
    emit_phase(parse_run(sglang_run, spec)["prefill"], out,
               ReportContext(spec=spec, run_dir=sglang_run))
    text = (out / "report.md").read_text()
    assert "unknown collective name: 1" in text
    assert "share one stdout" not in text


def test_hitting_a_sanity_bound_says_how_to_raise_it(sglang_run: Path, tmp_path: Path):
    """The bound protects against spliced digits, but a bigger run may legitimately exceed it."""
    log = sglang_run / "decode_NODE2.log"
    log.write_text(log.read_text() + coll_line(count=97920854624) + "\n")
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "rejected for exceeding a sanity bound" in text
    assert "--max-msg-bytes" in text


def test_a_flood_of_damage_is_called_a_warning_not_a_footnote(tmp_path: Path):
    run = tmp_path / "run"
    write(run / "prefill_NODE0.log",
          [coll_line(grank=r) for r in range(8)] + [coll_line(coll="prllReduce")] * 8)
    text = build(run, tmp_path / "out")["prefill"]
    assert "**Warning:" in text and "of records were discarded**" in text


def test_a_training_run_reports_iterations_and_its_own_metrics(primus_run: Path, tmp_path: Path):
    reports = build(primus_run, tmp_path / "out")
    assert sorted(reports) == ["BF16", "FP8"]
    text = reports["BF16"]
    assert "Iterations per rank: 1, median iteration 250.5 ms" in text
    assert "Throughput reported in log: 1200.5 tokens/s/GPU" in text
    assert "Compute reported in log: 410.2 TFLOP/s/GPU" in text
    assert "Volume per iteration per rank" in text


def test_a_serving_run_quotes_no_per_iteration_figure(sglang_run: Path, tmp_path: Path):
    """Serving has no iterations; inventing one would turn a window into a rate."""
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "Volume per iteration" not in text
    assert "Iterations per rank" not in text


def test_each_engine_only_makes_its_own_claims(sglang_run: Path, primus_run: Path, tmp_path: Path):
    serving = build(sglang_run, tmp_path / "sgl")["prefill"]
    training = build(primus_run, tmp_path / "primus")["BF16"]
    assert SGLANG_ONLY in serving and PRIMUS_ONLY not in serving
    assert PRIMUS_ONLY in training and SGLANG_ONLY not in training


def test_a_new_engine_does_not_inherit_another_engines_scope_note(sglang_run: Path, tmp_path: Path):
    """The trap this design exists to close: phases named prefill/decode used to select the prose.

    A different engine that happens to call its phases prefill and decode must not have sglang's
    claims about mooncake RDMA or `--disable-custom-all-reduce` put in its report.
    """
    spec = EngineSpec(
        name="hypothetical-engine",
        summary="a third engine that also calls its phases prefill and decode",
        logs=LogLayout(globs=("prefill_NODE*.log", "decode_NODE*.log"),
                       phase_from=PHASE_FROM_FILENAME, node_from=NODE_FROM_STEM),
        notes=ReportNotes(),
    )
    phases = parse_run(sglang_run, spec)
    out = tmp_path / "hypothetical_prefill"
    emit_phase(phases["prefill"], out, ReportContext(spec=spec, run_dir=sglang_run))
    text = (out / "report.md").read_text()
    assert "prefill" in text
    assert SGLANG_ONLY not in text
    assert "disable-custom-all-reduce" not in text
    assert "Engine: **hypothetical-engine**" in text


def test_an_empty_trace_directory_is_reported_and_skipped(sglang_run: Path, tmp_path: Path, capsys):
    """An idle replica captured nothing, and that one empty directory cost a whole job's reports.

    Decode keeps a live capture beside the empty one, prefill has only the empty one: the first must
    still get its trace section, the second must fall back to a volume-only report.
    """
    live, dead = "2000.100000", "1000.100000"
    write(sglang_run / "torchprof" / live / "worker0.trace.json",
          ['{"traceEvents": [', trace_event(), "]}"])
    (sglang_run / "torchprof" / dead).mkdir(parents=True)
    for role, epochs in (("prefill", [dead]), ("decode", [live, dead])):
        log = next(sglang_run.glob(f"*_PROFILE_{role}.log"))
        log.write_text("".join(f"'output_dir': '/run_logs/25999/torchprof/{e}'\n" for e in epochs))

    reports = build(sglang_run, tmp_path / "out")
    console = capsys.readouterr().out
    assert f"trace directories with no trace files, skipped: {dead}" in console
    assert "no usable traces for prefill" in console
    assert sorted(reports) == ["decode", "prefill"]
    assert "torch profiler" not in reports["prefill"]
    assert "torch profiler" in reports["decode"]
    assert dead in reports["decode"], "the report, not just the console, names what is absent"


def test_captures_from_two_replicas_say_that_the_numbers_are_an_average(sglang_run: Path,
                                                                       tmp_path: Path):
    """Both replicas of a role now contribute, and unevenly loaded ones average into one mix.

    A real decode pair captured 224 KiB AllReduces on one node and 112 KiB on the other, so the
    average message described neither. The rows show both sizes; this sentence says why.
    """
    dirs = ("2000.100000", "2000.200000")
    for epoch, nelems in zip(dirs, (512, 256)):
        write(sglang_run / "torchprof" / epoch / "worker0.trace.json",
              ['{"traceEvents": [', trace_event(nin=nelems, nout=nelems), "]}"])
    log = next(sglang_run.glob("*_PROFILE_decode.log"))
    log.write_text("".join(f"'output_dir': '/run_logs/25999/torchprof/{e}'\n" for e in dirs))
    log = next(sglang_run.glob("*_PROFILE_prefill.log"))
    log.write_text(f"'output_dir': '/run_logs/25999/torchprof/{dirs[0]}'\n")

    reports = build(sglang_run, tmp_path / "out")
    assert "capture directories" in reports["decode"]
    assert "cover 2 replicas of the phase" in reports["decode"]
    assert "capture directories" not in reports["prefill"], "one replica needs no caveat"


def test_rank_local_communicators_get_their_own_section(sglang_run: Path, tmp_path: Path):
    log = sglang_run / "prefill_NODE0.log"
    log.write_text(log.read_text()
                   + coll_line(coll="Broadcast", count=8192, dtype=4, nranks=1, stream="(nil)")
                   + "\n")
    text = build(sglang_run, tmp_path / "out")["prefill"]
    assert "## Single-rank communicators (excluded above)" in text
    assert "local no-ops that move no data between ranks" in text


def test_the_csvs_beside_a_report_are_the_ones_this_run_wrote(sglang_run: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(sglang_run, out)
    stale = out.parent / f"{out.name}_prefill" / "left_over.csv"
    stale.write_text("collective,calls\nAllReduce,1\n")
    build(sglang_run, out)
    assert not stale.exists()
    assert (out.parent / f"{out.name}_prefill" / "collective_totals.csv").exists()


def test_a_phase_that_logged_nothing_is_skipped_with_a_reason(sglang_run: Path, tmp_path: Path,
                                                              capsys):
    main(["--run-dir", str(sglang_run), "--out-dir", str(tmp_path / "out"),
          "--phases", "prefill", "nonexistent"])
    assert "skip nonexistent: no such phase" in capsys.readouterr().out


def test_restricting_phases_produces_only_those(sglang_run: Path, tmp_path: Path):
    reports = build(sglang_run, tmp_path / "out", ["--phases", "decode"])
    assert sorted(reports) == ["decode"]
