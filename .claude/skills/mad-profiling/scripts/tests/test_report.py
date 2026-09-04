"""End to end: a synthetic run in, a report out -- and no engine's claims leaking into another's."""

from __future__ import annotations

import dataclasses
import shlex
from pathlib import Path

from conftest import (coll_line, decode_batch_line, server_args_line, topo_line, trace_event,
                      write)

from collprof import engines
from collprof.cli import build_parser, main
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


def test_the_recorded_command_is_one_the_parser_accepts(sglang_run: Path, tmp_path: Path):
    """The report says rerunning that command reproduces the file, so it has to be runnable.

    Built through regen_reports.py the arguments arrive as an explicit argv while sys.argv[0] is the
    campaign driver, whose own parser knows none of these flags.
    """
    text = build(sglang_run, tmp_path / "out")["prefill"]
    line = next(ln for ln in text.splitlines() if ln.startswith("> Produced by"))
    command = shlex.split(line.split("`")[1])
    assert command[0] == "collective_report.py"
    build_parser().parse_args(command[1:])


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


def test_per_rank_logs_are_stated_and_not_blamed_on_a_shared_stdout(tmp_path: Path):
    """Torn records in a per-rank file mean something else, so the engine's excuse must not run."""
    run = tmp_path / "run"
    write(run / "decode_NODE2.log", ["starting decode server"])
    for rank in range(8):
        write(run / "rccl" / f"decode_NODE2.host.{3000 + rank}.log",
              [coll_line(grank=rank, pid=3000 + rank)] * 20)
    write(run / "rccl" / "decode_NODE2.host.3999.log", [coll_line(coll="prllReduce")])
    text = build(run, tmp_path / "out")["decode"]
    assert "one file per process (`NCCL_DEBUG_FILE`)" in text
    assert "unknown collective name: 1" in text
    assert "share one stdout" not in text
    assert "per-rank files, where nothing interleaves" in text


def test_a_rank_coverage_caveat_is_dropped_where_every_rank_logged(tmp_path: Path):
    """`--local-ranks-filter` decides who reaches stdout; it decides nothing about per-rank files.

    This is the shape of a real run measured with NCCL_DEBUG_FILE: the stdout keeps the phase
    marker and the iteration line and holds no RCCL record at all.
    """
    run = tmp_path / "26615"
    write(run / "node_0" / "stdout.out",
          ["[INFO] [main] Executing: bash runner/primus-cli-direct.sh -- train pretrain "
           "--config examples/megatron/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml",
           " iteration 1/10 | elapsed time per iteration (ms): 250.5 |"])
    rccl = tmp_path / "rccl"
    for rank in range(8):
        write(rccl / f"BF16.node_0.host0.{4000 + rank}.log",
              [coll_line(grank=rank, pid=4000 + rank), topo_line(rank, (rank + 1) % 8)])
    text = build(run, tmp_path / "out", ["--rccl-dir", str(rccl)])["BF16"]
    assert "local-ranks-filter" not in text
    assert "Iterations per rank: 1" in text, "the metrics still come from the shared stdout"


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


# -- configuration and step time -----------------------------------------------------------------
# Both sections exist because of one Kimi-K2 MoRI-vs-DeepEP comparison whose arms also differed
# in graph capture and static memory fraction, with no artifact saying so.


def test_the_configuration_a_phase_ran_with_is_reported(sglang_run: Path, tmp_path: Path):
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "## Configuration this phase ran with" in text
    assert "`moe_a2a_backend` | `mori`" in text
    assert (tmp_path / "out_decode" / "run_config.csv").exists()


def test_a_node_configured_unlike_its_role_is_named(sglang_run: Path, tmp_path: Path):
    """Three nodes profiled and one not is a real failure mode only the settings show."""
    log = sglang_run / "decode_NODE3.log"
    log.write_text(server_args_line(disable_cuda_graph="True") + "\n" + log.read_text())
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "### Nodes of this phase disagree" in text
    assert "decode_NODE3=`True`" in text


def test_comparing_a_graph_free_arm_against_a_graphed_one_flags_the_difference(
        sglang_run: Path, tmp_path: Path):
    """The confound this section was written for: it must not come out looking comparable."""
    reference = tmp_path / "reference"
    for node in (2, 3):
        write(reference / f"decode_NODE{node}.log",
              [server_args_line(disable_cuda_graph="True", mem_fraction_static="0.92",
                                moe_a2a_backend="'deepep'"),
               coll_line(count=128, grank=0, pid=2000)])
    text = build(sglang_run, tmp_path / "out", ["--compare-config", str(reference)])["decode"]
    assert "### Difference from" in text
    assert "`disable_cuda_graph` | `False` | `True`" in text
    assert "moves throughput" in text
    assert "not attributable to anything else in this report" in text


def test_two_runs_configured_alike_are_declared_comparable(sglang_run: Path, tmp_path: Path):
    text = build(sglang_run, tmp_path / "out", ["--compare-config", str(sglang_run)])["decode"]
    assert "The two runs are comparable on configuration." in text


def test_step_time_is_reported_per_node_with_the_graph_state(sglang_run: Path, tmp_path: Path):
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "## Step time, from the engine's own accounting" in text
    assert "| decode_NODE2 |" in text and "replayed" in text
    assert (tmp_path / "out_decode" / "step_times.csv").exists()


def test_graphs_being_off_is_called_out_as_a_per_step_cost(sglang_run: Path, tmp_path: Path):
    for node in (2, 3):
        log = sglang_run / f"decode_NODE{node}.log"
        write(log, [server_args_line(disable_cuda_graph="True"),
                    coll_line(count=128, grank=0, pid=2000),
                    decode_batch_line(batch=16, rate=80.0, graphed=False)])
    text = build(sglang_run, tmp_path / "out")["decode"]
    assert "Graph replay is off" in text
    assert "does on purpose" in text, "the engine explains why, the core only reports the state"


def test_a_silent_node_does_not_become_a_claim_that_every_step_was_ungraphed(
        sglang_run: Path, tmp_path: Path):
    """`graph_states` drops `None`, so `{False, None}` reduced to `{False}` and the report claimed
    every step paid launch cost, including those of the node that never said."""
    write(sglang_run / "decode_NODE2.log",
          [server_args_line(disable_cuda_graph="True"), coll_line(count=128, grank=0, pid=2000),
           decode_batch_line(batch=16, rate=80.0, graphed=False)])
    # An older sglang, printing the interval without the `cuda graph` field.
    write(sglang_run / "decode_NODE3.log",
          [server_args_line(disable_cuda_graph="True"), coll_line(count=128, grank=0, pid=2001),
           decode_batch_line(batch=16, rate=80.0, graphed=None)])

    text = build(sglang_run, tmp_path / "out")["decode"]

    assert "Graph replay is off" in text
    assert "Every step here" not in text
    assert "1 node(s) did not state theirs" in text


def test_a_phase_without_steps_has_no_step_section(sglang_run: Path, tmp_path: Path):
    assert "## Step time" not in build(sglang_run, tmp_path / "out")["prefill"]


def test_an_engine_that_explains_neither_keeps_the_core_silent(sglang_run: Path, tmp_path: Path):
    """The core reports the state; the wording around it belongs to the engine that knows why."""
    base = engines.REGISTRY["sglang-disagg"]
    spec = dataclasses.replace(
        base, notes=dataclasses.replace(base.notes, step_basis="", graphs_off=""))
    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, spec)["decode"], out,
               ReportContext(spec=spec, run_dir=sglang_run))
    text = (out / "report.md").read_text()
    assert "## Step time" in text
    assert "does on purpose" not in text, "the engine's reason for graphs being off"
    assert "one token per running request" not in text, "the engine's basis for the step time"


def test_a_run_without_collectives_says_so_not_zeros(sglang_run: Path, tmp_path: Path):
    """A tuned run leaves NCCL_DEBUG_SUBSYS without COLL, so the channel is absent, not zero, and
    an empty "Traffic by collective" table reads as a run that issued none."""
    for node in (2, 3):
        write(sglang_run / f"decode_NODE{node}.log",
              [server_args_line(), decode_batch_line(batch=16, rate=80.0)])

    text = build(sglang_run, tmp_path / "out")["decode"]

    assert "**Not available.** This phase logged no collective records" in text
    assert "## Step time" in text, "the channels that were collected still report"
    for gone in ("collective_message_sizes.csv", "collective_totals.csv", "per_rank.csv"):
        assert not (tmp_path / "out_decode" / gone).exists(), gone
    assert (tmp_path / "out_decode" / "step_times.csv").exists()


def test_the_file_list_names_only_files_that_exist(sglang_run: Path, tmp_path: Path):
    """The list stayed fixed while the writing became conditional, so a run with no collective
    records advertised four CSVs that were deliberately never written."""
    for node in (2, 3):
        write(sglang_run / f"decode_NODE{node}.log",
              [server_args_line(), decode_batch_line(batch=16, rate=80.0)])

    text = build(sglang_run, tmp_path / "out")["decode"]
    out = tmp_path / "out_decode"
    listed = {line.split("`")[1] for line in text.splitlines()
              if line.startswith("- `") and line.split("`")[1].endswith(".csv")}

    assert listed, "something was written"
    for name in listed:
        assert (out / name).exists(), f"{name} is listed but absent"


def test_a_phase_without_collectives_does_not_report_zero_traffic(tmp_path: Path):
    """A tuned run logs no COLL rows while still carrying configuration and step times. The
    summary printed `Collectives parsed: 0` and `Traffic per rank: 0 B` for it, contradicting the
    notice further down that the channel is unavailable."""
    from collprof.engines import sglang_disagg

    run = tmp_path / "tuned"
    write(run / "decode_NODE2.log",
          [server_args_line(), decode_batch_line(batch=16, rate=100.0),
           decode_batch_line(batch=16, rate=80.0), topo_line()])

    out = tmp_path / "out_decode"
    phase = parse_run(run, sglang_disagg.SPEC)["decode"]
    assert not phase.sizes, "nothing collective was logged"
    emit_phase(phase, out, ReportContext(spec=sglang_disagg.SPEC, run_dir=run))

    text = (out / "report.md").read_text()
    assert "Collectives parsed" not in text
    assert "Traffic per rank" not in text
    assert "Collective traffic: **not available**" in text
