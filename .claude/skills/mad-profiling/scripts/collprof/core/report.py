"""Composing one phase's report: the CSVs, the markdown, and the workbook that collects both.

Every claim this module makes holds for any engine. Anything true of only one engine arrives as
prose in :class:`~collprof.core.spec.ReportNotes` and is inserted, never asserted here. If a
sentence about mooncake, HIP graphs or ``--local-ranks-filter`` ever appears in this file, the
separation has been broken -- those belong to an engine module.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .phase import BOUND_DAMAGE, Phase
from .rccl_log import transport_scope
from .compare import EXCLUDED_FROM_DIFF
from .rdma_counters import by_kind, kind_order
from .runconfig import NOT_STATED, merge_nodes
from .spec import LOG_SHARED, EngineSpec
from .steps import by_node as steps_by_node
from .steps import invalidators
from .steps import graph_state
from .steps import summarise as summarise_steps
from .units import LATENCY_BOUND_BYTES, MIB, fmt_bytes, fmt_per_rank_calls
from .workbook import write_workbook


@dataclass
class ReportContext:
    """Everything a report needs besides the phase itself."""

    spec: EngineSpec
    run_dir: Path
    top: int = 20
    rocprof: dict | None = None
    rocprof_dir: Path | None = None
    torch_trace: dict | None = None
    #: Capture directories behind ``torch_trace``. Named because a phase can be captured on several
    #: replicas at once, and two replicas of the same role are not always loaded alike: their
    #: message sizes then appear side by side in one table.
    trace_dirs: tuple[str, ...] = ()
    #: ``{node: CounterSeries}`` from the RDMA adapter counters, when the run sampled them.
    #: Empty means the channel is absent, not that it measured zero.
    counters: dict = field(default_factory=dict)
    #: Capture directories that held no traces, per phase. Stated in the report rather than only on
    #: the console: a phase whose second replica captured nothing covers fewer nodes than it looks.
    empty_trace_dirs: tuple[str, ...] = ()
    #: How this report was produced, recorded in the header so a number can be traced back.
    command: str = ""
    parse_version: int = 0
    #: Settings separating this run from a reference one. None means no comparison; an empty
    #: list means one was made and found nothing.
    config_diff: list | None = None
    #: The run the diff was taken against.
    config_diff_source: str = ""
    #: Why a requested comparison could not be made; without it the report looks like one where
    #: none was asked for.
    config_diff_unavailable: str = ""
    #: ``{which run: {setting: {node: value}}}`` for settings a run's own nodes disagreed on.
    #: Such a key is absent from the merged configuration, so the diff cannot see it.
    config_diff_split: dict = field(default_factory=dict)


class CsvSink:
    """The CSVs of one report, remembering what it wrote.

    A rerun into an existing directory must not leave the previous run's CSVs behind: the workbook
    collects whatever CSV sits next to the report, so a file this run did not write would come back
    as a sheet dated differently from every other number.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.written: set = set()

    def write(self, name: str, header: list, rows: list) -> None:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        self.written.add(path.resolve())

    def drop_stale(self, only: set | None = None) -> None:
        """Delete CSVs this run did not write, optionally restricted to names it may own.

        ``only`` exists because a comparison writes into a directory the caller names, which may
        be a run directory holding the `perf_*.csv` it reads. Deleting those destroys input.
        """
        for path in self.out_dir.glob("*.csv"):
            if only is not None and path.name not in only:
                continue
            if path.resolve() not in self.written:
                path.unlink()


@dataclass
class PhaseView:
    """Per-rank normalisation of a phase, computed once and shared by every section."""

    phase: Phase
    spec: EngineSpec
    totals: dict = field(init=False)
    active: dict = field(init=False)
    idle: int = field(init=False)
    reps: int = field(init=False)
    grand_calls: int = field(init=False)
    grand_bytes: int = field(init=False)
    per_rank_bytes: float = field(init=False)
    #: Whether any of these records came off a stream several ranks shared. What an engine says
    #: about torn records and about ranks missing from stdout only holds for those.
    shared_logs: bool = field(init=False)

    def __post_init__(self):
        self.totals = self.phase.collective_totals()
        # Everything parsed from the logs is the sum over the ranks that logged, so divide by their
        # number for per-rank figures -- counting only ranks that carried traffic, since an idle
        # replica would otherwise drag every average down by a factor of two.
        self.active = self.phase.active_ranks(self.spec.limits.idle_rank_fraction)
        self.idle = len(self.phase.per_rank) - len(self.active)
        self.reps = max(len(self.active), 1)
        self.grand_calls = sum(r["calls"] for r in self.totals.values())
        self.grand_bytes = sum(r["bytes"] for r in self.totals.values())
        self.per_rank_bytes = self.grand_bytes / self.reps
        self.shared_logs = LOG_SHARED in self.phase.writers or not self.phase.writers


# -- CSV tables ----------------------------------------------------------------------------------


def write_tables(view: PhaseView, sink: CsvSink, ctx: ReportContext) -> dict:
    """Write every CSV of a report and return what the markdown still needs from them."""
    phase, reps = view.phase, view.reps

    hist_rows = []
    # Written only when the channel produced something: an empty CSV reads in a workbook as a
    # channel measured as zero.
    for (coll, nr, msg_bytes, dt), calls in sorted(phase.sizes.items(),
                                                   key=lambda kv: -kv[1] * kv[0][2]):
        vol = calls * msg_bytes / reps
        share = round(100.0 * vol / view.per_rank_bytes, 3) if view.per_rank_bytes else 0.0
        hist_rows.append([coll, nr, msg_bytes, round(msg_bytes / MIB, 4), dt, round(calls / reps),
                          round(vol), share])
    if hist_rows:
        sink.write("collective_message_sizes.csv",
                   ["collective", "nranks", "msg_bytes", "msg_mib", "datatype", "calls_per_rank",
                    "bytes_per_rank", "pct_of_bytes"], hist_rows)

    total_rows = []
    for coll, row in sorted(view.totals.items(), key=lambda kv: -kv[1]["bytes"]):
        total_rows.append([coll, round(row["calls"] / reps), round(row["bytes"] / reps),
                           len(row["sizes"]), min(row["sizes"]), max(row["sizes"]),
                           round(100.0 * row["bytes"] / view.grand_bytes, 3)
                           if view.grand_bytes else 0.0])
    if total_rows:
        sink.write("collective_totals.csv",
                   ["collective", "calls_per_rank", "bytes_per_rank", "distinct_sizes", "min_bytes",
                    "max_bytes", "pct_of_bytes"], total_rows)

    if phase.per_node:
        sink.write("per_node.csv", ["node", "collective", "calls", "total_bytes"],
                   [[node, coll, calls, nbytes]
                    for (node, coll), (calls, nbytes) in sorted(phase.per_node.items())])
    if phase.per_rank:
        sink.write("per_rank.csv", ["node", "rank", "calls", "total_bytes"],
                   [[node, rank, calls, nbytes]
                    for (node, rank), (calls, nbytes) in sorted(phase.per_rank.items())])

    if phase.damage:
        sink.write("discarded_records.csv", ["reason", "records"],
                   [[reason, count] for reason, count in phase.damage.most_common()])

    if phase.config:
        perf = ctx.spec.run_config.perf_relevant
        agreed, disagreed = merge_nodes(phase.config, phase.config_nodes)
        sink.write("run_config.csv", ["setting", "value", "perf_relevant", "agreed_across_nodes"],
                   [[key, agreed[key], key in perf, True] for key in sorted(agreed)]
                   + [[key, "; ".join(f"{n}={v}" for n, v in sorted(spread.items())),
                       key in perf, False]
                      for key, spread in sorted(disagreed.items())])

    step_rows = []
    # A withheld channel is withheld everywhere: writing the CSV would put the numbers the section
    # refuses to state into a workbook sheet under `median_ms`.
    for node, st in ({} if invalidators(phase.config, ctx.spec.steps)
                     else steps_by_node(phase.steps)).items():
        step_rows.append([node, st.intervals, st.batch_min, st.batch_max, round(st.median_ms, 3),
                          round(st.p95_ms, 3), round(st.min_ms, 3), round(st.max_ms, 3),
                          graph_state(st)])
    if step_rows:
        sink.write("step_times.csv",
                   ["node", "intervals", "batch_min", "batch_max", "median_ms", "p95_ms", "min_ms",
                    "max_ms", "graph_replay"], step_rows)

    counter_rows = counter_table(view, ctx)
    if counter_rows:
        sink.write("fabric_counters.csv",
                   ["node", "device", "port", "counter", "kind", "delta", "window_s", "samples"],
                   counter_rows)

    a2a_rows = []
    trace = ctx.torch_trace or {}
    files = max(trace.get("files", 1), 1)
    for (stage, cat), (calls, dur_us) in sorted((trace.get("a2a") or {}).items()):
        total = (trace.get("category_us") or {}).get(cat, 0.0)
        # No absolute duration: trace durations fail a cross-check against rocprofv3 by orders of
        # magnitude, and a CSV column outlives the caveat beside it. Only the share survives.
        a2a_rows.append([stage, cat, round(calls / files, 2),
                         round(100.0 * dur_us / total, 3) if total else ""])
    if a2a_rows:
        sink.write("expert_a2a.csv",
                   ["stage", "category", "calls_per_trace", "pct_of_category"], a2a_rows)

    edge_rows, matrix_cells = [], defaultdict(int)
    for (src, dst, transport), channels in sorted(phase.edges.items()):
        scope = transport_scope(transport)
        edge_rows.append([src, dst, transport, scope, len(channels),
                          ",".join(str(c) for c in sorted(channels))])
        matrix_cells[(src, dst)] += len(channels)
    if edge_rows:
        sink.write("comm_edges.csv",
                   ["src_rank", "dst_rank", "transport", "scope", "channels", "channel_ids"],
                   edge_rows)
        ranks = sorted({r for pair in matrix_cells for r in pair})
        sink.write("rank_matrix.csv", ["src \\ dst"] + [str(r) for r in ranks],
                   [[src] + [matrix_cells.get((src, dst), 0) for dst in ranks] for src in ranks])

    # rocprofv3 sums over every profiled process of the whole job (all ranks, all phases), so
    # durations are reported per process rather than as job totals.
    timing_rows = []
    if ctx.rocprof:
        for coll, row in sorted(view.totals.items(), key=lambda kv: -kv[1]["bytes"]):
            stats = ctx.rocprof["rccl"].get("nccl" + coll)
            if not stats or not stats["ns"]:
                continue
            procs = max(stats["pids"], 1)
            timing_rows.append([coll, procs, round(stats["calls"] / procs),
                                round(stats["ns"] / procs / 1e9, 4),
                                round(stats["ns"] / max(stats["calls"], 1) / 1000.0, 2),
                                round(stats["max_ns"] / 1000.0, 2)])
        sink.write("collective_enqueue_times.csv",
                   ["collective", "processes", "calls_per_process", "enqueue_s_per_process",
                    "avg_us", "max_us"], timing_rows)

    return {"edge_rows": edge_rows, "matrix_cells": dict(matrix_cells), "timing_rows": timing_rows}


# -- markdown sections ---------------------------------------------------------------------------


def section_header(view: PhaseView, ctx: ReportContext) -> list:
    phase, notes = view.phase, ctx.spec.notes
    ranks_per_node = phase.ranks_per_node()
    lines = [
        f"# Collective profile — {phase.name}",
        "",
        f"Source run: `{ctx.run_dir}`  ",
        f"Engine: **{ctx.spec.name}** — {ctx.spec.summary}  ",
        # Counted from every channel, not the collectives alone: `Phase.nodes` fills only when a
        # collective record parses, so a tuned run would report zero contributing nodes.
        f"Nodes contributing log data: {len(_contributing(phase))} "
        f"({', '.join(sorted(_contributing(phase))) or 'none'})  ",
        f"Ranks present in the log: {len(phase.per_rank)} "
        f"({', '.join(f'{n}: {c}' for n, c in sorted(ranks_per_node.items()))})"
        + (f", of which {view.reps} carried traffic" if view.idle else "")
        + (f" — {notes.rank_coverage}  " if notes.rank_coverage and view.shared_logs else "  "),
        "RCCL records read from: "
        + ("a stream shared by the ranks of a node"
           if view.shared_logs else "one file per process (`NCCL_DEBUG_FILE`)"),
        f"Communicator size seen: nranks={phase.nranks}"
        + (f", {notes.communicator.format(nranks=phase.nranks)}" if notes.communicator else ""),
    ]
    if ctx.command:
        lines += ["", f"> Produced by `{ctx.command}` with parser version {ctx.parse_version}. "
                      "Rerunning that command on the same inputs reproduces this file."]
    return lines


def counter_table(view, ctx: ReportContext) -> list:
    """The rows of `fabric_counters.csv`, so the writer and the file list cannot disagree.

    A sampled node with no deltas -- a header-only file, one reading, a driver exposing nothing --
    is a phase with counters and an empty table, and listing the file on the strength of the first
    pointed at an artifact the writer had not created.
    """
    rows = []
    for node, series in sorted(phase_counters(view.phase, ctx).items()):
        for kind, group in sorted(by_kind(series, ctx.spec.counters).items()):
            for (dev, port, counter), delta in sorted(group.items()):
                rows.append([node, dev, port, counter, kind, delta,
                             round(series.seconds, 1), series.samples])
    return rows


def section_summary(view: PhaseView, ctx: ReportContext) -> list:
    phase, spec = view.phase, ctx.spec
    lines = ["", "## Summary", ""]
    # Only when collectives parsed. A run reporting configuration, step times and adapter counters
    # without a single COLL row is now a supported report, and printing "Collectives parsed: 0"
    # and "Traffic per rank: 0 B" for it states a measurement that was never taken -- directly
    # against the notice further down saying the channel is unavailable.
    if phase.sizes:
        lines += [
            f"- Collectives parsed (nranks>1): **{view.grand_calls}** over {view.reps} ranks "
            f"= {view.grand_calls // view.reps} per rank"
            + (f", leaving out {view.idle} rank(s) that carried under "
               f"{spec.limits.idle_rank_fraction:.0%} of the busiest rank" if view.idle else ""),
            f"- Traffic per rank: **{fmt_bytes(view.per_rank_bytes)}**"]
    else:
        lines.append("- Collective traffic: **not available** — no collective record parsed for "
                     "this phase, which is a channel that did not report rather than a run that "
                     "sent nothing")

    iters = 0
    if spec.iteration_metric:
        series = phase.metric(spec.iteration_metric)
        iters = len(series)
        if series:
            label = next((m.label for m in spec.metrics if m.key == spec.iteration_metric), "")
            median = statistics.median(series)
            lines.append(f"- Iterations per rank: {iters}, "
                         + (label.format(value=median) if label else f"median {median:.1f}")
                         + " (median, so warmup outliers do not shift it)")
            lines.append("- Volume per iteration per rank: "
                         f"**{fmt_bytes(view.per_rank_bytes / iters)}**")

    for metric in spec.metrics:
        if metric.key == spec.iteration_metric or not metric.label:
            continue
        values = phase.metric(metric.key)
        if values:
            lines.append(f"- {metric.label.format(value=max(values))}")

    lines += section_data_quality(view, ctx)
    if spec.notes.scope:
        lines += [""] + [f"> {line}" for line in spec.notes.scope]
    if not ctx.rocprof:
        # An engine printing its own step time has a duration without a profiler, so the notice
        # must say what is missing is a *device* duration, not duration as such.
        step_note = (" The engine's own step time is reported below, which is a duration for the "
                     "step and not for any collective in it."
                     if view.phase.steps else "")
        lines += ["", "> No rocprofv3 RCCL stats supplied, so this report carries no per-"
                      "collective or per-kernel durations and no bandwidth: message sizes and "
                      "call counts only." + step_note]
    else:
        lines += ["", "> rocprofv3 stats cover the whole job, including initialisation and every "
                      "phase, while the volumes above are per phase. The two therefore cannot be "
                      "divided into a per-phase bandwidth, and none is quoted anywhere in this "
                      "report."]
    return lines


def section_data_quality(view: PhaseView, ctx: ReportContext) -> list:
    """Records that were dropped, and whether that means torn logs or bounds set too low.

    A silent discard is the failure mode this section exists to prevent: bounds are calibrated to a
    run's scale, so a larger topology or model can have perfectly real records rejected. The counts
    are per reason, and hitting a bound is called out with the flag that raises it.
    """
    phase, limits = view.phase, ctx.spec.limits
    if not phase.damage:
        return []

    total_records = phase.damaged + view.grand_calls
    share = phase.damaged / max(total_records, 1)
    lines = [f"- Discarded as unusable: {phase.damaged} of {total_records} collective records "
             f"({share:.2%}), broken down in `discarded_records.csv`:"]
    for reason, count in phase.damage.most_common():
        lines.append(f"  - {reason}: {count}")
    if ctx.spec.notes.damage_cause and view.shared_logs:
        lines.append(f"  {ctx.spec.notes.damage_cause}")
    elif not view.shared_logs:
        lines.append("  These came out of per-rank files, where nothing interleaves, so tearing "
                     "is not the explanation: look at the records themselves before trusting the "
                     "volume.")

    if share >= limits.damage_warn_fraction:
        lines.append(f"- **Warning: {share:.1%} of records were discarded**, above the "
                     f"{limits.damage_warn_fraction:.0%} this engine expects. Treat the volumes as "
                     "a lower bound until the cause is understood.")
    bound_hits = sum(phase.damage[reason] for reason in BOUND_DAMAGE)
    if bound_hits:
        lines.append(
            f"- **{bound_hits} record(s) were rejected for exceeding a sanity bound** "
            f"(max message {fmt_bytes(limits.max_msg_bytes)}, max nranks {limits.max_nranks}). "
            "Those bounds catch spliced digits, but they are calibrated to a run's scale: if this "
            "run legitimately moves larger messages or builds wider communicators, raise "
            "`--max-msg-bytes` / `--max-nranks` and reparse, or the volume above is missing them.")
    return lines


def section_run_config(view: PhaseView, ctx: ReportContext) -> list:
    """What the run reported about its own configuration, and how it differs from a reference.

    First because these settings explain more differences between two runs than the communication
    numbers do; omitting them can attribute a graph-free arm's per-step cost to its transport.
    """
    config = view.phase.config
    if not config:
        # A requested comparison that failed must still reach the artifact, including here: a
        # missing section reads as a comparison that was never asked for.
        return section_diff_unavailable(ctx, "## Configuration comparison")

    agreed, disagreed = merge_nodes(config, view.phase.config_nodes)
    lines = ["", "## Configuration this phase ran with", "",
             f"As {len(config)} node(s) of this phase reported themselves, defaults already "
             "applied. This is the effective configuration, which a command line is not.", ""]

    shown = sorted(k for k in agreed if k in ctx.spec.run_config.perf_relevant)
    if shown:
        lines += ["| setting | value |", "|---|---|"]
        lines += [f"| `{key}` | `{agreed[key]}` |" for key in shown]
        lines += ["", f"Performance-relevant settings only, {len(agreed)} recorded in total; the "
                      "full set is in `run_config.csv`."]
    else:
        lines.append("No performance-relevant setting was recognised; the full set is in "
                     "`run_config.csv`.")

    # `host`, `port`, `node_rank` and the seeds differ between nodes by construction, so listing
    # them makes the warning fire on every healthy multi-node run. The full spread stays in
    # `run_config.csv`, which is not a warning.
    notable = sorted(set(disagreed) - ctx.spec.run_config.noise)
    if notable:
        lines += ["", "### Nodes of this phase disagree", "",
                  "One role is meant to run one configuration. A launcher that assembles rank 0's "
                  "command separately from the rest is how a profiled run once carried the "
                  "measurement flags on three nodes out of four, and this is the only witness to "
                  "it. Settings that differ per node by design are left out; `run_config.csv` "
                  "carries every one of them.", "",
                  "| setting | per node |", "|---|---|"]
        for key in notable:
            spread = ", ".join(f"{node}=`{value}`"
                               for node, value in sorted(disagreed[key].items()))
            lines.append(f"| `{key}` | {spread} |")

    lines += section_measurement_check(agreed, ctx, disagreed)
    if ctx.config_diff is not None:
        lines += section_config_diff(ctx)
    else:
        lines += section_diff_unavailable(ctx, "### Difference from the reference run")
    return lines


def section_measurement_check(agreed: dict, ctx: ReportContext,
                              disagreed: dict | None = None) -> list:
    """Whether the run bears out what this engine's scope note claims about it.

    A report that claims a measurement configuration while showing the traffic of a tuned one is
    worse than one that claims nothing.
    """
    # `disagreed` too, not agreed values alone: a key three nodes set one way and one the other is
    # a definite failure on that node, and checking only agreed keys would say nothing.
    contradictions = []
    for key, expected in ctx.spec.measurement_assumptions:
        if key in agreed:
            if agreed[key] != expected:
                contradictions.append((key, expected, agreed[key], ""))
            continue
        spread = (disagreed or {}).get(key) or {}
        against = {node: value for node, value in spread.items()
                   if value not in (expected, NOT_STATED)}
        if against:
            shown = ", ".join(f"{node}=`{value}`" for node, value in sorted(against.items()))
            contradictions.append((key, expected, shown,
                                   f" on {len(against)} of {len(spread)} nodes"))
    if not contradictions:
        return []
    lines = ["", "### The scope note above does not hold for this run", "",
             "This engine's scope note says which profiler-bypassing paths were disabled so the "
             "collectives could be observed. The run reports otherwise:", "",
             "| setting | scope note assumes | this run | ", "|---|---|---|"]
    for key, expected, actual, where in contradictions:
        shown = actual if where else f"`{actual}`"
        lines.append(f"| `{key}` | `{expected}` | {shown}{where} |")
    lines += ["", "Treat the volumes above as a floor rather than the phase's traffic: a path the "
                  "scope note assumed was routed through RCCL was not, so whatever crossed it is "
                  "missing here. This is the difference between a report that under-counts and one "
                  "that is wrong about what it counted."]
    return lines


def section_diff_unavailable(ctx: ReportContext, heading: str) -> list:
    """The notice for a comparison that was asked for and could not be made.

    The heading is the caller's, since the two paths that reach this sit at different depths. The
    notice exists because a missing section reads as a comparison that was never asked for.
    """
    if not ctx.config_diff_unavailable:
        return []
    return ["", heading, "",
            f"**Not available.** {ctx.config_diff_unavailable} The comparability check this report "
            "would otherwise carry is therefore missing, and its absence is stated rather than "
            "left to look like a comparison that was never asked for."]


def section_config_diff(ctx: ReportContext) -> list:
    """The settings separating this run from the reference it was compared against."""
    diff = ctx.config_diff
    lines = ["", f"### Difference from `{ctx.config_diff_source}`", ""]
    split_seen = False
    perf = ctx.spec.run_config.perf_relevant
    for which, split in sorted((ctx.config_diff_split or {}).items()):
        # Every non-noise key, not only the throughput-relevant ones: a split key is absent from
        # the diff and from the agreed CSV rows, so this is the last chance to mention it.
        # Relevance is marked per key instead of deciding what is shown.
        keys = sorted(k for k in split if k not in ctx.spec.run_config.noise)
        if keys:
            moving = [k for k in keys if k in perf]
            # Any non-noise split, not only a throughput-relevant one: the diff is blind to every
            # split key, and "no setting differs" over a blind spot is not a finding.
            split_seen = True
            named = ", ".join(f"`{k}`" + (" **(moves throughput)**" if k in perf else "")
                              for k in keys)
            lines += [f"**{which} has {len(keys)} setting(s) its own nodes disagree on: {named}.** "
                      "They are absent from the merged configuration and therefore from the diff "
                      "below, so the comparison cannot speak to them"
                      + (f", and the runs are not comparable on the {len(moving)} marked above "
                         "whatever it says." if moving else "."), ""]
    if not diff:
        # Only when nothing above already said the opposite: a split makes the diff empty
        # *because* of it, so declaring the runs comparable would contradict the paragraph above.
        lines.append(f"No setting differs between the values both runs agreed on -- "
                     f"{EXCLUDED_FROM_DIFF}. The split above still stands, so this is not a "
                     "declaration that the runs are comparable." if split_seen else
                     f"No setting differs; {EXCLUDED_FROM_DIFF}. The two runs are comparable "
                     "on configuration.")
        return lines

    critical = [s for s in diff if s.perf_relevant]
    lines += ["| setting | this run | reference | |", "|---|---|---|---|"]
    for setting in diff:
        # A setting one side never reported is absent rather than null: not having it is a
        # different statement from having it unset.
        left = f"`{setting.left}`" if setting.left is not None else "*absent*"
        right = f"`{setting.right}`" if setting.right is not None else "*absent*"
        mark = " **moves throughput** " if setting.perf_relevant else " "
        lines.append(f"| `{setting.key}` | {left} | {right} |{mark}|")
    if critical:
        keys = ", ".join(f"`{s.key}`" for s in critical)
        lines += ["", f"**{len(critical)} of these move throughput on their own: {keys}.** A "
                      "throughput or latency difference between these two runs is not attributable "
                      "to anything else in this report until they are matched. Nothing here says "
                      "which arm is the misconfigured one -- that is a question about intent."]
    else:
        lines += ["", "None of these is known to move throughput on its own."]
    return lines


def section_steps(view: PhaseView, ctx: ReportContext) -> list:
    """The step-time distribution the engine's own logging implies.

    Often the only duration channel a serving run has, and the server computes it rather than an
    instrument, so it is comparable between a profiled and an unprofiled run.
    """
    per_node = steps_by_node(view.phase.steps)
    if not per_node:
        return []

    broken = invalidators(view.phase.config, ctx.spec.steps)
    if broken:
        named = "; ".join(f"`{setting}={value}`, so {why}" for setting, value, why in broken)
        return ["", "## Step time, from the engine's own accounting", "",
                f"**Withheld.** This run reported {named}. The quotient is still a number and "
                "still looks like a step time, and the batch-invariance check that guards this "
                "channel cannot see the breakage, so it is not reported at all rather than "
                "reported with a caveat. Read per-token latency from the benchmark's own ITL "
                "instead, which measures the interval rather than deriving it."]

    unit = ctx.spec.steps.unit
    pooled = summarise_steps([r for records in view.phase.steps.values() for r in records])
    basis = ctx.spec.notes.step_basis
    lines = ["", "## Step time, from the engine's own accounting", "",
             f"One record per {unit}. " + (basis + " " if basis else "")
             + "Reported per node because a replica running a batch of 8 against another's 512 has "
               "a step time that says nothing about the run.", "",
             "| node | intervals | batch | median ms | p95 ms | p95/median | graphs |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for node, st in per_node.items():
        batch = (f"{st.batch_min}" if st.batch_min == st.batch_max
                 else f"{st.batch_min}-{st.batch_max}")
        graphed = graph_state(st)
        graphed = "**off**" if graphed == "off" else graphed
        lines.append(f"| {node} | {st.intervals} | {batch} | {st.median_ms:.1f} | "
                     f"{st.p95_ms:.1f} | {st.spread:.2f} | {graphed} |")

    if pooled:
        lines += ["", f"Pooled median {pooled.median_ms:.1f} ms, p95 {pooled.p95_ms:.1f} ms over "
                      f"{pooled.intervals} interval(s)."]
        # The shape of the distribution separates a fixed per-step cost from a volume-limited one.
        if pooled.spread < 1.15:
            lines.append("The distribution is flat (p95 within 15% of the median), so this run's "
                         "step time is not carried by occasional stragglers. Whether a "
                         "*difference* from another run is a fixed per-step cost is a separate "
                         "question one run cannot answer: it needs the gap observed across "
                         "batches or concurrency, which is what a comparison is for.")
        else:
            lines.append("The distribution has a tail (p95 well above the median), so a mean "
                         "difference against another run may be a few slow steps rather than a "
                         "per-step cost. Compare the medians.")

    # `None` is a node that did not state its graph mode, not a third state: reporting it as
    # disagreement would make a definite claim out of the unknown.
    graph_states = {st.graphed for st in per_node.values()} - {None}
    # A node whose own intervals disagreed reports `graphed is None`, so the set above cannot see
    # it; asking the nodes directly keeps a batch-dependent mixture from reading as "not stated".
    mixed_within = any(st.graphs_mixed for st in per_node.values())
    if False in graph_states or mixed_within:
        # Three claims because there are three states. `graph_states` drops `None`, so
        # `{False, None}` reads as `{False}` and silence becomes a claim about the silent node;
        # the per-node summaries are asked directly instead.
        uniform_off = (all(st.graphed is False for st in per_node.values()) and not mixed_within)
        silent = [node for node, st in per_node.items() if st.graphed is None]
        if uniform_off:
            claim = ("Every step here pays host-side launch cost that a graphed run does not, so "
                     "the difference is fixed per step.")
        elif not mixed_within and silent and graph_states == {False}:
            claim = (f"The nodes that stated it ran without replay; {len(silent)} node(s) did not "
                     "state theirs, so how much of this distribution pays host-side launch cost "
                     "is unknown rather than all of it.")
        else:
            claim = ("The intervals that ran without replay pay host-side launch cost that a "
                     "graphed run does not, and the rest do not, so the cost falls on part of "
                     "this distribution rather than on all of it.")
        lines += ["", "**Graph replay is off for at least one node of this phase.** " + claim
                  + " Either way it is the first thing to rule out before attributing a step-time "
                    "gap to transport."]
        if ctx.spec.notes.graphs_off:
            lines.append(ctx.spec.notes.graphs_off)
    if len(graph_states) > 1 or mixed_within:
        lines.append("")
        lines.append("Intervals of this phase disagree on graph replay"
                     + (" within a single node" if mixed_within else " between nodes")
                     + ", so these step times mix replayed and ungraphed steps and their "
                       "distribution is not one population.")
    return lines


def _contributing(phase) -> set:
    """Every node this phase has data from, by any channel.

    A node that logged no collective still contributed if it reported a configuration or a step.
    """
    return set(phase.nodes) | set(phase.config_nodes) | set(phase.steps)


def phase_counters(phase, ctx: ReportContext) -> dict:
    """This phase's share of a run-wide counter map.

    Samples are per node for the whole job while a report is per phase, so the split lives here
    alone or the markdown and `fabric_counters.csv` disagree. Split by the engine's declared rule,
    not by which nodes reached the logs: a node whose log was lost still carried traffic.
    """
    return {node: s for node, s in (ctx.counters or {}).items()
            if ctx.spec.counters.phase_of_node(node, phase.name)}


def section_counters(view: PhaseView, ctx: ReportContext) -> list:
    """What this phase's nodes put on the wire, from the adapters' own counters.

    The only channel that says how much traffic reached the wire, and the only one cheap enough to
    collect from a tuned run. It reports verbs, not causality, and counts every user of the NIC,
    so it bounds the exchange rather than measuring it; its value is in comparing two arms.
    """
    series = phase_counters(view.phase, ctx)
    if not series:
        return []

    lines = ["", "## What crossed the fabric", "",
             "From the adapters' own counters, sampled while the servers ran. Per node and per "
             "adapter -- never per rank and never per kernel -- and every user of the NIC is "
             "included, the KV transfer among them. Read these as a **ceiling** for any one "
             "operation, and compare them against another arm rather than in isolation.", ""]

    sampled = [s for s in series.values() if s.samples > 1]
    if not sampled:
        lines.append("Only one sample per node was recorded, so there is no window to take a "
                     "difference over. The counters are cumulative since the adapter came up, "
                     "which says nothing about this run.")
        return lines

    # Only the nodes with a window: a node sampled once has no delta, and a row of zeros for it is
    # partial collection presenting itself as a measurement. Those are named below the table.
    measured = {node: s for node, s in series.items() if s.samples > 1}
    omitted = sorted(set(series) - set(measured))

    # Only the kinds something was counted under: a column of zeros for a kind this run never saw
    # is noise in a table read for its shape.
    grouped = {node: by_kind(s, ctx.spec.counters) for node, s in measured.items()}
    present = {k for g in grouped.values() for k in g}
    # Present, not non-zero: a kind measured as zero is a fact about this run, and dropping the
    # column would say instead that the counter was never there.
    shown = [k for k in kind_order(ctx.spec.counters, present) if k in present]
    lines += ["| node | samples | window s | " + " | ".join(shown) + " |",
              "|---|---:|---:|" + "---:|" * len(shown)]
    for node, s in sorted(measured.items()):
        cells = [f"{sum(grouped[node].get(k, {}).values()):,}" for k in shown]
        lines.append(f"| {node} | {s.samples} | {s.seconds:.0f} | " + " | ".join(cells) + " |")

    # The table is read for its shape, and the asymmetry is the point: reads and atomics in
    # quantity are evidence of a protocol that waits, their absence is not evidence of one that
    # does not.
    lines += ["", "These are **verb counts, not causality**. A reply can itself be an RDMA write "
                  "or a SEND, and transport acknowledgements appear as neither a read nor an "
                  "atomic, so a write-only profile is consistent with a one-sided protocol "
                  "without establishing one. Reads or atomics in quantity are positive evidence "
                  "of a protocol that waits; their absence proves nothing on its own. Counters "
                  "the engine has not classified keep their own names, which is where a "
                  "driver's own word for something interesting shows up first."]

    if omitted:
        lines += ["", f"**{len(omitted)} node(s) sampled no window and are not in the table** "
                      f"({', '.join(omitted)}). Their files exist -- a header, or a single "
                      "reading that is cumulative since the adapter came up -- so this phase is "
                      "covered less completely than the rows suggest, and a zero row for them "
                      "would have said the opposite."]

    torn = {node: s.damaged for node, s in series.items() if s.damaged}
    if torn:
        named = "; ".join(f"{node}: {n}" for node, n in sorted(torn.items()))
        lines += ["", f"**{sum(torn.values())} sampled line(s) were unreadable and skipped** "
                      f"({named}). The samples land on a shared filesystem that returns zeros "
                      "rather than an error when a read falls in a bad window, so a line can come "
                      "back as NUL bytes. One lost line costs one counter of one sample; the "
                      "window is defined by the samples that did parse."]
    wrapped = {node: s.wrapped for node, s in series.items() if s.wrapped}
    if wrapped:
        named = "; ".join(f"{node}: {len(keys)}" for node, keys in sorted(wrapped.items()))
        lines += ["", f"**Counters that went backwards were dropped** ({named}). A counter "
                      "decreasing means it wrapped or the adapter was reset mid-run; the work it "
                      "represented is not recoverable, so those columns are floors."]
    return lines


def section_a2a(view: PhaseView, ctx: ReportContext) -> list:
    """The expert all-to-all, which reaches no RCCL log.

    A backend carrying its own transport leaves the report's strongest channel silent about the
    operation under comparison. Classification is by event name, so this is a discovery aid: it
    says what matched, and says nothing rather than zero when nothing did.
    """
    trace = ctx.torch_trace
    if not trace or not ctx.spec.a2a.patterns:
        return []
    a2a = trace.get("a2a") or {}
    category_us = trace.get("category_us") or {}
    files = max(trace["files"], 1)

    outside = ctx.spec.notes.a2a_outside_rccl
    lines = ["", "## Expert all-to-all (torch trace, by event name)", ""]
    # Repeated here because the collective section returns early when a capture holds no
    # `record_param_comms` events -- exactly the capture this section exists to describe.
    if not (trace.get("events") or {}):
        lines += trace_quality_notes(trace)
    if not a2a:
        top = sorted((trace.get("unmatched_us") or {}).items(), key=lambda kv: -kv[1])[:10]
        lines += ["No event name matched this engine's all-to-all patterns, so this section "
                  "reports nothing rather than zero. The traffic exists but under names these "
                  "patterns do not know." + (f" {outside}" if outside else ""), ""]
        if top:
            kernel_us = (trace.get("category_us") or {}).get("kernel", 0.0)
            lines += ["The busiest unclassified device events, for extending "
                      "`A2A_PATTERNS` in the engine module. Ranked by share of this capture's "
                      "kernel time, since the absolute durations are not trustworthy:", "",
                      "| event | share of kernel time |", "|---|---:|"]
            lines += [f"| `{name}` | {100.0 * us / kernel_us:.1f}% |" if kernel_us else
                      f"| `{name}` | n/a |" for name, us in top]
        return lines

    lines += ["Classified by event name, per captured trace."
              + (f" {outside}" if outside else ""), "",
              "| stage | category | calls per trace | share of category |",
              "|---|---|---:|---:|"]
    for (stage, cat), (calls, dur_us) in sorted(a2a.items(), key=lambda kv: -kv[1][1]):
        total = category_us.get(cat, 0.0)
        share = f"{100.0 * dur_us / total:.1f}%" if total else "n/a"
        lines.append(f"| {stage} | {cat} | {calls / files:.1f} | {share} |")

    named = trace.get("a2a_names") or {}
    if named:
        lines += ["", "### Which kernels these were", "",
                  "The kernel name says which variant of the exchange ran. A backend on its "
                  "low-latency kernels and one on its throughput kernels are not comparable as "
                  "backends, and this is where that shows without having to trust a flag.", "",
                  "| stage | variant | kernel | calls per trace | share of exchange |",
                  "|---|---|---|---:|---:|"]
        spent = sum(dur for _calls, dur in named.values()) or 1.0
        for (stage, name), (calls, dur_us) in sorted(named.items(), key=lambda kv: -kv[1][1])[:12]:
            variant = next((label for label, pattern in ctx.spec.a2a.variants
                            if pattern.search(name)), "not stated")
            lines.append(f"| {stage} | {variant} | `{name}` | {calls / files:.1f} | "
                         f"{100.0 * dur_us / spent:.1f}% |")

    lines += ["", "Durations are the trace's own and `references/interpretation.md` rules them out "
                  "for absolute claims: they fail a cross-check against rocprofv3 by orders of "
                  "magnitude. The share of one category within one trace is what this table is "
                  "for -- comparable between two runs captured the same way, not against a "
                  "wall-clock step time.",
              "", "Call counts do not carry the *duration* caveat: per rank per step they should "
                  "resemble the model, one dispatch and one combine per MoE layer. They carry the "
                  "capture's own, though -- a trace that stopped mid-stream omits a suffix, so "
                  "where this report says a trace was cut short its counts are floors as well.",
              ""]
    return lines


def section_traffic(view: PhaseView) -> list:
    lines = ["", "## Traffic by collective (per rank)", "",
             "| collective | calls | volume | share | distinct sizes | size range |",
             "|---|---:|---:|---:|---:|---|"]
    for coll, row in sorted(view.totals.items(), key=lambda kv: -kv[1]["bytes"]):
        share = 100.0 * row["bytes"] / view.grand_bytes if view.grand_bytes else 0.0
        lines.append(f"| {coll} | {fmt_per_rank_calls(row['calls'], view.reps)} | "
                     f"{fmt_bytes(row['bytes'] / view.reps)} | {share:.1f}% | "
                     f"{len(row['sizes'])} | "
                     f"{fmt_bytes(min(row['sizes']))} .. {fmt_bytes(max(row['sizes']))} |")
    return lines


def section_imbalance(view: PhaseView, ctx: ReportContext) -> list:
    phase = view.phase
    if len(phase.per_rank) <= 1:
        return []

    vols = [v[1] for v in view.active.values()]
    calls = [v[0] for v in view.active.values()]
    spread = (max(vols) - min(vols)) / max(vols) * 100.0 if max(vols) else 0.0
    lines = ["", "## Imbalance across logged ranks", "",
             f"- Volume per rank: {fmt_bytes(min(vols))} min, {fmt_bytes(max(vols))} max, "
             f"spread **{spread:.2f}%**",
             f"- Calls per rank: {min(calls)} min, {max(calls)} max",
             "- Full per-rank breakdown in `per_rank.csv`; this covers only the ranks that reach "
             "stdout, so it is a sanity check on symmetry rather than a full imbalance profile."]

    # One number over all ranks hides which of two effects is present: the exchange inside a node is
    # symmetric by construction, while a router is free to send one node more work than another, and
    # only the second is a finding.
    by_node: dict = defaultdict(list)
    for (node, _rank), (_calls, nbytes) in phase.per_rank.items():
        by_node[node].append(nbytes)
    if len(by_node) > 1:
        floor = ctx.spec.limits.idle_rank_fraction * max(sum(w) for w in by_node.values())
        busy = [v for v in by_node.values() if sum(v) >= floor]
        worst = max(((max(v) - min(v)) / max(v) * 100.0 if max(v) else 0.0) for v in busy)
        node_totals = {n: sum(v) for n, v in by_node.items()}
        hi, lo = max(node_totals.values()), min(node_totals.values())
        lines += [
            f"- Within a node that carried traffic, worst spread across its ranks: "
            f"**{worst:.2f}%** — the tensor-parallel exchange, which is expected to be even"
            + ("" if worst < 5.0 else ", so a spread this wide means one rank carries work the "
               "others do not, e.g. an unevenly placed expert or a KV receiver"),
            f"- Across nodes, total volume: {fmt_bytes(lo)} to {fmt_bytes(hi)}, "
            f"ratio **{hi / max(lo, 1):.1f}x** ("
            + ", ".join(f"{n}: {fmt_bytes(t)}" for n, t in sorted(node_totals.items()))
            + ") — each node runs its own replica, so this is how requests were spread, not a "
            "communication problem",
        ]
    return lines


def section_size_distribution(view: PhaseView, ctx: ReportContext) -> list:
    lines = ["", f"## Message-size distribution, per rank (top {ctx.top} by volume)", "",
             "| collective | message size | bytes | datatype | calls | volume | share |",
             "|---|---:|---:|---|---:|---:|---:|"]
    ranked = sorted(((k, v) for k, v in view.phase.sizes.items() if k[1] > 1),
                    key=lambda kv: -kv[1] * kv[0][2])[:ctx.top]
    for (coll, _nr, msg_bytes, dt), calls in ranked:
        vol = calls * msg_bytes / view.reps
        share = 100.0 * vol / view.per_rank_bytes if view.per_rank_bytes else 0.0
        lines.append(f"| {coll} | {fmt_bytes(msg_bytes)} | {msg_bytes} | {dt} | "
                     f"{fmt_per_rank_calls(calls, view.reps)} | {fmt_bytes(vol)} | {share:.1f}% |")
    return lines


def section_single_rank(view: PhaseView) -> list:
    single: dict = defaultdict(lambda: [0, 0])
    for (coll, nr, msg_bytes, _dt), calls in view.phase.sizes.items():
        if nr <= 1:
            single[coll][0] += calls
            single[coll][1] += calls * msg_bytes
    if not single:
        return []
    scalls = sum(v[0] for v in single.values()) / view.reps
    sbytes = sum(v[1] for v in single.values()) / view.reps
    lines = ["", "## Single-rank communicators (excluded above)", "",
             f"{round(scalls)} calls per rank, {fmt_bytes(sbytes)} — `nranks=1` communicators, "
             "i.e. local no-ops that move no data between ranks.", "",
             "| collective | calls | nominal volume |", "|---|---:|---:|"]
    for coll, (calls, nbytes) in sorted(single.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| {coll} | {round(calls / view.reps)} | {fmt_bytes(nbytes / view.reps)} |")
    return lines


def section_rocprof(view: PhaseView, ctx: ReportContext, timing_rows: list) -> list:
    if not timing_rows:
        return []
    rocprof = ctx.rocprof
    lines = ["", "## Host-side enqueue time per collective (rocprofv3 `--rccl-trace`)", "",
             f"Merged from `{ctx.rocprof_dir}`, normalised per profiled process. These are **host "
             "API durations**: RCCL collectives are asynchronous, so this is the cost of "
             "enqueueing the operation, not the time on the wire. Blocking shows up in "
             "`ncclGroupEnd` and in the device kernel below.", "",
             "| collective | procs | calls/proc | enqueue s/proc | avg us | max us |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in timing_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")

    ck = rocprof["comm_kernels"]
    if ck:
        lines += ["", "### Device-side communication time", "",
                  "RCCL fuses all collectives into a small set of generic device kernels, so "
                  "device time cannot be attributed to individual collectives — only the "
                  "aggregate is available.", "",
                  "| kernel | procs | calls/proc | s/proc | share of kernel time |",
                  "|---|---:|---:|---:|---:|"]
        ktot = rocprof["kernel_ns"] or 1
        comm_ns_per_proc = 0.0
        for name, v in sorted(ck.items(), key=lambda kv: -kv[1]["ns"]):
            procs = max(v["pids"], 1)
            comm_ns_per_proc += v["ns"] / procs
            lines.append(f"| `{name}` | {procs} | {round(v['calls'] / procs)} | "
                         f"{v['ns'] / procs / 1e9:.2f} | {100.0 * v['ns'] / ktot:.2f}% |")
        if comm_ns_per_proc:
            lines += ["", f"That is {comm_ns_per_proc / 1e9:.1f} s of device communication time "
                          "per rank across the **whole job**, every phase included. No bandwidth "
                          "is derived from it on purpose: the volume above is per phase while this "
                          "time is not, and the fused kernel prevents per-collective attribution "
                          "regardless. This is the only trustworthy device time in the report — "
                          "the torch profiler contributes sizes, not durations."]

    dom = rocprof["domain"]
    if dom:
        tot = sum(v["ns"] for v in dom.values()) or 1
        lines += ["", "### Time domains (compute vs communication)", "",
                  "| domain | calls | total s | share |", "|---|---:|---:|---:|"]
        for name, v in sorted(dom.items(), key=lambda kv: -kv[1]["ns"]):
            lines.append(f"| {name} | {v['calls']} | {v['ns'] / 1e9:.2f} | "
                         f"{100.0 * v['ns'] / tot:.2f}% |")

    notes = []
    init = rocprof["rccl"].get("ncclCommInitRankConfig")
    group = rocprof["rccl"].get("ncclGroupEnd")
    if init:
        notes.append(f"`ncclCommInitRankConfig` accounts for {init['ns'] / 1e9:.1f} s of RCCL "
                     "time; exclude it before quoting any communication share.")
    if group:
        notes.append(f"`ncclGroupEnd` accounts for {group['ns'] / 1e9:.1f} s — grouped collectives "
                     "actually block there, so per-collective API durations understate their cost.")
    if notes:
        lines += ["", "### Caveats", ""] + [f"- {n}" for n in notes]
    return lines


def trace_quality_notes(trace: dict | None) -> list:
    """What was unreadable in a capture, as lines any section rendering it can insert.

    Shared because the collective section returns early when a capture holds no
    ``record_param_comms`` events, which would leave a2a shares with nothing said about the
    traces that failed to read.
    """
    if not trace:
        return []
    lines: list = []
    files = max(trace.get("files", 1), 1)
    cut = trace.get("truncated") or []
    if cut:
        names = ", ".join(f"`{name}`" for name in sorted(cut))
        lines += [f"**{len(cut)} of these {files} trace(s) stop being readable** and were read up "
                  f"to that point: {names}. Two causes, and the report does not claim to tell them "
                  "apart: a rank killed at teardown leaves its last gzip member without a trailer, "
                  "and damaged bytes mid-stream fail to decompress. Those ranks contribute fewer "
                  "events than they ran, so every per-trace count below is a floor, and a per-rank "
                  "imbalance that implicates one of these ranks is the capture's before it is the "
                  "run's.",
                  "", "A failed read is not by itself a damaged file. Repeated parses of one "
                  "unchanged capture on NFS reported 1, then 10, then 12, then 0 unreadable files "
                  "of 16. Reparse before concluding anything from this list, and quote a parse "
                  "that reported none.", ""]
    bad = trace.get("malformed") or {}
    if bad:
        total = sum(count for count, _example in bad.values())
        worst = max(bad.items(), key=lambda kv: kv[1][0])
        lines += [f"**{total} event(s) across {len(bad)} trace(s) carry a duration that is not a "
                  f"number**, e.g. `{worst[1][1]}` in `{worst[0]}`. That shape is two durations "
                  "spliced together, which is what a profiler flushing from several threads "
                  "leaves behind. Neither half is recoverable, and the event may or may not be "
                  "one of the classified ones -- the count above is over every event with an "
                  "unreadable duration. Losing a classified duration understates its share; "
                  "losing an unclassified one shrinks only the denominator and overstates it. "
                  "Treat the shares as **possibly biased in either direction**; which happened is "
                  "not recorded. Calls parsed from a later args line are unaffected.", ""]
    return lines


def section_traces(view: PhaseView, ctx: ReportContext, sink: CsvSink) -> list:
    trace = ctx.torch_trace
    if not trace or not trace["events"]:
        return []

    notes = ctx.spec.notes
    files = max(trace["files"], 1)
    steps = max(trace["steps"], 1)
    group = trace["group_size"] or view.phase.nranks
    ev = trace["events"]
    total_calls = sum(v[0] for v in ev.values())
    total_call_us = sum(v[1] for v in ev.values())
    trace_avg_us = total_call_us / max(total_calls, 1)

    roc_comm_avg_us = 0.0
    if ctx.rocprof and ctx.rocprof["comm_kernels"]:
        roc_calls = sum(v["calls"] for v in ctx.rocprof["comm_kernels"].values())
        roc_ns = sum(v["ns"] for v in ctx.rocprof["comm_kernels"].values())
        roc_comm_avg_us = roc_ns / roc_calls / 1000.0 if roc_calls else 0.0

    captures = (f" from {len(ctx.trace_dirs)} capture directories"
                if len(ctx.trace_dirs) > 1 else "")
    lines = ["", "## Per-collective message size and mix (torch profiler)", "",
             f"From {files} trace(s){captures} for ranks {trace['ranks']}, "
             + (f"{steps} profiled iteration(s) each" if trace["steps"] else notes.unmarked_window)
             + f", group size {group}. The RCCL kernel event carries `Collective name`, "
               "`In/Out msg nelems`, `dtype` and `Group size`, which makes this the only channel "
               "that attributes a message size and a process group to each individual collective.",
             "",
             "Sizes here are the **total message across the group**, following nccl-tests, whereas "
             "the log-derived sections above count the per-rank shard — the two differ by a factor "
             f"of {group}.", ""]
    lines += trace_quality_notes(trace)
    if len(ctx.trace_dirs) > 1:
        lines += ["Counts and volumes are averaged over every trace, and these captures cover "
                  f"{len(ctx.trace_dirs)} replicas of the phase ({', '.join(ctx.trace_dirs)}). "
                  "Replicas are not always loaded alike, so one collective can appear as two rows "
                  "of different size below, and the average message then falls between them rather "
                  "than describing either replica.", ""]
    if ctx.empty_trace_dirs:
        count = len(ctx.empty_trace_dirs)
        lines += [f"{count} further capture "
                  + ("directory held no trace files and contributes"
                     if count == 1 else "directories held no trace files and contribute")
                  + " nothing here: "
                  + f"{', '.join(ctx.empty_trace_dirs)}. An idle replica captures nothing, so "
                  "check whether that process did any work before reading this as a coverage gap.",
                  ""]
    if notes.trace_vs_log:
        lines += [notes.trace_vs_log, ""]

    # The kernel durations in these traces do not survive a cross-check against rocprofv3, so this
    # section reports no time and no bandwidth: a size without a trustworthy duration is still
    # useful, a rate built on a wrong duration is not.
    if roc_comm_avg_us:
        lines += [f"**No timing is derived from this channel.** The same kernel averages "
                  f"{trace_avg_us:.1f} us in the trace and {roc_comm_avg_us / 1000.0:.1f} ms in "
                  f"rocprofv3, a factor of {roc_comm_avg_us / max(trace_avg_us, 1e-9):,.0f}. The "
                  "trace-side value is the one that fails a physical check — it puts multi-GiB "
                  "collectives in tens of microseconds — and summing every kernel in a trace "
                  "accounts for a few percent of its own `ProfilerStep` wall time, while rocprofv3 "
                  "kernel time fills the step. Device time therefore comes from the rocprofv3 "
                  "section above; sizes and counts come from here.", ""]
    else:
        lines += ["**No timing is derived from this channel.** The kernel durations in these "
                  "traces are understated by orders of magnitude against rocprofv3, so only sizes, "
                  "counts and process groups are taken from the trace.", ""]

    by_coll: dict = defaultdict(lambda: [0, 0])
    for (coll, _pg, nbytes, _dt), (calls, _dur) in ev.items():
        by_coll[coll][0] += calls
        by_coll[coll][1] += calls * nbytes
    vol_total = sum(v[1] for v in by_coll.values()) or 1
    lines += ["| collective | calls/iter | total volume/iter | share of volume | avg message |",
              "|---|---:|---:|---:|---:|"]
    for coll, (calls, nbytes) in sorted(by_coll.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| {coll} | {calls / files / steps:.1f} | "
                     f"{fmt_bytes(nbytes / files / steps)} | "
                     f"{100.0 * nbytes / vol_total:.1f}% | {fmt_bytes(nbytes / calls)} |")

    rows = []
    for (coll, pg, nbytes, dt), (calls, _dur) in ev.items():
        rows.append([coll, pg, nbytes, dt, round(calls / files / steps, 1),
                     calls * nbytes / files / steps,
                     round(100.0 * calls * nbytes / vol_total, 2),
                     "latency-bound" if nbytes < LATENCY_BOUND_BYTES else "", calls])
    rows.sort(key=lambda r: -r[5])
    lines += ["", f"### Message size mix (top {ctx.top} by volume)", "",
              "| collective | process group | total size | dtype | calls/iter | volume/iter | "
              "share of volume | note |", "|---|---|---:|---|---:|---:|---:|---|"]
    for r in rows[:ctx.top]:
        lines.append(f"| {r[0]} | {r[1]} | {fmt_bytes(r[2])} | {r[3]} | {r[4]} | "
                     f"{fmt_bytes(r[5])} | {r[6]}% | {r[7]} |")
    lines += ["", f"Messages below {fmt_bytes(LATENCY_BOUND_BYTES)} are marked **latency-bound**: "
                  "their cost follows the number of calls, not the volume, so they matter to the "
                  "step time far more than their share of bytes suggests."]

    sink.write("torch_collective_sizes.csv",
               ["collective", "process_group", "total_bytes", "dtype", "calls_per_iter",
                "bytes_per_iter", "pct_of_volume", "note", "calls_total"], rows)

    # Cross-check the two independent channels: trace volume divided by the group size should land
    # on the per-rank shard volume counted from the log. Only meaningful where iterations exist,
    # since otherwise the two cover different windows by construction.
    iters = len(view.phase.metric(ctx.spec.iteration_metric)) if ctx.spec.iteration_metric else 0
    if iters:
        trace_total = sum(c * b for (_c, _p, b, _d), (c, _u) in ev.items())
        trace_per_rank = trace_total / files / steps / group
        log_per_rank = view.per_rank_bytes / iters
        if log_per_rank:
            lines += ["", f"Cross-check against the log: {fmt_bytes(trace_per_rank)} per rank per "
                          f"iteration from the trace vs {fmt_bytes(log_per_rank)} from the RCCL "
                          f"log — {100.0 * trace_per_rank / log_per_rank:.1f}% agreement between "
                          "two fully independent channels."]

    by_pg: dict = defaultdict(lambda: [0, 0])
    for (_coll, pg, nbytes, _dt), (calls, _dur) in ev.items():
        by_pg[pg][0] += calls
        by_pg[pg][1] += calls * nbytes
    lines += ["", "### Volume by process group", "",
              "| process group | calls/iter | volume/iter | share |", "|---|---:|---:|---:|"]
    for pg, (calls, nbytes) in sorted(by_pg.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| {pg} | {calls / files / steps:.1f} | "
                     f"{fmt_bytes(nbytes / files / steps)} | "
                     f"{100.0 * nbytes / vol_total:.1f}% |")
    return lines


def section_connectivity(view: PhaseView, ctx: ReportContext, tables: dict) -> list:
    edge_rows = tables["edge_rows"]
    if not edge_rows:
        return []
    by_scope: dict = defaultdict(lambda: [0, 0])
    for _src, _dst, _t, scope, channels, _ids in edge_rows:
        by_scope[scope][0] += 1
        by_scope[scope][1] += channels
    matrix_cells = tables["matrix_cells"]
    observed = sorted({r for pair in matrix_cells for r in pair})
    lines = ["", "## Rank-to-rank connectivity", "",
             f"{len(matrix_cells)} directed rank pairs among {len(observed)} ranks (of "
             f"{len(observed) * (len(observed) - 1)} possible), {len(edge_rows)} (pair, transport) "
             "connections in total, parsed from the `NCCL INFO Channel .. : src -> dst via ..` "
             "lines that RCCL emits while building each communicator. Only ranks whose log "
             "survived contribute rows"
             + (f", so coverage follows the same limit as everything else: "
                f"{ctx.spec.notes.rank_coverage}."
                if ctx.spec.notes.rank_coverage and view.shared_logs else "."), "",
             "| scope | (pair, transport) rows | channels |", "|---|---:|---:|"]
    for scope, (conns, channels) in sorted(by_scope.items()):
        lines.append(f"| {scope} | {conns} | {channels} |")
    if view.phase.topo_damaged:
        lines += ["", f"A further {view.phase.topo_damaged} topology line(s) named no transport "
                      "RCCL can print and were dropped as torn, so this table understates the "
                      "connectivity rather than inventing edges. Before that check, their spliced "
                      "transports counted as connections in their own right."]
    lines += ["", "This is measured **connectivity**, not measured traffic: RCCL logs which peers "
                  "a rank connects to and over which transport, but never how many bytes crossed "
                  "each edge. Per-edge volume can only be modelled from the ring or tree "
                  "structure. The matrix lives in `rank_matrix.csv` and, with a colour scale "
                  "applied, on the `rank_matrix` sheet of `profile.xlsx` — that sheet is the "
                  "heatmap."]
    return lines


def section_files(view: PhaseView, ctx: ReportContext, tables: dict) -> list:
    """What this report actually wrote, which is not the same as what it can write.

    Writing is conditional, so the list must be too: a file list is read as an index, and naming
    a file that is not there is worse than omitting it.
    """
    lines = ["", "## Files", ""]
    if view.phase.sizes:
        lines += ["- `collective_message_sizes.csv` — full histogram, one row per "
                  "(collective, nranks, size, datatype)",
                  "- `collective_totals.csv` — per-collective calls and volume"]
    if view.phase.per_node:
        lines.append("- `per_node.csv` — volume per node")
    if view.phase.per_rank:
        lines.append("- `per_rank.csv` — volume per logged rank, for the imbalance check; the "
                     "rank is RCCL's globalrank where the log prints it, otherwise the process id")
    if view.phase.damage:
        lines.append("- `discarded_records.csv` — records dropped, by reason")
    if view.phase.config:
        lines.append("- `run_config.csv` — every setting the nodes reported, flagged for whether "
                     "it moves throughput and whether the nodes agreed on it")
    if view.phase.steps and not invalidators(view.phase.config, ctx.spec.steps):
        lines.append("- `step_times.csv` — step-time distribution per node, with whether graphs "
                     "were replayed")
    if counter_table(view, ctx):
        lines.append("- `fabric_counters.csv` — adapter counters per node, device and port, with "
                     "the kind each one was grouped under and the window it was measured over")
    if ctx.torch_trace and ctx.torch_trace.get("a2a"):
        lines.append("- `expert_a2a.csv` — expert all-to-all stages found in the trace by name")
    if tables["timing_rows"]:
        lines.append("- `collective_enqueue_times.csv` — host-side API durations per collective")
    if ctx.torch_trace and ctx.torch_trace["events"]:
        lines.append("- `torch_collective_sizes.csv` — calls and volume per "
                     "(collective, process group, size)")
    if tables["edge_rows"]:
        lines += ["- `comm_edges.csv` — one row per (src, dst, transport) with the channels used",
                  "- `rank_matrix.csv` — rank x rank channel counts"]
    lines += ["- `profile.xlsx` — one workbook holding this report as cells plus every CSV above "
              "as a sortable Excel table, with the rank matrix rendered as a heatmap", ""]
    return lines


# -- entry point ---------------------------------------------------------------------------------


def emit_phase(phase: Phase, out_dir: Path, ctx: ReportContext) -> Path:
    """Write one phase's CSVs, markdown report and workbook. Returns the report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    view = PhaseView(phase, ctx.spec)
    sink = CsvSink(out_dir)
    tables = write_tables(view, sink, ctx)

    lines = section_header(view, ctx)
    lines += section_summary(view, ctx)
    lines += section_run_config(view, ctx)
    lines += section_steps(view, ctx)
    # The collective sections only when there are collectives: an empty "Traffic by collective"
    # reads as a run that issued none, which for a tuned run is the opposite of the truth.
    if phase.sizes:
        lines += section_traffic(view)
        lines += section_imbalance(view, ctx)
        lines += section_size_distribution(view, ctx)
        lines += section_single_rank(view)
    else:
        lines += ["", "## Traffic by collective (per rank)", "",
                  "**Not available.** This phase logged no collective records, so the volume, "
                  "imbalance, size-distribution and connectivity sections are absent rather than "
                  "empty: nothing here says the run issued no collectives. An unprofiled run "
                  "leaves `NCCL_DEBUG_SUBSYS` at its default without `COLL`, which is the usual "
                  "reason; the channels below were collected and are reported as measured."]
    lines += section_rocprof(view, ctx, tables["timing_rows"])
    lines += section_traces(view, ctx, sink)
    lines += section_a2a(view, ctx)
    lines += section_counters(view, ctx)
    if phase.sizes:
        lines += section_connectivity(view, ctx, tables)
    lines += section_files(view, ctx, tables)

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    sink.drop_stale()
    write_workbook(out_dir, lines)
    return report
