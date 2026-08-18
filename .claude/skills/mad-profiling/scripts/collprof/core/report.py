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
from .spec import EngineSpec
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
    #: Capture directories that held no traces, per phase. Stated in the report rather than only on
    #: the console: a phase whose second replica captured nothing covers fewer nodes than it looks.
    empty_trace_dirs: tuple[str, ...] = ()
    #: How this report was produced, recorded in the header so a number can be traced back.
    command: str = ""
    parse_version: int = 0


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

    def drop_stale(self) -> None:
        for path in self.out_dir.glob("*.csv"):
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


# -- CSV tables ----------------------------------------------------------------------------------


def write_tables(view: PhaseView, sink: CsvSink, ctx: ReportContext) -> dict:
    """Write every CSV of a report and return what the markdown still needs from them."""
    phase, reps = view.phase, view.reps

    hist_rows = []
    for (coll, nr, msg_bytes, dt), calls in sorted(phase.sizes.items(),
                                                   key=lambda kv: -kv[1] * kv[0][2]):
        vol = calls * msg_bytes / reps
        share = round(100.0 * vol / view.per_rank_bytes, 3) if view.per_rank_bytes else 0.0
        hist_rows.append([coll, nr, msg_bytes, round(msg_bytes / MIB, 4), dt, round(calls / reps),
                          round(vol), share])
    sink.write("collective_message_sizes.csv",
               ["collective", "nranks", "msg_bytes", "msg_mib", "datatype", "calls_per_rank",
                "bytes_per_rank", "pct_of_bytes"], hist_rows)

    total_rows = []
    for coll, row in sorted(view.totals.items(), key=lambda kv: -kv[1]["bytes"]):
        total_rows.append([coll, round(row["calls"] / reps), round(row["bytes"] / reps),
                           len(row["sizes"]), min(row["sizes"]), max(row["sizes"]),
                           round(100.0 * row["bytes"] / view.grand_bytes, 3)
                           if view.grand_bytes else 0.0])
    sink.write("collective_totals.csv",
               ["collective", "calls_per_rank", "bytes_per_rank", "distinct_sizes", "min_bytes",
                "max_bytes", "pct_of_bytes"], total_rows)

    sink.write("per_node.csv", ["node", "collective", "calls", "total_bytes"],
               [[node, coll, calls, nbytes]
                for (node, coll), (calls, nbytes) in sorted(phase.per_node.items())])
    sink.write("per_rank.csv", ["node", "rank", "calls", "total_bytes"],
               [[node, rank, calls, nbytes]
                for (node, rank), (calls, nbytes) in sorted(phase.per_rank.items())])

    if phase.damage:
        sink.write("discarded_records.csv", ["reason", "records"],
                   [[reason, count] for reason, count in phase.damage.most_common()])

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
        f"Nodes contributing log data: {len(phase.nodes)} ({', '.join(sorted(phase.nodes))})  ",
        f"Ranks present in the log: {len(phase.per_rank)} "
        f"({', '.join(f'{n}: {c}' for n, c in sorted(ranks_per_node.items()))})"
        + (f", of which {view.reps} carried traffic" if view.idle else "")
        + (f" — {notes.rank_coverage}  " if notes.rank_coverage else "  "),
        f"Communicator size seen: nranks={phase.nranks}"
        + (f", {notes.communicator.format(nranks=phase.nranks)}" if notes.communicator else ""),
    ]
    if ctx.command:
        lines += ["", f"> Produced by `{ctx.command}` with parser version {ctx.parse_version}. "
                      "Rerunning that command on the same inputs reproduces this file."]
    return lines


def section_summary(view: PhaseView, ctx: ReportContext) -> list:
    phase, spec = view.phase, ctx.spec
    lines = ["", "## Summary", "",
             f"- Collectives parsed (nranks>1): **{view.grand_calls}** over {view.reps} ranks "
             f"= {view.grand_calls // view.reps} per rank"
             + (f", leaving out {view.idle} rank(s) that carried under "
                f"{spec.limits.idle_rank_fraction:.0%} of the busiest rank" if view.idle else ""),
             f"- Traffic per rank: **{fmt_bytes(view.per_rank_bytes)}**"]

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
        lines += ["", "> No rocprofv3 RCCL stats supplied, so this report is volume-only: "
                      "message sizes and call counts, no durations or bandwidth."]
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
    if ctx.spec.notes.damage_cause:
        lines.append(f"  {ctx.spec.notes.damage_cause}")

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
                f"{ctx.spec.notes.rank_coverage}." if ctx.spec.notes.rank_coverage else "."), "",
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
    lines = ["", "## Files", "",
             "- `collective_message_sizes.csv` — full histogram, one row per "
             "(collective, nranks, size, datatype)",
             "- `collective_totals.csv` — per-collective calls and volume",
             "- `per_node.csv` — volume per node",
             "- `per_rank.csv` — volume per logged rank, for the imbalance check; the rank is "
             "RCCL's globalrank where the log prints it, otherwise the process id"]
    if view.phase.damage:
        lines.append("- `discarded_records.csv` — records dropped, by reason")
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
    lines += section_traffic(view)
    lines += section_imbalance(view, ctx)
    lines += section_size_distribution(view, ctx)
    lines += section_single_rank(view)
    lines += section_rocprof(view, ctx, tables["timing_rows"])
    lines += section_traces(view, ctx, sink)
    lines += section_connectivity(view, ctx, tables)
    lines += section_files(view, ctx, tables)

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    sink.drop_stale()
    write_workbook(out_dir, lines)
    return report
