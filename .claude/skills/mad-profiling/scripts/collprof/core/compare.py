"""Two runs, side by side, decomposed.

The mechanical parts of a comparison, which by hand is how a graph-free arm ends up quoted
against a graphed one: **what differed**, from what each run reported about itself; **where the
time went**, split through ``E2E = TTFT + (OSL-1) * ITL``; and **what the exchange cost**, per
stage and per kernel variant.

Nothing here decides which arm is right or reports a bandwidth. Arms differing in a setting that
moves throughput are reported as not comparable rather than explained.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .rdma_counters import (INCOMPARABLE_OP_RATIO, by_kind, bytes_per_op,
                            kind_order, kinds_of)
from .runconfig import diff_configs, merge_nodes
from .spec import BenchmarkLayout, EngineSpec
from .units import fmt_bytes
from .steps import batch_invariance_by_node
from .steps import by_node as steps_by_node
from .steps import invalidators
from .steps import summarise as summarise_steps

#: Kernels shown in the markdown table per arm. A display cut only: every classified kernel reaches
#: `expert_a2a_kernels.csv` and decides the variant verdict.
SHOWN_KERNELS = 8


@dataclass
class Arm:
    """One side of a comparison: what it was called, and what was collected for it."""

    name: str
    run_dir: Path
    config: dict          # setting -> value, as the nodes agreed it
    steps: dict           # node -> StepStats, for the node count and for per-node reporting
    trace: dict | None    # parse_traces output, or None
    points: dict          # (isl, osl, con) -> {metric: value}
    #: One StepStats over every node's intervals pooled. A quantile of the pooled sample, not a
    #: mean of per-node quantiles: a mean of two medians is a number neither node ever saw.
    pooled: object | None = None
    #: Settings the nodes of this arm did not agree on, ``key -> {node: value}``.
    disagreed: dict = field(default_factory=dict)
    #: Worst per-node spread between the batch groups' median step times. Per node rather than
    #: pooled: pooling confounds node identity with batch.
    batch_spread_ms: float | None = None
    #: ``{node: CounterSeries}`` from this arm's RDMA adapter counters, when it sampled them. The
    #: only channel that says what an exchange put on the wire.
    counters: dict = field(default_factory=dict)
    #: Set by the caller to assert that this arm served the same requests as the other. Nothing
    #: here can verify a request count, so the comparison asks rather than assumes.
    counters_comparable: bool = False
    #: Capture directories of this phase that held no trace file, by name. With one directory per
    #: replica, an arm missing a capture reads as fully covered over fewer replicas.
    empty_trace_dirs: tuple = ()
    #: ``(setting, value, why)`` for each engine-declared setting that makes this arm's step times
    #: meaningless. Non-empty withholds the channel for both arms.
    steps_invalid: tuple = ()


def load_benchmark(run_dir: Path, layout: BenchmarkLayout) -> dict:
    """Per-configuration metrics from a run's benchmark CSV, keyed ``(isl, osl, con)``.

    Long format, one row per metric, both columns named by the engine. A measurement can appear in
    two matching CSVs with different values: the first is kept and the collision named on stderr.
    """
    if not layout.globs or layout.point is None:
        return {}
    points: dict = {}
    clashes: dict = {}
    for pattern in layout.globs:
        for path in sorted(run_dir.glob(pattern)):
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    key = layout.point.search(row.get(layout.model_column, ""))
                    metric = row.get(layout.metric_column)
                    # A CSV without `metric` is not this schema; skipped like any other unusable
                    # row rather than raising a bare KeyError.
                    if not key or not metric or not row.get(layout.value_column):
                        continue
                    try:
                        value = float(row[layout.value_column])
                    except ValueError:
                        continue
                    point = tuple(int(key.group(g)) for g in ("isl", "osl", "con"))
                    seen = points.setdefault(point, {})
                    if metric in seen:
                        if seen[metric] != value:
                            clashes.setdefault((point, metric), (seen[metric], value, path.name))
                        continue
                    seen[metric] = value
    for (point, metric), (kept, dropped, path_name) in sorted(clashes.items()):
        print(f"warning: {run_dir}: {metric} at isl{point[0]}_osl{point[1]}_con{point[2]} is "
              f"reported twice with different values, {kept} and {dropped} (the second in "
              f"{path_name}); the first is used.")
    return points


#: What a "no setting differs" verdict has and has not looked at. Written once because the same
#: sentence appears in the comparison and in the single-run report.
EXCLUDED_FROM_DIFF = ("ports, hosts and seeds excluded, and model and tokenizer paths compared "
                      "by their final component rather than in full")


def pct(new: float, ref: float) -> str:
    return f"{100.0 * (new - ref) / ref:+.1f}%" if ref else "n/a"


# Every table below is built once, as numbers, then rendered into the markdown and into a CSV:
# building it twice is how a document and its spreadsheet come to disagree.

def build_steps(left: Arm, right: Arm) -> tuple:
    header = ["arm", "nodes", "intervals", "median_ms", "p95_ms", "p95_over_median", "graphs"]
    rows = []
    if left.steps_invalid or right.steps_invalid:
        # Withheld rather than caveated, for the reason on `Arm.steps_invalid`. The rows do not
        # exist, so no CSV or workbook sheet carries them either.
        return header, rows
    if left.pooled is None and right.pooled is None:
        # An engine that declares no StepTimingLayout has no step channel. Two n/a rows would
        # credit figures to an accounting it never printed. One arm missing it is still shown.
        return header, rows
    for arm in (left, right):
        pooled = arm.pooled
        if pooled is None:
            rows.append([arm.name, len(arm.steps), 0, None, None, None, "not stated"])
            continue
        # Per node only for what is additive or categorical; the quantiles come from the pool.
        # `None` is "this node did not say", so it makes no mixture and no definite label either.
        states = {s.graphed for s in arm.steps.values()}
        known = states - {None}
        # A node whose own intervals disagreed reports `graphed is None`, so `states` alone cannot
        # tell that mixture from silence; `graphs_mixed` is what does.
        within = any(s.graphs_mixed for s in arm.steps.values())
        label = ("mixed" if within or len(known) > 1 else
                 "not stated" if None in states or not known else
                 "replayed" if known == {True} else "off")
        rows.append([arm.name, len(arm.steps), pooled.intervals,
                     round(pooled.median_ms, 1), round(pooled.p95_ms, 1),
                     round(pooled.p95_ms / pooled.median_ms, 2) if pooled.median_ms else None,
                     label])
    return header, rows


def build_decomposition(left: Arm, right: Arm, spec: EngineSpec) -> tuple:
    header = ["isl", "osl", "concurrency", "e2e_gap_ms", "from_ttft_ms", "from_itl_ms",
              "residual_ms", "decode_share_pct", "itl_delta_ms"]
    rows = []
    # The identity is arithmetic over three harness-named metrics. An engine that names none gets
    # no rows rather than a split over columns guessed for it.
    need = spec.benchmark.identity_metrics
    if not need:
        return header, rows
    e2e_key, ttft_key, itl_key = need
    for isl, osl, con in sorted(set(left.points) & set(right.points)):
        lv, rv = left.points[(isl, osl, con)], right.points[(isl, osl, con)]
        if any(k not in lv or k not in rv for k in need):
            continue
        e2e = rv[e2e_key] - lv[e2e_key]
        ttft = rv[ttft_key] - lv[ttft_key]
        itl = rv[itl_key] - lv[itl_key]
        decode = itl * (osl - 1)
        rows.append([isl, osl, con, round(e2e, 1), round(ttft, 1), round(decode, 1),
                     round(e2e - ttft - decode, 1),
                     round(100.0 * decode / e2e, 1) if e2e else None, round(itl, 2)])
    return header, rows


def build_points(left: Arm, right: Arm, spec: EngineSpec) -> tuple:
    header = ["isl", "osl", "concurrency", "metric", left.name, right.name, "delta_pct"]
    rows = []
    for point in sorted(set(left.points) & set(right.points)):
        for metric, label, _fmt in spec.benchmark.metrics:
            lv, rv = left.points[point].get(metric), right.points[point].get(metric)
            if lv is None or rv is None:
                continue
            delta = round(100.0 * (rv - lv) / lv, 1) if lv else None
            rows.append([point[0], point[1], point[2], label, lv, rv, delta])
    return header, rows


def build_config_diff(left: Arm, right: Arm, spec: EngineSpec) -> tuple:
    """The settings that differ, as rows, so the CSV and the markdown cannot disagree about them.

    This is the comparability evidence, so it belongs beside the numbers it qualifies.
    """
    header = ["setting", left.name, right.name, "moves_throughput"]
    # `parse_config_line` returns "" for `key=''`, a value a run really reported, so "" cannot
    # also mean "not stated" without the two collapsing.
    absent = "<not stated>"
    cfg = spec.run_config
    rows = [[s.key,
             s.left if s.left is not None else absent,
             s.right if s.right is not None else absent,
             s.perf_relevant]
            for s in diff_configs(left.config, right.config,
                                  perf_relevant=cfg.perf_relevant, noise=cfg.noise,
                                  path_valued=cfg.path_valued)]
    # One row per key, whichever arm is split: a marker in both columns would claim the other arm
    # was split too, and a row per arm could list the same key twice.
    seen = {row[0] for row in rows}
    for key in sorted({k for arm in (left, right) for k in set(arm.disagreed) - cfg.noise}):
        cells = []
        for arm in (left, right):
            if key in arm.disagreed:
                cells.append("; ".join(f"{n}={v}"
                                       for n, v in sorted(arm.disagreed[key].items())))
            else:
                cells.append(arm.config.get(key, absent))
        if key in seen:  # already in the diff; the split is the more specific statement
            rows = [r for r in rows if r[0] != key]
        rows.append([key, cells[0], cells[1], key in cfg.perf_relevant])
    return header, rows


def matched_cats(arm: Arm) -> set:
    """The trace event categories in which this arm classified any all-to-all stage."""
    return {cat for _stage, cat in ((arm.trace or {}).get("a2a") or {})}


def build_stages(left: Arm, right: Arm) -> tuple:
    header = ["stage", f"{left.name}_calls_per_trace", f"{left.name}_share_pct",
              f"{right.name}_calls_per_trace", f"{right.name}_share_pct"]

    # Each arm's denominator is its own matched categories. A union across both put the other
    # arm's categories into this arm's divisor, sinking its share by an order of magnitude.

    def per_arm(arm: Arm) -> dict:
        """Per stage, summed over the categories it appears in rather than keyed by one.

        A stage can appear in more than one category, so keying by one let the later overwrite the
        earlier.
        """
        if not arm.trace:
            return {}
        files = max(arm.trace.get("files", 1), 1)
        totals = arm.trace.get("category_us") or {}
        base = sum(totals.get(cat, 0.0) for cat in matched_cats(arm))
        calls_by_stage: dict = defaultdict(float)
        dur_by_stage: dict = defaultdict(float)
        for (stage, cat), (calls, dur) in (arm.trace.get("a2a") or {}).items():
            calls_by_stage[stage] += calls
            dur_by_stage[stage] += dur
        out: dict = {}
        for stage, calls in calls_by_stage.items():
            out[stage] = (calls / files,
                          100.0 * dur_by_stage[stage] / base if base else 0.0)
        return out

    lrows, rrows = per_arm(left), per_arm(right)
    rows = []
    # An arm that classified nothing has an unknown mix, not a measured zero. One that classified
    # something and lacks a stage ran none of it, so that is a zero.
    absent = {}
    for arm, got in ((left, lrows), (right, rrows)):
        absent[id(arm)] = (None, None) if not got else (0.0, 0.0)
    for stage in sorted(set(lrows) | set(rrows)):
        lc, ls = lrows.get(stage, absent[id(left)])
        rc, rs = rrows.get(stage, absent[id(right)])
        rows.append([stage,
                     round(lc, 1) if lc is not None else None,
                     round(ls, 1) if ls is not None else None,
                     round(rc, 1) if rc is not None else None,
                     round(rs, 1) if rs is not None else None])
    return header, rows


def variant_of(name: str, spec: EngineSpec) -> str:
    """Which variant of the exchange a kernel name says ran, first pattern winning.

    The patterns overlap on purpose, so two call sites resolving it two ways is how a summary
    comes to name variants its own table does not contain.
    """
    return next((label for label, pattern in spec.a2a.variants if pattern.search(name)),
                "not stated")


def build_kernels(arm: Arm, spec: EngineSpec, limit: int | None = None) -> tuple:
    """Which kernel ran each stage, and what share of this arm's own exchange it took.

    ``limit`` truncates for display only: a table cut at the busiest few cannot answer "did this
    arm run a second variant".
    """
    header = ["arm", "stage", "variant", "kernel", "calls_per_trace", "share_of_exchange_pct"]
    named = (arm.trace or {}).get("a2a_names") or {}
    if not named:
        return header, []
    files = max((arm.trace or {}).get("files", 1), 1)
    # Share within this arm's own exchange, not milliseconds: absolute trace durations fail a
    # cross-check against rocprofv3, but a share is normalised inside one capture.
    spent = sum(dur for _calls, dur in named.values()) or 1.0
    rows = []
    ordered = sorted(named.items(), key=lambda kv: -kv[1][1])
    for (stage, name), (calls, dur) in (ordered[:limit] if limit else ordered):
        # One decimal, as in `section_a2a`'s `cell`: one call across two traces is 0.5 a trace,
        # and an integer round reports 0 calls beside a 90% share of the exchange.
        rows.append([arm.name, stage, variant_of(name, spec), name, round(calls / files, 1),
                     round(100.0 * dur / spent, 1)])
    return header, rows


def tables(left: Arm, right: Arm, spec: EngineSpec) -> dict:
    """Every table of the comparison as ``name -> (header, rows)``, for CSV and the workbook."""
    out = {"configuration_diff": build_config_diff(left, right, spec),
           "step_times": build_steps(left, right),
           "decomposition": build_decomposition(left, right, spec),
           "expert_a2a_stages": build_stages(left, right),
           "benchmark_points": build_points(left, right, spec),
           "fabric_counters": build_counters(left, right, spec)}
    kernel_header, kernel_rows = build_kernels(left, spec)
    kernel_rows += build_kernels(right, spec)[1]
    out["expert_a2a_kernels"] = (kernel_header, kernel_rows)
    return {name: table for name, table in out.items() if table[1]}


def section_comparability(left: Arm, right: Arm, spec: EngineSpec) -> list:
    """What differed between the arms, and whether that leaves them comparable."""
    cfg = spec.run_config
    # The same rows the CSV gets; recomputing the diff here is how the two came to disagree.
    rows = build_config_diff(left, right, spec)[1]
    lines = ["## What differed", ""]

    # Before any verdict about the pair: an arm whose own nodes disagree has no single
    # configuration to compare. The NOISE set keeps out keys that differ per node by construction.
    split_note = []
    for arm in (left, right):
        split = sorted(set(arm.disagreed) - cfg.noise)
        if not split:
            continue
        keys = ", ".join(f"`{k}`" for k in split)
        moves = [k for k in split if k in cfg.perf_relevant]
        split_note += [f"**{arm.name}'s own nodes did not agree on {len(split)} setting(s): "
                       f"{keys}.** They appear in the table with their per-node values rather "
                       "than a single one, because an arm with no agreed value for a setting has "
                       "none to compare against the other arm's. A launcher that "
                       "assembles rank 0 separately is how a profiled run carried the measurement "
                       "flags on three nodes out of four, so this is a finding about the arm "
                       "before it is a caveat about the pair."
                       + (f" {len(moves)} of them move throughput on their own, so the pair is "
                          "not comparable on those regardless of what the diff shows."
                          if moves else ""), ""]

    # `Arm.config` holds only what an arm's nodes *agreed*, so an arm that said plenty can have an
    # empty one. Silence is both dicts being empty.
    missing = [arm.name for arm in (left, right) if not arm.config and not arm.disagreed]
    if len(missing) == 2:
        lines.append("Neither arm reported a configuration, so there is no table to build and "
                     "nothing was checked. Every number in this document rests on the assumption "
                     "that the arms differed only in what was intended.")
    elif missing:
        lines.append(f"{missing[0]} reported no configuration, so the rows below are one-sided: "
                     "they are the settings the other arm stated, and a *not stated* cell means "
                     "this arm was silent rather than that the setting was absent. Nothing was "
                     "checked, and every number in this document rests on the assumption that the "
                     "arms differed only in what was intended.")
    if not rows:
        # Only when there is genuinely nothing to show. A missing configuration withholds the
        # *verdict*, not the rows, which the CSV carries either way.
        if not missing:
            lines.append(f"No setting differs between the values the arms agreed on -- "
                         f"{EXCLUDED_FROM_DIFF}." if split_note else
                         f"No setting differs; {EXCLUDED_FROM_DIFF}.")
        return lines + [""] + split_note

    lines += ["", f"| setting | {left.name} | {right.name} | |", "|---|---|---|---|"]
    for key, lv, rv, moves in rows:
        lcell = "*not stated*" if lv == "<not stated>" else f"`{lv}`"
        rcell = "*not stated*" if rv == "<not stated>" else f"`{rv}`"
        mark = " **moves throughput** " if moves else " "
        lines.append(f"| `{key}` | {lcell} | {rcell} |{mark}|")

    if split_note:
        lines += [""] + split_note

    critical = [] if missing else [key for key, _l, _r, moves in rows if moves]
    if missing:
        # The rows are shown; the verdict is not. With one side silent, "these move throughput"
        # would read as a finding rather than a gap in the evidence.
        return lines
    if critical:
        keys = ", ".join(f"`{k}`" for k in critical)
        lines += ["", f"**{len(critical)} of these move throughput on their own: {keys}.** If more "
                      "than the one under test is in that list, the numbers below measure the "
                      "difference in those as well, and the comparison does not hold."]
    else:
        lines += ["", "None of these is known to move throughput on its own."]
    return lines


def section_steps(left: Arm, right: Arm) -> list:
    """Step time side by side, one row per arm over that arm's nodes."""
    step_rows = build_steps(left, right)[1]
    broken = {arm.name: arm.steps_invalid for arm in (left, right) if arm.steps_invalid}
    if broken:
        named = "; ".join(f"{name} reported `{setting}={value}`, so {why}"
                          for name, found in sorted(broken.items())
                          for setting, value, why in found)
        return ["", "## Step time", "",
                f"**Withheld.** {named}. The quotient remains a plausible number and the "
                "batch-invariance check that guards this channel cannot see the breakage, so it "
                "is not reported at all rather than reported with a caveat -- for both arms, "
                "since a gap between a duration and something else is not a gap. The benchmark "
                "ITL in the decomposition below measures the interval rather than deriving it."]
    if not step_rows:
        # The builder returns nothing when neither arm has a step channel; rendering the heading
        # anyway advertises "each engine's own accounting" above an empty table.
        return []
    lines = ["", "## Step time", "",
             "From each engine's own accounting, so it is present in a profiled and an unprofiled "
             "run alike. A gap constant across intervals is a fixed per-step cost; one that grows "
             "with the batch is volume-limited, and a mean cannot tell them apart.", "",
             "| arm | nodes | intervals | median ms | p95 ms | p95/median | graphs |",
             "|---|---:|---:|---:|---:|---:|---|"]
    medians = {}
    for name, nodes, intervals, median, p95, ratio, graphs in step_rows:
        if median is None:
            lines.append(f"| {name} | {nodes} | {intervals} | n/a | n/a | n/a | n/a |")
            continue
        medians[name] = median
        shown = f"**{graphs}**" if graphs == "off" else graphs
        lines.append(f"| {name} | {nodes} | {intervals} | {median:.1f} | {p95:.1f} | "
                     f"{ratio:.2f} | {shown} |")
    # One-sided, and worded that way: a flat spread rules out the rate and the batch describing
    # different windows, but a large one does not establish it.
    for arm in (left, right):
        # Per node, then the worst: the pooled figure mixes node identity into the batch grouping,
        # so two replicas with disjoint batch ranges look like a broken estimate.
        spread = arm.batch_spread_ms
        if spread is None:
            continue
        median = arm.pooled.median_ms or 1.0
        verdict = ("which is flat against a median of "
                   f"{median:.0f} ms. That rules out the rate and the batch covering different "
                   "windows, since that error would scale with the batch, so the estimate holds"
                   if spread < 0.05 * median else
                   f"which is **batch-sensitive** against a median of {median:.0f} ms. This "
                   "channel cannot resolve that on its own: a "
                   "volume-limited workload's step time grows with its batch, and a rate averaged "
                   "over a window in which the batch changed produces the same shape. Compare the "
                   "medians at a matched batch, or read the benchmark's own ITL, before treating "
                   "a difference here as a per-step cost")
        lines += ["", f"{arm.name}: the median step time varies by {spread:.2f} ms across its "
                      f"batch groups, {verdict}."]

    if len(medians) == 2:
        (an, a), (bn, b) = medians.items()
        lines += ["", f"Per-step difference: **{b - a:+.1f} ms** ({bn} against {an}), "
                      f"{pct(b, a)}."]
    return lines


def counters_usable(arm: Arm) -> bool:
    """Whether this arm sampled a window worth differencing over.

    The counters are cumulative since the adapter came up, so a lone reading says nothing about
    this run.
    """
    return any(series.samples > 1 for series in arm.counters.values())


def counter_kinds_seen(arm: Arm, spec: EngineSpec) -> set:
    """Kinds this arm's adapters actually reported, over the series that carry a window."""
    return {kind for series in arm.counters.values() if series.samples > 1
            for kind in by_kind(series, spec.counters)}


def counter_kinds_moved(arm: Arm, spec: EngineSpec) -> set:
    """Kinds whose counters actually changed, as opposed to kinds that were merely reported."""
    return {kind for series in arm.counters.values() if series.samples > 1
            for kind, group in by_kind(series, spec.counters).items() if sum(group.values())}


def same_workload(left: Arm, right: Arm) -> bool:
    """Whether the two arms measured the same set of benchmark points.

    Necessary and **not sufficient**: the perf CSV keeps no request count, so two runs can agree
    on every point key while one served twice as many. Hence `--counters-same-workload` too.
    """
    return bool(left.points) and bool(right.points) and set(left.points) == set(right.points)


def counters_damaged(arm: Arm) -> int:
    """Rows this arm's samples lost to the filesystem, over the series that carry a window."""
    return sum(s.damaged for s in arm.counters.values() if s.samples > 1)


def counter_coverage(arm: Arm) -> set:
    """The ``(node, device, port)`` an arm's totals are summed over -- the hardware, not its size.

    Identities rather than a count: two arms can each sample eight NICs and sample eight
    *different* ones, and the totals then differ by the adapters, not the backend.
    """
    covered: set = set()
    for node, series in arm.counters.items():
        if series.samples < 2:
            continue
        # What was sampled, not what moved: an idle adapter still counts as covered, or it would
        # be indistinguishable from an unsampled one.
        covered |= {(node, dev, port) for dev, port in series.adapters}
        covered.add((node, "", ""))
    return covered


def coverage_size(covered: set) -> tuple:
    """``(nodes, adapters)`` of a coverage set, for saying how much hardware is behind a total."""
    return (len({node for node, _dev, _port in covered}),
            len({c for c in covered if c[1] or c[2]}))


def build_counters(left: Arm, right: Arm, spec: EngineSpec) -> tuple:
    """Adapter-counter totals per kind, per arm, as rows.

    Built once: the section below reads these rows and `tables()` writes `fabric_counters.csv`.
    """
    header = ["kind", left.name, right.name, "delta_pct"]
    rows: list = []
    # Both arms or neither: an arm whose sampling failed contributes zeros, which read as a
    # backend difference. Likewise unequal coverage.
    if not (counters_usable(left) and counters_usable(right)):
        return header, rows
    if counter_coverage(left) != counter_coverage(right):
        return header, rows
    # A dropped sample can be the endpoint of a counter's window, leaving one arm's totals short
    # by an unboundable amount -- the shape of a backend difference.
    if counters_damaged(left) or counters_damaged(right):
        return header, rows
    # Whole-window totals compare the work as much as the backend, and the benchmark points are
    # the only available proxy for "the same work".
    if not same_workload(left, right):
        return header, rows
    if not (left.counters_comparable and right.counters_comparable):
        return header, rows
    grouped = {arm.name: [by_kind(s, spec.counters) for s in arm.counters.values()
                          if s.samples > 1]
               for arm in (left, right)}
    # Only the kinds both arms reported: a counter a driver omits is absent, not zero, and a zero
    # would read as a -100% backend delta. One-sided kinds are named below the table.
    seen = {name: {k for g in groups for k in g} for name, groups in grouped.items()}
    shared = seen[left.name] & seen[right.name]
    order = [k for k in kind_order(spec.counters, shared) if k in shared]
    totals = {}
    for arm in (left, right):
        per_kind = {k: 0 for k in order}
        for g in grouped[arm.name]:
            for kind, group in g.items():
                if kind in per_kind:
                    per_kind[kind] += sum(group.values())
        totals[arm.name] = per_kind
    # After the wrap check: if every counter that moved also wrapped, the totals are empty because
    # the deltas were discarded, not because the run put nothing on the wire.
    if not any(sum(v.values()) for v in totals.values()) and not wrapped_kinds(left, right, spec):
        return header, rows
    # A kind whose counter wrapped on either arm is left out: its delta was dropped at parse time,
    # so it would appear as a zero and read as a backend difference that never happened.
    unusable = wrapped_kinds(left, right, spec) | set(incomparable_ops(left, right, spec))
    for kind in order:
        if kind in unusable:
            continue
        lv, rv = totals[left.name][kind], totals[right.name][kind]
        if not lv and not rv:
            continue
        rows.append([kind, lv, rv, pct(rv, lv) if lv else None])
    return header, rows


def incomparable_ops(left: Arm, right: Arm, spec: EngineSpec) -> dict:
    """``kind -> (left bytes/op, right bytes/op)`` where the two arms cannot be counting alike.

    Both counts can look sane while each arm's volume over its own count comes out orders of
    magnitude apart: one transport posts through a path these verb counters do not increment. The
    link-level volume counters are the check, and carry such a pair instead.
    """
    lhs = bytes_per_op(left.counters.values(), spec.counters)
    rhs = bytes_per_op(right.counters.values(), spec.counters)
    out = {}
    for kind in set(lhs) & set(rhs):
        big, small = max(lhs[kind], rhs[kind]), min(lhs[kind], rhs[kind])
        if small and big / small >= INCOMPARABLE_OP_RATIO:
            out[kind] = (lhs[kind], rhs[kind])
    return out


def wrapped_kinds(left: Arm, right: Arm, spec: EngineSpec) -> set:
    """Kinds that wrapped on either arm, and so cannot be compared on this pair."""
    return (kinds_of(left.counters.values(), spec.counters)
            | kinds_of(right.counters.values(), spec.counters))


def section_counters(left: Arm, right: Arm, spec: EngineSpec) -> list:
    """What each arm put on the wire, from the adapters' own counters.

    The KV transfer shares these adapters, so a column bounds the exchange rather than measuring
    it -- but two arms serving the same requests differ by the backend and little else, so the
    *difference* is close to the exchange even though neither column is.
    """
    if not (left.counters or right.counters):
        return []

    header, rows = build_counters(left, right, spec)
    if not rows:
        missing = [arm.name for arm in (left, right) if not counters_usable(arm)]
        if not missing and counters_usable(left) and counters_usable(right) \
                and same_workload(left, right) \
                and not (left.counters_comparable and right.counters_comparable):
            return ["", "## What crossed the fabric", "",
                    "**Withheld.** The arms measured the same benchmark points, which is "
                    "necessary and not sufficient: the perf CSV keeps one row per point and "
                    "metric, not a request count, so a larger `BENCHMARK_ITR`, a retry or extra "
                    "profile-point traffic makes one arm do more work while every point key still "
                    "matches. Nothing this tooling reads can tell those apart. Pass "
                    "`--counters-same-workload` to assert that the two runs served the same "
                    "requests, and the totals will be compared."]
        if not missing and counters_usable(left) and counters_usable(right) \
                and not same_workload(left, right):
            shared_points = sorted(set(left.points) & set(right.points))
            return ["", "## What crossed the fabric", "",
                    "**Withheld.** These are whole-window totals, so they compare the work as "
                    "much as the backend -- and the two arms are not known to have served the "
                    f"same requests ({len(left.points)} benchmark point(s) against "
                    f"{len(right.points)}, {len(shared_points)} in common). Nothing in a "
                    "configuration dump states a request count; the sweep does, and until it "
                    "matches on both sides a difference here has an obvious second explanation."]
        if not missing and (counters_damaged(left) or counters_damaged(right)):
            named = "; ".join(f"{arm.name}: {counters_damaged(arm)} row(s)"
                              for arm in (left, right) if counters_damaged(arm))
            return ["", "## What crossed the fabric", "",
                    f"**Withheld.** Sampled rows were lost to the filesystem ({named}). A lost "
                    "row can be the endpoint of a counter's window, so one arm's totals are short "
                    "by an amount nobody can bound -- and short totals on one side are the shape "
                    "of a backend difference. Reparse, or rerun the arm: the samples land on a "
                    "mount that returns zeros instead of an error, and a second read usually "
                    "succeeds."]
        if not missing and wrapped_kinds(left, right, spec):
            named = ", ".join(sorted(wrapped_kinds(left, right, spec)))
            return ["", "## What crossed the fabric", "",
                    f"**Withheld.** Every counter that moved also wrapped or was reset ({named}), "
                    "so what they carried is unrecoverable. An empty table here would read as a "
                    "run that put nothing on the wire, which is a different statement and not "
                    "one these samples support."]
        if not missing and counter_coverage(left) != counter_coverage(right):
            lhs, rhs = counter_coverage(left), counter_coverage(right)
            ln, la = coverage_size(lhs)
            rn, ra = coverage_size(rhs)

            def _named(own):
                return ", ".join(sorted(f"{n}/{d}/{p}" if d else n for n, d, p in own))
            only = "; ".join(f"{arm.name} only: {_named(own)}"
                             for arm, own in ((left, lhs - rhs), (right, rhs - lhs)) if own)
            return ["", "## What crossed the fabric", "",
                    f"**Withheld.** The arms did not sample the same hardware -- "
                    f"{left.name} {ln} node(s) and {la} adapter(s), {right.name} {rn} and {ra}"
                    + (f" ({only})" if only else "") + ". "
                    "Totals over different sets are not comparable, and the difference between "
                    "them would read as the backend's. Sample the same adapters on both arms, or "
                    "read each arm's own table in its single-run report."]
        if not missing:
            # "Nothing moved" has to be checked, not inferred from an empty table: rows are
            # dropped for reasons that are not stillness.
            moved = {arm.name: counter_kinds_moved(arm, spec) for arm in (left, right)}
            if any(moved.values()):
                why = []
                one_side = {arm.name: moved[arm.name] - moved[other.name]
                            for arm, other in ((left, right), (right, left))}
                for name, kinds in one_side.items():
                    if kinds:
                        why.append(f"moved only on {name} ({', '.join(sorted(kinds))})")
                apart = incomparable_ops(left, right, spec)
                if apart:
                    why.append("cannot be counted alike by the two arms "
                               f"({', '.join(sorted(apart))})")
                return ["", "## What crossed the fabric", "",
                        "**Withheld.** Counters moved on both arms, but no kind survives as "
                        f"comparable: every one that changed {' or '.join(why) or 'was dropped'}. "
                        "An empty table is not a statement that the fabric was idle, and each "
                        "arm's own totals are in its single-run report."]
            return ["", "## What crossed the fabric", "",
                    "Both arms sampled their adapters and no counter moved between the first "
                    "sample and the last, which is not a result about either backend: it is what "
                    "a run whose traffic never reached these adapters looks like."]
        return ["", "## What crossed the fabric", "",
                f"**Withheld.** {' and '.join(missing)} sampled no window to difference over -- no "
                "files, a header only, or a single reading, which is cumulative since the adapter "
                "came up and says nothing about this run. The other arm's counts are not shown "
                "beside a zero: a comparison against a channel that failed to collect reads as a "
                "difference the backend caused, and this one has no way to tell a reader "
                "otherwise."]

    windows = {arm.name: max((s.seconds for s in arm.counters.values() if s.samples > 1),
                             default=0.0)
               for arm in (left, right)}
    lines = ["", "## What crossed the fabric", "",
             "Adapter counters, summed over each arm's nodes. **Totals, not rates**: both arms "
             "serve the same requests, so the work is the same and only the backend differs -- "
             "dividing by each arm's wall clock would turn "
             f"({windows[left.name]:.0f} s against {windows[right.name]:.0f} s) into a difference "
             "of its own and undo the very cancellation this comparison rests on. Per node and "
             "per adapter, and the KV transfer shares those adapters, so a column bounds the "
             "exchange rather than measuring it.", "",
             f"| counter kind | {left.name} | {right.name} | {right.name} vs {left.name} |",
             "|---|---:|---:|---:|"]
    for kind, lv, rv, delta in rows:
        lines.append(f"| {kind} | {lv:,} | {rv:,} | {delta if delta is not None else 'n/a'} |")
    # Coverage before interpretation: both arms can pass `counters_usable` over different amounts
    # of hardware, and nothing in the numbers themselves says so.
    named = "; ".join(f"{arm.name}: {n} node(s), {a} adapter(s)"
                      for arm, (n, a) in ((arm, coverage_size(counter_coverage(arm)))
                                          for arm in (left, right)))
    lines += ["", f"Coverage -- {named}, and the same ones on both arms. The table exists only "
                  "because they match; when they do "
                  "not, it is withheld rather than shown with a footnote."]

    apart = incomparable_ops(left, right, spec)
    if apart:
        named = "; ".join(f"{kind} ({fmt_bytes(lv)} against {fmt_bytes(rv)} per operation)"
                          for kind, (lv, rv) in sorted(apart.items()))
        lines += ["", f"**Left out, because the two arms cannot be counting alike** ({named}). "
                      "Each arm's byte volume divided by its own operation count comes out orders "
                      "of magnitude apart, which no choice of message size explains -- one "
                      "transport posts its traffic through a path these verb counters do not "
                      "increment. The volume and packet rows are link-level and count everything, "
                      "which is how the discrepancy shows and why they carry the comparison here."]

    unusable = wrapped_kinds(left, right, spec)
    if unusable:
        lines += ["", f"**Left out: {', '.join(sorted(unusable))}.** A counter of each wrapped or "
                      "was reset on one of the arms, so what it lost is unrecoverable; shown as a "
                      "zero it would read as a difference the backend made. The remaining rows "
                      "are unaffected -- a wrap is per counter, not per adapter."]

    only_one = {arm.name: counter_kinds_seen(arm, spec) - counter_kinds_seen(other, spec)
                for arm, other in ((left, right), (right, left))}
    if any(only_one.values()):
        named = "; ".join(f"{name}: {', '.join(sorted(kinds))}"
                          for name, kinds in only_one.items() if kinds)
        lines += ["", f"**Reported by one arm only, and therefore not compared** ({named}). A "
                      "counter a driver omits or could not read is absent, not zero, and a zero "
                      "here would have shown as a complete difference."]

    lines += ["", "These are **verb counts, not causality**. A reply can itself be an RDMA write "
                  "or a SEND, and transport acknowledgements are neither a read nor an atomic, so "
                  "a write-only profile is consistent with a one-sided protocol without "
                  "establishing one. Reads or atomics in quantity are positive evidence of a "
                  "protocol that waits; their absence proves nothing on its own. What a "
                  "difference here does establish is that two arms doing the same work put "
                  "different amounts of it on the wire. Counters this engine has not classified "
                  "keep their own names, each on its own row, because pooling them would sum "
                  "packets, errors and gauges into a number with no unit."]
    return lines


def section_points(left: Arm, right: Arm, spec: EngineSpec) -> list:
    """Every benchmark point, every metric, side by side."""
    shared = sorted(set(left.points) & set(right.points))
    if not shared:
        return ["", "## Benchmark points", "",
                "No configuration was measured by both arms, so there is nothing to put side by "
                "side. Each arm's own perf CSV still stands on its own."]
    lines = ["", "## Benchmark points", "",
             f"{len(shared)} configuration(s) measured by both arms.", "",
             f"| ISL/OSL | con | metric | {left.name} | {right.name} | {right.name} vs "
             f"{left.name} |", "|---|---:|---|---:|---:|---:|"]
    formats = {label: fmt for _metric, label, fmt in spec.benchmark.metrics}
    for isl, osl, con, label, lv, rv, _delta in build_points(left, right, spec)[1]:
        fmt = formats[label]
        lines.append(f"| {isl}/{osl} | {con} | {label} | {fmt.format(lv)} | "
                     f"{fmt.format(rv)} | {pct(rv, lv)} |")
    return lines


def section_decomposition(left: Arm, right: Arm, spec: EngineSpec) -> list:
    """Split the end-to-end gap into a prefill-side term and a decode-side one.

    ``E2E = TTFT + (OSL-1) * ITL`` is an identity over the benchmark's own metrics, so the residual
    is a check on it rather than a fitted term. It does **not** point at queueing -- waiting is
    already inside TTFT or the inter-token intervals -- but at aggregates that do not satisfy it.
    """
    rows = build_decomposition(left, right, spec)[1]
    if not rows:
        return []

    lines = ["", "## Where the difference comes from", "",
             "`E2E = TTFT + (OSL-1) x ITL`, an identity over the metrics above, so the residual "
             "column is a check on the split rather than a fitted term.", "",
             "| ISL/OSL | con | E2E gap ms | from TTFT | from ITL x (OSL-1) | residual | "
             "decode share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for isl, osl, con, e2e, ttft, decode, residual, share, _itl in rows:
        shown = f"{share:.0f}%" if share is not None else "n/a"
        lines.append(f"| {isl}/{osl} | {con} | {e2e:+.0f} | {ttft:+.0f} | {decode:+.0f} | "
                     f"{residual:+.0f} | {shown} |")

    itl_deltas = [row[-1] for row in rows]
    lines += ["", f"Per-step cost across {len(itl_deltas)} point(s): "
                  f"{min(itl_deltas):+.1f} to {max(itl_deltas):+.1f} ms, mean "
                  f"{sum(itl_deltas) / len(itl_deltas):+.1f} ms."]
    # The interpretation needs a narrow spread over enough concurrency points to call a range.
    # Grouped by input shape: points at different ISL/OSL are not a concurrency sweep.
    by_shape: dict = defaultdict(list)
    for row in rows:
        by_shape[(row[0], row[1])].append((row[2], row[-1]))
    swept = max(by_shape.values(), key=lambda pts: len({c for c, _d in pts}))
    concurrencies = {c for c, _d in swept}
    deltas = [d for _c, d in swept]
    spread = max(deltas) - min(deltas)
    if len(concurrencies) >= 3 and abs(sum(deltas) / len(deltas)) > 2 * spread:
        lines += ["", "A spread that narrow across that range of concurrency is what makes it a "
                      "per-step cost rather than a volume-limited one."]
    elif len(concurrencies) < 3:
        lines += ["", f"{len(concurrencies)} concurrency point(s) is too few to tell a per-step "
                      "cost from a volume-limited one; that needs the delta held across a range."]
    return lines


def unmatched_notes(left: Arm, right: Arm) -> list:
    """For each arm that classified nothing, the busiest names it did see.

    An arm with no device kernels has nothing to list, so no caller may assume this appears.
    """
    lines: list = []
    for arm in (left, right):
        if not arm.trace or (arm.trace.get("a2a") or {}):
            continue
        busiest = sorted((arm.trace.get("unmatched_us") or {}).items(), key=lambda kv: -kv[1])[:5]
        if not busiest:
            continue
        named = ", ".join(f"`{n}`" for n, _us in busiest)
        lines += ["", f"**No event name in {arm.name} matched this engine's patterns**, so its "
                      "cells are unavailable rather than zero -- nothing says the exchange did "
                      f"not happen. Its busiest unclassified kernels are {named}, which is what "
                      "extending the engine's patterns is read off."]
    return lines


def section_a2a(left: Arm, right: Arm, spec: EngineSpec) -> list:
    """The expert exchange, per stage and per kernel variant, side by side."""
    # Traces alone are not enough: the prose below claims backends carry their own transport,
    # which is false for an engine that declares no patterns.
    if not spec.a2a.patterns or not any(arm.trace for arm in (left, right)):
        return []

    lines = ["", "## The expert exchange", "",
             "Classified from the trace by event name, because this traffic reaches no RCCL log. "
             "The trace's absolute durations do not survive a cross-check, so only shares appear. "
             "A stage's share is of the trace-event categories that arm itself classified, so its "
             "numerator and denominator are the same clock; the per-kernel tables below are "
             "within classified kernel time alone.", ""]

    # Capture coverage first: an arm with no trace at all is not an arm whose names failed to
    # match, and the "nothing matched" wording below speaks for both.
    uncaptured = [arm.name for arm in (left, right) if not arm.trace]
    if uncaptured:
        lines += [f"**{' and '.join(uncaptured)} has no trace for this phase**, so this section "
                  "describes the other arm alone. Nothing below is a statement about the "
                  "uncaptured arm: a capture that was never taken and a capture whose names "
                  "matched no pattern look the same in a table and are not the same fact.", ""]
    for arm in (left, right):
        if arm.empty_trace_dirs:
            count = len(arm.empty_trace_dirs)
            lines += [f"{arm.name} had {count} capture "
                      + ("directory that held" if count == 1 else "directories that held")
                      + " no trace file and so contributed nothing: "
                      + f"{', '.join(sorted(arm.empty_trace_dirs))}. With one directory per "
                      "replica this is a coverage gap rather than a detail -- an arm counted over "
                      "fewer replicas than the other is not measured over the same thing -- "
                      "though an idle replica genuinely captures nothing, so check whether that "
                      "process did any work before reading it as a loss.", ""]
        cut = (arm.trace or {}).get("truncated") or []
        if cut:
            lines += [f"{arm.name} had {len(cut)} of {(arm.trace or {}).get('files', 0)} trace(s) "
                      f"stop being readable ({', '.join(sorted(cut))}) -- either a missing gzip "
                      "trailer or damage mid-stream -- and each was read up to that point. Its "
                      "call counts are floors, and its shares are **possibly biased**: normalising "
                      "removes how long the capture lasted, not which part of it was lost, and a "
                      "cut between dispatch and combine drops a suffix that is not representative "
                      "of the mix. Reparse before comparing these shares across arms -- reads on "
                      "NFS flap, and one unchanged capture gave 1, then 10, then 12, then 0 "
                      "unreadable of 16.", ""]
        bad = (arm.trace or {}).get("malformed") or {}
        if bad:
            total = sum(count for count, _example in bad.values())
            example = max(bad.items(), key=lambda kv: kv[1][0])
            lines += [f"{arm.name} has {total} event(s) whose duration is not a number, e.g. "
                      f"`{example[1][1]}` in `{example[0]}` -- two durations spliced together by "
                      "an interleaved flush. The calls are still counted; the unreadable durations "
                      "contribute nothing, which makes this arm's shares **possibly biased in "
                      "either direction**: the count is over every event, so a lost exchange "
                      "duration understates the share while a lost unclassified one shrinks only "
                      "the denominator and overstates it. Which happened is not recorded.", ""]
    stage_rows = build_stages(left, right)[1]
    if not stage_rows:
        # An empty table reads as "measured, and there was none", so say nothing rather than zero.
        # `unmatched_us` can be empty too, so the notices below cannot be relied on to speak.
        captured = [arm.name for arm in (left, right) if arm.trace]
        whose = " and ".join(captured) if captured else "either arm"
        lines += [f"**No event name in {whose} matched this engine's all-to-all patterns**, so "
                  "this section reports nothing rather than zero. The traffic exists; the names "
                  "these patterns know do not appear in the captures that were taken.", ""]
        return lines + unmatched_notes(left, right)
    lines += [f"| stage | {left.name} calls | {left.name} share | {right.name} calls | "
              f"{right.name} share |", "|---|---:|---:|---:|---:|"]

    def cell(calls, share):
        # Not %.0f: one event across two traces is 0.5 a trace, and rounding it to zero reports
        # that it never happened.
        return "n/a | n/a" if calls is None else f"{calls:g} | {share:.1f}%"

    for stage, lc, ls, rc, rs in build_stages(left, right)[1]:
        lines.append(f"| {stage} | {cell(lc, ls)} | {cell(rc, rs)} |")

    lcats, rcats = matched_cats(left), matched_cats(right)
    if lcats and rcats and lcats != rcats:
        # When the arms matched different categories, both shares are correct and no longer of the
        # same quantity: one may be of wall time where the other is of device time.
        lines += ["", f"**The two shares are not of the same quantity**: {left.name} classified "
                      f"{', '.join(sorted(lcats))} and {right.name} classified "
                      f"{', '.join(sorted(rcats))}. Each share is a fraction of that arm's own "
                      "categories, so compare the stages within an arm rather than the "
                      "percentages across them."]

    lines += unmatched_notes(left, right)

    variants: dict = {}
    hidden: dict = {}
    for arm in (left, right):
        rows = build_kernels(arm, spec)[1]
        if not rows:
            continue
        # Every classified kernel decides the verdict below; only the table is cut. A kernel too
        # small to be shown is how an arm that mixed two variants passes for a single-variant one.
        variants[arm.name] = {row[2] for row in rows} - {"not stated"}
        shown, rest = rows[:SHOWN_KERNELS], rows[SHOWN_KERNELS:]
        hidden[arm.name] = ({row[2] for row in rest} - {"not stated"}) - {
            row[2] for row in shown}
        lines += ["", f"### {arm.name}: which kernels", "",
                  "| stage | variant | kernel | calls per trace | share of exchange |",
                  "|---|---|---|---:|---:|"]
        for _arm, stage, variant, name, calls, share in shown:
            short = name if len(name) <= 70 else name[:67] + "..."
            lines.append(f"| {stage} | {variant} | `{short}` | {calls:g} | {share:.1f}% |")
        if rest:
            lines += ["", f"{len(rest)} smaller kernel(s) are in `expert_a2a_kernels.csv` rather "
                          "than above"
                      + (", and one of them ran a variant this table does not show: "
                         + ", ".join(sorted(hidden[arm.name])) + "."
                         if hidden.get(arm.name) else ".")]

    named_variants = [v for v in variants.values() if v]
    if len(named_variants) == 2 and named_variants[0] != named_variants[1]:
        lines += ["", "**The arms ran different variants of the exchange** "
                      f"({left.name}: {', '.join(sorted(variants[left.name]))}; "
                      f"{right.name}: {', '.join(sorted(variants[right.name]))}). A backend on its "
                      "low-latency kernels and one on its throughput kernels are not comparable as "
                      "backends; what is measured is the variant that was available."
                  + (" Some of this evidence is below the display cut; the CSV carries it."
                      if any(hidden.values()) else "")]
    return lines


def build(left: Arm, right: Arm, spec: EngineSpec, phase: str, command: str = "") -> str:
    """The whole comparison document."""
    head = [f"# {left.name} against {right.name} — {phase}", "",
            f"Left: `{left.run_dir}`  ", f"Right: `{right.run_dir}`  ",
            f"Engine: **{spec.name}**", ""]
    if command:
        head += [f"> Produced by `{command}`. Rerunning it on the same inputs reproduces this "
                 "file.", ""]
    parts = (section_comparability(left, right, spec)
             + section_steps(left, right)
             + section_decomposition(left, right, spec)
             + section_a2a(left, right, spec)
             + section_counters(left, right, spec)
             + section_points(left, right, spec))
    return "\n".join(head + parts) + "\n"


def make_arm(name: str, run_dir: Path, phases: dict, traces: dict, phase: str,
             spec: EngineSpec | None = None, empty_trace_dirs: tuple = (),
             counters: dict | None = None, counters_comparable: bool = False) -> Arm:
    """Assemble one side from an already-parsed run."""
    parsed = phases.get(phase)
    agreed, disagreed = (merge_nodes(parsed.config, parsed.config_nodes)
                         if parsed and parsed.config else ({}, {}))
    records = [r for recs in parsed.steps.values() for r in recs] if parsed else []
    return Arm(name=name, run_dir=run_dir,
               config=agreed,
               steps=steps_by_node(parsed.steps) if parsed else {},
               trace=traces.get(phase),
               points=load_benchmark(run_dir, spec.benchmark) if spec else {},
               pooled=summarise_steps(records) if records else None,
               disagreed=disagreed,
               batch_spread_ms=batch_invariance_by_node(parsed.steps) if parsed else None,
               steps_invalid=tuple(invalidators(parsed.config, spec.steps))
               if parsed and spec else (),
               empty_trace_dirs=tuple(empty_trace_dirs),
               counters=dict(counters or {}),
               counters_comparable=counters_comparable)
