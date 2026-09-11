"""Per-step timing recovered from what a serving engine prints about itself.

A server logging its running batch `n` with its generation rate `r` has stated its step time as
`n / r`, since a decode step emits one token per running request. It is the only duration channel
serving has: trace durations do not survive a cross-check against rocprofv3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .spec import StepTimingLayout


@dataclass(frozen=True)
class StepRecord:
    """One logging interval: how many requests were running and how fast they advanced."""

    batch: int
    rate: float
    #: Whether the engine replayed a captured graph over these steps, when it says so.
    graphed: bool | None = None

    @property
    def step_ms(self) -> float:
        """Milliseconds per decode step, **estimated** from the batch and rate the engine reported.

        An estimate because the two fields need not describe the same interval;
        :func:`batch_invariance` checks that. It also assumes one token per request per forward,
        which speculative decoding breaks invisibly.
        """
        return 1000.0 * self.batch / self.rate if self.rate else 0.0


def invalidators(config: dict, layout: StepTimingLayout) -> list:
    """The engine-declared settings this run stated that make its step times meaningless.

    One node stating a value is enough, since the channel is pooled. Returns ``(setting, value,
    why)`` triples, empty when the channel is sound or the run stated no configuration at all.
    """
    found: dict = {}
    for inv in layout.invalidated_by:
        for settings in config.values():
            value = settings.get(inv.setting)
            if value is None or value in inv.benign:
                continue
            found.setdefault(inv.setting, (inv.setting, value, inv.why))
    return [found[k] for k in sorted(found)]


def parse_step_line(line: str, layout: StepTimingLayout) -> StepRecord | None:
    """The interval a log line reports, or None when it reports none."""
    if not layout.pattern:
        return None
    hit = layout.pattern.search(line)
    if not hit:
        return None
    fields = hit.groupdict()
    try:
        batch, rate = int(fields["batch"]), float(fields["rate"])
    except (KeyError, TypeError, ValueError):
        return None
    # A rate of zero is the first interval after startup: no step time, and a division by zero.
    if batch <= 0 or rate <= 0:
        return None
    graphed = fields.get("graphed")
    return StepRecord(batch, rate, None if graphed is None else graphed.strip().lower() == "true")


@dataclass(frozen=True)
class StepStats:
    """The step-time distribution of one node, or of a whole phase."""

    intervals: int
    batch_min: int
    batch_max: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    #: True/False when every interval agreed, None when the engine did not say or nodes disagreed.
    graphed: bool | None
    #: True when intervals in this sample explicitly disagreed about graph replay. Distinct from
    #: ``graphed is None``, which means nothing was said; a mixture is a confound worth showing.
    graphs_mixed: bool = False

    @property
    def spread(self) -> float:
        """p95 over median. A flat distribution sits near 1; stragglers push it up."""
        return self.p95_ms / self.median_ms if self.median_ms else 0.0


def graph_state(st) -> str:
    """One node's graph-replay state as a word, for every renderer that shows it.

    Four states, not three: a node whose intervals disagreed is "mixed", and rendering that as
    "not stated" loses the confound.
    """
    if st.graphs_mixed:
        return "mixed"
    return {True: "replayed", False: "off", None: "not stated"}[st.graphed]


def percentile(values: list, fraction: float) -> float:
    """Nearest-rank percentile: the smallest observation with at least ``fraction`` below it.

    Nearest-rank rather than interpolating, so the figure is a step time some interval took.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def batch_invariance_by_node(steps: dict, min_group: int = 200) -> float | None:
    """The worst per-node spread, or None when no node had enough groups.

    Per node because pooling confounds node identity with batch: replicas serving disjoint batch
    ranges fan the groups out, and the report then calls a sound channel unreliable.
    """
    spreads = [s for s in (batch_invariance(recs, min_group) for recs in steps.values())
               if s is not None]
    return max(spreads) if spreads else None


def batch_invariance(records: list, min_group: int = 200) -> float | None:
    """Spread in ms between the median step times of the batch groups, or None if too few.

    **One-sided.** A flat spread rules out a rate accumulated over a window whose batch changed,
    which errs proportionally to the batch; a large spread is also what a volume-limited workload
    looks like, so only the absence of batch sensitivity is evidence.
    """
    groups: dict = {}
    for r in records:
        if r.step_ms > 0:
            groups.setdefault(r.batch, []).append(r.step_ms)
    medians = [median(v) for v in groups.values() if len(v) >= min_group]
    return max(medians) - min(medians) if len(medians) >= 2 else None


def summarise(records: list) -> StepStats | None:
    """Collapse a node's or a phase's intervals into a distribution."""
    if not records:
        return None
    times = [r.step_ms for r in records if r.step_ms > 0]
    if not times:
        return None
    # `None` stays in `flags`: `StepStats.graphed` promises every interval agreed, and an interval
    # that stated nothing is not evidence that it matched the ones that did.
    stated = {r.graphed for r in records} - {None}
    flags = {r.graphed for r in records}
    return StepStats(
        intervals=len(times),
        batch_min=min(r.batch for r in records),
        batch_max=max(r.batch for r in records),
        median_ms=median(times),
        p95_ms=percentile(times, 0.95),
        min_ms=min(times),
        max_ms=max(times),
        graphed=flags.pop() if len(flags) == 1 else None,  # a lone None means "unstated"
        graphs_mixed=len(stated) > 1,
    )


def by_node(steps: dict) -> dict:
    """``{node: StepStats}``, skipping nodes whose intervals carried no usable timing.

    Per node rather than pooled: a disaggregated proxy may leave one replica nearly idle, and its
    batch of 8 against another's 512 says nothing about the run.
    """
    return {node: stats for node, records in sorted(steps.items())
            if (stats := summarise(records)) is not None}
