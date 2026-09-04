"""Per-step timing recovered from what a serving engine prints about itself.

This is the duration channel serving does not otherwise have: a torch trace's durations do not
survive a cross-check against rocprofv3, and rocprofv3 produced no stats at all for these servers.

A server logging its running batch size with its generation rate has already stated its step time:
`n` requests advancing at `r` tokens per second spend `n / r` seconds per step, since a decode step
emits one token per running request. It is the server's own accounting, so it survives profiling
and is comparable between a profiled and an unprofiled run. Two properties earn it a section:

- **It is a distribution.** A constant gap is a fixed per-step cost; one growing with the batch is
  volume-limited. A single mean cannot tell those apart.
- **It carries whether the steps were graphed**, in band -- the configuration difference most
  likely to explain a per-step gap.
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

        A decode step emits one token per running request, so the rate divided by the batch is the
        step frequency however the engine batches internally.

        It is an estimate because the two fields need not describe the same interval: a rate
        accumulated over a window in which the batch changed is divided by a batch that did not
        hold throughout. :func:`batch_invariance` measures whether that correspondence holds --
        such a mismatch scales with the batch, so the group medians would fan out.

        The remaining condition is one token per request per forward, which speculative decoding
        breaks and this check cannot see. Run the check on a new engine before trusting it.
        """
        return 1000.0 * self.batch / self.rate if self.rate else 0.0


def invalidators(config: dict, layout: StepTimingLayout) -> list:
    """The engine-declared settings this run stated that make its step times meaningless.

    ``config`` is per node because one node stating a value is enough: the channel is pooled and a
    single speculative decoder poisons the pool.

    Returns ``(setting, value, why)`` triples, empty when the channel is sound or when the run
    stated no configuration at all -- withholding on an unmentioned setting would withhold from
    every run of an engine that never prints it.
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
    # A rate of zero is the first interval after startup: no step time, and a division by zero
    # downstream.
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
    #: ``graphed is None``, which means nothing said: a mixture of replayed and ungraphed steps is
    #: a confound, and collapsing it onto "not stated" hid it.
    graphs_mixed: bool = False

    @property
    def spread(self) -> float:
        """p95 over median. A flat distribution sits near 1; stragglers push it up."""
        return self.p95_ms / self.median_ms if self.median_ms else 0.0


def graph_state(st) -> str:
    """One node's graph-replay state as a word, for every renderer that shows it.

    Four states, not three: a node whose intervals disagreed has ``graphed is None`` and
    ``graphs_mixed`` set, and calling that "not stated" loses the confound. One function because
    the CSV and the markdown table rendered this separately and only one was fixed.
    """
    if st.graphs_mixed:
        return "mixed"
    return {True: "replayed", False: "off", None: "not stated"}[st.graphed]


def percentile(values: list, fraction: float) -> float:
    """Nearest-rank percentile: the smallest observation with at least ``fraction`` below it.

    Nearest-rank rather than interpolating: every value here is a measured step time, and a figure
    no interval actually took is harder to defend than a marginally biased one.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def batch_invariance_by_node(steps: dict, min_group: int = 200) -> float | None:
    """The worst per-node spread, or None when no node had enough groups.

    Pooling every node's records first confounds node identity with batch: replicas serving
    disjoint batch ranges fan the groups out for a reason unrelated to the estimate, and the
    report then calls a sound channel unreliable. Each node is judged on its own.
    """
    spreads = [s for s in (batch_invariance(recs, min_group) for recs in steps.values())
               if s is not None]
    return max(spreads) if spreads else None


def batch_invariance(records: list, min_group: int = 200) -> float | None:
    """Spread in ms between the median step times of the batch groups, or None if too few.

    **A one-sided check.** A flat spread rules out the failure this exists for -- a rate
    accumulated over a window in which the batch changed errs proportionally to the batch. A large
    spread does *not* establish it, since a volume-limited workload legitimately grows with its
    batch and the two look identical from here. Only the absence of batch sensitivity is evidence.
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
    # `None` stays in `flags`: an interval that did not state its graph mode is not evidence that
    # it matched the ones that did, and `StepStats.graphed` promises every interval agreed.
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

    Per node rather than pooled because a disaggregated proxy may leave one replica nearly idle,
    and a replica running a batch of 8 against another's 512 says nothing about the run.
    """
    return {node: stats for node, records in sorted(steps.items())
            if (stats := summarise(records)) is not None}
