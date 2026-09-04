"""What the adapters actually put on the wire, from their own counters.

No other channel reaches this: the expert all-to-all appears in no RCCL log, a torch trace names
kernels but not their network behaviour, and rocprofv3 gives durations at best. The counters give
**verbs, not causality** -- reads and atomics in quantity are evidence of a protocol that waits,
their absence is not evidence of one that does not.

A sampler writes ``epoch_ns,device,port,counter,value`` rows while the servers run. Three limits
decide how the numbers may be read:

* **Per adapter and per node**, never per rank or per kernel.
* **Every user of the NIC is in there**, so an absolute count is a ceiling, not a measurement.
* **A zero is a measurement**: "sampled and stayed at zero" and "this driver has no such counter"
  are different facts.
* **Cumulative and unsynchronised.** Deltas are summed sample to sample so a mid-run wrap cannot
  hide behind a plausible end-to-end difference; any key that ever fell is dropped and named.

What survives is the difference between two arms serving the same requests, where the other
traffic is common-mode.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .spec import CounterLayout


@dataclass(frozen=True)
class CounterSeries:
    """One node's counters: what changed between its first and last sample."""

    node: str
    #: ``(device, port, counter) -> delta`` over the sampled window, wrapped counters excluded.
    deltas: dict
    #: Seconds between the first and last sample, 0.0 when a node has only one.
    seconds: float
    #: Number of samples, so a single-sample node is visibly not a rate.
    samples: int
    #: ``(device, port, counter)`` seen to decrease, which is a wrap or a reset rather than work.
    wrapped: tuple = ()
    #: Lines that could not be read at all -- torn by the filesystem rather than by the sampler.
    damaged: int = 0
    #: Every ``(device, port)`` this node sampled, moved or not. Coverage has to come from here:
    #: in the deltas an adapter that was sampled and stayed idle looks like one never sampled.
    adapters: tuple = ()

    @property
    def total(self) -> int:
        return sum(self.deltas.values())

    def per_second(self, key) -> float | None:
        return self.deltas[key] / self.seconds if self.seconds else None


def parse_counters(paths: list, layout: CounterLayout) -> dict:
    """``{node: CounterSeries}`` from the sampler's CSVs, one file per node.

    A header-only file is kept as a node with no counters rather than dropped: "sampled, found
    nothing" is a different fact from "never sampled", and a coverage note has to say which.
    """
    out: dict = {}
    for path in paths:
        node = layout.node_of_name(Path(path).stem)
        adapters: set = set()
        previous: dict = {}
        deltas: dict = {}
        wrapped: set = set()
        first_t = last_t = None
        samples = 0
        seen_t: set = set()
        with open(path, newline="", errors="ignore") as fh:
            for row in csv.DictReader(_readable(fh, damaged := [])):
                try:
                    stamp = int(row["epoch_ns"])
                    value = int(row["value"])
                except (KeyError, TypeError, ValueError):
                    # A torn line from a sampler killed mid-write is still damage: counting it only
                    # in `_readable` let the report claim zero damaged rows after dropping one.
                    damaged.append(str(row)[:64])
                    continue
                key = (row.get("device", ""), row.get("port", ""), row.get("counter", ""))
                adapters.add((key[0], key[1]))
                # Summed sample to sample, not last minus first: a counter that wrapped mid-run and
                # climbed back past its start gives a plausible end-to-end difference and an
                # invisible loss, whereas a decrease between consecutive samples cannot hide.
                if key in previous:
                    # A second observation makes the key measured, even at zero: 0 reads against
                    # N reads is the contrast a backend comparison exists to show, and it vanished
                    # when only non-zero steps were recorded.
                    deltas.setdefault(key, 0)
                    step = value - previous[key]
                    if step < 0:
                        wrapped.add(key)
                    elif step:
                        deltas[key] += step
                previous[key] = value
                if stamp not in seen_t:
                    seen_t.add(stamp)
                    samples += 1
                first_t = stamp if first_t is None else min(first_t, stamp)
                last_t = stamp if last_t is None else max(last_t, stamp)

        # A key that ever went backwards is reported rather than counted: what it lost across the
        # wrap is not recoverable, so its total would be a floor pretending to be a measurement.
        for key in wrapped:
            deltas.pop(key, None)
        seconds = (last_t - first_t) / 1e9 if first_t is not None and last_t is not None else 0.0
        out[node] = CounterSeries(node=node, deltas=deltas, seconds=seconds, samples=samples,
                                  wrapped=tuple(sorted(wrapped)), damaged=len(damaged),
                                  adapters=tuple(sorted(adapters)))
    return out


#: Longest line the sampler can legitimately write. Anything beyond this is damage, not data.
MAX_LINE = 4096


def _readable(fh, damaged: list):
    """Yield the lines that are lines, and record the ones that are not.

    The shared filesystem the counters land on returns **zeros instead of an error** when a read
    falls in a bad window, and a megabyte of NUL bytes mid-file stops `csv` with "field larger
    than field limit", taking the whole report down with it.
    """
    for line in fh:
        if len(line) > MAX_LINE or "\x00" in line:
            damaged.append(line[:64])
            continue
        yield line


def kind_order(layout: CounterLayout, present: set | None = None) -> list:
    """The engine's own kinds in the order it declared them, then whatever else was seen.

    The names are a property of the driver, so a core that hardcodes them decides for the next
    fabric what is worth showing. Unclassified counters keep their own names: pooling packets,
    errors, gauges and timers into one `other` produced a number with no unit.
    """
    declared = [name for name, _pattern in layout.kinds]
    extra = sorted((present or set()) - set(declared))
    return declared + extra


def bytes_per_op(counters, layout: CounterLayout) -> dict:
    """``kind -> bytes moved per operation of that kind``, over one arm's series.

    The ratio between two arms is what says whether their operation counts mean the same thing.
    An absolute threshold cannot: an RDMA write may carry a byte or a megabyte, and a legitimately
    rare operation looks exactly like a counter nobody is incrementing.
    """
    if not (layout.volume_kinds and layout.operation_kinds):
        return {}
    volume = 0
    ops: dict = {}
    for series in counters:
        for kind, group in by_kind(series, layout).items():
            if kind in layout.volume_kinds:
                volume += sum(group.values())
            elif kind in layout.operation_kinds:
                ops[kind] = ops.get(kind, 0) + sum(group.values())
    return {kind: volume / count for kind, count in ops.items() if count and volume}


#: How far two arms' bytes-per-operation may differ before their counts are taken to be measuring
#: different things. An order of magnitude can be a genuine difference in message size; three
#: orders is one side's counter not being incremented.
INCOMPARABLE_OP_RATIO = 100


def kinds_of(counters, layout: CounterLayout) -> set:
    """The kinds whose counters wrapped, across a set of series.

    A wrapped counter is dropped from the deltas, so its kind reads as zero -- a difference the
    backend did not make. Naming the kinds lets a comparison withhold exactly those rows.
    """
    affected: set = set()
    for series in counters:
        for _dev, _port, counter in series.wrapped:
            label = counter
            for name, pattern in layout.kinds:
                if pattern.search(counter):
                    label = name
                    break
            affected.add(label)
    return affected


def by_kind(series: CounterSeries, layout: CounterLayout) -> dict:
    """Group a node's deltas into the kinds an engine declared, e.g. writes against reads.

    The names are the driver's and differ between vendors, so the grouping is engine data rather
    than a table in here. A counter matching no kind is kept under ``other``: an unclassified
    name should show up rather than vanish.
    """
    kinds: dict = {}
    for (dev, port, counter), delta in series.deltas.items():
        # The counter's own name when nothing claims it. Pooling them lost the unit: packets,
        # errors and a gauge sum to a number the comparison then took a percentage of.
        label = counter
        for name, pattern in layout.kinds:
            if pattern.search(counter):
                label = name
                break
        # The engine's factor, where the counter is not in the unit its label claims: the IB data
        # counters count 4-octet words, so a raw delta reported as bytes is out by four.
        kinds.setdefault(label, {})[(dev, port, counter)] = delta * layout.scale.get(label, 1)
    return kinds
