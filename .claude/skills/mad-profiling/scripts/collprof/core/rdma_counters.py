"""What the adapters actually put on the wire, from their own counters.

A sampler writes ``epoch_ns,device,port,counter,value`` rows per node. Counts are per adapter and
cover every user of the NIC, so a total is a ceiling; only the difference between two arms serving
the same requests is comparable.
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
    #: The sampler reported it could not finish, via a `.failed` marker beside the file. The
    #: samples that landed are real, but the window is a floor rather than a measurement.
    sampler_failed: bool = False
    #: The machines the samples came from. Provenance, not an identity check: the two arms of an
    #: A/B always land on different physical nodes, so requiring equality would withhold every
    #: comparison. Empty for files written before the column existed.
    hosts: tuple = ()
    #: Every ``(device, port)`` this node sampled, moved or not. In the deltas an adapter that was
    #: sampled and stayed idle is indistinguishable from one never sampled.
    adapters: tuple = ()

    @property
    def total(self) -> int:
        return sum(self.deltas.values())

    def per_second(self, key) -> float | None:
        return self.deltas[key] / self.seconds if self.seconds else None


def parse_counters(paths: list, layout: CounterLayout) -> dict:
    """``{node: CounterSeries}`` from the sampler's CSVs, one file per node.

    A header-only file is kept as a node with no counters, so the coverage note can distinguish
    "sampled, found nothing" from "never sampled".
    """
    out: dict = {}
    for path in paths:
        node = layout.node_of_name(Path(path).stem)
        adapters: set = set()
        hosts: set = set()
        previous: dict = {}
        deltas: dict = {}
        wrapped: set = set()
        first_t = last_t = None
        samples = 0
        seen_t: set = set()
        with open(path, "rb") as fh:
            for row in csv.DictReader(_readable(fh, damaged := [])):
                try:
                    stamp = int(row["epoch_ns"])
                    value = int(row["value"])
                except (KeyError, TypeError, ValueError):
                    # A torn line from a sampler killed mid-write is damage too; counting it only
                    # in `_readable` let the report claim zero damaged rows after dropping one.
                    damaged.append(str(row)[:64])
                    continue
                if row.get("host"):
                    hosts.add(row["host"])
                key = (row.get("device") or "", row.get("port") or "",
                       row.get("counter") or "")
                if not all(key):
                    # A row whose numbers parse but whose identity does not: `...,,,rx_write_req,
                    # 100` would otherwise contribute traffic under an empty adapter, counted as
                    # real and absent from the damaged tally.
                    damaged.append(str(row)[:64])
                    continue
                adapters.add((key[0], key[1]))
                # Summed sample to sample, not last minus first: a counter that wrapped and climbed
                # back past its start would give a plausible end-to-end difference.
                if key in previous:
                    # A second observation makes the key measured, even at zero, so a counter that
                    # stayed at 0 still contrasts with the other arm's N.
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
        # wrap is unrecoverable, so its total would be a floor pretending to be a measurement.
        for key in wrapped:
            deltas.pop(key, None)
        seconds = (last_t - first_t) / 1e9 if first_t is not None and last_t is not None else 0.0
        out[node] = CounterSeries(node=node, deltas=deltas, seconds=seconds, samples=samples,
                                  wrapped=tuple(sorted(wrapped)), damaged=len(damaged),
                                  adapters=tuple(sorted(adapters)),
                                  hosts=tuple(sorted(hosts)),
                                  sampler_failed=Path(f"{path}.failed").exists())
    return out


#: Longest line the sampler can legitimately write. Anything beyond this is damage, not data.
MAX_LINE = 4096


def _readable(fh, damaged: list):
    """Yield the lines that decode and are lines; record the rest as damage.

    Read as bytes and decoded strictly. `errors="ignore"` dropped undecodable bytes instead, so
    `1\xff2` arrived as `12`: a fabricated value, counted as real, with nothing added to
    `damaged` for the comparability gate to refuse on. The shared filesystem returns zeros rather
    than an error on a bad read, so a run of NUL bytes is expected and skipped the same way.
    """
    for raw in fh:
        if len(raw) > MAX_LINE or b"\x00" in raw:
            damaged.append(repr(raw[:64]))
            continue
        try:
            yield raw.decode("utf-8")
        except UnicodeDecodeError:
            damaged.append(repr(raw[:64]))


def kind_order(layout: CounterLayout, present: set | None = None) -> list:
    """The engine's own kinds in the order it declared them, then whatever else was seen.

    Unclassified counters keep their own names, since pooling packets, errors, gauges and timers
    into one `other` produces a number with no unit.
    """
    declared = [name for name, _pattern in layout.kinds]
    extra = sorted((present or set()) - set(declared))
    return declared + extra


def bytes_per_op(counters, layout: CounterLayout) -> dict:
    """``kind -> bytes moved per operation of that kind``, over one arm's series.

    Only the ratio between two arms says whether their operation counts mean the same thing; an
    RDMA write may carry a byte or a megabyte, so no absolute threshold works.
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
#: different things. An order of magnitude can be a real difference in message size; three orders
#: is one side's counter not being incremented.
INCOMPARABLE_OP_RATIO = 100


def kinds_of(counters, layout: CounterLayout) -> set:
    """The kinds whose counters wrapped, across a set of series.

    A wrapped counter is dropped from the deltas, so its kind reads as zero; naming the kinds lets
    a comparison withhold exactly those rows instead of reporting a difference nobody made.
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

    Counter names are the driver's and differ between vendors, so the grouping is engine data; a
    counter matching no kind keeps its own name rather than vanishing.
    """
    kinds: dict = {}
    for (dev, port, counter), delta in series.deltas.items():
        # The counter's own name when nothing claims it. Pooling them loses the unit: packets,
        # errors and a gauge sum to a number the comparison would take a percentage of.
        label = counter
        for name, pattern in layout.kinds:
            if pattern.search(counter):
                label = name
                break
        # The engine's factor, where the counter is not in the unit its label claims: IB data
        # counters count 4-octet words, so a raw delta reported as bytes is out by four.
        kinds.setdefault(label, {})[(dev, port, counter)] = delta * layout.scale.get(label, 1)
    return kinds
