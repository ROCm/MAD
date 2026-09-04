"""Message sizes and process groups from torch profiler traces.

This is the second independent channel of the report. The RCCL kernel event carries the collective
name, the message sizes, the dtype and the group size in its args, which makes it the only place
where a size and a process group describe the same individual operation.

Durations are read but deliberately not reported: they do not survive a cross-check against
rocprofv3, which puts multi-GiB collectives in tens of microseconds. See
references/interpretation.md.
"""

from __future__ import annotations

import gzip
import re
import zlib
from collections import defaultdict
from functools import partial
from pathlib import Path

from .spec import A2AKernels, TraceLayout
from .units import TORCH_DTYPE_BYTES

RE_TRACE_DUR = re.compile(r'"ts":\s*[\d.]+,\s*"dur":\s*([\d.]+)')
RE_TRACE_CAT = re.compile(r'"cat":\s*"([^"]+)"')
RE_TRACE_NAME = re.compile(r'"name":\s*"([^"]+)"')
RE_TRACE_ARGS = re.compile(
    r'"Collective name":\s*"(?P<coll>[^"]+)".*?"In msg nelems":\s*(?P<nin>\d+).*?'
    r'"Out msg nelems":\s*(?P<nout>\d+).*?"Group size":\s*(?P<group>\d+).*?'
    r'"dtype":\s*"(?P<dtype>[^"]+)"'
)
RE_TRACE_PG = re.compile(r'"Process Group Description":\s*"([^"]+)"')
RE_TRACE_STEP = re.compile(r'"ProfilerStep#(\d+)"')


def canonical_collective(name: str) -> str:
    """Map a torch collective name onto the RCCL name used elsewhere in the report."""
    n = name.lower()
    if "all_gather" in n or "allgather" in n:
        return "AllGather"
    if "reduce_scatter" in n or "reducescatter" in n:
        return "ReduceScatter"
    if "all_reduce" in n or "allreduce" in n:
        return "AllReduce"
    if "broadcast" in n:
        return "Broadcast"
    if "all_to_all" in n or "alltoall" in n:
        return "AllToAll"
    return name


def classify_a2a(name: str, a2a: A2AKernels) -> str:
    """The expert-all-to-all stage an event name belongs to, or "" when it belongs to none."""
    for label, pattern in a2a.patterns:
        if pattern.search(name):
            return label
    return ""


def trace_files(path: Path, layout: TraceLayout) -> list:
    """The trace files a path stands for: itself if a file, everything matching if a directory.

    A path that is neither raises. Both callers (``compare_cli._holds``, ``cli.holds_traces``)
    use this as their existence check and treat a raise as "nothing usable here"; returning a
    name for a missing file instead crashes later in ``parse_traces`` with a bare traceback.
    """
    if not path.is_dir():
        # is_file(), not "exists": a FIFO or device node would also pass and fail at open time.
        if not path.is_file():
            raise FileNotFoundError(f"no trace file at {path}")
        return [path]
    files: list = []
    for pattern in layout.file_globs:
        files.extend(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {' / '.join(layout.file_globs)} under {path}")
    return sorted(set(files))


def rank_of(name: str, layout: TraceLayout) -> int:
    """Rank a trace file belongs to, or -1 when its name does not say."""
    for pattern in layout.rank_patterns:
        m = pattern.search(name)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                return int(groups[0])
    return -1


def _until_truncated(fh, trace: Path, truncated: list):
    """Yield a trace's lines, stopping where the stream stops being readable instead of raising.

    One file per rank, so one rank's file going bad must not cost the whole capture; whatever was
    parsed before the bad point is kept and the caller reports which ranks were affected.

    Two failure modes needing different exceptions: a missing trailer (``EOFError``, a rank killed
    at teardown) and corrupt bytes mid-stream (``zlib.error``, which is not an ``OSError``).
    """
    try:
        yield from fh
    except (EOFError, OSError, zlib.error) as exc:
        truncated.append(trace.name)
        print(f"warning: {trace.name} stops being readable ({type(exc).__name__}), keeping what "
              "it held up to that point")


def parse_traces(paths: list, layout: TraceLayout, a2a: A2AKernels | None = None) -> dict:
    """Extract per-collective sizes, process groups and ranks from one phase's traces.

    Chrome-trace events span several lines and the duration sits on the line right before the args,
    so a line scan keyed on the args line is enough -- and far cheaper than loading a 300 MB JSON
    document. A single serving trace runs to ~15M lines, of which a handful carry a collective.

    When the engine declares :class:`~collprof.core.spec.A2AKernels`, events are also classified by
    name into the stages of the expert all-to-all. That traffic reaches no RCCL log, so on a
    backend comparison this is the only channel that names it.
    """
    files: list = []
    for path in paths:
        files.extend(trace_files(Path(path), layout))

    # (collective, process_group, total_bytes, dtype) -> [calls, dur_us]
    events: dict = defaultdict(lambda: [0, 0.0])
    # (a2a stage, category) -> [calls, dur_us], and the same category's total, for the share.
    a2a_events: dict = defaultdict(lambda: [0, 0.0])
    # (a2a stage, event name) -> [calls, dur_us]. The name says which variant of the exchange ran.
    a2a_names: dict = defaultdict(lambda: [0, 0.0])
    category_us: dict = defaultdict(float)
    # Event name -> device time, for the names no pattern claimed. A pattern set that matches
    # nothing is the expected state on a new backend, and this is what it takes to extend it.
    unmatched_us: dict = defaultdict(float)
    ranks: list = []
    group_size = 0
    steps: set = set()
    # Files whose stream ended early. Named rather than counted: which rank was cut short says
    # whether a per-rank imbalance is the run's or the capture's.
    truncated: list = []
    # File -> how many durations in it did not parse as a number, and one example. Interleaved
    # writes can splice two durations into a field like `4.200.347`; neither half is recoverable,
    # but a silently dropped event lowers a share.
    malformed: dict = defaultdict(lambda: [0, ""])

    for trace in files:
        ranks.append(rank_of(trace.name, layout))
        last_dur = 0.0
        last_cat = ""
        last_name = ""
        opener = (partial(gzip.open, trace, "rt") if trace.suffix == ".gz"
                  else partial(trace.open))
        with opener(errors="ignore") as fh:
            for line in _until_truncated(fh, trace, truncated):
                # Each step shows up in more than one event, so collect ids rather than count.
                if "ProfilerStep#" in line:
                    step = RE_TRACE_STEP.search(line)
                    if step:
                        steps.add(step.group(1))
                # A new event begins, so nothing of the previous one carries into it. Category,
                # name and duration come from separate lines and `cat` is optional in the
                # Chrome-trace format, so without this reset a category-less event inherits the
                # previous `cat` and inflates that denominator.
                if '"ph"' in line:
                    last_cat = ""
                    last_name = ""
                    last_dur = 0.0
                if '"cat"' in line:
                    cat = RE_TRACE_CAT.search(line)
                    if cat:
                        last_cat = cat.group(1)
                # Independently of the category: the two are separate keys and a pretty-printer
                # may break the line between them, which would leave a kernel counted into the
                # category total with no name. `"Collective name"` does not match, since the
                # pattern requires the quote immediately before `name`.
                if '"name"' in line:
                    name = RE_TRACE_NAME.search(line)
                    if name:
                        last_name = name.group(1)
                if '"dur"' in line:
                    dur = RE_TRACE_DUR.search(line)
                    if dur:
                        try:
                            last_dur = float(dur.group(1))
                        except ValueError:
                            # Zero rather than the previous event's duration, and fall through
                            # rather than skip: an a2a kernel is counted on this line and nowhere
                            # else, so skipping would drop the call along with the duration.
                            last_dur = 0.0
                            seen = malformed[trace.name]
                            seen[0] += 1
                            seen[1] = seen[1] or dur.group(1)
                        if a2a and a2a.patterns and last_cat in a2a.categories:
                            category_us[last_cat] += last_dur
                            stage = classify_a2a(last_name, a2a)
                            if stage:
                                row = a2a_events[(stage, last_cat)]
                                row[0] += 1
                                row[1] += last_dur
                                # Kept per name as well: the name says which variant ran, which
                                # the stage does not. Device events only -- a host-side `cpu_op`
                                # overlaps the kernel it launched, so counting both double-counts
                                # the overlap. The stage aggregate above still counts both.
                                if last_cat == "kernel":
                                    named = a2a_names[(stage, last_name)]
                                    named[0] += 1
                                    named[1] += last_dur
                            elif last_cat == "kernel":
                                unmatched_us[last_name] += last_dur
                        continue
                if '"Collective name"' not in line:
                    continue
                # The same args block is attached twice: once to the host-side `record_param_comms`
                # cpu_op and once to the RCCL device kernel. Counting both doubles every call and
                # mixes enqueue time into device time, so only the kernel event is taken.
                if last_cat != "kernel":
                    continue
                m = RE_TRACE_ARGS.search(line)
                if not m:
                    continue
                dsize = TORCH_DTYPE_BYTES.get(m.group("dtype"), 4)
                # nccl-tests measures bandwidth against the total message, not the per-rank shard.
                total_bytes = max(int(m.group("nin")), int(m.group("nout"))) * dsize
                pg = RE_TRACE_PG.search(line)
                key = (canonical_collective(m.group("coll")),
                       pg.group(1) if pg else "unknown", total_bytes, m.group("dtype"))
                entry = events[key]
                entry[0] += 1
                entry[1] += last_dur
                group_size = max(group_size, int(m.group("group")))

    # Distinct rank ids, not one per file: several replicas of a phase are captured at once and
    # each numbers its ranks from zero, so the raw list read as [0, 0, 1, 1, ...] for two replicas.
    return {"events": dict(events), "ranks": sorted(set(ranks)), "files": len(files),
            "steps": len(steps), "group_size": group_size,
            "a2a": dict(a2a_events), "a2a_names": dict(a2a_names),
            "category_us": dict(category_us), "unmatched_us": dict(unmatched_us),
            "truncated": truncated, "malformed": dict(malformed)}
