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
from collections import defaultdict
from functools import partial
from pathlib import Path

from .spec import TraceLayout
from .units import TORCH_DTYPE_BYTES

RE_TRACE_DUR = re.compile(r'"ts":\s*[\d.]+,\s*"dur":\s*([\d.]+)')
RE_TRACE_CAT = re.compile(r'"cat":\s*"([^"]+)"')
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


def trace_files(path: Path, layout: TraceLayout) -> list:
    """The trace files a path stands for: itself if a file, everything matching if a directory."""
    if not path.is_dir():
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


def parse_traces(paths: list, layout: TraceLayout) -> dict:
    """Extract per-collective sizes, process groups and ranks from one phase's traces.

    Chrome-trace events span several lines and the duration sits on the line right before the args,
    so a line scan keyed on the args line is enough -- and far cheaper than loading a 300 MB JSON
    document. A single serving trace runs to ~15M lines, of which a handful carry a collective.
    """
    files: list = []
    for path in paths:
        files.extend(trace_files(Path(path), layout))

    # (collective, process_group, total_bytes, dtype) -> [calls, dur_us]
    events: dict = defaultdict(lambda: [0, 0.0])
    ranks: list = []
    group_size = 0
    steps: set = set()

    for trace in files:
        ranks.append(rank_of(trace.name, layout))
        last_dur = 0.0
        last_cat = ""
        opener = (partial(gzip.open, trace, "rt") if trace.suffix == ".gz"
                  else partial(trace.open))
        with opener(errors="ignore") as fh:
            for line in fh:
                # Each step shows up in more than one event, so collect ids rather than count.
                if "ProfilerStep#" in line:
                    step = RE_TRACE_STEP.search(line)
                    if step:
                        steps.add(step.group(1))
                if '"cat"' in line:
                    cat = RE_TRACE_CAT.search(line)
                    if cat:
                        last_cat = cat.group(1)
                if '"dur"' in line:
                    dur = RE_TRACE_DUR.search(line)
                    if dur:
                        last_dur = float(dur.group(1))
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
            "steps": len(steps), "group_size": group_size}
