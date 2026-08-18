"""Reading collectives, topology and engine metrics out of RCCL debug logs.

The log is the only channel that reports a message size for every collective of a whole run, which
is why it carries the report despite being text written concurrently by eight ranks. It needs
``NCCL_DEBUG=INFO`` and a ``NCCL_DEBUG_SUBSYS`` that includes ``COLL``; see
references/measurement-setup.md.

Nothing here names an engine: the log layout, the phase source, the metrics to harvest and the
sanity bounds all arrive in an :class:`~collprof.core.spec.EngineSpec`.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from .phase import (DAMAGE_MSG_CAP, DAMAGE_NO_TAIL, DAMAGE_NRANKS_RANGE, DAMAGE_RANK_RANGE,
                    DAMAGE_TOPO_TRANSPORT, DAMAGE_TWO_RECORDS, DAMAGE_UNKNOWN_COLL, Phase)
from .spec import NODE_FROM_PARENT, PHASE_FROM_MARKER, EngineSpec
from .units import datatype

# host:pid:tid [dev] NCCL INFO <Coll>: opCount <hex> ... count <n> datatype <d> ... [nranks=<n>]
RE_COLL = re.compile(
    r"^(?P<host>[\w.-]+):(?P<pid>\d+):(?P<tid>\d+)\s+\[(?P<dev>\d+)\].*?"
    r"NCCL INFO (?P<coll>\w+): opCount (?P<opcount>[0-9a-f]+).*?"
    r"count (?P<count>\d+) datatype (?P<dtype>\d+).*?nranks=(?P<nranks>\d+)"
)

# The tail RCCL prints after the arguments. Intact records always end this way and a
# half-overwritten one almost never does, which makes it the sharpest damage test available. It
# also carries globalrank, a better rank identity than the host:pid prefix, which tearing
# duplicates.
# `stream (nil)` is accepted alongside a stream address: a rank-local communicator does its work
# without a stream, and requiring an address discarded every one of them -- 17200 records per node
# on a training run -- as damaged.
RE_COLL_TAIL = re.compile(
    r"comm 0x[0-9a-f]+ \[nranks=(?P<nranks>\d+)\] stream (?:0x[0-9a-f]+|\(nil\)) "
    r"task \d+ globalrank (?P<grank>\d+)\s*$"
)

# Connection topology, logged once per channel while a communicator is built. The bracketed value is
# a device or bus id, not a rank, and direction is already encoded by the arrow: a `[receive]` line
# logged by rank 15 for `7 -> 15` still means data flows from 7 to 15.
RE_TOPO = re.compile(
    r"NCCL INFO Channel (?P<channel>\d+)/\d+ : (?P<src>\d+)\[[0-9a-f]+\] -> "
    r"(?P<dst>\d+)\[[0-9a-f]+\](?: \[(?P<dir>\w+)\])? via (?P<transport>[A-Za-z0-9/ ]+?)"
    # Stop at the record tail, at the first character no transport name contains, or at the end of
    # the line. Spaces are inside the name (`P2P/direct pointer`), so the end cannot be a space --
    # and a torn line has to keep matching here, or it would be dropped without being counted.
    r"(?= comm 0x|[^A-Za-z0-9/ ]|\s*$)"
)

#: Transports RCCL can name on a topology line, and which side of the node boundary each one is.
#: These lines tear like every other line of a shared stdout, and the transport is where the damage
#: shows: one serving report carried 20 edges over `PCCL`, `P50` and `localRank`, every one of them
#: counted as inter-node on a role that has no inter-node communicator at all.
TRANSPORT_SCOPES = (
    (re.compile(r"^P2P/(?:IPC|CUMEM|direct pointer|indirect)(?:/read)?$"), "intra-node"),
    (re.compile(r"^SHM(?:/\w+){0,2}$"), "intra-node"),
    (re.compile(r"^LOC$"), "intra-node"),
    (re.compile(r"^NET/\w+/\d+(?:/GDRDMA)?(?:/Shared)?$"), "inter-node"),
    (re.compile(r"^COLLNET(?:/\w+)*$"), "inter-node"),
    (re.compile(r"^(?:MNNVL|NVLS)(?:/\w+)*$"), "inter-node"),
)


def transport_scope(transport: str) -> str | None:
    """``intra-node`` / ``inter-node`` for a transport RCCL can print, None for a torn one.

    Anything unrecognised is treated as damage rather than as an unknown-but-real transport: the
    strings arriving here are spliced prefixes of the real ones (``P2P/IPCrank``, ``P2P/Iproxy``),
    so a prefix match would let most of them through.
    """
    for pattern, scope in TRANSPORT_SCOPES:
        if pattern.match(transport):
            return scope
    return None

#: Collective names RCCL can print. Anything else on an ``opCount`` line means the record was
#: damaged in flight, which happens because a role's eight ranks share one stdout: a 2.7M-line
#: decode log arrives with names like ``prllReduce`` and rank ids spliced from two writes.
KNOWN_COLLS = frozenset({
    "AllReduce", "AllGather", "ReduceScatter", "Broadcast", "Reduce", "AllToAll", "AllToAllv",
    "Gather", "Scatter", "Send", "Recv", "SendRecv",
    "mscclFuncAllReduce", "mscclFuncAllGather", "mscclFuncReduceScatter", "mscclFuncAllToAll",
    "mscclFuncSendRecv", "mscclFuncBroadcast", "mscclFuncReduce",
})

#: Sampled lines that must carry an intact tail before a log is held to that standard. Older RCCL
#: builds print no tail at all, and those logs stay on the weaker checks.
TAIL_SAMPLE = 200
TAIL_MAJORITY = 0.9


def open_log(log: Path):
    """Open a node log, gzipped or not."""
    if log.suffix == ".gz":
        return gzip.open(log, "rt", errors="ignore")
    return log.open(errors="ignore")


def log_stem(log: Path) -> str:
    """File name without the trace/compression suffixes, so ``x.log`` and ``x.log.gz`` agree."""
    name = log.name[:-3] if log.suffix == ".gz" else log.name
    return Path(name).stem


def discover_logs(run_dir: Path, spec: EngineSpec, rccl_dir: Path | None = None) -> list:
    """Find the logs of a job and say which node, phase and writer each one belongs to.

    Returns ``(path, node_label, phase_or_None, layout)`` tuples; the phase is None when the engine
    announces phases inside the log, in which case parsing picks them up from the markers.

    A run measured with ``NCCL_DEBUG_FILE`` has two sets: the shared stdout the engine always
    writes, and one file per process under ``rccl_dir`` (the run directory unless the files live
    elsewhere, as they do for training, where they land beside the traces). Both are read, since
    only the first carries the phase markers and the framework's own metrics.
    """
    found: list = []
    for layout, root in ((spec.logs, run_dir), (spec.rccl_logs, rccl_dir or run_dir)):
        if layout is None:
            continue
        for log in sorted({p for pattern in layout.globs for p in root.glob(pattern)}):
            stem = log_stem(log)
            node = log.parent.name if layout.node_from == NODE_FROM_PARENT else layout.node_of_name(
                stem)
            phase = None if layout.phase_from == PHASE_FROM_MARKER else layout.phase_of_name(stem)
            found.append((log, node, phase, layout))

    if not found:
        raise FileNotFoundError(
            f"no {' / '.join(spec.logs.globs)} under {run_dir} (engine {spec.name})")
    return found


def log_has_record_tails(log: Path) -> bool:
    """Sample a log to see whether its collective records end in the RE_COLL_TAIL shape."""
    seen = intact = 0
    with open_log(log) as fh:
        for line in fh:
            if "opCount" not in line:
                continue
            seen += 1
            if RE_COLL_TAIL.search(line):
                intact += 1
            if seen >= TAIL_SAMPLE:
                break
    return seen > 0 and intact / seen >= TAIL_MAJORITY


def damage_reason(m: re.Match, tail: re.Match | None, max_nranks: int) -> str | None:
    """Why a matched collective line cannot have come from one write, or None if it can."""
    if m.group("coll") not in KNOWN_COLLS:
        return DAMAGE_UNKNOWN_COLL
    # nranks=1 is a real value, not a splice: a rank-local communicator moves no data between ranks.
    # The report counts those separately and excludes them from every total, so they are kept rather
    # than discarded -- a lower bound of 2 here silently dropped tens of thousands of them per node.
    if not 1 <= int(m.group("nranks")) <= max_nranks:
        return DAMAGE_NRANKS_RANGE
    # A rank outside its own communicator means the digits were spliced, which is how ranks 12 and
    # 22 turned up in an 8-GPU decode log.
    if tail is not None and int(tail.group("grank")) >= int(tail.group("nranks")):
        return DAMAGE_RANK_RANGE
    # A second header inside the match means the regex bridged two records, taking the collective
    # from one and the count from the other.
    if "NCCL INFO" in m.group(0)[m.end("coll") - m.start():]:
        return DAMAGE_TWO_RECORDS
    return None


def parse_run(run_dir: Path, spec: EngineSpec, rccl_dir: Path | None = None) -> dict:
    """Parse every log of a run into ``{phase name: Phase}``."""
    limits = spec.limits
    phases: dict = {}

    def phase_named(name: str) -> Phase:
        return phases.setdefault(name, Phase(name, spec.name))

    for log, node, log_phase, layout in discover_logs(run_dir, spec, rccl_dir):
        marker_guard = layout.marker_guard if layout.phase_marker else ""
        current = phase_named(log_phase) if log_phase else None
        strict_tail = log_has_record_tails(log)
        with open_log(log) as fh:
            # A decode log reaches 2 GB, so every regex is guarded by a cheap substring its pattern
            # requires anyway. Same set of matches, roughly an order of magnitude less time.
            for line in fh:
                if marker_guard and marker_guard in line:
                    marker = layout.phase_marker.search(line)
                    if marker:
                        current = phase_named(marker.group(1))
                if current is None:
                    continue

                m = RE_COLL.match(line) if "opCount" in line else None
                if m:
                    current.writers.add(layout.written_by)
                    tail = RE_COLL_TAIL.search(line) if strict_tail else None
                    reason = damage_reason(m, tail, limits.max_nranks)
                    if reason is None and strict_tail and tail is None:
                        reason = DAMAGE_NO_TAIL
                    if reason:
                        current.damage[reason] += 1
                        continue
                    dt_name, dt_size = datatype(int(m.group("dtype")))
                    msg_bytes = int(m.group("count")) * dt_size
                    if msg_bytes > limits.max_msg_bytes:
                        current.damage[DAMAGE_MSG_CAP] += 1
                        continue
                    # The tail survives tearing far more often than the prefix, so when it is there
                    # it decides both the communicator size and which rank owns the line.
                    nranks = int(tail.group("nranks")) if tail else int(m.group("nranks"))
                    # Keyed by node rather than by the host:pid prefix: the file already says which
                    # machine this is, and that prefix is the part tearing corrupts most often,
                    # which used to invent extra ranks.
                    rank_id = tail.group("grank") if tail else m.group("pid")
                    coll = m.group("coll")
                    current.sizes[(coll, nranks, msg_bytes, dt_name)] += 1
                    current.nodes.add(node)
                    current.ranks.add((node, rank_id))
                    if nranks > 1:
                        row = current.per_node[(node, coll)]
                        row[0] += 1
                        row[1] += msg_bytes
                        rank_row = current.per_rank[(node, rank_id)]
                        rank_row[0] += 1
                        rank_row[1] += msg_bytes
                    continue

                if " via " in line:
                    topo = RE_TOPO.search(line)
                    if topo:
                        transport = topo.group("transport")
                        if transport_scope(transport) is None:
                            current.topo_damage[DAMAGE_TOPO_TRANSPORT] += 1
                            continue
                        key = (int(topo.group("src")), int(topo.group("dst")), transport)
                        current.edges[key].add(int(topo.group("channel")))
                        continue

                # No early exit: engines print several of these on one line, e.g. Megatron reports
                # tokens/s/GPU and TFLOP/s/GPU side by side.
                for metric in spec.metrics:
                    if metric.guard in line:
                        hit = metric.pattern.search(line)
                        if hit:
                            current.add_metric(metric.key, node, float(hit.group(1)))

    return phases
