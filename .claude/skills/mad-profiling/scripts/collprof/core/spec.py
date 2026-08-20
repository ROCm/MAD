"""The contract between the engine-agnostic core and one engine's conventions.

Everything an engine knows -- how its logs are named, where a phase comes from, which throughput
lines it prints, how its trace files are named, and what a reader must be told about its numbers --
is data in an :class:`EngineSpec`. The core reads that data and never names an engine.

That invariant is the point of this module. Before it existed, report prose branched on
``phase.name in ("prefill", "decode")``, so a new inference engine that happened to call its phases
prefill and decode inherited sglang's claims about mooncake RDMA -- wrong output that looked right.
Adding an engine now means adding one module under ``collprof/engines/`` and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MIB = 1024 ** 2

#: Where the name of a phase comes from.
#:
#: ``MARKER``  -- the log itself announces a phase change, so one log covers several phases in
#:               sequence. Primus training runs BF16 then FP8 through the same stdout.
#: ``FILENAME`` -- the phase is part of the log's name, so each log belongs to exactly one phase.
#:               sglang writes one log per role per node and prints no markers.
PHASE_FROM_MARKER = "marker"
PHASE_FROM_FILENAME = "filename"

#: How the node a log came from is labelled.
#:
#: ``PARENT``  -- the log sits in a per-node directory (``node_0/stdout.out``).
#: ``STEM``    -- the node is in the file name itself (``decode_NODE2.log``).
NODE_FROM_PARENT = "parent"
NODE_FROM_STEM = "stem"

#: What wrote the records of a log, which decides what may be said about their damage.
#:
#: ``SHARED``   -- many ranks share one stream, so records tear and a fraction is unusable.
#: ``PER_RANK`` -- ``NCCL_DEBUG_FILE`` gave each process its own line-buffered file, so nothing
#:                interleaves and a torn record means something else went wrong.
LOG_SHARED = "shared"
LOG_PER_RANK = "per-rank"


@dataclass(frozen=True)
class LogLayout:
    """How to find an engine's per-node logs and what each one is.

    ``globs`` are matched against the run directory. Compressed variants belong here too: a decode
    log at ``NCCL_DEBUG=INFO`` reaches 2 GB, and these runs keep them gzipped, so ``*.log.gz`` is a
    normal input rather than a special case.
    """

    globs: tuple[str, ...]
    phase_from: str
    node_from: str
    #: Required when ``phase_from`` is ``MARKER``: matches the line announcing a phase, group 1
    #: being the phase name.
    phase_marker: re.Pattern | None = None
    #: Cheap substring every marker line contains, checked before the regex. A 2 GB log is scanned
    #: line by line, so each regex is guarded by a literal it requires anyway.
    marker_guard: str = ""
    #: Turns a matched log path into the phase name, when ``phase_from`` is ``FILENAME``.
    phase_of_name: Callable[[str], str] = lambda stem: stem.split("_")[0]
    #: Turns a matched log path into the node label, when ``node_from`` is ``STEM``. A per-rank file
    #: names the process as well as the node, and the report wants the node.
    node_of_name: Callable[[str], str] = lambda stem: stem
    #: Whether ranks shared this stream. Only :data:`LOG_SHARED` may be blamed for torn records.
    written_by: str = LOG_SHARED


@dataclass(frozen=True)
class LogMetric:
    """A scalar an engine prints into its log and the report should carry.

    Keeping these declarative is what let the parse loop stop knowing about Megatron: the loop
    harvests whatever an engine declares into ``Phase.metrics[key]``.
    """

    key: str
    guard: str
    pattern: re.Pattern
    #: How the report words it. ``{value}`` is the maximum observed, already formatted.
    label: str = ""


@dataclass(frozen=True)
class TraceLayout:
    """How to find torch profiler traces and decide which phase each one belongs to.

    Both supported engines put traces in directories that are not named after the phase, which used
    to be resolved by a human reading timestamps out of a log and pasting them into a shell script.
    That is where the second node of each sglang role silently went missing, so resolution is code.
    """

    #: Glob for a directory (or file) holding traces, relative to the run directory.
    dir_glob: str = ""
    #: Maps candidate trace directories to ``{phase: [paths]}``. Engine-specific by nature: an
    #: engine that names trace dirs after the phase needs a regex, one that names them after an
    #: epoch second needs to be told which phase was profiled when.
    resolve: Callable[[Path], dict[str, list[Path]]] | None = None
    #: Patterns yielding the rank from a trace file name; the first group that matches wins.
    #: Megatron writes ``...rank[3]...``, sglang ``<epoch>-TP-3.trace.json.gz``.
    rank_patterns: tuple[re.Pattern, ...] = ()
    #: Suffixes a trace file can have.
    file_globs: tuple[str, ...] = ("*.trace.json", "*.trace.json.gz")


@dataclass(frozen=True)
class SanityLimits:
    """Bounds separating a real RCCL record from one that arrived spliced.

    These are properties of a run's scale, not of the parser, which is why they live per engine and
    are overridable from the command line. A run whose real messages or communicators are larger
    than the bounds does not silently lose them: every rejection is counted by reason and the report
    says so where a reader cannot miss it. See references/data-quality.md.
    """

    #: A communicator wider than this means the digits were spliced. Eight GPUs per node here;
    #: expert parallelism across nodes legitimately goes higher.
    max_nranks: int = 64
    #: Largest single message treated as real. Tearing inside the ``count`` field concatenates two
    #: decimal counts: 97920 and 854624 arrived as 97920854624, a 91 GiB "AllReduce" that carried
    #: 16% of a decode report's volume until it was caught.
    max_msg_bytes: int = 512 * MIB
    #: A rank below this fraction of the busiest rank's volume is idle rather than a symmetric peer,
    #: and averaging over it would halve every per-rank figure. Set well below any real imbalance
    #: between working ranks, which stays within 2%.
    idle_rank_fraction: float = 0.05
    #: Fraction of records that must be rejected before the report warns about the input rather
    #: than just footnoting the count.
    damage_warn_fraction: float = 0.02


@dataclass(frozen=True)
class ReportNotes:
    """What a reader must be told about this engine's numbers to not misread them.

    Prose, not code, but prose that has to travel with the engine: each sentence here is a claim
    about how one engine was measured. The core inserts them and asserts nothing itself.
    """

    #: Appended to the "ranks present" line: why some ranks may be missing from the log.
    rank_coverage: str = ""
    #: Appended to the communicator-size line.
    communicator: str = ""
    #: Blockquote after the summary: what configuration these numbers describe and what they omit.
    scope: tuple[str, ...] = ()
    #: Why this engine's logs carry damaged records at all. Without it the report states the count
    #: and the breakdown and leaves the cause open, which is the honest default for a new engine.
    damage_cause: str = ""
    #: Added to the torch-trace section when the trace holds no ``ProfilerStep`` markers.
    unmarked_window: str = ("an unmarked window each: no ProfilerStep annotations were emitted, so "
                            "the counts below are per capture rather than per iteration")
    #: Added to the torch-trace section: why trace and log volumes are not meant to agree.
    trace_vs_log: str = ""


@dataclass(frozen=True)
class EngineSpec:
    """One engine's conventions, in full."""

    name: str
    summary: str
    logs: LogLayout
    notes: ReportNotes
    #: Where ``NCCL_DEBUG_FILE`` puts one log per process, when a run was measured that way. Read in
    #: addition to ``logs``, because the two carry different things: the per-rank files hold every
    #: collective of every rank untorn, while the shared stdout keeps the phase markers and the
    #: throughput lines the framework prints. A run without those files is unaffected.
    rccl_logs: LogLayout | None = None
    metrics: tuple[LogMetric, ...] = ()
    #: Key of the metric whose values count iterations, when the engine has iterations at all.
    #: With it the report divides volume per rank by iteration count; without it -- serving has no
    #: iterations, and its throughput lives in a benchmark CSV rather than in the server log -- no
    #: per-iteration figure is quoted anywhere.
    iteration_metric: str = ""
    traces: TraceLayout = field(default_factory=TraceLayout)
    limits: SanityLimits = field(default_factory=SanityLimits)
