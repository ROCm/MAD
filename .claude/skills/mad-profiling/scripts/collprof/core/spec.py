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
class RunConfigLayout:
    """How to read the configuration a run actually ran with, out of its own log.

    Not the command line: a launcher assembles one command per role and a framework fills in its
    defaults afterwards, so the only authoritative statement of what ran is the one the process
    prints about itself. sglang prints ``server_args=ServerArgs(k=v, ...)`` once per server.
    See references/interpretation.md.
    """

    guard: str = ""
    #: Matches the line; group 1 is the ``k=v, k=v`` body to split into fields.
    pattern: re.Pattern | None = None
    #: Which of this engine's settings move throughput, which differ between any two runs without
    #: meaning, and which must never reach an artifact. The engine's vocabulary -- `deepep_mode`
    #: means nothing to a training engine -- so declared here rather than in `core/`.
    perf_relevant: frozenset = frozenset()
    noise: frozenset = frozenset()
    secret: frozenset = frozenset()
    #: Settings whose value is a path to the same thing under a different mount. Compared by last
    #: component, so a remount is not a difference; calling them noise let two runs of different
    #: models compare as identical.
    path_valued: frozenset = frozenset()
    #: Splits the body into ``(key, value)`` pairs. Bracketed values are matched whole: stopping at
    #: the first comma reduced ``cuda_graph_bs=[1, 2, 4]`` to ``[1``, so two runs capturing
    #: different batch sizes compared equal.
    field_pattern: re.Pattern = re.compile(
        r"(\w+)=('[^']*'|\"[^\"]*\"|\[[^]]*\]|\([^)]*\)|\{[^}]*\}|[^,()]+)")


@dataclass(frozen=True)
class StepInvalidator:
    """A setting whose value breaks the derivation of a step time from batch and rate.

    ``batch / rate`` is a step time only while a step emits one token per running request.
    Speculative decoding emits several, so the quotient stays plausible and stops being a duration
    -- and :func:`steps.batch_invariance` cannot see it, since the breakage does not scale with the
    batch. The engine names such settings so the core can withhold the channel rather than lie.

    ``benign`` are the values that leave the channel intact; any other stated value withholds it.
    """

    setting: str
    benign: tuple[str, ...]
    #: What is wrong with the number, in the report's own words.
    why: str


@dataclass(frozen=True)
class StepTimingLayout:
    """How to recover per-step timing from what the engine already prints while serving.

    Serving has no iteration counter and, on this cluster, no rocprofv3 stats, so without this the
    report has no duration channel at all. A server logging its running batch and generation rate
    states its own step time, one record per logging interval per node, and it survives profiling
    because the server computes it rather than an instrument.
    """

    guard: str = ""
    #: Matches one logging interval. Named groups ``batch`` and ``rate`` are required; ``graphed``
    #: is optional and records whether the engine replayed a captured graph for those steps.
    pattern: re.Pattern | None = None
    #: What one record covers, for the report's wording.
    unit: str = "logging interval"
    #: Settings that invalidate the derived step time outright. Declared per engine because the
    #: setting's name and its harmless value are the engine's vocabulary.
    invalidated_by: tuple[StepInvalidator, ...] = ()


@dataclass(frozen=True)
class A2AKernels:
    """Patterns naming the expert all-to-all in a torch trace, per backend.

    The MoE all-to-all of an expert-parallel model does not go through RCCL: MoRI drives its own
    IBGDA path and DeepEP goes over rocSHMEM. Neither appears in an RCCL debug log, so the trace is
    the only place the operations are named at all.

    Classification is by name, so it is a discovery aid rather than a measurement: the report says
    which patterns matched and stays silent rather than reporting a zero when none did.
    """

    #: ``(label, pattern)``; the first pattern matching an event name wins.
    patterns: tuple[tuple[str, re.Pattern], ...] = ()
    #: Trace event categories worth classifying. Host-side ops carry the readable names.
    categories: tuple[str, ...] = ("kernel", "cpu_op")
    #: ``(label, pattern)`` naming which variant of the all-to-all a kernel implements, e.g.
    #: low-latency against throughput. Reported beside the stage because backends on different
    #: paths are not comparable, and the kernel name shows it without trusting a flag.
    variants: tuple[tuple[str, re.Pattern], ...] = ()


@dataclass(frozen=True)
class BenchmarkLayout:
    """Where a run's benchmark numbers live, and what that engine's CSV calls them.

    Throughput and latency come from a benchmark harness rather than the server log, and each
    harness names its own columns. Declared here per ``SKILL.md``: if adding an engine requires
    editing ``core/``, the spec is missing a field. An engine declaring nothing here loses the
    benchmark sections rather than the run.
    """

    #: Files holding the numbers, matched against the run directory. A glob because the harness
    #: writes the model name into the file name.
    globs: tuple[str, ...] = ()
    #: Matches the per-configuration key in ``model_column``; named groups ``isl``, ``osl`` and
    #: ``con``. ``4p4d_isl8192_osl1024_con256`` is one harness's shape, not every harness's.
    point: re.Pattern | None = None
    model_column: str = "model"
    metric_column: str = "metric"
    value_column: str = "performance"
    #: ``(metric, label, format)`` for the side-by-side table, in the order it should read.
    metrics: tuple[tuple[str, str, str], ...] = ()
    #: The three metrics ``E2E = TTFT + (OSL-1) x ITL`` is written over. Named per engine; leaving
    #: any empty yields no decomposition rather than one over guessed columns.
    e2e_metric: str = ""
    ttft_metric: str = ""
    itl_metric: str = ""

    @property
    def identity_metrics(self) -> tuple:
        """The three names, or ``()`` when the engine did not declare all of them."""
        named = (self.e2e_metric, self.ttft_metric, self.itl_metric)
        return named if all(named) else ()


@dataclass(frozen=True)
class CounterLayout:
    """Where the RDMA adapter counters of a run were sampled to, and how to group them.

    Declared per engine only because the file names are the launcher's. The counter *names* are
    the driver's -- mlx5 and bnxt_re spell the same operation differently -- so ``kinds`` is a
    classification the engine offers, and anything unmatched is reported under ``other``.
    """

    globs: tuple[str, ...] = ()
    #: Turns a matched file's stem into the node label the rest of the report uses.
    node_of_name: Callable[[str], str] = lambda stem: stem
    #: Which phase a sampled node belongs to, from its label: a comparison is per phase and the
    #: samples are per node. The engine decides because the convention is the launcher's; the
    #: default matches this launcher's `decode_NODE2`.
    phase_of_node: Callable[[str, str], bool] = lambda node, phase: node.startswith(phase)
    #: ``(label, pattern)`` over counter names; first match wins.
    kinds: tuple[tuple[str, re.Pattern], ...] = ()
    #: Kinds whose value is a byte volume, and kinds whose value is a count of operations. The
    #: core divides one by the other to ask whether an operation counter tracks that adapter's
    #: traffic at all: terabytes through four "write requests" is a counter blind to its data path.
    volume_kinds: tuple[str, ...] = ()
    operation_kinds: tuple[str, ...] = ()
    #: ``label -> factor`` for counters not in the unit their label claims. `port_rcv_data` and
    #: `port_xmit_data` are defined in **4-octet words**, so calling their raw delta "bytes" is out
    #: by four. Per engine, since it is a property of the fabric's counters.
    scale: dict = field(default_factory=dict)


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
    #: What one step-timing record of this engine covers, and how it was derived.
    step_basis: str = ""
    #: What graph replay being off means for this engine: only it knows whether that is a
    #: deliberate measurement setting or a fault.
    graphs_off: str = ""
    #: Why this engine's expert all-to-all reaches no RCCL log. Naming the transports is the
    #: engine's business.
    a2a_outside_rccl: str = ""


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
    #: How to read back the configuration the run actually used. An engine that declares none keeps
    #: the report as it was.
    run_config: RunConfigLayout = field(default_factory=RunConfigLayout)
    #: How to recover per-step timing from the engine's own logging.
    steps: StepTimingLayout = field(default_factory=StepTimingLayout)
    #: How the expert all-to-all is named in this engine's traces.
    a2a: A2AKernels = field(default_factory=A2AKernels)
    #: Where this engine's benchmark numbers are and what its CSV calls them.
    benchmark: BenchmarkLayout = field(default_factory=BenchmarkLayout)
    #: Where the RDMA adapter counters were sampled to, when the run sampled them at all.
    counters: CounterLayout = field(default_factory=CounterLayout)
    #: Settings ``notes.scope`` assumes were applied for measurement, as ``(setting, expected)``.
    #: Checked against what the run reported, so the scope note cannot assert a configuration the
    #: run contradicts. Needs :class:`RunConfigLayout`; without it nothing is checked.
    measurement_assumptions: tuple[tuple[str, str], ...] = ()
