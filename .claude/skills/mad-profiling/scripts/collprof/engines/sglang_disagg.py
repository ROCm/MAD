"""sglang PD-disaggregated serving (prefill and decode servers on separate nodes).

Layout: one log per role per node, ``prefill_NODE0.log`` / ``decode_NODE2.log``, with no phase
markers -- the role is the phase, and it is in the file name. Kept gzipped as a matter of course: a
decode log at ``NCCL_DEBUG=INFO`` reaches 2 GB and has filled a shared home directory once.

There are no iterations to divide by and the server log carries no throughput, so those numbers come
from the benchmark CSV of a separate run without profiling. What this engine's numbers do and do not
describe is the scope note below, and it is the reason this prose lives with the engine: a report
that inherited it by phase name would claim things about a run that never happened.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.spec import (LOG_PER_RANK, NODE_FROM_STEM, PHASE_FROM_FILENAME, EngineSpec, LogLayout,
                         ReportNotes, SanityLimits, TraceLayout)

#: `benchmark_<job>_..._PROFILE_<role>.log`, written once per profile point by bench_serving.
RE_PROFILE_LOG = re.compile(r"_PROFILE_(\w+)\.log$")


def resolve_traces(root: Path) -> dict:
    """Map sglang's timestamp-named trace directories onto the role each one profiled.

    sglang names a trace directory after the epoch second the capture started and never mentions the
    role. The authoritative mapping is in the profile-point log itself: bench_serving writes one
    ``*_PROFILE_<role>.log`` per profile point and records the ``output_dir`` it asked each worker
    of that role for, so the role's directories are named in its own log, one per node.

    Timestamps are deliberately not used to match them. The directory names come from the
    container's clock and the file mtimes from the shared filesystem's, and on this cluster the two
    are about 460 seconds apart, which is enough to attribute a whole role's traces to the other
    role.

    This used to be done by reading timestamps by eye and pasting one directory per role into a
    shell script, which is how the second node of every role went missing from three reports.
    """
    trace_dirs = {p.name: p for p in root.glob("torchprof/*") if p.is_dir()}
    role_logs = {}
    for log in sorted(root.glob("*_PROFILE_*.log")):
        m = RE_PROFILE_LOG.search(log.name)
        if m:
            role_logs[m.group(1)] = log
    if not trace_dirs or not role_logs:
        return {}

    found: dict = {}
    for role, log in role_logs.items():
        text = log.read_text(errors="ignore")
        claimed = sorted(name for name in trace_dirs if name in text)
        if claimed:
            found[role] = [trace_dirs[name] for name in claimed]

    # A role that was profiled but ends up with no trace directory means the mapping is unknown --
    # an older sglang that does not log output_dir, or artifacts copied without their logs. Guessing
    # would mislabel a whole report, so this stops instead.
    missing = set(role_logs) - set(found)
    if missing:
        raise ValueError(
            f"cannot map trace directories to roles under {root}: {sorted(missing)} got none. "
            f"Trace dirs {sorted(trace_dirs)}, profile points {sorted(role_logs)}. The "
            "profile-point log should name the output_dir of each worker; pass --torch-trace "
            "ROLE=PATH explicitly.")
    return found


SPEC = EngineSpec(
    name="sglang-disagg",
    summary="sglang PD-disaggregated serving, one log per role per node, the role being the phase",
    logs=LogLayout(
        globs=("prefill_NODE*.log", "decode_NODE*.log",
               "prefill_NODE*.log.gz", "decode_NODE*.log.gz"),
        phase_from=PHASE_FROM_FILENAME,
        node_from=NODE_FROM_STEM,
        phase_of_name=lambda stem: stem.split("_")[0],
    ),
    # `rccl/prefill_NODE0.<host>.<pid>.log`, one per server process, written when the launcher was
    # given RCCL_LOG_DIR. The role and node label are the ones the shared logs already use, so a
    # report reads the same whichever way the run was measured.
    rccl_logs=LogLayout(
        globs=("rccl/*_NODE*.log", "rccl/*_NODE*.log.gz"),
        phase_from=PHASE_FROM_FILENAME,
        node_from=NODE_FROM_STEM,
        phase_of_name=lambda stem: stem.split("_")[0],
        node_of_name=lambda stem: stem.split(".")[0],
        written_by=LOG_PER_RANK,
    ),
    traces=TraceLayout(
        dir_glob="torchprof/*",
        resolve=resolve_traces,
        rank_patterns=(re.compile(r"-TP-(\d+)"),),
    ),
    limits=SanityLimits(),
    notes=ReportNotes(
        communicator="so each node runs its own TP={nranks} replica",
        damage_cause=("A role's ranks share one stdout, so at INFO verbosity some records "
                      "overwrite each other mid-write and cannot be attributed; under a percent is "
                      "normal."),
        scope=(
            "Scope of an sglang PD-disaggregated profile, all three points are by design:",
            "the numbers above describe a *measurement* configuration, not the tuned one. TP is "
            "routed through RCCL with `--disable-custom-all-reduce` and decode runs without HIP "
            "graphs, because sglang's own all-reduce kernel and graph replay both bypass every "
            "profiler. Read throughput from a run without `PROFILE_ENABLE` instead.",
            "KV cache transfer between the prefill and decode groups goes over mooncake RDMA, "
            "never through RCCL, so the inter-node traffic that defines this topology does not "
            "appear here at all. What is measured is the intra-node TP exchange of one role.",
            "Each prefill and decode node is an independent TP replica, so per-rank figures carry "
            "over; totals across the group do not.",
        ),
        unmarked_window=("an unmarked window each: sglang's /start_profile emits no ProfilerStep "
                         "annotations, so the counts below are per capture, which held roughly one "
                         "forward pass"),
        trace_vs_log=("The two channels also cover different windows, so their absolute volumes "
                      "are not meant to agree: the RCCL log spans the whole run including "
                      "communicator setup and weight loading, while the trace covers only the few "
                      "steps the profile point requested. Compare the mix and the sizes, not the "
                      "totals."),
    ),
    fingerprints=("perf_sglang-disagg-*.csv", "proxy_NODE*.log"),
)
