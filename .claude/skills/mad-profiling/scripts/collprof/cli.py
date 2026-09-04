"""Command line for one job: parse its logs, then write one report per phase.

    collective_report.py --run-dir <job dir> --out-dir reports/<name>

Everything else has a default that works. The engine is detected from the log layout, torch traces
are found and matched to phases by the engine, and the report says which engine it used and why.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import sys
from functools import partial
from pathlib import Path

from . import engines
from .core.cache import PARSE_VERSION, ParseCache, file_signature
from .core.phase import Phase
from .core.rccl_log import discover_logs, parse_run, scan_run_config
from .core.rdma_counters import parse_counters
from .core.report import ReportContext, emit_phase
from .core.rocprof import parse_rocprof
from .core.runconfig import diff_configs, merge_nodes
from .core.spec import LOG_PER_RANK, EngineSpec
from .core.torch_trace import parse_traces, trace_files
from .core.units import fmt_bytes


#: The script that owns the arguments below, and the only name a reader can rerun them with. A
#: caller passing ``argv`` -- regen_reports.py walking a catalog -- runs under its own name, whose
#: parser knows none of these flags.
ENTRY_POINT = "collective_report.py"


def recorded_command(argv: list | None) -> str:
    """The command the report claims a reader can rerun to reproduce it."""
    if argv is not None:
        return shlex.join([ENTRY_POINT] + argv)
    return shlex.join([Path(sys.argv[0]).name] + sys.argv[1:])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="job directory holding the per-node logs")
    ap.add_argument("--out-dir", type=Path,
                    help="output root; one directory per phase, named <out-dir>_<phase>")
    ap.add_argument("--engine", default="auto",
                    help="engine that produced the run, or auto to detect it from the log layout "
                         f"(known: {', '.join(sorted(engines.REGISTRY))})")
    ap.add_argument("--list-engines", action="store_true", help="print the known engines and exit")
    ap.add_argument("--phases", "--dtypes", dest="phases", nargs="*",
                    help="restrict to these phases: datatypes for training (BF16, FP8), roles for "
                         "serving (prefill, decode). Default: every phase that logged collectives")
    ap.add_argument("--rocprof-dir", type=Path,
                    help="directory with rocprofv3 <pid>_rccl_api_stats.csv / _domain_stats.csv")
    ap.add_argument("--rccl-dir", type=Path,
                    help="where NCCL_DEBUG_FILE wrote one RCCL log per process (default: the run "
                         "directory). Serving keeps them beside the server logs, training beside "
                         "the traces, since its run directory is madengine's and not writable "
                         "from the container")
    ap.add_argument("--trace-root", type=Path,
                    help="where to look for torch profiler traces (default: the run directory). "
                         "The engine maps what it finds to phases")
    ap.add_argument("--torch-trace", action="append", default=[], metavar="PHASE=PATH",
                    help="pin a phase's traces explicitly, repeatable. Overrides what the engine "
                         "resolves for that phase")
    ap.add_argument("--no-auto-traces", action="store_true",
                    help="do not look for traces; use only --torch-trace")
    ap.add_argument("--top", type=int, default=20, help="rows in the message-size tables")
    ap.add_argument("--parse-cache", type=Path,
                    help="reuse the parse stored here, or write it when absent or stale. Turns a "
                         "rerun for wording changes from tens of minutes into seconds")
    ap.add_argument("--max-msg-bytes", type=int,
                    help="override the engine's cap on a single message, in bytes. Raise it when a "
                         "run legitimately moves larger messages than the default expects")
    ap.add_argument("--max-nranks", type=int,
                    help="override the engine's cap on communicator width")
    ap.add_argument("--compare-config", type=Path, metavar="RUN_DIR",
                    help="another run to diff this one's configuration against, per phase. The "
                         "settings that differ decide a comparison more often than the "
                         "communication numbers do, and the report flags the ones that move "
                         "throughput on their own. Only that run's startup lines are read")
    ap.add_argument("--compare-noise", action="store_true",
                    help="keep ports, hosts, seeds and the like in the configuration diff; they "
                         "differ on any two runs and are dropped by default. It does not affect "
                         "model or tokenizer paths, which are always compared by their final "
                         "component so a remount is quiet and a different model is not")
    return ap


def apply_limit_overrides(spec: EngineSpec, args) -> EngineSpec:
    """A spec with the command line's sanity bounds, if any were given."""
    changes = {}
    if args.max_msg_bytes:
        changes["max_msg_bytes"] = args.max_msg_bytes
    if args.max_nranks:
        changes["max_nranks"] = args.max_nranks
    if not changes:
        return spec
    return dataclasses.replace(spec, limits=dataclasses.replace(spec.limits, **changes))


#: An incomplete parse is not cached: stored under the files' unchanged signature it would be
#: reused forever. Recomputed each time until one pass reads them all.
def _trace_is_complete(parsed: dict) -> bool:
    return not parsed.get("truncated")


def no_config_reason(name: str, reference: dict, reference_nodes: dict) -> str:
    """Why a requested `--compare-config` produced nothing for this phase.

    Three outcomes, not two. `scan_run_config` returns the nodes it discovered for a phase
    separately from the configurations it parsed, so a reference that ran the phase and stated
    nothing readable is distinguishable from one that never ran it -- and only the first is worth
    rereading the reference's logs over.
    """
    if name in reference:
        return "this run states none, so there is nothing to compare it against."
    if name in reference_nodes:
        return (f"the reference ran that phase on {len(reference_nodes[name])} node(s) but none "
                "of them stated a configuration this tooling could read.")
    return "that phase is absent from the reference run."


def _counters_of(counters: dict, phase: str, spec: EngineSpec) -> dict:
    """This phase's share of a run-wide counter map, by the engine's own rule."""
    return {node: series for node, series in counters.items()
            if spec.counters.phase_of_node(node, phase)}


def load_counters(run_dir: Path, spec: EngineSpec) -> dict:
    """The RDMA counter samples of a run, or nothing when it did not take any.

    Not cached: the files are kilobytes against the gigabytes the log parse handles, so a stale
    entry would cost more than the reparse.
    """
    paths = [p for pattern in spec.counters.globs for p in sorted(run_dir.glob(pattern))]
    if not paths:
        return {}
    series = parse_counters(paths, spec.counters)
    sampled = sum(1 for s in series.values() if s.samples > 1)
    print(f"RDMA counters: {len(paths)} file(s), {sampled} with a window to measure over")
    return series


def resolve_traces(args, spec: EngineSpec) -> dict:
    """Trace paths per phase: what the engine finds, with explicit --torch-trace winning.

    A discovered directory that holds no traces is reported and dropped rather than raised on. It
    happens for real -- a role's idle replica had its capture produce an empty directory -- and it
    is not a reason to lose a whole job's reports. A path given by hand stays strict: it was asked
    for.
    """
    found: dict = {}
    skipped: dict = {}
    root = args.trace_root or args.run_dir
    if not args.no_auto_traces and spec.traces.resolve and root.exists():
        for phase, paths in sorted(spec.traces.resolve(root).items()):
            usable, empty = [], []
            for path in paths:
                (usable if holds_traces(path, spec.traces) else empty).append(path)
            if empty:
                skipped[phase] = tuple(p.name for p in empty)
                print(f"warning: {phase} trace directories with no trace files, skipped: "
                      + ", ".join(skipped[phase]))
            if usable:
                found[phase] = usable
                print(f"traces for {phase}: {', '.join(p.name for p in usable)}")
            else:
                print(f"warning: no usable traces for {phase}; its report will be volume-only")

    for item in args.torch_trace:
        phase, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--torch-trace needs PHASE=PATH, got {item!r}")
        found[phase] = [Path(raw)]
        skipped.pop(phase, None)
    return found, skipped


def holds_traces(path: Path, layout) -> bool:
    try:
        return bool(trace_files(path, layout))
    except FileNotFoundError:
        return False


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.list_engines:
        for name, spec in sorted(engines.REGISTRY.items()):
            print(f"{name:16} {spec.summary}")
            print(f"{'':16} logs: {', '.join(spec.logs.globs)}")
        return
    if not args.run_dir or not args.out_dir:
        raise SystemExit("--run-dir and --out-dir are required")

    if args.engine == "auto":
        spec, reason = engines.detect(args.run_dir)
        print(f"engine {spec.name}: {reason}")
    else:
        spec = engines.get(args.engine)
        reason = "forced with --engine"
    spec = apply_limit_overrides(spec, args)

    cache = ParseCache(args.parse_cache)
    found = discover_logs(args.run_dir, spec, args.rccl_dir)
    per_rank = [log for log, _n, _p, layout in found if layout.written_by == LOG_PER_RANK]
    if per_rank:
        print(f"{len(per_rank)} per-rank RCCL log(s) from NCCL_DEBUG_FILE under "
              f"{args.rccl_dir or args.run_dir}")
    phases = cache.get(
        f"logs of {args.run_dir}",
        file_signature([log for log, _n, _p, _l in found])
        + [spec.name, spec.limits.max_msg_bytes, spec.limits.max_nranks],
        partial(parse_run, args.run_dir, spec, args.rccl_dir),
        encode=lambda ps: {name: p.to_state() for name, p in ps.items()},
        decode=lambda st: {name: Phase.from_state(s) for name, s in st.items()},
    )

    trace_paths, empty_trace_dirs = resolve_traces(args, spec)
    traces = {
        phase: cache.get(
            f"traces of {[str(p) for p in paths]}",
            # The engine belongs in the key: a trace parse carries this engine's a2a
            # classification and rank patterns, so a cache filled under one `--engine` and read
            # under another answers the wrong question.
            file_signature([f for p in paths for f in trace_files(Path(p), spec.traces)])
            + [spec.name],
            partial(parse_traces, paths, spec.traces, spec.a2a),
            keep=_trace_is_complete)
        for phase, paths in trace_paths.items()
    }
    rocprof = parse_rocprof(args.rocprof_dir) if args.rocprof_dir else None
    counters = load_counters(args.run_dir, spec)
    cache.flush()

    reference: dict = {}
    reference_nodes: dict = {}
    if args.compare_config:
        # No --rccl-dir: that path belongs to the run being reported on, so passing it here would
        # discover this job's logs while claiming to read the other's.
        reference, reference_nodes = scan_run_config(args.compare_config, spec)
        if reference:
            print(f"configuration to compare against, from {args.compare_config}: "
                  f"{', '.join(sorted(reference))}")
        else:
            print(f"warning: no configuration found under {args.compare_config}; the engine reads "
                  "it from a startup line the reference run may predate")

    ctx = ReportContext(spec=spec, run_dir=args.run_dir, top=args.top, rocprof=rocprof,
                        rocprof_dir=args.rocprof_dir, counters=counters,
                        command=recorded_command(argv),
                        parse_version=PARSE_VERSION)

    produced = []
    for name in (args.phases or sorted(phases)):
        phase = phases.get(name)
        if phase is None:
            print(f"skip {name}: no such phase in {args.run_dir} "
                  f"(found: {', '.join(sorted(phases)) or 'none'})")
            continue
        # Collectives are one channel of five: an unprofiled run logs no COLL rows while still
        # carrying step times, configuration and adapter counters. Skipped only when all are empty.
        phase_counts = _counters_of(counters, name, spec)
        if not (phase.sizes or phase.steps or phase.config or traces.get(name) or phase_counts):
            print(f"skip {name}: nothing to report -- no collectives, no step records, no "
                  "configuration, no traces and no adapter counters")
            continue
        if not phase.sizes:
            print(f"note {name}: no collectives logged (NCCL_DEBUG_SUBSYS without COLL, or an "
                  "early failure); the volume sections will be absent and the rest will not")
        out = args.out_dir.parent / f"{args.out_dir.name}_{name}"
        ctx.torch_trace = traces.get(name)
        ctx.trace_dirs = tuple(Path(p).name for p in trace_paths.get(name, ()))
        ctx.empty_trace_dirs = empty_trace_dirs.get(name, ())
        ctx.counters = phase_counts
        # Compared phase by phase: prefill and decode run different configurations, so diffing a
        # decode against a reference prefill would report every setting as changed.
        if args.compare_config and name in reference and phase.config:
            here, here_split = merge_nodes(phase.config, phase.config_nodes)
            there, there_split = merge_nodes(reference[name],
                                             reference_nodes.get(name))
            ctx.config_diff = diff_configs(here, there, include_noise=args.compare_noise,
                                           perf_relevant=spec.run_config.perf_relevant,
                                           noise=spec.run_config.noise,
                                           path_valued=spec.run_config.path_valued)
            ctx.config_diff_source = str(args.compare_config)
            ctx.config_diff_unavailable = ""
            # A key its own nodes disagree on is absent from the merged configuration, so the pair
            # would read as comparable on it. Carried per side so the report names the split run.
            ctx.config_diff_split = {"this run": here_split, str(args.compare_config): there_split}
        else:
            # A requested comparison that could not be made must say so in the saved report, not
            # only on the console where nobody rereads it.
            ctx.config_diff = None
            ctx.config_diff_unavailable = (
                f"`--compare-config {args.compare_config}` was requested but no configuration "
                f"could be read for phase {name!r}: "
                + no_config_reason(name, reference, reference_nodes)
                if args.compare_config else "")
        emit_phase(phase, out, ctx)
        totals = phase.collective_totals()
        reps = max(len(phase.active_ranks(spec.limits.idle_rank_fraction)), 1)
        produced.append((name, out, round(sum(r["calls"] for r in totals.values()) / reps),
                         sum(r["bytes"] for r in totals.values()) / reps))

    if not produced:
        raise SystemExit("no phase produced a report")
    print(f"{'phase':<10}{'calls/rank':>12}{'volume/rank':>14}   report")
    for name, out, calls, nbytes in produced:
        print(f"{name:<10}{calls:>12}{fmt_bytes(nbytes):>14}   {out / 'report.md'}")
