"""Command line for an A/B: two runs in, one decomposed comparison out.

    compare_runs.py --left <run> --right <run> --out reports/<name>_comparison.md

Each side is parsed the way ``collective_report.py`` parses one, so the two documents cannot
disagree about the same run. The engine is detected from the left run's layout and the right one is
required to match: comparing a training run against a serving one is a mistake worth refusing
rather than rendering.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from functools import partial
from pathlib import Path

from . import engines
from .core.cache import PARSE_VERSION, ParseCache, file_signature
from .core.compare import build, make_arm, tables
from .core.phase import Phase
from .core.rccl_log import discover_logs, parse_run
from .core.rdma_counters import parse_counters
from .core.report import CsvSink
from .core.torch_trace import parse_traces, trace_files
from .core.workbook import write_workbook

ENTRY_POINT = "compare_runs.py"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True, help="run directory of the first arm")
    ap.add_argument("--right", type=Path, required=True, help="run directory of the second arm")
    ap.add_argument("--left-name", default="", help="label for the first arm (default: its dir)")
    ap.add_argument("--right-name", default="", help="label for the second arm")
    ap.add_argument("--phase", default="decode",
                    help="phase to compare, e.g. decode or prefill (default: decode)")
    # Mutually exclusive rather than a precedence rule, so a command naming both is refused
    # instead of silently ignoring one.
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--out", type=Path, help="markdown to write (default: stdout)")
    out.add_argument("--out-dir", type=Path,
                     help="directory for comparison.md, one CSV per table and comparison.xlsx; "
                          "the workbook carries the prose as its first sheet, so the caveats "
                          "travel with the numbers instead of being left behind in the markdown")
    ap.add_argument("--engine", default="auto", help="engine, or auto to detect it")
    ap.add_argument("--no-traces", action="store_true",
                    help="skip trace parsing, which drops the expert-exchange section")
    ap.add_argument("--counters-same-workload", action="store_true",
                    help="assert that the two runs served the same requests, which the adapter "
                         "counters need before their whole-window totals can be compared. The "
                         "perf CSV carries one row per point and metric and no request count, so "
                         "a different BENCHMARK_ITR or a retry is invisible here; without this "
                         "flag the fabric table is withheld rather than guessed at")
    ap.add_argument("--parse-cache", type=Path, help="reuse or write the parse stored here")
    return ap


#: An incomplete parse is not cached: stored under the files' unchanged signature it would be
#: reused forever. Recomputed each time until one pass reads them all.
def _trace_is_complete(parsed: dict) -> bool:
    return not parsed.get("truncated")


#: What the previous run of this command wrote into `--out-dir`, so this one knows what is its to
#: delete: the directory belongs to the caller and may hold files this command never wrote.
MANIFEST = ".comparison_manifest"


def read_manifest(out_dir: Path) -> set:
    """Names the previous run recorded, or an empty set when there was none."""
    try:
        return {line.strip() for line in (out_dir / MANIFEST).read_text().splitlines()
                if line.strip()}
    except OSError:
        return set()


def write_manifest(out_dir: Path, names: list) -> None:
    (out_dir / MANIFEST).write_text("\n".join(sorted(names)) + "\n")


def load_counters(run_dir: Path, spec) -> dict:
    """The RDMA counter samples of one arm, keyed by the node label the report already uses."""
    paths = [p for pattern in spec.counters.globs for p in sorted(run_dir.glob(pattern))]
    return parse_counters(paths, spec.counters) if paths else {}


def _counters_of(counters: dict, phase: str, spec) -> dict:
    """The counter samples belonging to one phase's nodes, by the engine's own rule.

    The naming convention is the launcher's, which is why `CounterLayout` carries the predicate.
    Matched on the label, not the phase's node set: a node that logged no collectives still has
    an adapter carrying this run's traffic.
    """
    return {node: series for node, series in counters.items()
            if spec.counters.phase_of_node(node, phase)}


def parse_side(run_dir: Path, spec, cache: ParseCache, want_traces: bool) -> tuple:
    """``(phases, traces, empty_dirs)`` for one arm, through the same cache as a single-run report.

    ``empty_dirs`` are the capture directories that held no trace file. Dropped from the parse
    either way, but not silently: with one directory per replica, an arm whose second replica
    captured nothing would otherwise read as fully covered against an arm with two.
    """
    found = discover_logs(run_dir, spec, run_dir)
    phases = cache.get(
        f"logs of {run_dir}",
        file_signature([log for log, _n, _p, _l in found])
        + [spec.name, spec.limits.max_msg_bytes, spec.limits.max_nranks],
        partial(parse_run, run_dir, spec, run_dir),
        encode=lambda ps: {name: p.to_state() for name, p in ps.items()},
        decode=lambda st: {name: Phase.from_state(s) for name, s in st.items()},
    )
    traces: dict = {}
    empty_dirs: dict = {}
    if want_traces and spec.traces.resolve and run_dir.exists():
        for phase, paths in spec.traces.resolve(run_dir).items():
            usable = [p for p in paths if _holds(p, spec)]
            empty = [p.name for p in paths if p not in usable]
            if empty:
                empty_dirs[phase] = tuple(empty)
                print(f"warning: {run_dir}: {phase} capture directories with no trace files: "
                      + ", ".join(empty))
            if usable:
                traces[phase] = cache.get(
                    f"traces of {[str(p) for p in usable]}",
                    # With the engine, as in `cli`: the cached value holds this engine's stage
                    # and variant classification, which the files alone do not identify.
                    file_signature([f for p in usable
                                    for f in trace_files(Path(p), spec.traces)]) + [spec.name],
                    partial(parse_traces, usable, spec.traces, spec.a2a),
                    keep=_trace_is_complete)
    return phases, traces, empty_dirs


def _holds(path: Path, spec) -> bool:
    try:
        return bool(trace_files(path, spec.traces))
    except FileNotFoundError:
        return False


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.engine == "auto":
        spec, reason = engines.detect(args.left)
        right_spec, _ = engines.detect(args.right)
        if right_spec.name != spec.name:
            raise SystemExit(f"the arms look like different engines ({spec.name} and "
                             f"{right_spec.name}); pass --engine to force one")
        print(f"engine {spec.name}: {reason}")
    else:
        spec = engines.get(args.engine)

    cache = ParseCache(args.parse_cache)
    left_phases, left_traces, left_empty = parse_side(args.left, spec, cache, not args.no_traces)
    right_phases, right_traces, right_empty = parse_side(args.right, spec, cache,
                                                         not args.no_traces)
    left_counters = load_counters(args.left, spec)
    right_counters = load_counters(args.right, spec)
    cache.flush()

    for name, phases in (("--left", left_phases), ("--right", right_phases)):
        if args.phase not in phases:
            raise SystemExit(f"{name} has no phase {args.phase!r} "
                             f"(found: {', '.join(sorted(phases)) or 'none'})")

    # The arms are keyed by label in several places (step medians, variant sets, workbook
    # headers), so two arms with one label overwrite each other and differences go missing.
    left_name = args.left_name or args.left.name
    right_name = args.right_name or args.right.name
    if left_name == right_name:
        raise SystemExit(f"both arms resolve to the label {left_name!r}; pass --left-name and "
                         "--right-name so the two can be told apart in the report")

    command = shlex.join([ENTRY_POINT] + (argv if argv is not None else sys.argv[1:]))
    left = make_arm(left_name, args.left, left_phases, left_traces, args.phase, spec,
                    empty_trace_dirs=left_empty.get(args.phase, ()),
                    counters=_counters_of(left_counters, args.phase, spec),
                    counters_comparable=args.counters_same_workload)
    right = make_arm(right_name, args.right, right_phases, right_traces, args.phase, spec,
                     empty_trace_dirs=right_empty.get(args.phase, ()),
                     counters=_counters_of(right_counters, args.phase, spec),
                     counters_comparable=args.counters_same_workload)
    text = build(left, right, spec, args.phase, command)

    if args.out_dir:
        # Writing into a run directory would put the comparison beside the logs it read and, with
        # the stale-CSV sweep below, delete the perf CSV it parsed. Refused rather than handled.
        for side, run in (("--left", args.left), ("--right", args.right)):
            if args.out_dir.resolve() == run.resolve():
                raise SystemExit(f"--out-dir is {side}'s run directory; write the comparison "
                                 "somewhere of its own so it cannot overwrite what it read")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        produced = tables(left, right, spec)
        # Nothing is overwritten that this command did not write: the directory is caller-chosen
        # and shareable, so the manifest decides ownership for writes as it does for deletions.
        previous = read_manifest(args.out_dir)
        planned = (["comparison.md", "comparison.xlsx"]
                   + [f"{name}.csv" for name in sorted(produced)])
        clashes = [name for name in planned
                   if name not in previous and (args.out_dir / name).exists()]
        if clashes:
            raise SystemExit(
                f"--out-dir {args.out_dir} already holds {', '.join(clashes)}, and this command "
                "did not write them: no manifest of a previous run claims them. Choose another "
                "directory, or move those files aside; refusing rather than overwriting work "
                "whose provenance is unknown.")
        (args.out_dir / "comparison.md").write_text(text)
        sink = CsvSink(args.out_dir)
        for name, (header, rows) in sorted(produced.items()):
            sink.write(f"{name}.csv", header, rows)
        # Stale CSVs from an earlier run would come back as sheets dated differently from every
        # other number in the workbook. What may be deleted is decided by the manifest that run
        # left behind, not by the name: a familiar name is no proof this command wrote the file.
        sink.drop_stale(only=previous)
        # Drop any workbook from a previous run first: if openpyxl is missing, write_workbook
        # returns None and the stale file would survive beside CSVs it no longer matches.
        stale_book = args.out_dir / "comparison.xlsx"
        if stale_book.name in previous:
            stale_book.unlink(missing_ok=True)
        # Only this run's tables: `write_workbook` otherwise collects every CSV beside it, and
        # this directory is explicitly allowed to hold unrelated files.
        book = write_workbook(args.out_dir, text.splitlines(), filename="comparison.xlsx",
                              only={f"{name}.csv" for name in produced})
        # This run's artifacts, not the directory listing: the directory may hold unrelated files
        # and reporting them as this command's output would be wrong.
        mine = (["comparison.md"] + [f"{name}.csv" for name in produced]
                + (["comparison.xlsx"] if book else []))
        write_manifest(args.out_dir, mine)
        written = ", ".join(sorted(mine))
        print(f"comparison written to {args.out_dir}: {written} (parser version {PARSE_VERSION})")
        if book is None:
            print("the workbook was skipped; comparison.md and the CSVs hold every number")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"comparison written to {args.out} (parser version {PARSE_VERSION})")
    else:
        print(text)
