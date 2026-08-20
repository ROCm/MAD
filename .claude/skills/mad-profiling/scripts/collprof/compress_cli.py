"""Command line for reclaiming the disk a measured run took.

    compress_logs.py --run-dir <job dir> [--rccl-dir <dir>]

Compresses exactly the logs the report tooling would read -- the engine's own globs decide -- and
leaves everything else in the run directory alone. Run it after the job has finished: a log that is
still being appended to would lose whatever arrives after the copy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import engines
from .core.compress import DEFAULT_JOBS, JOBS_CAP, compress_all, plan
from .core.rccl_log import discover_logs
from .core.units import fmt_bytes


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="job directory holding the per-node logs")
    ap.add_argument("--rccl-dir", type=Path,
                    help="where NCCL_DEBUG_FILE wrote one RCCL log per process, when that is not "
                         "the run directory (the training case)")
    ap.add_argument("--engine", default="auto",
                    help="engine that produced the run, or auto to detect it from the log layout "
                         f"(known: {', '.join(sorted(engines.REGISTRY))})")
    ap.add_argument("--level", type=int, default=6, choices=range(1, 10), metavar="1-9",
                    help="gzip level; 6 already gets most of the ratio on RCCL text")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"files compressed in parallel (default {DEFAULT_JOBS} here: one per core, "
                         f"capped at {JOBS_CAP})")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be compressed and touch nothing")
    return ap


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.engine == "auto":
        spec, reason = engines.detect(args.run_dir)
        print(f"engine {spec.name}: {reason}")
    else:
        spec = engines.get(args.engine)

    found = discover_logs(args.run_dir, spec, args.rccl_dir)
    todo, doubled = plan([log for log, _n, _p, _l in found])

    for log in doubled:
        print(f"warning: {log} sits beside its own .gz, so the parser would read those records "
              "twice; leaving both alone until one is removed")

    if not todo:
        print("nothing to compress" + (f", {len(doubled)} pair(s) to resolve by hand"
                                       if doubled else ""))
        if doubled:
            raise SystemExit(1)
        return

    total = sum(log.stat().st_size for log in todo)
    print(f"{len(todo)} log(s) to compress, {fmt_bytes(total)}")
    if args.dry_run:
        for log in todo:
            print(f"  {log} ({fmt_bytes(log.stat().st_size)})")
        return

    started = time.monotonic()
    results = compress_all(todo, args.level, args.jobs)
    elapsed = time.monotonic() - started

    failed = [r for r in results if not r.ok]
    before = sum(r.before for r in results if r.ok)
    after = sum(r.after for r in results if r.ok)
    for r in failed:
        print(f"error: {r.path}: {r.error} (left uncompressed)")
    if before:
        print(f"compressed {len(results) - len(failed)} log(s): {fmt_bytes(before)} -> "
              f"{fmt_bytes(after)} ({before / max(after, 1):.0f}x) in {elapsed:.1f} s")
    if failed or doubled:
        raise SystemExit(1)
