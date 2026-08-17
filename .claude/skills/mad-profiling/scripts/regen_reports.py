#!/usr/bin/env python3
"""Rebuild every report of a campaign from a catalog file.

A campaign is a handful of jobs, each becoming one report per phase. The catalog is the only place
job ids, artifact paths and output names live, so rerunning after a change to the tooling is one
command and nobody edits a script to add a job.

    ./regen_reports.py --catalog reports/jobs.json
    ./regen_reports.py --catalog reports/jobs.json --only sgl_gptoss_2p2d_balanced
    ./regen_reports.py --catalog reports/jobs.json --list

Catalog format (see assets/jobs.example.json):

    {
      "defaults": {"parse_cache_dir": "reports/.parse-cache", "top": 20},
      "reports": [
        {"name": "...", "run_dir": "...", "out_dir": "reports/...",
         "engine": "auto", "phases": ["prefill", "decode"],
         "rocprof_dir": null, "trace_root": null, "torch_trace": {},
         "max_msg_bytes": null, "max_nranks": null, "note": "why this run is in the catalog"}
      ]
    }

Sequential on purpose: two parsers at once thrash a shared filesystem and each one slows down about
fivefold. One failing report does not stop the rest -- the failures are listed at the end and the
exit code is non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collprof.cli import main as report_main  # noqa: E402

#: Catalog keys that map onto a command line flag taking one value.
VALUE_FLAGS = {"engine": "--engine", "rocprof_dir": "--rocprof-dir", "trace_root": "--trace-root",
               "top": "--top", "max_msg_bytes": "--max-msg-bytes", "max_nranks": "--max-nranks"}


def argv_for(entry: dict, defaults: dict, root: Path) -> list:
    """Turn one catalog entry into the argument list collective_report.py would take."""
    merged = {**defaults, **entry}
    name = entry.get("name") or entry["run_dir"]

    def resolve(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else root / path)

    argv = ["--run-dir", resolve(merged["run_dir"]), "--out-dir", resolve(merged["out_dir"])]

    for key, flag in VALUE_FLAGS.items():
        value = merged.get(key)
        if value in (None, ""):
            continue
        is_path = key.endswith("_dir") or key == "trace_root"
        argv += [flag, resolve(value) if is_path else str(value)]

    if merged.get("phases"):
        argv += ["--phases"] + list(merged["phases"])
    for phase, path in (merged.get("torch_trace") or {}).items():
        argv += ["--torch-trace", f"{phase}={resolve(path)}"]
    if merged.get("no_auto_traces"):
        argv += ["--no-auto-traces"]

    cache_dir = merged.get("parse_cache_dir")
    if cache_dir:
        argv += ["--parse-cache", str(Path(resolve(cache_dir)) / f"{name}.pkl")]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, type=Path, help="JSON catalog of reports to build")
    ap.add_argument("--only", action="append", default=[],
                    help="build only these catalog entries, by name; repeatable")
    ap.add_argument("--list", action="store_true", help="list the catalog and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--root", type=Path,
                    help="base for relative paths in the catalog (default: the catalog's directory "
                         "parent, i.e. the run directory the catalog lives in)")
    args = ap.parse_args()

    catalog = json.loads(args.catalog.read_text())
    defaults = catalog.get("defaults", {})
    entries = catalog.get("reports", [])
    root = args.root or args.catalog.resolve().parent.parent

    if args.only:
        entries = [e for e in entries if e.get("name") in args.only]
        missing = set(args.only) - {e.get("name") for e in entries}
        if missing:
            raise SystemExit(f"no catalog entry named {', '.join(sorted(missing))}")

    if args.list:
        for entry in catalog.get("reports", []):
            print(f"{entry.get('name', '?'):<32} {entry['run_dir']}")
            if entry.get("note"):
                print(f"{'':<32} {entry['note']}")
        return 0

    failed = []
    for entry in entries:
        name = entry.get("name") or entry["run_dir"]
        argv = argv_for(entry, defaults, root)
        print(f"\n=== {name} ===\ncollective_report.py {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        try:
            report_main(argv)
        except SystemExit as exc:
            if exc.code:
                failed.append((name, str(exc)))
        except Exception:
            traceback.print_exc()
            failed.append((name, "exception, see traceback above"))

    if failed:
        print("\nfailed:")
        for name, why in failed:
            print(f"  {name}: {why}")
        return 1
    print(f"\n{len(entries)} report set(s) rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
