#!/usr/bin/env python3
"""Compress the logs of a finished madengine run, in place.

A profiled run leaves the largest artifact it will ever have as plain text: with one RCCL log per
process nothing is filtered away, which is about 8 GB for a 4-node serving job. That text repeats,
so gzip takes it under 100 MB, and every parser in this skill reads ``.log.gz`` as readily as
``.log`` -- there is no reason to keep the plain copy once the job is over.

Usage:
    ./compress_logs.py --run-dir <job dir>                       # serving: logs live in the run dir
    ./compress_logs.py --run-dir <job dir> --rccl-dir <prof>/rccl # training: per-rank files elsewhere
    ./compress_logs.py --run-dir <job dir> --dry-run

The engine's log globs decide what counts as a log, so this compresses exactly what
collective_report.py would read and nothing else.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collprof.compress_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
