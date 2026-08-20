#!/usr/bin/env python3
"""Build a collective-communication profile for one madengine run.

The report joins channels that each hold only half of the picture:

  * RCCL debug lines in the per-node logs (needs ``NCCL_DEBUG=INFO`` and a ``NCCL_DEBUG_SUBSYS``
    including ``COLL``) carry collective name, element count, datatype and nranks -- message
    sizes -- for the whole run, but no durations.
  * torch profiler traces carry a message size, a dtype and a process group per individual
    collective, for the few steps a profile point covered.
  * ``rocprofv3 --rccl-trace`` CSVs carry per-API and per-kernel durations, but no message sizes.

Usage:
    ./collective_report.py --run-dir <job dir> --out-dir reports/<name>
    ./collective_report.py --run-dir <job dir> --out-dir reports/<name> --phases prefill decode
    ./collective_report.py --list-engines

The engine is detected from the log layout; pass ``--engine`` to force it. Run ``--help`` for the
rest, and see the skill's SKILL.md for what the numbers mean.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collprof.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
