#!/usr/bin/env python3
"""Put two madengine runs side by side and decompose what separates them.

``collective_report.py`` describes one run. This describes the difference between two, which is
the question a backend comparison actually asks and the one the reports template calls "the one
report the tooling cannot generate". Three parts of it are mechanical and are generated here:

  * **what differed** -- diffed from what each run reported about itself, not from what the
    manifests intended, and flagged when a differing setting moves throughput on its own;
  * **where the difference comes from** -- split through ``E2E = TTFT + (OSL-1) * ITL``, an
    identity over the benchmark's own metrics, separating a per-step decode cost from a
    prefill-side one. Its residual is a check that the exported aggregates satisfy the identity,
    not a queueing term: waiting before the first token is already inside TTFT and waiting after
    it is already inside the intervals;
  * **what the exchange cost** -- the expert all-to-all per stage and per kernel variant, which
    reaches no RCCL log and is therefore invisible to every other channel.

Usage:
    ./compare_runs.py --left <run dir> --right <run dir> --out reports/<name>_comparison.md
    ./compare_runs.py --left <run> --right <run> --phase prefill --left-name MoRI

Both arms are parsed by the same code that builds a single-run report, so the two documents
cannot disagree about the same run. Run ``--help`` for the rest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collprof.compare_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
