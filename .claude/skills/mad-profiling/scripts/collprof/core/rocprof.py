"""Aggregating rocprofv3 stats CSVs: the only trustworthy durations in the report.

``rocprofv3 --rccl-trace`` writes one set of CSVs per profiled process. They carry per-API and
per-kernel durations but no message sizes, which is the mirror image of what the RCCL log carries.

Note what this cannot become: the stats cover a whole process, initialisation and every phase
included, while volumes are per phase, so the two are never divided into a bandwidth. Serving runs
in particular have produced no CSVs at all -- see references/interpretation.md.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def parse_rocprof(rocprof_dir: Path) -> dict:
    """Aggregate per-PID RCCL API, domain and kernel stats under a directory."""
    rccl: dict = defaultdict(lambda: {"calls": 0, "ns": 0, "pids": 0, "max_ns": 0})
    domain: dict = defaultdict(lambda: {"calls": 0, "ns": 0})
    comm_kernels: dict = defaultdict(lambda: {"calls": 0, "ns": 0, "pids": 0})
    kernel_ns = 0

    def load(pattern: str, sink, track_pids: bool):
        for path in sorted(rocprof_dir.rglob(pattern)):
            with path.open() as fh:
                for row in csv.DictReader(fh):
                    entry = sink[row["Name"]]
                    entry["calls"] += int(row["Calls"])
                    entry["ns"] += int(row["TotalDurationNs"])
                    if track_pids:
                        entry["pids"] += 1
                        entry["max_ns"] = max(entry["max_ns"], int(row.get("MaxNs", 0)))

    load("*_rccl_api_stats.csv", rccl, True)
    load("*_domain_stats.csv", domain, False)

    for path in sorted(rocprof_dir.rglob("*_kernel_stats.csv")):
        with path.open() as fh:
            for row in csv.DictReader(fh):
                name, ns = row["Name"], int(row["TotalDurationNs"])
                kernel_ns += ns
                if "nccl" in name.lower() or "rccl" in name.lower():
                    entry = comm_kernels[name.split("(")[0]]
                    entry["calls"] += int(row["Calls"])
                    entry["ns"] += ns
                    entry["pids"] += 1

    return {"rccl": dict(rccl), "domain": dict(domain),
            "comm_kernels": dict(comm_kernels), "kernel_ns": kernel_ns}
