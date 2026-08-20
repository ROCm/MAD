"""Collective-communication profiles for madengine runs.

Two layers, and the boundary between them is the point of the package:

* ``collprof.core`` knows RCCL logs, torch profiler traces, rocprofv3 stats and how to turn them
  into a report. It contains no engine names and no claims that hold for only one workload.
* ``collprof.engines`` holds one module per engine: where its logs live, where a phase name comes
  from, which metrics it prints, how its traces are named, and what a reader must be told about its
  numbers. Adding an engine touches nothing else.

Entry points are ``scripts/collective_report.py`` (one job) and ``scripts/regen_reports.py`` (a
catalog of jobs). See the skill's SKILL.md and references/engines.md.
"""

__all__ = ["core", "engines"]
