---
name: mad-report
description: Analyze and summarize MAD benchmark results (perf.csv / perf_entry_super.json) — summarize a run, compare two runs, or flag regressions. Use when the user asks about benchmark results, performance numbers, or comparing runs.
argument-hint: "[csv-or-json path] [compare-to path]"
context: fork
agent: mad-perf-analyst
allowed-tools: Read Grep Glob Bash(python3 *) Bash(madengine report *)
---

Analyze MAD results: $ARGUMENTS

## Task
1. Load the result file(s). Default to `perf.csv` if no path is given. Other sources:
   `perf_entry.csv`, `perf_entry_super.json`, or any `multiple_results` CSV named.
2. If two paths are provided, compare them: per-model delta and % change, flagging
   regressions vs improvements (respect each row's `metric`/unit — never compare
   across different units).
3. Otherwise summarize: models run, performance + unit, and pass/fail status.

A higher number is usually better for throughput (tokens/s, samples/s) and worse for
latency (ms, s) — state which direction you assumed. Lead with the headline (biggest
regression / overall pass rate), then a compact table.

To generate the HTML dashboard instead:
`madengine report to-html --csv-file-path <perf.csv>` (verify flags with `--help`).
