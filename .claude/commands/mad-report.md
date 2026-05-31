---
description: Analyze and summarize MAD benchmark results (perf.csv / perf_entry_super.json)
argument-hint: [csv-or-json path] [compare-to path]
---

Analyze MAD results: $ARGUMENTS

Use the `mad-perf-analyst` subagent (read-only). It should:
1. Load the result file(s). Default to `perf.csv` if no path is given.
2. If two paths are provided, compare them: per-model delta and % change,
   flagging regressions vs improvements (respecting each row's `metric`/unit).
3. Otherwise summarize: models run, performance + unit, and pass/fail status.

Lead with the headline finding, then a compact table. To generate the HTML
dashboard instead, run `madengine report to-html --csv-file-path <perf.csv>`
(verify flags with `madengine report to-html --help`).
