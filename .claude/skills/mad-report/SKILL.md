---
name: mad-report
description: Analyze and summarize MAD benchmark results (perf.csv / perf_entry_super.json) — summarize a run, compare two runs, or flag regressions. Use when the user asks about benchmark results, performance numbers, or comparing runs.
argument-hint: "[csv-or-json path] [compare-to path]"
context: fork
agent: mad-perf-analyst
background: false
allowed-tools: Read Grep Glob Bash(madengine report *)
---

Analyze MAD results: $ARGUMENTS

## Task
1. Load the result file(s). Default to `perf.csv` if no path is given. Other sources:
   `perf_entry.csv`, `perf_entry_super.json`, or any `multiple_results` CSV named.
2. If two paths are provided, compare them: per-model delta and % change, flagging
   regressions vs improvements (respect each row's `metric`/unit — never compare
   across different units). Match models by the last `/` component of `model`:
   madengine names per-directory models `dir/name` (e.g. `vllm/pyt_vllm_qwen3-8b`),
   while CSVs from before the per-directory registries hold the bare name.
3. Otherwise summarize: models run, performance + unit, and pass/fail status.

A higher number is usually better for throughput (tokens/s, samples/s) and worse for
latency (ms, s) — state which direction you assumed. Lead with the headline (biggest
regression / overall pass rate), then a compact table.

To generate the HTML dashboard instead:
`madengine report to-html --csv-file-path <perf.csv>` (verify flags with `--help`).

Operator and kernel reports live in `tracelens_output/tracelens_summary.json`.
Summarize that file when the user asks about a TraceLens report. Diff two
operator workbooks with
`madengine report tracelens-compare <before> <after> --names before --names after`.
A profiled `perf.csv` row is not a clean score.

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-perf-analyst` subagent. In Cursor, delegate
  to the `mad-perf-analyst` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_perf_analyst` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-perf-analyst.md` first for the conventions it assumes.
