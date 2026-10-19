---
name: mad-benchmark
description: Build and run a madengine benchmark for a MAD model/tag. Use when the user wants to benchmark, run, or measure a model on AMD GPUs.
argument-hint: <tag-or-model>[,<tag-or-model>...] [extra options]
disable-model-invocation: true
context: fork
agent: mad-benchmark-runner
allowed-tools: Bash(madengine discover *) Bash(madengine report *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Read Grep Glob
---

Benchmark `$ARGUMENTS` with madengine.

The first whitespace-delimited token of `$ARGUMENTS` is the tag, or several
tags joined by commas (`bert,pyt_vllm_qwen3-8b`). Later tokens are options
(output path, timeout, deploy). Pass that token unchanged to every `madengine`
command: madengine splits it on commas, so one `madengine run --tags a,b`
runs every model the tags resolve to and writes one `perf.csv` row per model.

## Pre-flight
Claude Code runs this before the skill starts. Other agents: run it first from
the repo root and stop if it fails.
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. Confirm the tag(s) match real models via `madengine discover --tags <tag> --json`
   (read-only, no GPU). If a tag matches nothing, say which one and stop. If the
   intent is fuzzy, grep `scripts/**/models.json` for candidates.
2. Assemble `madengine run --tags <tag> --live-output`. Add as the request implies:
   - `-o <path>` to keep results separately,
   - `--timeout <s>` for long training runs (default 7200),
   - profiling/deploy via `--additional-context` (a `"slurm"`/`"k8s"` key selects
     the target; neither → local Docker).
   - Build-once/run-many split: `madengine build --tags <tag> [-r REGISTRY]` writes
     `build_manifest.json`, then `madengine run -m build_manifest.json` executes it.
3. List required env vars (e.g. `export MAD_SECRETS_HFTOKEN=...` for HF models).
4. Check for AMD GPUs (`rocm-smi`/`amd-smi`). If none are present, DO NOT run —
   print the exact command(s) to run on a GPU host and stop. If GPUs exist, run it
   detached with `bash .claude/skills/mad-common/mad-run.sh start <run-id> -- <command>`
   and repeat `mad-run.sh wait <run-id>` until it reports `succeeded` or `failed`.
   Never run `madengine run`/`build` in the foreground: a tool call ends after
   10 minutes.

Report: resolved model list, the exact command, required env vars, and where
results landed (`perf.csv` by default).

For a matrix across GPU counts, or a separate result file per tag, Claude Code
has the `mad-benchmark-sweep` workflow.

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-benchmark-runner` subagent. In Cursor, delegate
  to the `mad-benchmark-runner` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_benchmark_runner` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-benchmark-runner.md` first for the conventions it assumes.
