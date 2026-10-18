---
name: mad-benchmark
description: Build and run a madengine benchmark for a MAD model/tag. Use when the user wants to benchmark, run, or measure a model on AMD GPUs.
argument-hint: <tag-or-model> [extra options]
disable-model-invocation: true
context: fork
agent: mad-benchmark-runner
allowed-tools: Bash(madengine discover *) Bash(madengine report *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Read Grep Glob
---

Benchmark `$ARGUMENTS` with madengine.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`. Later
tokens are options (output path, timeout, deploy). Use that tag token in every
`madengine` command.

## Pre-flight
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. Confirm the tag matches real models via `madengine discover --tags <tag> --json`
   (read-only, no GPU). If the intent is fuzzy, grep `scripts/**/models.json`
   for candidates.
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
