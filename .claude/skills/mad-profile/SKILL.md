---
name: mad-profile
description: Profile a MAD training or inference model at one depth — operator (TraceLens), dispatch (rocm-trace-lite), or kernel (rocprofv3). Use when the user wants to profile or trace a MAD model.
argument-hint: "<tag> [operator|dispatch|kernel] [tool]"
disable-model-invocation: true
context: fork
agent: mad-benchmark-runner
allowed-tools: Bash(madengine discover *) Bash(madengine report *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Read Grep Glob
---

Profile `$ARGUMENTS`.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`. The next
token, when it is `operator`, `dispatch`, or `kernel`, selects the depth.
Anything after that is an explicit madengine tool name. Use the tag token in
every `madengine` command.

## Pre-flight
Claude Code runs this before the skill starts. Other agents: run it first from
the repo root and stop if it fails.
```!
bash .claude/skills/mad-common/preflight.sh
```

## Depths
madengine is the harness. One depth is one run. Never put `rocm_trace_lite` and
any `rocprofv3_*` tool in the same `tools` array.

1. `operator` — PyTorch op and module breakdown. Tools:
   `torch_profiler_dynolog` plus `tracelens` with `TRACELENS_MODE=pytorch`.
   Training keeps `TORCH_PROFILE_ITERATIONS=5`. Inference (vLLM, SGLang, serving)
   sets `TORCH_PROFILE_ITERATIONS=0` and `TORCH_PROFILE_DURATION_MS=500`, because
   there is no `optimizer.step()`. Skip this depth for a non-PyTorch model and
   say why. After the run, read `tracelens_output/tracelens_summary.json`, or
   run `madengine report tracelens` on the host.
2. `dispatch` — kernel dispatch timeline for any HIP workload, training or
   inference. Tool: `rocm_trace_lite` only. Read `rocm_trace_lite_output/`.
3. `kernel` — hardware counters. Default tool `rocprofv3_compute`. Use
   `rocprofv3_memory` or `rocprofv3_communication` when the user names them.
   Read `rocprof_output/`.

When the user names a tool and no depth, map it: `rocm_trace_lite` /
`rocm_trace_lite_default` / `rpd` → dispatch; `rocprof*` → kernel;
`torch_profiler_dynolog` / `tracelens*` → operator. When they name neither,
default to `dispatch` (`rocm_trace_lite`), which covers training and inference.

## Task
1. `madengine discover --tags <tag>`.
2. Build one `madengine run --tags <tag> --live-output --additional-context '{"tools":[...]}'`
   for the selected depth. Keep the user's other context keys (`docker_env_vars`,
   `slurm`, `k8s`) and replace `tools`.
3. Check for AMD GPUs (`rocm-smi` / `amd-smi`). If none, print the command and
   stop. Otherwise start it with
   `bash .claude/skills/mad-common/mad-run.sh start <run-id> -- <command>`, repeat
   `mad-run.sh wait <run-id>` until it reports `succeeded` or `failed`, and
   report the trace directory. Say that the perf
   number under a profiler is not a clean benchmark number.

For a multi-depth search that also measures clean candidates, use the
`mad-tune-search` workflow (`--depth auto`).

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-benchmark-runner` subagent. In Cursor, delegate
  to the `mad-benchmark-runner` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_benchmark_runner` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-benchmark-runner.md` first for the conventions it assumes.
