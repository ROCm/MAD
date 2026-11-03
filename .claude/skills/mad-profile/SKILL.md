---
name: mad-profile
description: Profile a MAD training or inference model at one depth — operator (TraceLens), dispatch (rocm-trace-lite), kernel (rocprofv3 counters and a kernel card), assembly (code-object summary, no GPU run), or PC sampling. Use when the user wants to profile or trace a MAD model.
argument-hint: "<tag> [operator|dispatch|kernel|assembly|pc_sampling] [kernel name or tool]"
disable-model-invocation: true
context: fork
agent: mad-benchmark-runner
allowed-tools: Bash(madengine discover *) Bash(madengine report *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Bash(python3 .claude/skills/mad-common/kernel-tools.py *) Read Grep Glob
---

Profile `$ARGUMENTS`.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`. The next
token, when it is `operator`, `dispatch`, `kernel`, `assembly`, or
`pc_sampling`, selects the depth. Anything after that is a kernel name (for the
kernel, assembly and PC-sampling depths) or an explicit madengine tool name.
Use the tag token in every `madengine` command.

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
   To name the top kernel for the kernel depth, run `kernel-tools.py context
   --mode warm` once, then `--mode dispatch` in the same run directory, then
   `kernel-tools.py rank` (see "Warm the cache first" in kernel-analysis.md).
   A cold `torch.compile` trace shows autotuning (`triton_mm` candidates that
   never run in steady state), and `rocm_trace_lite` misses kernels replayed
   from HIP graphs.
3. `kernel` — hardware counters for one kernel, ending in a kernel card. Read
   `.claude/skills/mad-common/kernel-analysis.md` first. The kernel comes from
   the user or from a dispatch trace (the top kernel by GPU time); without
   either, run `dispatch` first. Pick the counter group from what the kernel
   does (`compute` for GEMM, attention or convolution; `memory` otherwise),
   build the context with `kernel-tools.py context --mode counters`, run, then
   `kernel-tools.py card`. Only when the user names `rocprofv3_compute`,
   `rocprofv3_memory` or `rocprofv3_communication`, run that madengine preset
   instead and read `rocprof_output/`.
4. `assembly` — no GPU run. `kernel-tools.py asm --run-dir <kernel run dir>`
   on the code objects the kernel run kept. If there is no kernel run with a
   `co/` directory, run the `kernel` depth first.
5. `pc_sampling` — `kernel-tools.py context --mode pc_sampling`, run, then
   `kernel-tools.py pcsample`. It is a beta rocprofv3 feature: when it fails,
   report the error from the log and stop.

When the user names a tool and no depth, map it: `rocm_trace_lite` /
`rocm_trace_lite_default` / `rpd` → dispatch; `rocprof*` → kernel;
`rocprofv3_pc_sampling` → pc_sampling; `torch_profiler_dynolog` /
`tracelens*` → operator. When they name neither, default to `dispatch`
(`rocm_trace_lite`), which covers training and inference.

## Task
1. `madengine discover --tags <tag>`.
2. Build one `madengine run --tags <tag> --live-output --additional-context '{"tools":[...]}'`
   for the selected depth. Keep the user's other context keys (`docker_env_vars`,
   `docker_mounts`, `slurm`, `k8s`) and replace `tools`. For the kernel and
   PC-sampling depths, pass those keys as `--context` to `kernel-tools.py
   context` and use its output, with `--run-dir /var/tmp/mad-kernel/<run-id>`.
   Keep the run directory on local disk, outside the repo.
3. Check for AMD GPUs (`rocm-smi` / `amd-smi`). If none, print the command and
   stop. Otherwise start it with
   `bash .claude/skills/mad-common/mad-run.sh start <run-id> -- <command>`, repeat
   `mad-run.sh wait <run-id>` until it reports `succeeded` or `failed`, and
   report the trace directory. Say that the perf
   number under a profiler is not a clean benchmark number.
4. For the kernel, assembly and PC-sampling depths, end with the kernel card
   line from `kernel-analysis.md`: share, bound and the numbers behind it,
   registers, LDS, occupancy and its limiter, the instruction mix, and one
   lever from the levers table. Quote numbers from the JSON files only.

For a multi-depth search that also measures clean candidates, use the
`mad-tune-search` workflow (`depth: auto` climbs operator, dispatch, kernel
and assembly).

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-benchmark-runner` subagent. In Cursor, delegate
  to the `mad-benchmark-runner` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_benchmark_runner` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-benchmark-runner.md` first for the conventions it assumes.
