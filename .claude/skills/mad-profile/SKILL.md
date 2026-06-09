---
name: mad-profile
description: Run a madengine benchmark with a profiling/tracing tool attached (rocprofv3, rpd, rccl_trace, ...). Use when the user wants to profile or trace a MAD model.
argument-hint: "<tag-or-model> [tool: rocprofv3_compute|rpd|rccl_trace|...]"
disable-model-invocation: true
context: fork
agent: mad-benchmark-runner
allowed-tools: Bash(madengine *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash *) Read Grep Glob
---

Profile `$ARGUMENTS`.

## Pre-flight
```!
bash ${CLAUDE_SKILL_DIR}/../mad-common/preflight.sh
```

## Task
1. Pick the profiling tool (default `rocprofv3_compute` if unspecified). Common
   names: `rpd`, `rocprofv3`, `rocprofv3_compute`, `rocprofv3_memory`,
   `rocprofv3_communication`, `rocm_trace_lite`, `rccl_trace`,
   `gpu_info_power_profiler`. The authoritative full list (23+ tools, incl.
   `rocprofv3_full`, `rocblas_trace`, `hipblaslt_trace`, `miopen_trace`,
   `rocprof_sys`) lives in the madengine package at `scripts/common/tools.json`.
2. Build:
   `madengine run --tags $0 --live-output --additional-context '{"tools": [{"name": "<tool>"}]}'`
3. Check for AMD GPUs (`rocm-smi`/`amd-smi`). If none, print the command for a GPU
   host and stop. Otherwise run it and report where the trace output and `perf.csv`
   landed.

Note: profiling adds overhead — the perf number under profiling is NOT a clean
benchmark number.
