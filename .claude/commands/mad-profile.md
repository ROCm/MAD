---
description: Run a madengine benchmark with a profiling/tracing tool attached
argument-hint: <tag-or-model> [tool: rocprofv3_compute|rpd|rccl_trace|...]
---

Profile `$ARGUMENTS`.

Use the `mad-benchmark-runner` subagent with profiling enabled. It should:
1. Pick the profiling tool (default `rocprofv3_compute` if unspecified). Common
   names: `rpd`, `rocprofv3`, `rocprofv3_compute`, `rocprofv3_memory`,
   `rocprofv3_communication`, `rocm_trace_lite`, `rccl_trace`,
   `gpu_info_power_profiler`. The complete, authoritative list (23+ tools) lives
   in the madengine package at `scripts/common/tools.json` — consult it for
   names like `rocprofv3_full`, `rocblas_trace`, `hipblaslt_trace`, `rocprof_sys`.
2. Build:
   `madengine run --tags $1 --live-output --additional-context '{"tools": [{"name": "<tool>"}]}'`
3. Check for GPUs; if none, print the command for a GPU host. Otherwise run it
   and report where the trace/profile output and `perf.csv` landed.

Note: profiling adds overhead; the perf number under profiling is not a clean
benchmark number.
