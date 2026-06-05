---
description: Run a madengine benchmark with a profiling/tracing tool attached
argument-hint: <tag-or-model> [tool: rocprofv3_compute|rpd|rccl_trace|...]
---

Profile `$ARGUMENTS`.

Use the `mad-benchmark-runner` subagent with profiling enabled. It should:
0. Pre-flight: check madengine is installed and cwd is the MAD repo root.
   ```bash
   if ! command -v madengine &>/dev/null; then
     if [ -f requirements.txt ] && grep -q madengine requirements.txt; then
       echo "[pre-flight] madengine not found. Installing from requirements.txt..."
       pip install -r requirements.txt
     else
       echo "[pre-flight] madengine not found and requirements.txt is missing."
       echo "  Install:  pip install git+https://github.com/ROCm/madengine.git@main"
       echo "  Or clone MAD and run from its root (which has requirements.txt)."
       exit 1
     fi
   fi
   if [ ! -f models.json ]; then
     echo "[pre-flight] Warning: models.json not found — run from the MAD repo root."
   fi
   ```
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
