# Kernel analysis: from a dispatch trace to one lever

Shared by `/mad-profile`, `/mad-tune`, `mad-tune-search` and `mad-optimize`.
The helper is `kernel-tools.py` in this directory (standard library only).

## The ladder

| Rung | Question | Run | Output |
|------|----------|-----|--------|
| operator | Which PyTorch op or module dominates? | `torch_profiler_dynolog` + `tracelens` | op table |
| dispatch | Which kernels are on the timeline, and how much time each? | rocprofv3 kernel trace on a warmed cache (`kernel-tools.py context --mode dispatch`, then `rank`); `rocm_trace_lite` for a timeline | `kernel_rank.json` |
| kernel | Why is that kernel slow? | `kernel-tools.py context --mode counters` | `kernel_card.json` |
| assembly | What did the compiler emit? | no GPU run: `kernel-tools.py asm` | `kernel_asm_summary.json` |
| PC sampling (optional) | Which instructions stall, and on what? | `kernel-tools.py context --mode pc_sampling` | `kernel_pc_samples.json` |
| thread trace (optional) | Which compiled variant ran, and which instructions does it wait on? | `kernel-tools.py context --mode thread_trace` | `kernel_thread_trace.json` |

Climb one rung at a time and stop when the evidence names a lever. One tool per
run: never put `rocm_trace_lite` and a `rocprofv3_*` tool in the same run. A
number measured under a profiler is evidence, not a benchmark score.

## Warm the cache first

A model that uses `torch.compile` compiles and autotunes in every cold run.
Under a profiler that work is what you measure. In one FastWAM dispatch trace
the Inductor autotune candidates (`triton_mm`) took 65% of kernel time and never
ran in steady state. Under counters, eight of nine cold runs sat in Inductor
for more than 15 minutes with the GPU idle; on a warm cache the same counter
run took about 45 seconds per pass. So the dispatch and kernel rungs share one
run directory, and a warm-up run without a profiler fills its code-object cache:

```bash
D=/var/tmp/mad-kernel/<id>
python3 .claude/skills/mad-common/kernel-tools.py context --mode warm --run-dir $D --context '<base context>'
bash .claude/skills/mad-common/mad-run.sh start <id>_warm -- \
  madengine run --tags <tag> --live-output -o perf_warm.csv --additional-context-file $D/context_warm.json
python3 .claude/skills/mad-common/kernel-tools.py context --mode dispatch --run-dir $D --context '<base context>'
bash .claude/skills/mad-common/mad-run.sh start <id>_dispatch -- \
  madengine run --tags <tag> --live-output -o perf_dispatch.csv --additional-context-file $D/context_dispatch.json
python3 .claude/skills/mad-common/kernel-tools.py rank --run-dir $D
```

Each `context` call writes its context file to the run directory
(`context_warm.json`, `context_dispatch.json`, or `context.json` for counters,
PC sampling and thread trace), so `madengine` reads a file instead of `"$(...)"`: a headless agent
session cannot approve a command substitution.

`rank` writes `kernel_rank.json`: kernels by summed GPU time from the
rocprofv3 kernel trace. Take the top kernel from there, not from
`rocm_trace_lite`. In the same warm FastWAM GEAK run, rocprofv3 recorded about
296,000 dispatches and `rocm_trace_lite` 66,000; the missing ones were the
model's own kernels, which it replays from HIP graphs, so `rocm_trace_lite`
ranked a 3 µs fill kernel first while rocprofv3 and PC sampling both put
`_fw_skmm_kernel` on top.

`--mode warm` moves an existing `co/` aside to `co.prev-N`, so the autotune
results come from an unprofiled run. Every mode writes a `tools` list, empty
for the warm-up: `madengine run -m <manifest>` stores the context's `tools` in
the manifest and reuses them on a later run whose context has no `tools` key.
A "clean" run from a manifest that once carried a profiler is profiled again
unless it says `"tools": []`.

The cache is reused only when the model reads `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR` or `MAD_CODE_OBJECT_DIR`. A script that always compiles
into a fresh temporary directory needs to honor `MAD_CODE_OBJECT_DIR` first.

A counter run can still hang on a warm cache. The FastWAM eager arm (no
`torch.compile`) did so twice, once filtered to a hipBLASLt GEMM and once to
`attn_fwd`: GPU use stayed at 0% and `py-spy dump --native` showed the host in
`rocr::core::BusyWaitSignal::WaitRelaxed`. Choosing a different kernel did not
help. Treat a run as stalled when its status is `running` (not `queued`) and
`rocm-smi --showuse` reads 0% at every check for 10 minutes. That limit is a
judgment: longer than image start and model load, which also read 0%. Then:

1. Stop it with `mad-run.sh stop`, and do not run `card` on its output: the
   counter passes are incomplete.
2. Retry once with `--serialize` (see "A second stall" below). `mad-tune-search`
   and `mad-optimize` do exactly this, and mark the card `serialized`.
3. If the retry stalls too, stop it and fall back to PC sampling, which does not
   use counters. The workflows return status `error` for the kernel rung and go
   on without a bound.

Each profiling mode writes to its own `rocprof/<mode>/`, and moves the
previous one aside to `<mode>.prev-N`. `card`, `rank`, `pcsample` and
`threadtrace` read only their mode's directory, so one run directory can hold
several kernel runs.

Use a run directory on local disk such as `/var/tmp/mad-kernel/<id>`. The
code-object cache runs to hundreds of megabytes and does not belong in the
repo. Copy the JSON results into the repo afterwards if you need them there.

## Kernel rung

1. Take the kernel name from the warmed dispatch rung. Use the name exactly as the
   trace prints it; it becomes `--kernel-include-regex`, so only that kernel is
   counted and the run slows down less.
2. Pick the counter group. `compute` when the kernel is a GEMM, attention or
   convolution (or the trace shows MFMA kernels); `memory` for elementwise,
   reduction, normalization or copy kernels. Both groups collect bytes and MFMA
   operations, so either one places the kernel on the roofline.
3. Build the context in the same run directory (warm it first if the dispatch
   rung did not) and run:

   ```bash
   python3 .claude/skills/mad-common/kernel-tools.py context --mode counters \
     --group compute --kernel '<kernel>' --run-dir $D --context '<base context>'
   bash .claude/skills/mad-common/mad-run.sh start <id> -- \
     madengine run --tags <tag> --live-output -o perf_kernel.csv --additional-context-file $D/context.json
   ```

   `--run-dir` is mounted at `/mad_kernel`. rocprofv3 writes there,
   and the model keeps its Triton and Inductor code objects in `co/` when it
   honors `MAD_CODE_OBJECT_DIR`. rocprofv3 runs the workload once per counter
   pass (five passes per group), so allow five times the normal run time.

   rocprofv3 does not wrap the model's bash script: with counters on, a forked
   bash subshell deadlocks (it waits on a lock forever at 0% CPU). The context
   puts a `python3`/`python` shim first on `PATH` instead, so the first Python
   process the script starts runs under rocprofv3 and its children inherit the
   tool. A model whose script starts its workload some other way needs its own
   launcher.

   A second stall is in the model itself. A model that launches tens of
   thousands of tiny kernels before its first host wait (an eager PyTorch
   inference ending in `.to("cpu")` or `synchronize()`) can hang under counter
   collection with the GPU idle and the host in `BusyWaitSignal::WaitRelaxed`.
   A kernel filter that matches nothing does not avoid it; the kernel trace
   without counters does not show it. Add `--serialize` to `context`, which sets
   `AMD_SERIALIZE_KERNEL=3` so the host waits after every launch. Only use it
   after a counter run stalls: it slows the host side of the run, not the
   kernel's own counters. On the FastWAM eager arm it changed the kernel's
   duration under counters by about 4% from the kernel trace, so read a
   serialized card as ratios (bound, percent of peak), not as a latency score.

   The card averages counters over every dispatch. On a long run, check that
   the matrix-op count (`SQ_INSTS_VALU_MFMA_MOPS_*`) is identical across the
   dispatches of one launch shape. The FastWAM eager run with 97,500 dispatches
   had about one in five inflated (wave counts were exact), which tripled the
   matrix-core share. Use a short run, or take the median per launch shape.
4. `python3 .claude/skills/mad-common/kernel-tools.py card --run-dir $D`.

### Reading the card

- `bytes` is `FETCH_SIZE + WRITE_SIZE` (memory-side traffic). `mfma_flops` is
  512 per MFMA operation counted by `SQ_INSTS_VALU_MFMA_MOPS_*`; non-matrix
  FLOPs are not included.
- `pct_peak_bandwidth` and `pct_peak_mfma` divide the achieved rates by public
  peaks (HBM TB/s and dense BF16 TFLOP/s for the product in `agent_info`).
  Override them with `--peak-bw` and `--peak-tflops` when the data type or the
  product differs.
- `bound` follows fixed rules:
  - **latency**: both fractions below 10%. Neither unit is busy, so launch
    overhead, low occupancy, dependency chains or serialization dominate.
  - **compute**: the MFMA fraction is the larger one.
  - **bandwidth**: the bandwidth fraction is the larger one.
  - **unknown**: a counter or the duration is missing. Say so; do not guess.
- `wait_share` (`SQ_WAIT_ANY / SQ_WAVE_CYCLES`) is the share of wave time spent
  waiting. `l2_hit_rate` is `TCC_HIT / (TCC_HIT + TCC_MISS)`.
  `lds_bank_conflict_ratio` is `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE`.
- `per_wave` is the instruction mix per wave, from `SQ_INSTS_*`.
- The duration comes from counter-collection dispatches, which run
  serialized. Use it for ratios, not as a latency score.
- Autotuning launches candidate kernels under one name. A card is one variant
  (name, registers, LDS, workgroup size), and `variants_with_this_name` says
  how many there were. The first card is the variant with the most profiled
  time; a candidate seen only during autotuning lacks most counters.

## Assembly rung

`python3 .claude/skills/mad-common/kernel-tools.py asm --run-dir /var/tmp/mad-kernel/<id>` reads
`.amdgcn` (Triton writes one per kernel) or disassembles `.hsaco` with
`llvm-objdump`. It matches the dispatched variant by name and the VGPR count
from the card; when autotuning left several variants with the same name and
registers, the status is `ambiguous` and all of them are listed. It reports:

- registers (VGPR, AGPR, SGPR), spills, scratch and LDS bytes (static plus
  Triton's dynamic `shared`);
- occupancy in waves per SIMD and what limits it (registers, LDS, hardware);
- the instruction mix by class (MFMA, AGPR move, VALU, SALU, SMEM, VMEM,
  scratch, LDS, waitcnt, barrier, branch);
- memory operations and their widths (`dwordx4`/`b128` is 16 bytes per lane);
- the three largest loops (a backward branch) and their mix.

Library kernels (hipBLASLt `Cijk_*`, rocBLAS, Composable Kernel) live inside
shared libraries, so the status is `no code object`. Use PC sampling for them.

## PC sampling

```bash
python3 .claude/skills/mad-common/kernel-tools.py context --mode pc_sampling \
  --run-dir /var/tmp/mad-kernel/<id> --context '<base context>'
# then madengine run ... --additional-context-file /var/tmp/mad-kernel/<id>/context.json
```

Stochastic sampling counted in cycles (interval 1048576 by default). It is a
beta rocprofv3 feature: record the run even when it fails, and quote the error.
`pcsample` reports the hottest instructions, the stall reasons, and the share
of samples that issued.

## Thread trace

```bash
python3 .claude/skills/mad-common/kernel-tools.py context --mode thread_trace \
  --kernel '^_fw_skmm_kernel$' --iteration-range '[300-305]' \
  --run-dir /var/tmp/mad-kernel/<id> --context '<base context>'
# then madengine run ... --additional-context-file /var/tmp/mad-kernel/<id>/context.json
python3 .claude/skills/mad-common/kernel-tools.py threadtrace --run-dir /var/tmp/mad-kernel/<id>
```

This is `rocprofv3 --kernel-trace --att` for one kernel, on one compute unit
(`--att-target-cu`, default 1), over `--kernel-iteration-range` (default
`[1-8]`). Set the range inside the steady state when the kernel launches
thousands of times: the FastWAM GEAK trace used `[300-305]`. An unfiltered
trace records every launch, so `--kernel` is required.

`threadtrace` groups the traced launches by the code object they ran. For each
variant it reports the instruction mix, the share of traced cycles and of stall
cycles by class, and the instructions that cost the most latency. It hashes the
`.out` code object `--att` saved and matches it to a file under `co/`, which
names the compiled variant when several share a kernel name and a register
count.

rocprofv3 documents thread trace for MI200 and MI300. The same command
completed on gfx950 with rocprofv3 1.3.2. `mad-tune-search` does not climb
this rung: it is a closer look at a kernel the dispatch rank already chose.

## The kernel card in a report

One line per kernel, built from the three JSON files:

> `<kernel>`: <share>% of GPU time (dispatch). <bound> (<pct bw>% peak
> bandwidth, <pct mfma>% peak MFMA, <flops/byte> FLOPs/byte against a ridge of
> <ridge>). <vgpr> VGPR, <lds> B LDS, occupancy <n> (limited by <limiter>).
> Mix: <top classes>. Lever: <one lever from the table below>.

## Levers by bound

| Bound | Evidence | Lever to try first |
|-------|----------|--------------------|
| bandwidth | low `l2_hit_rate`, narrow loads (`dword`, `b32`) | vectorize to 16-byte loads; fuse the producer or consumer to cut a round trip through memory; reuse through LDS tiles |
| bandwidth | high `l2_hit_rate`, wide loads, near peak | the kernel is at the roof: cut bytes (fusion, lower precision) or skip it |
| compute | MFMA near peak | larger tiles, better MFMA shape, lower precision (FP8) if accuracy allows |
| compute | MFMA low, VALU high | move the math onto MFMA; check `matrix_instr_nonkdim`, `kpack` |
| latency | occupancy limited by registers or spills | fewer registers per thread (`num_warps`, `waves_per_eu`, smaller tiles) |
| latency | occupancy limited by LDS | fewer `num_stages`, smaller tiles |
| latency | small grid or very short kernel | fuse launches, batch work, HIP graphs, `torch.compile` mode |
| latency | high `wait_share`, waitcnt-heavy loop | prefetch (more `num_stages`), interleave loads with MFMA |
| any | high `lds_bank_conflict_ratio` | pad or swizzle the LDS layout |

Library kernels take library levers: `PYTORCH_TUNABLEOP_ENABLED=1`, hipBLASLt
tuning, or a different GEMM backend. Change one lever per measurement, then
measure clean (no profiler) at least twice.
