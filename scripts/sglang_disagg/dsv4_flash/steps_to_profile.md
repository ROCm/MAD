# Steps to Profile MoRI-IO + MoRI-EP (DSV4-Flash, sglang_disagg) — field notes

Practical, tested notes for profiling the `sglang_disagg` + MoRI stack with `rocprofv3`
(ROCtx markers + kernel trace), following the ROCm blog *"Profiling MoRI-IO and MoRI-EP
Transfers with ROCtx"* — but **without** the ROCm/MAD-private RUN_PROFILE/profiling-image
plumbing. Written from a non-SLURM, 4-node MI308X (gfx942) + Broadcom Thor2 bring-up.

**TL;DR:** the MoRI ROCtx markers and `rocprofv3` are already usable on a stock DSV4 MoRI
image, but **wrapping all 8 DP workers under `rocprofv3` crashes the DP collective on MI308X**.
You must profile a *subset* of workers (rank-0), use marker-trace-only, or use a topology
without an 8-way DP all-reduce. Details + exactly what failed below.

---

## 0. What you need (all confirmed present in a stock DSV4 MoRI image)
- **MoRI ROCtx markers (ROCm/mori PR #510)** baked into the MoRI build. Verify:
  `ls /sgl-workspace/mori/src/io/roctx_mori.hpp` and
  `grep -rl MORI_ROCTX /sgl-workspace/mori/src/`. Our MoRI commit `7c51d18f` has them.
- **`rocprofv3` + the SDK ROCtx lib**: `/opt/rocm/bin/rocprofv3`,
  `/opt/rocm/lib/librocprofiler-sdk-roctx.so`. `rocprofv3 --marker-trace --kernel-trace -f pftrace`.
  (MoRI dlopens `librocprofiler-sdk-roctx.so` — the SDK one that rocprofv3 intercepts, NOT the
  legacy `libroctx64.so`. No link-time dependency; zero cost when the env gates are off.)
- **bnxt NIC sysfs counters** for the independent NIC poller:
  `/sys/class/infiniband/bnxt_re_bond*/ports/1/counters/{port_xmit_data,port_rcv_data}`.

## 1. The two MoRI marker gates (off by default → no cost when unset)
- `MORI_ROCTX=1` — synchronous host-submission ranges: `mori.io.*`, `mori.rdma.batch_post.*`
  (measure how long the CPU call stays on the stack; NOT transfer latency).
- `MORI_ROCTX_TRANSFER=1` — asynchronous post-to-completion range `mori.rdma.io_transfer`
  (the KV-transfer latency to use). Each marker carries `bytes=<payload>` then `id=<MoRI transfer id>`.
Set BOTH in the container env of the worker you profile. They must be set **before** the worker
process starts (the roctx library is dlopen'd once, in the `RoctxApi()` ctor, at process init).

## 2. How to invoke rocprofv3
```
MORI_ROCTX=1 MORI_ROCTX_TRANSFER=1 \
rocprofv3 --marker-trace --kernel-trace -f pftrace -d <out_dir> -- \
  python3 -m sglang.launch_server <server args>
```
- `-f pftrace` → Perfetto trace (open in https://ui.perfetto.dev).
- `-d <out_dir>` → traces land at `<out_dir>/<hostname>/<pid>_results.pftrace`, **one file per PID**.
- **rocprofv3 follows forks** (verified: a parent+child GPU test yields two `.pftrace` files). So
  wrapping the SGLang launcher *does* capture the forked DP worker kernels — that part works.

## 3. ★ THE BIG GOTCHA — do NOT wrap all 8 DP workers on MI308X
Wrapping `python3 -m sglang.launch_server` (which forks 8 DP scheduler workers) under
`rocprofv3 --kernel-trace` **kills the run** on MI308X:
```
RuntimeError: [gloo/transport/tcp/pair.cc:547] Connection closed by peer [<decode ip>]
... rocprofv3_error_signal_handler ... rocprofv3 caught signal 3
```
**Why:** DSV4 decode runs 8 DP workers that stay in lockstep on gloo collectives. `rocprofv3`'s
per-kernel HSA/HIP interception adds overhead to every worker; one falls behind, a collective
times out, the whole DP group tears down. This happens **during cudagraph capture** and again on
a **post-capture gloo sync**.

Things that DID NOT fix it (tested):
- **Raising timeouts** — `--dist-timeout 3600`, `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600`,
  `GLOO_TIMEOUT_SECONDS=3600`. Got past the *initial* rendezvous (all 8 cudagraphs captured, server
  briefly ROUTER_OK) but the group still died on a later collective. dist-timeout covers init
  *waits*, not an in-flight collective stall under load.
- **`-P` collection-period** — `rocprofv3 -P 600:120:1` (keep data collection OFF for the first
  600 s, past warmup). Survived longer (server reached ROUTER_OK) but the DP group STILL died.
  Reason: `-P` delays *data collection*, but the **instrumentation hooks are installed at attach
  time regardless** — the per-kernel hook overhead is what stalls the collective, not the
  collection. So delaying the window does not remove the overhead.
- **`rocprofv3 --attach <PID>`** on an already-running worker — "attach :: success", server stayed
  up, workload ran — but **no `.pftrace` was ever written**. rocprofv3 attach cannot retroactively
  instrument GPU kernels (the HSA/HIP layer must be present at process init). Attach connects at the
  ROCtx level only; useless for kernel/marker capture here.

## 4. What TO do instead (pick one)
The blog pools all 8 ranks because MI300X + AMD's private per-worker SLURM harness had the headroom.
On a stock image + MI308X, reduce how many workers carry the profiler:

- **(A) marker-trace ONLY** — drop `--kernel-trace`, keep `--marker-trace`. The ROCtx-only hook is
  much lighter and may keep the DP group in sync. Gets you the **MoRI-IO transfer ranges
  (bytes/id) + byte-accounting**, but NOT the kernel-category tables. Cheapest; try first.
- **(B) profile rank-0 only** — make just ONE DP worker launch under `rocprofv3`; the other 7 run
  unhooked so gloo stays healthy. This is what the private MAD RUN_PROFILE path does per-worker.
  On a stock SGLang there is **no env to gate the per-worker wrapper** (workers are forked
  internally by `data_parallel_controller.py` via `mp.Process`, not a shell wrapper), so this needs
  a small **SGLang patch**: in the DP worker entry, if `dp_rank == 0` and `RUN_PROFILE` is set,
  re-exec the worker under `rocprofv3 -- ...`. Single-worker analysis is still valid (blog says so).
- **(C) profile a TP-only / single-GPU serve** — no 8-way DP all-reduce to break. Good for a
  kernel + MoRI microbenchmark, but it is a different topology than the EP8/EP16 you deploy.

## 5. Keeping captures small (applies to all paths)
`rocprofv3` records one continuous trace for the whole server lifetime → size explodes with
tokens × requests × concurrency. Use: **con=1, one ISL/OSL point, 1 iteration, short OSL (8–32),
2–8 prompts**, and `SKIP_WARMUP` / `SKIP_CURL_TEST`. Decode OSL dominates trace size. Write traces
to a big local mount (e.g. `/mnt/md0`, `/mnt/nvme*`), NOT `/root` (small on some nodes).
No shared FS needed — `scp` each node's `<out>/<host>/<pid>_results.pftrace` back for analysis.

## 6. Byte-accounting for DSV4-Flash (differs from the blog's DSV3 BF16)
The blog's DSV3 uses BF16 KV: bytes/token/layer = (kv_lora_rank + qk_rope_head_dim) × 2.
**DSV4-Flash here uses `--kv-cache-dtype fp8_e4m3` (1 byte/elem) and 43 layers** (not 61), and
`qk_rope_head_dim=64`. So derive the expected `bytes=` per `mori.rdma.io_transfer` marker from the
DSV4 config at run time and confirm against the trace — do not reuse the blog's 1152 B/token/layer.
MoRI may also split a layer's KV into multiple transfers under a max-write-size threshold (marker
count becomes an integer multiple of the layer count).

## 7. Non-SLURM driver wiring used here (reference)
A `RUN_PROFILE=1` gate was added to a non-SLURM driver + `sglang_disagg_mori_io_ep.sh` that:
mounts a big local mount → `/prof`, sets the MoRI marker gates, and prefixes `launch_server` with
`rocprofv3 ... -- `. **This reproduces the crash in §3** if it wraps all 8 workers — it is included
only as the base to build path (A)/(B) on, not as a working full-DP capture. The correct next step
is the SGLang rank-0 patch (B).

## 8. Environment where these notes were taken
MI308X (gfx942) ×8/node, 4 nodes, Broadcom Thor2 bnxt RoCE. ROCm 7.2, SGLang v0.5.15 (patched
DSV4), MoRI `7c51d18f`. Image `rocmshared/sglang-disagg-dsv4:mori-mi308-pr2`. DSV4-Flash-FP8,
EP8 1P1D + EP16 2P2D, MoRI-IO KV transfer + MoRI-EP a2a, decode CUDA-graph, optional MTP.
