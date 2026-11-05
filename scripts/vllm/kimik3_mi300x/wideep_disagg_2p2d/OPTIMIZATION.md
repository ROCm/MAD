# Optimization notes — EP16 2P/2D disagg

The recipe is tuned for correctness + long-context accuracy first. These are the
serving-capacity levers, with the one validated change called out.

## ✅ Raise the KV cache (validated — big win, no downside seen)
`KV_CACHE_MEMORY_BYTES` is **pinned** (default `8e9`) to skip a boot `profile_run`
that hangs under TP2×DP8 + MoRI all2all — it is NOT a memory limit. At `8e9` the
GPU KV cache is only **542,372 tokens** (→ 1.69× concurrency at 320K ctx), while
there is **~72 GB/GPU free**.

Raising it to `40e9` measured:
- **GPU KV cache: 2,836,158 tokens** (5.2×)
- **Maximum concurrency at 1,000,000-token ctx: 2.84×** (was 1.69× at 320K)
- Also required for any **single request > ~600K tokens** (a 900K request needs
  900K tokens of KV — the 542K cache can't hold it).

```bash
KV_CACHE_MEMORY_BYTES=40000000000 ... bash run_2p2d_launch.sh
```
Leave ~8–10 GB/GPU slack for activations + the MoRI heap. Recommended default for
throughput-oriented deployments.

## `max_num_seqs` (raise with the KV cache)
Decode caps concurrent requests at `max_num_seqs` (default 8 = DP8 replicas × 1
seq). With the larger KV cache above, raising to 16–32 lets each replica batch
multiple decodes → higher aggregate tok/s under load. Bounded by KV capacity; too
high with a small cache causes preemption.

## `max_num_batched_tokens` — a dead end (keep 2048)
Raising 2048 → 8192 was measured **worse**: 200K single-stream 88s → 113s, and
20K conc=8 throughput 0.353 → 0.245 req/s. The prefill is compute-bound, not
chunk-overhead-bound, and larger prefill batches contend more across 8 replicas.
Keep `MAX_NUM_BATCHED_TOKENS=2048`.

## Single-stream latency — architectural, little to gain
A single request runs on one DP replica (TP2 = 2 GPUs); a colocated PP2×TP8 serve
spreads one request across all 16 GPUs, so it wins single-stream by ~4×. That is
the design trade for concurrent throughput (5.7× at conc 8, 7.3× at 16 — see
[RESULTS.md](RESULTS.md)). Use disagg for high-QPS/batch, colocated for
low-latency interactive. No config closes the single-stream gap.

## Residual write race → multi-needle 10/10 (correctness/quality, code change)
Multi-needle NIAH dips to ~9/10 at ≥ 20K (single-needle unaffected). Root cause:
decode can read a block before its RDMA write is globally visible in decode HBM
(`wait_for_layer_load()` is a no-op; `write_done` races the RDMA write). Proper
fix = a decode-side per-request KV-ready barrier before the model forward
(vLLM MoRIIO change, not a knob). Sender-side knobs (delay fence, `post_batch_size`
split) were measured useless and add latency.

## >500K prefill hang — FIXED (KDA gather sync-free)
Contexts above ~500K used to hang (GPUs 100%, no progress). Root cause found with
py-spy: `gather_initial_states` ran a `bool((indices>=n).any())` device→CPU sync
per KDA layer per prefill chunk (~25k full stream drains at 750K) purely to log a
warning; the index clamp above it already made the address safe. Gating that
diagnostic behind `K3_KDA_GATHER_LOG=1` (default OFF) removed the drains:
**750K and 900K now pass (542s / 717s), 500K unchanged (301s), sub-quadratic
scaling.** Baked into vLLM branch `-v3`; also shipped as
`patchers/apply_kimik3_kda_gather_nosync.py`. This was both the hang and a real
perf drag (fewer syncs = faster). It also removed the biggest remaining
high-context perf overhead, so no further prefill-latency lever is outstanding
beyond the architectural O(n²) of attention itself.

## Build / durability
- **Base image pin by digest for reproducibility.** The Dockerfile pins
  `BASE_IMAGE` to a dated ROCm CI/`-complete` tag (good — reproducible). For
  long-term durability, pin `@sha256:…` and mirror to a registry you control, so a
  rebuild survives upstream tag GC.
- **First-launch JIT compile** (AITER + MoRI-EP kernels) is the ~4–8 min warmup;
  it caches to node-local `/tmp/$USER/vllm_jit_cache` and is fast thereafter. A
  fresh image pays this once (and can transiently trip a WorkerProc init race —
  relaunch the affected pool once and it comes up clean).
