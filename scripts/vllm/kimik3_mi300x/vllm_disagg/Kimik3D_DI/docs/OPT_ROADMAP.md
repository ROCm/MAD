# Optimization roadmap — Kimi-K3 v4 disagg EP16

Verified against this stack (vLLM d626108b, MoRI 624002c8, K3-MXFP4). Ordered by leverage.

## 0. BLOCKER (must fix before any perf work is measurable)
**~150–170s per-request decode stall** (both GPUs 100%, no logs, after a fast matched
handshake). See `PERF_DIAGNOSIS.md`. Until this is gone, every latency/throughput number is
dominated by it and the optimizations below can't be measured. Prime suspect: MoRI 624002c8
InterNodeV1LL decode all2all barrier on bnxt.
- **Next tests:** (a) decode `all2all-backend=mori_high_throughput` (A/B vs low_latency);
  (b) MoRI bisect 624002c8 → v3's MoRI to find the regression commit; (c) profile the decode
  forward (rocprof) during the 150s to see which kernel/collective spins.

## 1. Decode speculative / MTP  — NOT AVAILABLE for K3 in this stack
- vLLM supports MTP self-speculation for deepseek/glm/mimo (`config/speculative.py:37`), but
  **K3 config has no `num_nextn_predict_layers` / MTP head** and the K3 vLLM model class
  defines no nextn path. So MTP is a no-op here without model changes.
- EAGLE / draft-model speculative: possible mechanically but needs a trained K3 draft head —
  not a near-term recipe lever. **Deferred to model team.**
- If/when K3 ships MTP layers: `--speculative-config '{"method":"...","num_speculative_tokens":N}'`.

## 2. Prefill context parallelism (long-ctx prefill) — AVAILABLE
- `prefill_context_parallel_size` (PCP) and `decode_context_parallel_size` (DCP) exist
  (`config/parallel.py:126,342`; DCP comm backend `ag_rs`/`a2a`). DCP shards the decode KV
  cache without expanding world size — directly helps long-ctx (250K+) KV pressure + attn.
- **Test after the stall is fixed:** `--decode-context-parallel-size 2` on the decode pool
  (shards KV across TP ranks) for the 250K/300K NIAH tail. PCP for prefill attention at
  very long ctx.

## 3. Chunked prefill — ALREADY ON, tune the chunk
- `enable_chunked_prefill=True` by default. The chunk is bounded by `max_num_batched_tokens`
  (we cap at 2048–4096 because a larger value blows the MoE profiling compile shape,
  `sorted-M = batched × topk8`). Larger chunks = better prefill throughput at long ctx but
  risk the compile blowup — needs the heuristic-kernel fix or a compile-shape cap first.
- **Test:** creep `MAX_NUM_BATCHED_TOKENS` 2048→4096→6144 watching TTFT vs compile time.

## 4. Reasoning-token control (latency) — DONE / recipe-level
- `THINKING_DEFAULT=false` (`--default-chat-template-kwargs '{"thinking":false}'`) confirmed
  suppresses the ~640-tok `<think>` chain (needle lands in `content`). For prod, expose
  per-request `chat_template_kwargs.thinking` and `thinking_effort` (low/high/max).

## 5. Decode concurrency / KV-transfer serialization — CODE FIX (tracked separately)
- MoRIIO write pipeline is single-threaded (`moriio_engine.py:92` one daemon drains all KV
  writes, blocks per-task on `event.synchronize()`). Config knobs (qp_per_transfer,
  num_workers) only tune the RDMA backend, not the Python worker — and setting them broke the
  WRITE notify path in testing. **True fix = thread-pool `ensure_worker_started`** (spawn N
  writer threads). Do AFTER the #0 stall is understood (they may be the same root cause).

## 6. Cudagraph / capture — formal setup kept
- prefill eager (CG=NONE) + mori_high_throughput; decode FULL_AND_PIECEWISE + mori_low_latency.
  F_A_P captures fine at 320K with the HANDSHAKE=30 patch. (READ mode + FULL graph breaks the
  per-layer KV-read barrier → garbage; keep WRITE mode with F_A_P.)

## 7. Model-load time (ops, not serving perf)
- WekaFS 1.5T model, 16 concurrent rank reads → ~18min load, 7–21s/shard throttle. tmpfs/NVMe
  would cut reloads to ~2min but load time is NOT the serving bottleneck — deferred.

### Testing protocol
Use **100K context** for iterations (fits < max_model_len 320000 with output headroom,
representative of long-ctx without the 300K per-request cost). NIAH: `niah_sweep.py`.
Throughput: `perf_sweep.py` (streaming TTFT/e2e/tok-s at con=16/32).
