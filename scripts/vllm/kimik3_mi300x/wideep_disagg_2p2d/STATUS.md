# Status — Kimi-K3 MI300X 2P/2D EP16 MoRIIO disagg — **VALIDATED**

**This recipe is validated.** Single-needle NIAH passes **deterministically
across the full native context range 10K–900K** (all depths) on the 2 prefill +
2 decode EP16 disagg serve. The decode-recall bug that previously blocked this is
**fixed**, and the >500K prefill hang is **fixed** (KDA gather sync-free — see
below + RESULTS.md).

**Validated from a scratch build.** The two fixes are folded into vLLM source on
`raviguptaamd/vllm` branch `kimi-k3-wideep-disagg-fullsource-v2`; the Dockerfile
builds that branch, so the **image ships the fixes baked in**. A from-scratch
image build was deployed and re-validated (NIAH 12/12 to 300K) with the runtime
patchers all reporting "already applied" — i.e. the image is self-sufficient and
the patchers are idempotent no-ops on it.

## Result

Single-needle NIAH (needle = `HELIOTROPE-7492`, greedy `temperature=0`, depths
0.1 / 0.5 / 0.9) — **PASS at every size 10K → 900K**, deterministic:

| ctx (tokens) | result | eval time / req |
|--------------|--------|-----------------|
| 10K  | 3/3 PASS | 5.3s  |
| 50K  | 3/3 PASS | 19.5s |
| 100K | 3/3 PASS | ~47s  |
| 200K | 3/3 PASS | ~88s  |
| **300K** | **3/3 PASS** | **~150s** |
| 500K | 3/3 PASS | ~301s |
| 750K | PASS | ~542s |
| **900K** | **PASS** | **~717s** |

Throughput (the point of DP disaggregation), 20K ctx, batched=2048:
conc=1 → 0.062 req/s; conc=8 → **0.353 req/s (5.7×)**; conc=16 → **0.455 req/s
(7.3×)**. See `RESULTS.md`.

## Root cause (two independent bugs, both fixed)

Kimi-K3's attention is hybrid — 24 MLA full-attention layers + 69 KDA
(Kimi-Delta-Attention, recurrent) layers → vLLM allocates **4 KV-cache groups**
(idx 0/1/2 = MambaSpec/KDA, 23 layers each; idx 3 = MLAAttentionSpec, 24 layers).
Both bugs stem from this multi-group hybrid.

### Fix #1 — 4-KV-cache-group block routing (`patchers/apply_kimik3_moriio_group_routing.py`)
The shipped connector assumed 2 groups and hardcoded group indices `[0]`/`[1]`
when computing RDMA transfer offsets, sending MLA KV (group 3) to *group-0
(mamba)* block-ids. Decode read the group-3 blocks, which were never written →
fluent but context-free output. Fix carries **all 4 groups' block-id lists
end-to-end** and routes each layer's transfer by its own group index. Always on
(`K3_GROUP_ROUTING=1`). This fixed short recall (≤ 1 block).

### Fix #2 — multi-chunk prefill transfer (`apply_kimik3_chunk_gate_fix.py` + `apply_kimik3_chunked_allgrp.py`)
A razor-sharp cliff: recall died at exactly `max_num_batched_tokens`. The
connector detected the "final prefill chunk" by **block count**
(`num_prompt_tokens > len(block_ids) * self.block_size`), but the padded scheduler
block (~5760) holds a whole prompt in one block, so the KV transfer fired after
**chunk 1** (only `max_num_batched_tokens` computed); tokens past chunk 1 (e.g. a
needle at the end) were never transferred. When a prompt fits in ≤ 1 block,
block-count can never detect chunk completion.

Fix: gate on **compute progress** from fresh `scheduler_output`
(`num_computed_tokens` + `num_scheduled_tokens`), not block count — applied at 4
points: (A) build a per-step progress map, (B) entry defer, (C) accumulation
final-detect, (D) post-loop sweep for the final-chunk-adds-no-new-block case
(else the request deadlocks: `unmap MISS table_size=0`). `chunked_allgrp`
accumulates every group's block-ids across chunks. `SLACK=2`
(`K3_CHUNK_GATE_SLACK`) absorbs the mamba N-1 truncation. Enabled with
`K3_EXTRA_FIXES=1`.

Verified on a 2611-token prompt:
```
[k3-chunk-gate-entry] nblk=1 npt=2611 done=False prog=(0,2048)    # chunk 1 -> defer
[k3-chunk-gate-sweep] nblk=1 npt=2611 done=True  prog=(2048,563)  # chunk 2 -> emit full blocks
```

### Fix #3 — KDA gather sync-free (`patchers/apply_kimik3_kda_gather_nosync.py`)
Contexts above ~500K hung. Root-caused with py-spy: the DP rank holding the real
request was stuck in `gather_initial_states` (the KDA recurrent-state gather)
while all other DP ranks waited at the `coordinate_batch_across_dp` all_reduce.
The native stack showed the block was a `bool((indices>=n).any())` device→CPU
sync (`_local_scalar_dense` → `memcpy_and_sync`, a full stream drain) run purely
to emit a diagnostic warning — **per KDA layer, per prefill chunk** (~25k drains
at 750K). The index clamp above it already guarantees a valid GPU address, so the
sync is pure overhead. Fix gates it behind `K3_KDA_GATHER_LOG=1` (default OFF);
correctness is unchanged (indices still clamped). With this, 750K (542s) and 900K
(717s) now pass where they previously hung indefinitely; 500K unchanged (301s),
scaling sub-quadratically. Folded into vLLM branch `-v3`.

## Known residual (does not block single-needle NIAH to 300K)

An RDMA write-visibility race: `write_done` travels ZMQ/TCP, a different path than
the RDMA write, and `wait_for_layer_load()` is a no-op — so at high context (many
blocks) decode can occasionally read a block before its RDMA write is globally
visible in decode HBM. Effect: the **stricter 10-needle** stress
(`benchmark_niah.py`) dips to ~9/10 at ≥ 20K; **single-needle** NIAH is
unaffected (deterministic to 300K). Sender-side mitigations (delay fence,
`post_batch_size` split) don't help and add latency. Proper fix = a decode-side
per-request KV-ready barrier before the model forward; tracked as future work.

## Prior connector fixes (still required, pre-existing)
mamba/KDA state routed by the mamba KV-cache group's block ids; degenerate
`remote_tp_size` normalized so writes fan out to all decode TP ranks; mamba N-1
prefill/decode boundary (producer computes h(N-1), decoder recomputes token N).
These are load-bearing and remain in `patchers/`.

## Diagnostic knobs (opt-in, default OFF)
Gated knobs from the investigation remain available for debugging (turn up/down
as needed): `K3_XFER_PROBE`, `K3_DECODE_RECV_PROBE`, `K3_KDA_STATE_PROBE`,
`K3_WRITE_BC`, `K3_HS_BC`, `K3_INPUTS_PROBE`, `K3_CHUNK_GATE_DEBUG`,
`K3_WRITE_FENCE`, `K3_ENABLE_CLAMP`. None change default behavior. See each
patcher's docstring and the README "Debugging" section.
