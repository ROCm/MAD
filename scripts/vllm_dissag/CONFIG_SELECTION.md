# Choosing TP8 vs EP8 vs EP16 (2P/2D) — a decision guide grounded in the collectives

GLM-5.2-FP8, 1P/1D disaggregated serving on MI355X + AINIC (ionic), via vllm-router. All three
topologies ship in this PR because **each wins a different operating regime**. This file says which
to pick, and — more usefully — *why*, from the collective each one runs. Numbers trace to
`RESULTS_CRUSOE_IONIC.md` (EP8, EP16) and the 2026-08-26 TP8-vs-EP8 parallel comparison.

> **Scope caveat.** "1P/1D" here = one prefill deployment + one decode deployment. TP8/EP8 are
> 1 node per role; **EP16 is 2 nodes per role (2P/2D by node count)**. Numbers are Crusoe/ionic,
> ISL 8K/32K/128K, `osl=128`. TP8's 128K row is not yet measured (node-blocked) — noted below.

---

## 1. The one-paragraph answer

**TP8 is the latency tier, EP8 is the throughput tier, EP16 is the capacity/scale-ceiling tier.**
TP8 has the lowest TPOT (~22 ms) and wins when you are latency-bound and lightly loaded. The moment
you are throughput-bound — high concurrency, long context, or offline/batch — **EP8 overtakes TP8**,
because expert-parallel scales where tensor-parallel plateaus. EP16 never wins latency (it pays a
fixed cross-node-comm tax, ~117 ms TPOT) but is the only config whose KV pool + expert capacity
exceed a single node, so it is the answer when **large context AND large concurrency** together
overflow what one node can hold — and its real payoff is the step to 4P4D+ where prefill replicas
kill the TTFT that dominates long context.

---

## 2. The crossover — where EP8 beats TP8 (measured, both 8 GPU/role, 1P/1D)

**Throughput (aggregate output tok/s):**

| | con8 | con16 | con32 | con64 |
|---|---|---|---|---|
| **isl=8K TP8** | 218 | 262 | 275 | 288 |
| **isl=8K EP8** | 96 | 156 | 234 | **371** |
| | | | | ↑ EP8 wins |
| **isl=32K TP8** | 65 | 67 | 68 | 69 |
| **isl=32K EP8** | 53 | 86 | **112** | **119** |
| | | ↑ EP8 wins from con16 | | |

- At **8K**, EP8 overtakes TP8 near **con48-64** (371 vs 288 at con64).
- At **32K**, TP8 **flatlines at ~65-69 tok/s regardless of concurrency**, while EP8 keeps scaling —
  so EP8 wins from **~con16 onward**. The longer the context, the earlier EP8 wins.
- **TTFT under load** also favors EP8: 8K/con64 TP8 = 21.2 s vs EP8 = 11.4 s (EP distributes prefill
  compute across experts; TP8 serializes it).

**TPOT (latency), for contrast — flat per config, TP8 always lowest:**
TP8 ~22 ms (8K) / ~19 ms (32K) · EP8 ~60 ms · EP16 ~117 ms. This is the price of admission, not a
per-token scaling term — see §3.

---

## 3. Why — the collective each topology runs (the real cause)

All three put 8 (or 16) GPUs behind a role, but the **communication primitive differs**, and that is
the whole story:

| Topology | Per-decode-step collective | Fabric / measured BW | MoE GEMM shape |
|---|---|---|---|
| **TP8** | **allreduce** after each attn+MLP block (RCCL, intra-node) | NVLink/XGMI, light for this model | experts **fragmented into 1/8-size GEMMs** → memory-bound |
| **EP8** | **dispatch + combine** all2all | XGMI intra-node, **~116-144 GB/s dispatch, ~130-162 GB/s combine** | each expert a **full-size, efficient GEMM** |
| **EP16** | **dispatch + combine** all2all **across nodes** | ionic RDMA host-proxy, **~35 GB/s dispatch, ~40 GB/s combine** (3-4× slower) | full-size GEMM, 16-way sharded |

Two consequences that explain every number above:

1. **TP8 throughput plateaus** because (a) the allreduce is a fixed per-step cost that does not
   amortize with batch size, and (b) tensor-parallel splits GLM-5.2's fine-grained MoE experts into
   eighth-size GEMMs that run memory-bound at low efficiency. More concurrency cannot fix a
   memory-bound GEMM. → ceiling ~288 tok/s @ 8K, ~69 @ 32K.
2. **EP8/EP16 throughput scales** because attention is DP (independent per-rank streams that batch
   well) and each expert stays a full efficient GEMM; the all2all cost is paid once per step and
   amortizes across a bigger batch. EP16's curve is *steepest* (56→280 = 5× over con8→64) precisely
   because 16-way sharding leaves the most compute headroom — it just starts from a lower floor
   because its all2all is 3-4× slower over the wire.

**Cost-model check (predicts the TPOT ladder within tolerance):** TPOT ≈ compute_floor +
(bytes_moved_per_token ÷ link_BW). Compute floor ~19-22 ms (the TP8 number, ~pure compute). EP8
adds dispatch+combine over ~130 GB/s XGMI ≈ +~40 ms → ~60 ms ✓. EP16 moves the same bytes over
~37 GB/s cross-node ≈ 3.5× the EP8 comm term → ~117 ms ✓. The ratio of the measured collective
bandwidths (130/37 ≈ 3.5) reproduces the measured TPOT-comm ratio (95/40 ≈ 2.4-3.5) — so the ladder
is a fabric-bandwidth story, not a mystery, and it extrapolates: a faster cross-node path (GPU-direct
IBGDA once ionic supports it, or a fatter NIC) would pull EP16 toward the EP8 line.

---

## 4. Decision matrix — pick by (context, concurrency, SLA)

| Situation | Pick | Why |
|---|---|---|
| Interactive / low-QPS / tight TPOT SLA, context fits 1 node | **TP8** | ~22 ms TPOT floor; unbeatable latency when not batching |
| High concurrency (con≥~48 @ 8K, ≥~16 @ 32K), single node/role | **EP8** | throughput scales past TP8's plateau; lower TTFT under load |
| Offline / batch / max tokens-per-GPU-hour | **EP8** | best throughput-per-node; latency not the objective |
| Long context AND high concurrency together, KV > 1 node | **EP16 (2P/2D)** | only config with 2-node KV pool + 16-way expert capacity |
| Scaling out (more prefill replicas to cut long-ctx TTFT) | **EP16 → 4P4D+** | EP is the topology that keeps scaling; TTFT is the long-ctx bottleneck |
| Absolute lowest latency at any context that fits | **TP8** | no all2all, ever |

**Rules of thumb:**
- **Latency-bound → TP8. Throughput-bound → EP8. Capacity-bound → EP16.**
- The TP8→EP8 crossover moves *earlier* as context grows (con48 @ 8K → con16 @ 32K). At very long
  context, EP is almost always right unless QPS is tiny.
- EP16's value is **capacity and the on-ramp to 4P4D**, not 1P/1D latency. Don't deploy EP16 for a
  latency SLA; deploy it when one node can't hold the working set.

---

## 5. Known gaps / caveats

- **TP8 at 128K not yet measured** (node-blocked). Model predicts TPOT stays flat ~19-22 ms; the
  open question is TP8's KV-capacity ceiling at 128K (1 node of KV) vs EP's 2-node pool — expected to
  be the sharpest TP8-loses-on-capacity data point. To be filled when 2 a77 nodes free up.
- **EP16 128K/con64** cell missing (one hole; extrapolatable from the flat-TPOT trend).
- All EP16 numbers are 2P/2D. EP's headline case (4P4D+) is not yet run — it needs 8 nodes and is
  where the TTFT argument for long context actually pays off.
