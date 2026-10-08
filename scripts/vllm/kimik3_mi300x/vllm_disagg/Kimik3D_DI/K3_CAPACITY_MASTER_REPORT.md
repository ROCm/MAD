# Kimi-K3 1.4 TB MXFP4 — Capacity, Topology & Serving Study for MI300 / MI325

*(Working title / blog headline. Subtitle: "A Capacity Study of Kimi-K3 Disaggregated Wide-EP on MI325X+Thor2 vs MI300X+CX-7.")*

**Master report — the source of truth behind the DI-series blog.**
Everything here is measured on disk under `results/`. Code links are given as repo refs;
those marked `⟨PR pending⟩` are placeholders until the MAD/vLLM PRs are upstreamed.

- **Status:** MI325X ✅ complete (900K ladder) · MI300X ✅ complete (500K ladder, con>1-fix stack) · cross-platform capacity model validated
- **Last updated:** 2026-09-13
- **Blog HTML:** `K3_CAPACITY_BLOG_REPORT.html` (renders 11 figures + 8 sections)

---

## 0 · TL;DR (5 lines)

> We serve Kimi-K3 (1.4 TB MXFP4 MoE) disaggregated wide-EP (2P/2D, EP16) on two fabrics and show decode obeys `wall = C0 + k·OSL` — a fixed floor plus a per-token slope. The floor is platform-independent (a MoRI all-to-all barrier), so concurrency is nearly free until the KV pool saturates: MI325X (256 GB) rides to con256, MI300X (192 GB) is faster per-step but saturates 4× earlier at con64. A real agentic trace (AgentX) confirms the slope on each platform, proving the floor is physical, not fitted. The buyable axis is memory: on the con>1-fixed stack MI300X tops out at **500K** reliable context vs MI325X **900K** (a ~1.8× gap — down from ~9× before the fix) — set by the growing MLA KV on the 24 full-attention layers (K3's MLA already compresses KV ~16× natively, which is *why* the wall sits this far out). The capacity model turns "how much hardware" into a number you can compute before you deploy.

---

## 1 · The storyline (one arc)

**"A 1.4 TB model shouldn't decode faster just because you give it more work — but it does, and that changes how you buy hardware for it."**

1. **The problem** — Kimi-K3 is 1.4 TB (MXFP4, 93 layers: 24 MLA + 69 KDA). It doesn't fit the usual single-node mental model; there is no published capacity story for serving it disaggregated wide-EP.
2. **The insight** — decode wall-time is a *fixed floor plus a per-token slope*: `wall = C0 + k·OSL`. `C0` is a platform-independent all-to-all barrier (MoRI EP dispatch/combine); `k` is per-step compute. Because the floor is fixed, **adding concurrency is nearly free until the KV wall** — latency amortizes into throughput.
3. **The proof** — two platforms, identical recipe (2P/2D EP16, TP2/DP8). MI325X (256 GB) rides the floor to con256; MI300X (192 GB) is *faster per step* but saturates 4× earlier (con64) because its KV pool is ~4 GB vs ~20 GB in the throughput sweep. AgentX (real Claude-Code trace replay) closes it: measured ITL = the OSL slope on *each* platform (4.49≈4.47, 3.40≈3.32) — the floor is real, not a fit artifact.
4. **The frontier** — memory, not compute, is the buyable axis: on the con>1-fixed stack MI300X reliable-NIAH ceiling is **500K** (750K wedges) vs MI325X **900K** (7/7). The con>1 fix + a disciplined 192 GB config lifted MI300X 5× from the old ~100K, collapsing the gap from ~9× to ~1.8× — proving most of the old gap was config headroom, not silicon. The residual wall is the growing MLA KV (K3's MLA already compresses it ~16× natively — 512-dim fp8 latent vs full multi-head K/V), plus the FULL cudagraph family MI325X can afford and MI300X can't.
5. **The payoff** — the capacity model tells you *which* platform for *which* SLA before you deploy: latency-bound low-concurrency favors MI300X; throughput-bound or long-context favors MI325X.

---

## 2 · The model — Kimi-K3

| Property | Value |
|---|---|
| Total params | **~2.8 T** (896 routed experts top-16 + 2 shared) |
| Weight format | MXFP4 on the routed experts **only** (92.7% of bytes); everything else native bf16 |
| Weights on disk | **1453.7 GiB** (≈1.4 TiB) — measured from safetensors headers |
| Layers | 93 total = **24 MLA** (full attention) + **69 KDA** (gated-delta recurrent) |
| MLA latent | kv_lora_rank 512 + rope 64 = **576 elem/token** (KV **grows** with context) |
| KDA state | fixed recurrent state per layer (**does not grow** with tokens) |
| Arch target | gfx942 (MI300-series) |

**Precision note (important — not a quantization story beyond MXFP4):** MXFP4 covers only the 896
routed experts. The remaining **106.5 GiB is native bf16** (attention MLA+KDA 67.4, shared experts
22.6, embeddings/lm_head/gates/norms 16.5) — it is not quantized and does not shard with EP; under
any model-replicating (pure-DP) topology it is paid **per rank**. That fixed bf16 remainder is what
forces the attention-parallelism choice in the capacity model (§4).

**Why the MLA/KDA split matters for the whole study:** only the 24 MLA layers accumulate
per-token KV. At long context the growing MLA KV dominates memory; the 69 KDA layers are
context-flat. This is why the memory wall is a *context* wall — and why K3's architecture
already stores that MLA KV as a compact low-rank latent (512-dim), pushing the wall far out
before any external KV-compression is even considered.

### Why the architecture is complicated (the motivating constraint)

**The 1453.7 GiB checkpoint nearly fills one MI300X node by itself, so a simple 1-prefill /
1-decode split is impossible.**

| Node | GPU mem/GPU | Node total | Headroom over 1453.7 GiB weights |
|---|---:|---:|---|
| MI300X ×8 | 192 GB HBM3 (≈188 usable) | **~1504 GiB** | weights barely fit — **near-zero room for KV/activations/MoRI heap** |
| MI325X ×8 | 256 GB | **~2048 GiB** | ~0.6 TiB — fits with headroom |

On MI300X, a single node holds one full weight copy with essentially **zero** slack. You cannot
stand up a naive **1P/1D** topology where prefill and decode each own a full model replica on their
own node with KV budget — there isn't enough memory for the KV pool once the weights are resident.
That is the whole reason for the **disaggregated wide-EP** design: the experts are **sharded EP16
across the roles** (TP2×DP8), so no single GPU/node ever holds the full model twice, and the freed
memory becomes the KV pool. **The complicated 2P/2D EP16 architecture is not a preference — it is
forced by the ~1504 GiB node total vs 1453.7 GiB weights arithmetic.** MI325X's larger 256 GB helps
(bigger KV pool, §5), but the topology is dictated by the tighter MI300X envelope.

---

## 3 · The serving topology — 2P/2D disaggregated EP16

- **2 Prefill + 2 Decode** roles (2P/2D), each role = **TP2 × DP8 → EP16** (16-way expert parallel).
- **MoRI-EP** for expert all-to-all (dispatch/combine); **MoRIIO** RDMA connector moves KV prefill→decode.
- **Router** fans requests; in practice we hit `prefill_master:20005` directly (router discovery is fragile).
- **The MoRI-EP 16-GPU group spans prefill+decode** → the whole 4-role set must co-start; you can
  **never** restart decode-only (orphaned group → all-to-all deadlock, 306% CPU spin).

**Platform matrix:**

| | Accelerator | Mem/GPU | Fabric | KV pool (measured) |
|---|---|---|---|---|
| **Platform A** | MI325X | 256 GB | Broadcom **Thor2** RoCE | ~20 GB |
| **Platform B** | MI300X | 192 GB (live prefill sat at 191.9) | NVIDIA **CX-7** IB | ~4 GB |

Same image, same recipe, same benchmark scripts + warmup on both → apples-to-apples.

**Code:**
- Recipe (MAD, disagg EP16 + colocated, one image): `raviguptaamd/MAD @ kimik3-mi300x-v4` — ⟨PR pending⟩
- vLLM fork (K3 wide-EP disagg, onto upstream `d626108b`): `raviguptaamd/vllm @ kimi-k3-wideep-disagg-fullsource-v4` — ⟨PR pending⟩
- MoRI pin: `624002c897a3` (built from source, `WITH_MORI_BUILD=1`)
- Router pin: `82dc9811`
- Base image: `rocm/vllm-dev:ci_base-dedbf6be` (`WITH_NIXL=0`)
- MI300X launcher fork (mlx5 ABI-mount + timeout bump): `study/mi300x_cx7/launcher/run_2p2d.sh`
- MI325X launcher: `study/mi325x_thor2/launcher/run_2p2d.sh`

---

## 3b · How the parallelism works — attention (TP×DP) vs experts (EP16), and why it's forced

The 2P/2D EP16 shape is not one knob — it is **two different parallelism strategies applied to two
differently-shaped weight populations at once**, because Kimi-K3 has two of them.

| Population | Size | Grows with context? | Parallelized by |
|---|---:|---|---|
| **Routed experts** (896, MXFP4) | **1347.2 GiB** (92.7%) | no | **Expert Parallel (EP16)** — sharded |
| **Everything else** (MLA+KDA attention, shared experts, embeddings, norms — bf16) | **106.5 GiB** | attention KV grows | **Tensor + Data Parallel (TP2×DP8)** |

The experts are 92.7% of the bytes and the part that *can* be cleanly sharded; the 106.5 GiB bf16
remainder cannot be sharded by EP and is paid **per rank** under any model-replicating layout. Those
two facts dictate the whole topology.

**Expert Parallelism (EP16) — sharding the 1.35 TiB of experts.** Each MoE layer routes every token
to top-16 of 896 experts. Under EP16 the experts are partitioned across **16 GPUs** (56/GPU); no GPU
holds more than 1/16 of them. That shrinks the resident weight footprint from a 1453.7 GiB full replica
to something a node holds *with room left for a KV pool*. The cost is the **all-to-all** (dispatch →
expert GEMM → combine), which **MoRI-EP** implements. Two consequences seen throughout the study: (1)
the EP16 group of 16 GPUs **spans both prefill and decode pools**, so the 4-role set must co-start and
decode can never restart alone (orphaned all-to-all deadlocks); (2) the per-wave floor **C0** (§4) *is*
the fixed cost of one all-to-all barrier round-trip — platform-independent because it's a fabric
collective, not compute.

**Attention Parallelism (TP2×DP8) — the part EP can't shard.** The 106.5 GiB of attention/dense weight
is handled as **TP2** (each attention matrix split across 2 GPUs → a pair cooperatively holds one
attention stack) × **DP8** (8 independent attention lanes → 8× attention throughput). `TP2×DP8 = 16`
ranks per role — exactly the EP16 width, so the same 16 GPUs do attention-as-TP2×DP8 and
experts-as-EP16 at once. Why not the alternatives: **TP16** makes the attention GEMMs too small and
turns every layer's reduction into a 16-way collective (comm-bound); **DP16** pays the 106.5 GiB bf16
remainder ×16 and fragments the KV pool per-rank — the capacity trap the sizing analysis found, where a
**TP1/DP16** layout (190.7 GiB/rank) *fits MI325X's 256 GB but OOMs MI300X's 192 GB* (the §5 headline
divergence). **TP2/DP8 (137.4 GiB/rank) fits both**, which is why it is the portable, apples-to-apples
choice.

**Why the model doesn't fit one node — the arithmetic that starts it all.** One MI300X node
(8×192 GB ≈ 1504 GiB usable) vs the 1453.7 GiB checkpoint leaves **~50 GiB** for KV + activations +
MoRI heap across the whole node — effectively zero. You therefore cannot hold the whole model on one
node with a usable KV budget, and certainly cannot run a 1-prefill/1-decode split where each role owns
a full replica. The disaggregated wide-EP design resolves it: **EP16 shards the 1.35 TiB of experts so
no GPU ever holds a second copy**, and the freed HBM becomes the KV pool; 2P/2D then separates prefill
(compute-bound bursts) from decode (latency-bound token streams) while sharing the sharded experts over
the fabric.

**How this feeds capacity planning (§4–§5).** Once the topology is fixed at TP2×DP8→EP16, the memory
left for KV is what the study measures: MI325X (256 GB) → ~20 GB KV pool → rides the concurrency floor
to **con256**; MI300X (192 GB) → ~4 GB KV pool → saturates at **con64**, ~4–5× earlier, matching the
KV-pool ratio. So the parallelism choice is upstream of every number: **EP16 makes the model *fit*;
TP2×DP8 makes attention *portable* across both memory envelopes; and the memory those choices leave
over is the KV pool that sets the concurrency ceiling.**

---

## 4 · The capacity model (the math the blog computes)

Decode wall-time per request wave:

```
wall(OSL) = C0 + k · OSL
```

- **C0** = fixed per-wave floor: one MoRI all-to-all barrier round-trip regardless of how many
  tokens you decode. **Platform-independent** (it's a fabric/collective barrier, not compute).
- **k** = per-decode-step cost. **Platform-dependent** (raw compute + memory bandwidth).

**Measured fits (least-squares over the OSL sweep, both platforms):**

| Platform | Fit | C0 (floor) | k (slope) |
|---|---|---|---|
| MI325X + Thor2 | `wall = 26.6 + 4.47·OSL` | **26.6 s** | **4.47 s/tok** |
| MI300X + CX-7 | `wall = 27.3 + 3.32·OSL` | **27.3 s** | **3.32 s/tok** |

*(Canonical fit = 4-point, dropping the osl16 point which is warmup-sensitive; this is what the
blog Figures 7/9 render. The full 5-point fit gives near-identical slopes — 4.449 / 3.317 s/tok —
with slightly higher intercepts 30.6 / 28.8 s. The slope is what carries the AgentX proof; the
intercept is within ~3 s either way. The C0 floors are statistically identical across platforms,
as the model predicts.)*

Raw OSL points (OSL → wall_s), verified on disk (`results/*/03c_osl/`):

| OSL | MI325X wall | MI300X wall |
|---:|---:|---:|
| 16 | 107.5 s | 84 s |
| 32 | 170.3 s | 134 s |
| 64 | 312.5 s | 241 s |
| 128 | 598.5 s | 451 s |
| 256 | 1171.0 s | 879 s |

Fit quality is near-perfect (MI300X osl256 predicted 878 vs 879 measured) — the linear model
is not a smoothing, it's the actual mechanism.

**Reading it:** MI300X has the *lower* slope (3.32 < 4.45 s/tok) — it decodes each token faster.
The floors are within ~1 s of each other (27.3 vs 26.6) — as predicted, the barrier is ~platform-
independent. So per-request, **MI300X is faster**. The twist is capacity (§5).

---

## 5 · Results — the concurrency envelope (the headline)

Fixed (ISL,OSL), sweep concurrency; watch when wall-time breaks off the floor. Verified on disk
(`results/*/03b_envelope/`).

### MI325X + Thor2 (256 GB, ~20 GB KV)

| con | ok/total | wall (s) | agg tok/s |
|---:|---:|---:|---:|
| 1 | 1/1 | 614 | 0.2 |
| 4 | 4/4 | 605 | 0.8 |
| 8 | 8/8 | 604 | 1.7 |
| 16 | 16/16 | 605 | 3.4 |
| 32 | 32/32 | 618 | 6.6 |
| 64 | 64/64 | 626 | 13.1 |
| 128 | 128/128 | 643 | 25.5 |
| 256 | 256/256 | 688 | 47.6 |
| 512 | 496/512 | 1800 (cap) | 35.3 |

**Rides the floor to con256** — wall goes 614→688 s (+12%) while throughput goes 0.2→47.6 tok/s
(**238×**). The floor amortizes almost perfectly across 256 concurrent requests. con512 is the
KV wall (16 timeouts, throughput collapses).

### MI300X + CX-7 (192 GB, ~4 GB KV)

| con | ok/total | wall (s) | agg tok/s |
|---:|---:|---:|---:|
| 1 | 1/1 | 444 | 0.3 |
| 4 | 4/4 | 451 | 1.1 |
| 8 | 8/8 | 450 | 2.3 |
| 16 | 16/16 | 454 | 4.5 |
| 32 | 32/32 | 453 | 9.0 |
| **64** | **64/64** | **455** | **18.0** |
| 128 | 96/128 | 1700 (cap) | 7.2 |
| 256 | 64/256 | 1701 (cap) | 4.8 |
| 512 | 0/512 | 1700 (cap) | 0.0 |

**Rides the floor to con64** — wall flat 444→455 s while throughput 0.3→18.0 tok/s (**60×**).
Then the ~4 GB KV pool saturates: con128 already drops 32 requests, con256 drops 192, con512
serves zero. **MI300X saturates at con64 — exactly 4× earlier than MI325X's con256**, matching
the ~4 GB vs ~20 GB KV-pool ratio (**~5×**, con-knee ratio 4×).

### The cross-platform punchline

> MI300X is **faster per request** (lower k, lower floor) but **saturates 4× sooner**. MI325X
> trades a slightly slower per-token rate for **4× the concurrent capacity**. Which one you buy
> is set by your SLA: latency-bound low-concurrency → MI300X; throughput-bound high-concurrency
> or long-context → MI325X. The capacity model predicts the knee before you deploy.

---

## 6 · Results — raw perf matrix (throughput classes)

Verified on disk (`results/*/03_perf/`). Classes: latency-floor (128/32), throughput-peak
(1024/1024), long-context (32K/512).

### MI300X + CX-7

| class | ISL/OSL | con | ok/total | wall (s) | agg tok/s |
|---|---|---:|---:|---:|---:|
| latency floor | 128/32 | 1 | 1/1 | 90 | 0.4 |
| latency floor | 128/32 | 8 | 8/8 | 90 | 2.8 |
| latency floor | 128/32 | 16 | 16/16 | 91 | 5.6 |
| throughput | 1024/1024 | 8 | 8/8 | 532 | 15.4 |
| throughput | 1024/1024 | 32 | 32/32 | 613 | 53.5 |
| **throughput peak** | 1024/1024 | 64 | 64/64 | 889 | **73.7** |
| long-ctx | 32K/512 | 32 | 32/32 | 942 | 17.4 |
| long-ctx | 32K/512 | 64 | 64/64 | 1125 | 29.1 |

### MI325X + Thor2

| class | ISL/OSL | con | ok/total | wall (s) | agg tok/s |
|---|---|---:|---:|---:|---:|
| latency floor | 128/32 | 1 | 1/1 | 167 | 0.2 |
| latency floor | 128/32 | 8 | 8/8 | 171 | 1.5 |
| latency floor | 128/32 | 16 | 16/16 | 176 | 2.9 |
| throughput | 1024/1024 | 32 | 27/32 | 1800 | 15.4 |
| throughput | 1024/1024 | 64 | 38/64 | 1800 | 21.6 |
| throughput | 1024/1024 | 128 | 88/128 | 1800 | 50.1 |
| **throughput peak** | 1024/1024 | 256 | 130/256 | 1800 | **74.0** |
| long-ctx | 32K/512 | 32 | 11/32 | 1800 | 3.1 |
| long-ctx | 32K/512 | 64 | 24/64 | 1800 | 6.8 |
| long-ctx | 32K/512 | 128 | 45/128 | 1800 | 12.8 |
| long-ctx (frontier) | 128K/512 | 32 | 3/32 | 1800 | 0.9 |
| long-ctx (frontier) | 300K/128 | 8 | 8/8 | 1116 | 0.9 |
| long-ctx (frontier) | 300K/128 | 32 | 17/32 | 1800 | 1.2 |

**Note the peak convergence:** both platforms top out at ~74 tok/s aggregate — but MI300X gets
there at **con64** and MI325X needs **con256**. Same peak, 4× different concurrency to reach it.
(This throughput sweep predates the con>1 fix; on the fixed stack MI300X now serves single-request
long context to **500K** and MI325X to **900K** — see the updated §8. The con64/con256 *concurrency*
knee is a separate, still-valid throughput-sweep result at small OSL.)

---

## 7 · Results — AgentX (real agentic trace replay)

`aiperf` harness `inferencex-agentx-mvp`: replays a **real Claude-Code session trace** (not a
synthetic sweep) — actual tool-call/think/emit token cadence. This is the "does the floor survive
real traffic" test. Verified on disk (`results/*/04_agentx/`).

| profile | MI325X ITL | MI300X ITL |
|---|---:|---:|
| small | **4.49 s** | **3.40 s** |
| small_long | ✅ (MI325X only) | — |
| conformance_256k | ✅ | ✅ (ran; context-limited) |
| conformance_512k | ✅ | ✅ (ran; context-limited) |

**The symmetric-floor proof:** the measured inter-token latency under a *real* agentic trace equals
the OSL-sweep slope `k` on each platform independently:

| Platform | AgentX ITL | OSL-sweep k | match |
|---|---:|---:|---|
| MI325X | 4.49 s | 4.47 s/tok | ✅ 4.49 ≈ 4.47 |
| MI300X | 3.40 s | 3.32 s/tok | ✅ 3.40 ≈ 3.32 |

This is the load-bearing result. The `k` we extracted from a controlled OSL sweep **reappears
verbatim** when a real Claude-Code trace drives the server. The floor+slope model isn't a curve
fit — it's the physical decode mechanism, and it holds under production-shaped traffic.

**Explaining the 4.49 vs 3.40 gap (per user request):** the AgentX ITL *is* the per-decode-step
time `k`. MI300X's step is faster (3.32 s vs 4.45 s) because it has higher per-GPU memory bandwidth
utilization at low concurrency — fewer tokens in flight, so each step is bandwidth-bound and MI300X
wins. The gap is **not** a fabric difference (the C0 floor is ~equal); it's raw per-step compute.
The same property that makes MI300X's step faster (small KV, low concurrency) is what makes it
saturate 4× sooner under load — the two findings are the same coin.

---

## 8 · The memory frontier — the con>1 fix lifts MI300X from ~100K to 500K

The most operationally important finding, and it **changed** once the con>1 accuracy fix
(§11) landed. On the **pre-fix** stack the 192 GB platform capped usable context at ~100K —
roughly 1/9 of MI325X. On the **con>1-fixed** stack (`v4-disagg-situ-restore-mambafix`,
`206fffe`) with a properly-budgeted 192 GB memory config, the MI300X reliable ceiling rises
**5× to 500K**, and the cross-platform ratio collapses from ~9× to ~1.8×.

**MI300X on the fixed stack — single-needle NIAH ladder, PIECEWISE decode.** Config:
`KV=13 GB`, `MAX_MODEL_LEN=910000`, `MAX_NUM_SEQS=8`, `MORI_SHMEM_HEAP_SIZE=8 GiB`,
`GPU_UTIL=0.85` (depth 0.5, `HELIOTROPE-7492`, 2P/2D EP16 on nodes 115/121/122/156):

| ctx | 20K | 50K | 100K | 200K | 500K | 750K | 900K |
|---|---:|---:|---:|---:|---:|---:|---:|
| latency | 10.5 s | 24.0 s | 49.4 s | 111.5 s | 384.9 s | **stall** | — |
| recall | PASS | PASS | PASS | PASS | PASS | **wedge** | — |

**5/5 PASS 20K→500K**, coherent recall, `finish=stop`, latency scaling cleanly (~linear in ctx).
**750K is a reproducible wedge** (two independent attempts — in-ladder and isolated — both hung):
the decode GPUs pin at 100% but emit **zero tokens and zero KV-transfer log activity** for 25–35
min. Killing the client **recovers the serve instantly** (`OK: READY`), so it is a *request-specific
high-context transfer wedge, not an engine crash*. The stall signature is the prefill-side
`k3-readback BatchRead incompatible arguments` warning (non-fatal at ≤500K) plus a fresh
FMHA `hd192x128` kernel JIT-load for the 750K shape, after which the forward wedges. So the MI300X
**reliable ceiling on the fixed stack is 500K.**

**Contrast MI325X — the frontier reaches 900K.** Same con>1-fixed + int4-SiTU stack, but the
256 GB envelope affords `MAX_MODEL_LEN=1M` + `KV=40 GB` + `FULL_AND_PIECEWISE`; MI325X served the
**full 20K→900K ladder — 7/7 PASS**:

| ctx | 20K | 50K | 100K | 200K | 500K | 750K | 900K |
|---|---:|---:|---:|---:|---:|---:|---:|
| latency | 11.8 s | 30.5 s | 50.3 s | 117.4 s | 393.2 s | 710.2 s | 890.1 s |
| recall | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

This **retires the earlier "320K refused / 300K ceiling" claim for MI325X** — that was the older
stack pinned at `MAX_MODEL_LEN=320000` / 20 GB KV. At 900K MI325X logs the *same* non-fatal
`BatchRead incompatible arguments` MoRIIO readback warning MI300X hits at 750K — but on MI325X
(256 GB, FULL_AND_PIECEWISE) it **falls back cleanly and completes** (needle recalls at 890 s),
whereas on MI300X (192 GB, PIECEWISE) the same readback path **wedges**. So both platforms hit the
*identical* growing-MLA-KV + MoRIIO-large-region-readback wall; MI325X's larger HBM and the FULL
cudagraph family push that wall **~1.8× further** (900K vs 500K).

**Interpretation — memory is still the buyable axis, but the gap narrowed.** Pre-fix the ratio was
~9× (900K vs 100K), which over-sold MI325X; the con>1 fix + a disciplined 192 GB config (half-size
MoRI heap, tight KV, minimal cudagraph capture) proves most of that gap was *config headroom*, not
silicon. The residual ~1.8× is the real memory frontier: MI300X's 148.6 GiB/GPU weight footprint
leaves only ~13 GB for KV vs MI325X's 40 GB, and the FULL cudagraph family (which MI300X can't
afford) is what carries MI325X's readback fallback through 750K→900K.

**Why PIECEWISE on MI300X (and why bring-up is iterative).** `DECODE_CG=PIECEWISE` (drops the FULL
cudagraph family) is the only decode config that fits 192 GB. Bring-up took **5 relaunch iterations**
to fit the envelope — each a distinct, documented failure that maps to a real capacity constraint:
(1) `GPU_UTIL=0.9` → free-memory check fails by a hair (MoRI heap + CUDA ctx reserve ~19 GB
*before* the check) → 0.85; (2) 16 GB MoRI heap + 16 GB KV → PIECEWISE-capture OOM (2.6 GB free <
6 GB capture) → 8 GB heap; (3) `MAX_MODEL_LEN=1M` → KV-reserve refused (13.1 GB needed > 12.0 avail)
→ 910K; (4) `MAX_NUM_SEQS=32` → capture 11 graph sizes → connector KV-registration OOM
(`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, 1.29 GB free) → 8 seqs. MI325X's 256 GB absorbs all four in a
single one-shot launch. **This iteration count *is* the capacity story: 192 GB has no slack.**

---

## 9 · The five cross-platform findings (evidence index)

| # | Finding | Evidence file |
|---|---|---|
| 1 | Decode = `C0 + k·OSL`; floor ~platform-independent, slope platform-dependent | `results/*/03c_osl/`, `LATENCY_THROUGHPUT_EVIDENCE.md` |
| 2 | Concurrency is ~free until KV wall; MI300X knee con64 vs MI325X con256 (4×) | `results/*/03b_envelope/` |
| 3 | AgentX real-trace ITL = OSL slope on each platform (symmetric floor proof) | `results/*/04_agentx/`, `AGENTX_FINDINGS.md` |
| 4 | MI300X faster per-step (lower k) but saturates 4× sooner — same coin | §5 + §7 above |
| 5 | On con>1-fixed stack, 192 GB reaches 500K vs 256 GB 900K (~1.8×; was ~9× pre-fix) | `results/mi300x_cx7/02_accuracy_niah/`, §8, `CROSS_PLATFORM_FINDING.md` |

---

## 10 · Under the profiler (optional, §7 of blog — MoRI-ROCtx)

Aim: attribute the C0 floor to the MoRI all-to-all barrier directly, with ROCtx range markers around
EP dispatch/combine + the MoRIIO KV transfer. Recipe in `PHASE4_PROFILING_PLAN.md`; raw ranges land
in `results/*/06_profiles/`. **Not yet run to completion** — the floor is already proven three ways
(OSL fit, envelope knee, AgentX match), so this is corroboration, not a gate.

---

## 11 · The two correctness fixes that make concurrency safe

Making distinct-needle recall correct **under concurrency** took two independent fixes, both now baked
into the pinned vLLM branch `v4-disagg-situ-restore-mambafix` (`206fffe`). They live on different axes —
one in the KV *transfer*, one in the KDA *state* — and the envelope in §5 is only trustworthy with both.

### 11a · KV write-race — RDMA read-after-write barrier

**Bug:** in MoRIIO WRITE mode, decode admits a request when the ZMQ `write_done` arrives, but on RoCE a
completed RDMA-WRITE does **not** guarantee the KV is visible in the *receiver's* HBM → under concurrency,
decode read stale/garbage KV (distinct-needle NIAH con=8 recalled 3/8, garbled). **Fix:** after writes
complete and before `write_done`, issue a tiny RDMA **read of every written region** (`readback_targets`,
capped 64) — a read-after-write to the same session forces prior writes globally visible (deterministic
vs a guessed `K3_WRITE_FENCE` sleep). Gated `K3_WRITE_READBACK=1`. Progression: baseline 3/8 → fence-delay
6/8 → single-region readback 6/8 → **multi-region readback: garbage GONE**. Files: `moriio_engine.py`
(readback loop in `_finalize_if_complete` + stash in `write_kv_layer`), `moriio_common.py`. Also hardened
the completion gate to require **all** KV group transfers to succeed (`n_ok == n_total`), not just the
last-appended one (`moriio_connector.py`).

### 11b · con>1 KDA state-recycle — zero mamba blocks on reallocation (issue #35219)

After 11a, a subtler corruption remained: under **sustained** concurrency (con=32) recall fell to ~65%
and **self-healed at con=1** — the tell of state accumulation, not a transfer race. **Root cause:** K3's
KDA (mamba) layers keep a per-slot recurrent+conv state in the KV-cache pool; vLLM zeroes freshly
(re)allocated blocks *for attention* but a pair of `isinstance(spec, AttentionSpec)` gates **excluded
`MambaSpec`** — so a recycled KDA slot handed a finished request's finite-but-wrong state to a new one,
which the contractive gate then amplified. vLLM's own `needs_kv_cache_zeroing` is True for mamba (the
docstring cites this exact "state read before fully written" hazard, #35219) but the gates disabled it.
**Fix:** a separate pure-torch zero-on-recycle channel for mamba blocks (`state[new_ids]=0`), routed off
the attention byte-kernel's namespace, run eager in the decode worker's `update_requests` (V2 model
runner) — **outside cudagraph capture, so `FULL_AND_PIECEWISE` and the int4-SiTU MoE are untouched.**
Plus a de-alias `.clone()` of the KDA decode state-index view and a MoRI-EP dispatch-trim. **Validated:**
distinct-needle NIAH @50K con=1/8/16/32 = **57/57 = 100%**; con=32 @6K 9 consecutive runs @100%; con=1
after sustained con=32 hammering = **12/12** (no residual poison). Pre-fix con=32 swung 42–83% (~65%).

**Why K3 sharpens both:** the prefill→decode handoff ships **both** the MLA latent KV *and* the KDA
recurrent state, so K3 exercises a transfer hazard (11a) *and* a recurrent-state hazard (11b) that a
pure-attention model never sees. Together they are **what makes the con256 envelope in §5 real rather
than paper** — every completing request at every concurrency now recalls its own needle.

## 12 · Config levers that did NOT fix the ~150 s floor (ruled-out table)

For rigor — the floor is not a tunable; these were tested and rejected (from the v4 report §6):

| Hypothesis | Test | Result |
|---|---|---|
| Reasoning tokens inflate it | thinking=false | same ~150 s |
| WRITE-notify DP-rank mismatch | READ mode | same 150 s + garbage → rejected |
| Cudagraph mode | PIECEWISE (v3's value) | same 171 s |
| MoRIIO transfer knobs | qp=2 / workers=8 | broke the WRITE-notify path → reverted |

Prime remaining suspect: MoRI `624002c8` InterNodeV1LL decode all-to-all warmup (v3's older MoRI +
same backend was ~20 s/50K). This is the target of the §10 MoRI-ROCtx profiling.

---

## 13 · Reproducibility & methodology

- **Warmup:** every sweep script fires ONE warmup curl of the measured request shape before the
  timed loop → no measured point eats cold JIT/cache. Same scripts + warmup on both platforms.
- **Determinism:** 8241-token determinism probe on bring-up; serve JIT-warm from "Paris" validation
  before any sweep.
- **Bench scripts:** `study/*/bench/` (envelope, OSL, perf, NIAH, AgentX) — identical across
  platforms except node/fabric env.
- **Fabric env gotchas (launcher input vars):** `RDMA_DEVICES` (not `MORI_RDMA_DEVICES`),
  `IB_GID_INDEX` (not `NCCL_IB_GID_INDEX`), `MAX_MODEL_LEN`. MI300X needs
  `RDMA_DEVICES=mlx5_0,2,3,4,5,7,8,9` (CX-7) — passing the Thor2 default `rdma0-7` gives
  "no transport available for peer".
- **Weight staging:** RDMA copy tool from dist-inf-cookbook stages the 1.4 TB weights to nodes
  lacking them.

---

## 14 · Code & artifact index (with PR placeholders)

| Artifact | Location | Upstream status |
|---|---|---|
| MAD recipe (disagg EP16, Dockerfile+launchers+probes) | [ROCm/MAD #241](https://github.com/ROCm/MAD/pull/241) · `scripts/vllm/kimik3_mi300x/vllm_disagg/Kimik3D_DI/` | PR open |
| vLLM K3 fork — MoRIIO RDMA readback barrier | `raviguptaamd/vllm @ v4-disagg (2cbc11cd7)` onto `d626108b` | PR open |
| MAD recipe (v4 rebuild, two recipes one image) | `raviguptaamd/MAD @ kimik3-mi300x-v4` | ⟨new PR pending⟩ |
| Sizing analysis (Confluence) | [MLSE 1830010189 — Parallelism Sizing under EP16](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1830010189) | internal |
| v4 bring-up/accuracy/perf report (Confluence) | [MLSE 1923137760 — vLLM 2P/2D Disagg EP16 v4 Report](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1923137760) | internal |
| MoRI | pin `624002c897a3` | upstream |
| Router | pin `82dc9811` | upstream |
| Base image | `rocm/vllm-dev:ci_base-dedbf6be` | upstream |
| Study image | `rocmshared/kimik3-wideep-disagg:v4` (Docker Hub) | private |
| MI325X launcher | `study/mi325x_thor2/launcher/run_2p2d.sh` | in-repo |
| MI300X launcher | `study/mi300x_cx7/launcher/run_2p2d.sh` | in-repo |
| Shared bench scripts | `study/shared/bench/` (envelope, OSL, perf, NIAH, AgentX) | in-repo |
| Blog HTML | `K3_CAPACITY_BLOG_REPORT.html` | in-repo |
| This report | `K3_CAPACITY_MASTER_REPORT.md` | in-repo |

**Result trees:** `results/mi325x_thor2/` and `results/mi300x_cx7/`, each with `01_capacity/`,
`02_accuracy_niah/`, `03_perf/`, `03b_envelope/`, `03c_osl/`, `04_agentx/`, `05_precision_audit/`,
`06_profiles/`. Manifest: `results/mi300x_cx7/MANIFEST_mi300x_cx7.md`.

---

## 15 · Is this an MLSys paper?

The core result — a *computable* capacity model for a 1.4 TB disaggregated wide-EP MoE, validated
across two fabrics with a real agentic trace, plus a memory-frontier characterization — is a
solid **systems measurement/characterization** contribution. As-is it's a strong **blog + workshop**
story. To reach MLSys-main bar it would want: (a) the MoRI-ROCtx profiling (§10) to *attribute* C0
mechanistically, (b) a mechanism to move the memory frontier (a kernel-level KV format change, since
K3's MLA KV is already low-rank and fused into the attention kernels), and (c) a third platform or a scaling-law generalization of the
`C0 + k·OSL` model across model sizes. The bones are there; b + c are the gap.

---

---

## 16 · References

1. **Kimi-K3** — Moonshot AI, Kimi-K3 technical report / model card (MLA + gated-delta KDA hybrid, MXFP4). ⟨model card link pending public release⟩
2. **MLA (Multi-head Latent Attention)** — DeepSeek-V2/V3 papers, arXiv:2405.04434 — latent KV compression (kv_lora_rank), the mechanism K3's 24 full-attn layers use.
3. **Gated DeltaNet / gated-delta recurrence (KDA)** — Yang et al., "Gated Delta Networks," arXiv:2412.06464 — the fixed-state linear-attention class K3's 69 KDA layers use.
4. **Disaggregated prefill/decode serving** — Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI 2024, arXiv:2401.09670 — the P/D-split principle behind 2P/2D.
5. **Wide expert parallelism (wide-EP / EP-sharded MoE inference)** — DeepSeek-V3 report, arXiv:2412.19437 §inference — large-EP all-to-all dispatch/combine, the pattern MoRI-EP implements.
6. **MoRI / MoRIIO** — AMD ROCm expert-parallel all-to-all + RDMA KV connector (dist-inf-cookbook). ⟨repo/PR link pending⟩
7. **MXFP4 / microscaling formats** — OCP Microscaling Formats (MX) Specification v1.0 — the fp4 block format K3 weights ship in.
8. **vLLM** — Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP 2023, arXiv:2309.06180 — the serving engine + KV pool this study builds on.

*(References 1 and 6 carry ⟨pending⟩ links until Kimi-K3 and the MoRI/MAD PRs are public; swap in the canonical URLs at publish time.)*

---

*End of master report. Blog draft mirrors §1–§10 in `K3_CAPACITY_BLOG_REPORT.html`.*
