# GLM-5.2-FP8 1P/1D disaggregated serving — results (Crusoe MI355X + AINIC/ionic)

> **Cluster/scope note.** These numbers are from the **Crusoe** MI355X cluster (Pensando
> AINIC/ionic fw 1.117.5-a-77, KVM/VFIO), driven through **vllm-router**, ISL 8K/32K/128K.
> They are a DIFFERENT cluster and shape than the **AAC** 1P/1D-EP8 sweep in `GLM52_MI355X.md`
> (ISL 28,672/1,024). Do not merge or cross-compare the two tables directly. See
> `GLM52_EP16_IONIC.md` for the EP16 recipe.

All runs: GLM-5.2-**FP8** (`glm_moe_dsa`, 78 layers, e4m3 block-128), 1 prefill / 1 decode,
`max-model-len=131072`, `block-size=1`, `kv-cache-dtype=fp8`, no prefix caching, via
**vllm-router** (`raviguptaamd/router` @ `82dc9811`, branch `ravgupta/dp-roundrobin-on-tip`).
Fabric: AMD Pensando AINIC (ionic, fw 1.117.5-a-77) under KVM/VFIO. Benchmark: `vllm bench
serve`, random dataset, `osl=128`, `num_prompts = 3*con` (8K/32K), `= con` (128K).

## Three configs

| Config | Topology | all2all | Notes |
|--------|----------|---------|-------|
| **TP8** | 1 prefill node (tp8) / 1 decode node (tp8) | none | tensor-parallel, NCCL/RCCL allreduce |
| **EP8** | DP8+EP, 1 node each role | MoRI intra-node (XGMI) | `mori_high_throughput` / `mori_low_latency` |
| **EP16** | DP16+EP, **2 nodes each role** | MoRI **cross-node over ionic** | needs `MORI_EP_OVER_RDMA=1` (MoRI PR#558 host-CPU proxy) |

## Headline: TPOT ladder (per-output-token latency)

Each all2all hop roughly **doubles** TPOT; TPOT is **concurrency-independent** (it is
per-decode-step comm latency, not queueing):

| Config | TPOT (flat, all contexts/con) | tok/s @ 8K con8 | tok/s @ 8K con64 |
|--------|-------------------------------|-----------------|------------------|
| **TP8** | **~19–22 ms** (latency king) | 218 | — |
| **EP8** | **~60 ms** | 96 | 371 |
| **EP16** | **~117 ms** | 56 | 280 |

## Full matrix — EP8 (clean, via vllm-router)

| ISL | con | tok/s | Mean TTFT (ms) | Median TPOT (ms) |
|-----|-----|-------|----------------|------------------|
| 8K   |  8  |  96.1 |   2725 | 59.4 |
| 8K   | 16  | 156.1 |   4794 | 59.5 |
| 8K   | 32  | 243.8 |   6880 | 61.4 |
| 8K   | 64  | 370.6 |  11179 | 62.4 |
| 32K  |  8  |  56.4 |   9443 | 59.5 |
| 32K  | 16  |  87.3 |  13902 | 59.7 |
| 32K  | 32  | 112.9 |  24973 | 59.7 |
| 32K  | 64  | 118.6 |  52569 | 59.6 |
| 128K |  8  |  20.1 |  40852 | 60.5 |
| 128K | 16  |  26.1 |  64444 | 60.4 |
| 128K | 32  |  26.9 | 125827 | 60.5 |
| 128K | 64  |  27.4 | 248372 | 60.5 |

## Full matrix — EP16 (clean, via vllm-router)

| ISL | con | tok/s | Mean TTFT (ms) | Median TPOT (ms) |
|-----|-----|-------|----------------|------------------|
| 8K   |  8  |  56.2 |   3264 | 115.3 |
| 8K   | 16  |  96.4 |   5170 | 116.7 |
| 8K   | 32  | 168.7 |   7004 | 116.4 |
| 8K   | 64  | 279.7 |  10671 | 118.0 |
| 32K  |  8  |  38.8 |  10547 | 115.6 |
| 32K  | 16  |  65.8 |  13573 | 116.3 |
| 32K  | 32  | 104.7 |  20542 | 116.3 |
| 32K  | 64  | 113.6 |  48457 | 115.9 |
| 128K |  8  |  17.4 |  42239 | 116.5 |
| 128K | 16  |  24.9 |  61214 | 116.8 |
| 128K | 32  |  26.0 | 123516 | 116.8 |
| 128K | 64  |  n/a  |   n/a  | n/a (shape not completed) |

## NIAH long-context retrieval — all three = 10/10

Fixed harness (`benchmark_niah_v2.py`), maxtok=256, `enable_thinking:false`, sizes within
the 131K context window:

| Config | 2K words | 20K | 50K | 90K |
|--------|----------|-----|-----|-----|
| TP8    | 10/10 | 10/10 | 10/10 | 10/10 |
| EP8    | 10/10 | 10/10 | 10/10 | 10/10 |
| EP16   | 10/10 | 10/10 | 10/10 | 10/10 |

**GLM-5.2-FP8 long-context retrieval is perfect; fp8 KV cache does NOT hurt accuracy.**
Earlier "low NIAH" scores were a harness bug (`NIAH_MAXTOK=64` truncated the model's
chain-of-thought before it emitted the answer; no `enable_thinking:false` let the reasoning
model over-think a trivial task). See `benchmark_niah_v2.py`.

## Interpretation / when to pick which

- **TP8 = latency king at 1P/1D** (~22 ms TPOT, 218 tok/s @ con8). No all2all; TP allreduce
  is cheap for this model size on 8×MI355X. Near-optimal for 1P/1D — little headroom.
- **EP8 = balanced.** ~60 ms TPOT but throughput scales cleanly to **371 tok/s @ con64**
  (once the vllm-router replaces the single-threaded toy proxy that wedged at con≥32).
- **EP16 = comm-bound at 1P/1D** (~117 ms, the cross-node all2all tax). Throughput still
  scales. EP's real payoff is at larger scale (2P2D/4P4D, more experts sharded), not 1P/1D
  latency.
- **TTFT** scales hard with `context × concurrency` for all three — the single prefill engine
  serializes prompts. Adding prefill replicas (xPyD) is the #1 TTFT lever.

## Known non-viable / blocked levers (documented so they are not re-attempted)

- **MXFP4**: GLM-5.2-MXFP4 hits **Triton Code-209 on gfx950** (MXFP4 MoE kernel). This is
  why the recipe uses the **FP8** checkpoint. Unresolved at the kernel level.
- **DBO (dual-batch overlap)**: NOT viable on the MoRI/ionic stack. MoRI's MoE
  `prepare_finalize` has `supports_async()=False` (no dbo/ubatch/yield hooks); DBO overlap
  requires `supports_async=True`, provided only by `deepep_*`/`nixl`. DeepEP is unusable here
  (its all2all uses the GPU IBGDA doorbell that fails under KVM on ionic — the reason we use
  MoRI + host-proxy). ROCm gates DBO overlap to `deepep_high_throughput` (vllm config).
- **MTP / speculative** (`num_nextn_predict_layers=1`): **serves on TP8 and EP8 disagg**
  (−41% / −44% TPOT via the MoRIIO draft-block fix). **EP16-MTP does NOT serve** — cross-node
  cudagraph capture lockstep (`MTP_EP16_BREAKTHROUGH.md`); documented known limitation.
