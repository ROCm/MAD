# GLM-5.2-FP8 EP16 cross-node disagg on Crusoe MI355X + AINIC (ionic)

Companion to `GLM52_MI355X.md`. That file documents **1P/1D EP8** (single node per role,
intra-node XGMI all2all) measured on **AAC**. This file documents **EP16** — expert-parallel
across **2 nodes per role** (the MoE all2all crosses the ionic fabric) — measured on a
**different cluster** (Crusoe MI355X, Pensando AINIC/ionic fw 1.117.5-a-77, KVM/VFIO), driven
through **vllm-router**. The two are separate clusters and shapes; do not merge the numbers.

Scripts: `vllm_pd_ep16_launch.sh` (per-node) + `ep16_orchestrate.sh` (4-node driver).

---

## 1. Why EP16 needs Tej's PR #558 (MoRI host-CPU proxy)

EP8's all2all is **intra-node** (8 GPUs on one host, over XGMI) — no NIC involved. EP16 shards
the experts across **two nodes**, so the dispatch/combine all2all must cross the ionic RDMA
fabric. On ionic under KVM/VFIO the **GPU-initiated IBGDA doorbell MMIO fails**, so the stock
GPU-driven MoRI-EP dispatch never rings the NIC.

**Tej's PR #558** (`itej89/mori @ feat/ep-rdma-sharing`, `MORI_EP_OVER_RDMA=1` ->
`TransportType::PROXY`) adds a **host-CPU proxy**: the host rings the NIC doorbell via
`ibv_post_send`. This is what makes cross-node EP work on ionic. The image must be built with
`--build-arg WITH_MORI_EP_OVER_RDMA=1` (see the Dockerfile), which switches the MoRI source to
Tej's fork and applies `docker/mori_pr558_ionic.patch` (the ionic atomic-MR strip).

> The default image (`WITH_MORI_EP_OVER_RDMA=0`, `ROCm/mori @ 42e895472b08`) is the **validated
> EP8** build and is unchanged — it has no host-proxy and cannot do cross-node EP.

## 2. Topology (1P/1D-EP16 = 4 nodes)

```
prefill role  = DP16 : master(rank 0, api+kv-transfer) + headless(rank 8)   -> kv_producer
decode  role  = DP16 : master(rank 0, api+kv-transfer) + headless(rank 8)   -> kv_consumer
router        = vllm-router on the prefill master (intra-node-data-parallel-size 8)
```

## 3. Four bring-up gotchas (each blocks boot; all handled in `vllm_pd_ep16_launch.sh`)

1. **Master must NOT pass `--data-parallel-start-rank`.** Setting it on a non-headless node
   flips vLLM into hybrid/external DP-LB, which then rejects the headless secondary
   (`RuntimeError: Remote engine N must not use --headless in external or hybrid dp lb mode`).
   Only the **headless** node carries `--data-parallel-start-rank 8 --headless`; the master
   runs internal LB (single API-server group + DP coordinator).
2. **NCCL on TCP, not IB.** DP16 spans 2 nodes, so the DP coordinator needs a cross-node NCCL
   comm. mlx5 IB is not routable between these ionic nodes -> `ncclCommInitRank` fails
   ("unhandled system error"). Set `NCCL_IB_DISABLE=1` (`NCCL_SOCKET_IFNAME=ens3`). Safe: the
   DP-coordinator traffic is tiny; the heavy expert-all2all + KV ride MoRI/ionic, not NCCL.
3. **MoRI symmetric heap eats the GPU util budget.** It is allocated on-GPU at connector init,
   BEFORE vLLM's profiling mem-check. A 32 GB heap left only 222/288 GiB free and failed the
   0.80 check. Use `MORI_SHMEM_HEAP_SIZE=16G`, prefill util 0.80 / decode util 0.85.
4. **Drain GPUs before relaunch.** teardown->relaunch races GPU memory release; the dying
   attempt still holds ~65 GB and the new one fails the mem-check. Wait for `rocm-smi` used
   < ~6 GB on all nodes before launching. (A transient NFS `[Errno 116] Stale file handle` on
   one DP rank during model load just needs a decode relaunch — not a config bug.)

## 4. Run it

```bash
# build the EP16-capable image (PR #558):
docker build -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile \
    --build-arg WITH_MORI_EP_OVER_RDMA=1 -t vllm-disagg:glm52-ep16-ionic .

# edit the node/IP map at the top of ep16_orchestrate.sh (4 nodes: prefill master+headless,
# decode master+headless), then one command brings up router + all 4 workers:
bash ep16_orchestrate.sh
# wait for router log "Add Prefill" + "Add Decode", then benchmark through the router.
```

Key env baked into `vllm_pd_ep16_launch.sh`: `MORI_EP_OVER_RDMA=1`, `MORI_IO_DISABLE_ATOMIC_MR=1`,
`MORI_SHMEM_HEAP_SIZE=16G`, `NCCL_IB_DISABLE=1`, prefill all2all `mori_high_throughput` /
decode `mori_low_latency`, cudagraph prefill `NONE` / decode `FULL_AND_PIECEWISE`.

## 5. Measured (Crusoe MI355X + ionic, GLM-5.2-FP8, via vllm-router)

`vllm bench serve`, osl=128, `num_prompts = 3*con` (8K/32K) / `= con` (128K). See `RESULTS.md`
for the full three-config table. Headline — TPOT is a **flat comm tax per decode step**:

| Config | TPOT (flat) | 8K con8 tok/s | 8K con64 tok/s | all2all |
|--------|-------------|---------------|----------------|---------|
| TP8    | ~22 ms      | 218 | —   | none |
| EP8    | ~60 ms      | 96  | 371 | MoRI intra-node XGMI |
| EP16   | ~117 ms     | 56  | 280 | MoRI cross-node ionic (host-proxy) |

Each all2all hop ~doubles TPOT; it is concurrency-independent (per-step comm latency).
**NIAH long-context retrieval = 10/10 for all three** (2K/20K/50K/90K words) — fp8 KV does not
hurt accuracy, and EP topology does not change correctness.

## 6. Non-viable / blocked levers (documented so they are not re-attempted)

- **DBO (dual-batch overlap)** — NOT viable on MoRI/ionic. MoRI's MoE `prepare_finalize` has
  `supports_async()=False` (no dbo/ubatch/yield hooks); DBO overlap requires
  `supports_async=True`, provided only by `deepep_*`/`nixl`. DeepEP is unusable here (its
  all2all uses the GPU IBGDA doorbell that fails under KVM on ionic — the reason we use MoRI +
  host-proxy). vLLM gates ROCm DBO overlap to `deepep_high_throughput`.
- **MXFP4** — GLM-5.2-MXFP4 hits **Triton Code-209 on gfx950** for the MoE kernel in this stack
  (blocked; FP8 is the working path). Tracked separately for a fix across TP8/EP8/EP16.
- **MTP / speculative** (`num_nextn_predict_layers=1`) — **works on TP8 and EP8 disagg**
  (−41% / −44% TPOT; the MoRIIO block-transfer fix handles the extra draft KV block). **EP16-MTP
  does NOT serve** — cross-node cudagraph capture lockstep (see `MTP_EP16_BREAKTHROUGH.md` §end);
  documented known limitation.
