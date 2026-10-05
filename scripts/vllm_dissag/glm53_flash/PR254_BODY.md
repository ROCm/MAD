# GLM-5.3-Flash-FP8 disaggregated serving (TP4 / EP8) on MI355X (gfx950) + ionic

Disaggregated (1P/1D) prefill/decode serving of **GLM-5.3-Flash-FP8**
(`Glm5NextForConditionalGeneration` — FP8 E4M3, MLA + DSA sparse-indexer +
KDA-linear hybrid) over the **MoRIIO** RDMA KV connector on AMD **MI355X (gfx950)**
+ Pensando **ionic** (AMD AI NIC), MoRI **WRITE** mode. Verified exact
needle-in-haystack recall to **433K+ tokens**, all depths (TP4 on the self-contained
v3 image; EP8 recall verified on the earlier overlay recipe — see Verification status).

## What this PR ships
A **self-contained** Docker recipe + the runtime/orchestration scripts. The image
carries every code fix in-source (no runtime `.py` overlay mounts, no external
`.so`/router-binary mounts). The only launch-time host dependencies are
node-specific and documented (glibc closure + a warm JIT cache).

### Stack manifest — where each fix lives (baked vs host)

| Sub-component | Source / pin | In the image? |
|---|---|---|
| **Base image** | `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix` (`sha256:2a359be8…`) — ROCm/torch/vLLM(`fdd64a3db`)/aiter-tip/clr | FROM |
| **vLLM connector + model fixes** | 8 overlays in `patches/` → `COPY` into site-packages | ✅ baked |
| **aiter gfx950 kernels** | 3 overlays in `patches/` (a8w8 blockscale, fused-MoE, a16wfp4) → `COPY` | ✅ baked |
| **mori** (ionic RDMA) | built from `raviguptaamd/mori@b271ba2` (branch `ionic-atomic-mr-strip`, on ROCm/mori `v1.2.3.post1`) | ✅ built in image |
| **vllm-router** (pd-disagg) | built from `raviguptaamd/router@82dc9811` (Rust 1.88) → `/usr/local/bin/vllm-router` | ✅ built in image |
| glibc-2.39 ionic closure | host mount (`GLIBC_SWAP=1 HOSTLIBS=…`) | ⬛ host (node's own glibc) |
| aiter JIT warm cache | host cache / `AITER_KSPLIT=1` | ⬛ host (or prewarm) |

Built image: `rocmshared/vllm-glm53-flash:glm53-flash-disagg-v3`.

### The fixes, briefly
- **vLLM MoRIIO connector** (`moriio_connector`/`engine`/`common`/`layout`):
  per-group hybrid-KV block routing on the WRITE path + **cross-chunk KV
  accumulation** (chunked prefill keeps every chunk's KV, not just chunk 1) +
  HMA hybrid-KV support. Without these, disagg serves fluent but recall is garbage.
- **vLLM hybrid-KV / DSA / model** (`attn_utils`, `indexer`, `glm5next attention`):
  KDA(Mamba)+MLA shared KV-pool alloc, DSA sparse-indexer on the disagg path,
  fp32 head-gate for long-context sparse-MLA ranking.
- **aiter gfx950 kernels**: CK a8w8 blockscale GEMM, fused-MoE, batched a16wfp4.
- **mori (2 ionic C++ fixes)**: strip `IBV_ACCESS_REMOTE_ATOMIC` from MR access
  flags (ionic reports `IBV_ATOMIC_NONE` → `ibv_reg_mr` EINVAL otherwise), gated by
  `MORI_IO_DISABLE_ATOMIC_MR=1` (baked ON); restore the caller's HIP device across
  the RDMA backend (else the model's HIP context is left mutated → next DSA indexer
  load fails HIP-209). **Also proposed upstream** — see `MORI_PR_DRAFT.md`.

### Two host-infra launch pieces (node-specific, cannot be baked)
1. `GLIBC_SWAP=1 HOSTLIBS=<glibc-2.39 closure>` — the node's ionic libibverbs
   provider requires glibc ≥ 2.38; the image ships 2.35, so the provider fails to
   load and mori sees 0 devices (`availDevices.size() > 0`). The closure is the
   node's own glibc, so it is mounted, not baked.
2. `AITER_KSPLIT=1` or a warm aiter JIT cache — the CK-2-stage MoE `.so` is JIT-built
   on first use; one split-k variant fails to compile cold on gfx950, and TP workers
   can race the build. `AITER_KSPLIT=1` avoids the broken path; a warm cache avoids
   the race.

## Verified results — TP4 (image, `OVERLAYS=0`, no external mounts, WRITE mode, gold pair 014↔021)

| context | prompt tokens | depths | TTFT | recall |
|---|---|---|---|---|
| 8K | 8,684 | 0.1/0.5/0.9 | ~1s | DELTA-9931 (all) |
| 60K | 65,026 | 0.1/0.9 | 2.8–6.0s | DELTA-9931 (all) |
| 400K | 433,355 | 0.1/0.9 | 18.8–21.6s | DELTA-9931 (all) |

Ceiling = `max_model_len` (940K); over → clean HTTP-400, not mis-recall. Full grid +
before/after in `RESULTS.md`; reproduction gates + benchmark plan in `TEST_PLAN.md`.

### Verification status (be precise)
- **TP4 1P/1D**: verified on the fully self-contained image (`glm53-flash-disagg-v3`,
  no MORI_PATCHED, no ROUTER_BIN) — table above, re-confirmed 2026-09-23.
- **EP8 1P/1D**: recall verified on the earlier **base-image + overlays** recipe (gold
  pair 030↔038, to 871K — see `RESULTS.md`); the connector/mori fixes it depends on
  are config-agnostic (in the connector, not the parallelism). **Not yet re-run on the
  v3 self-contained image** — pending a second gold pair. The `orch_ep8.sh` path
  (`MODE=ep`, `ROUTER_DP_LOCAL=8`, util 0.40, `SPARSE_IDX_MB=4096`) is included.
- **Benchmarks** (TTFT/TPOT/throughput sweep): planned, not yet run — see `TEST_PLAN.md`.

## Files
- `docker/vllm_disagg_inference.glm53flash.overlay.amd.Dockerfile` — the self-contained image.
- `scripts/vllm_dissag/glm53_flash/patches/` — the 11 verified overlays + manifest.
- `scripts/vllm_dissag/glm53_flash/{vllm_pd_launch,run_flash_disagg_tp4,orch_ep8}.sh` — launcher + orchestrators.
- `scripts/vllm_dissag/glm53_flash/{README,RESULTS,TEST_PLAN,MORI_PR_DRAFT}.md`.

## Related
- Upstream mori PR (the 2 ionic fixes): `raviguptaamd/mori:ionic-atomic-mr-strip` → ROCm/mori (draft in `MORI_PR_DRAFT.md`).
