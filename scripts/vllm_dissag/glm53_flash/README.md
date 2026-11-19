# GLM-5.3-Flash-FP8 disaggregated (1P/1D) on MI355X (gfx950) + AMD AI NIC (ionic)

`GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`), **FP8** (E4M3, dynamic
activation scaling), is a DeepSeek-DSA / MLA +
KDA-linear-attention hybrid. This recipe serves it **disaggregated (prefill /
decode split)** over the **MoRIIO** KV-transfer connector, with verified
needle-in-haystack recall (see `RESULTS.md`).

Two configurations are provided, **both served by one image** (they differ only
in launch env, not in the build):
- **TP4 1P/1D** — tensor-parallel 4 per leg, MoRIIO KV transfer only.
- **EP8 1P/1D** — DP8 + expert-parallel, MoRIIO KV transfer + allgather/reducescatter
  MoE dispatch.

---

## TL;DR — what this is and why it exists

Disaggregated inference splits prefill and decode onto separate GPUs/nodes and
ships the KV cache between them. On this stack the KV hop runs over **MoRIIO**
(mori's RDMA connector) on **ionic** NICs. Getting GLM-5.3-Flash to serve
disaggregated with **correct long-context recall** took two connector code fixes:

1. the MoRIIO connector's **MLA KV block-mapping** (garbage recall — decode read
   zero KV), and
2. the connector's **cross-chunk KV accumulation** — under chunked prefill it
   must keep every chunk's per-group blocks, not just the last chunk's.

With both applied and **chunked prefill** as the recipe: **exact needle recall to
871K tokens (TP4)** — near the model's `max_model_len` — all needle depths. (EP8:
see RESULTS.md.) Details below.

### How this recipe is delivered (READ THIS)
The **currently-proven, reproducible** recipe = a pinned **base image** plus a set
of **11 Python overlays** bind-mounted at container start by `vllm_pd_launch.sh`
(`OVERLAYS=1`, the default). The overlays live in `./patches/` (see
`patches/README.md` for the file→destination manifest) and are the verified
connector / model / aiter fixes. This is what was measured to 871K, in MoRI
**WRITE** mode, on gold pair 014↔021.
- **Base image:** `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix` (Dockerhub).
- **Overlays:** the 11 files in `./patches/` (connector per-group WRITE routing +
  CHUNKFIX, layout, indexer, glm5next attn, hybrid-KV attn_utils, gfx950 aiter
  kernels, dynamo/inductor guard).

### Self-contained image (overlays + mori + router baked in — no runtime mounts)
For a single shippable artifact, `docker/vllm_disagg_inference.glm53flash.overlay.amd.Dockerfile`
starts `FROM` the proven base image and **(a) rebuilds mori** from the fork branch
that carries the ionic fixes (atomic-MR strip + HIP-device restore), **(b) builds
`vllm-router`** from source into `/usr/local/bin`, and **(c) COPYs the 11
sha256-verified overlays** into site-packages. So the image carries the ionic mori
strip AND the router — **no `MORI_PATCHED` / mori-`.so` mount and no `ROUTER_BIN`
host binary needed at runtime**.
Build (context = this recipe dir) and run with `OVERLAYS=0`:
```
docker build -f docker/vllm_disagg_inference.glm53flash.overlay.amd.Dockerfile \
  -t rocmshared/vllm-glm53-flash:glm53-flash-disagg-v3 \
  scripts/vllm_dissag/glm53_flash/
# then serve with the built image, no runtime overlays, no mori/router mounts:
IMG=rocmshared/vllm-glm53-flash:glm53-flash-disagg-v3 OVERLAYS=0 \
INFRA_ENV="GLIBC_SWAP=1 HOSTLIBS=<glibc-2.39 closure dir> AITER_KSPLIT=1" \
  ... bash run_flash_disagg_tp4.sh
```
(`ROUTER_BIN` defaults to the in-image `/usr/local/bin/vllm-router`; set it to a
host path only to override with an external build.)
Verified live on gold pair 014↔021, `OVERLAYS=0`, **no MORI_PATCHED**, MoRI WRITE
mode: exact `DELTA-9931` recall at 8K (depths 0.1/0.5/0.9), 60K, and 400K tokens
(433K prompt, TTFT ~20s). The baked mori is confirmed to carry the strip
(`libmori_application.so`/`libmori_cco.so` contain `MORI_IO_DISABLE_ATOMIC_MR`,
absent from the base image's mori) and `MORI_IO_DISABLE_ATOMIC_MR=1` is baked ON.

The mori fork branch (`raviguptaamd/mori:ionic-atomic-mr-strip`, the 2 ionic fixes
rebased onto ROCm/mori v1.2.3) is also proposed upstream — see `MORI_PR_DRAFT.md`.

**Two host-infra pieces still required at launch** (node-specific, cannot be
baked):
1. `GLIBC_SWAP=1 HOSTLIBS=<dir>` — the node's ionic libibverbs **provider**
   (`libionic-rdmav34.so`) requires **glibc ≥ 2.38**; the image ships 2.35, so
   without the host glibc-2.39 closure the provider fails to load → 0 ionic devices
   → mori aborts `availDevices.size() > 0`. (Host-specific — the closure is the
   node's own glibc; it can't be portably baked.)
2. `AITER_KSPLIT=1` **or a warm aiter JIT cache** — the CK-2-stage MoE `.so` is
   JIT-built on first use (not baked). Cold, one split-k variant fails to compile on
   gfx950 (`half_t`→`__half` in the pinned CK) and TP workers race the build baton.
   `AITER_KSPLIT=1` avoids the broken split-k path; a warm `jitcache_*/aiter/`
   avoids the race entirely (the `.so` is portable across identical gfx950 nodes).
   Baking prewarmed kernels into the image is a further follow-up.

> The earlier `glm53-flash-disagg-overlays-v1` tag (overlays baked, mori NOT) also
> works but additionally needs `MORI_PATCHED=1 MORI_SO_DIR=<atomic-stripped mori .so>`
> at launch. `v2` supersedes it by baking mori in.

> A **truly from-source** image (fixes committed into a vLLM fork, no clrfix base)
> is tracked as a separate FOLLOW-UP — fork `raviguptaamd/vllm` branch
> `glm53-flash-disagg-v0.30` is the WIP; its depth-recall is still open. Until it's
> verified, the two recipes above (overlays, or overlays baked onto the proven
> base) are the proven ones.

---

## The investigation (how we found the two fixes)

**Symptom.** Disagg served fluent *local* continuation but recall was garbage at
every length — e.g. a prompt stating "the password is banana … the password is"
decoded to "is is is is". Short greedy prompts ("The capital of France is" →
"Paris") looked fine, which masked it at first.

**Localization.** The *same image* run **colocated** (no prefill/decode split, no
MoRIIO) recalled correctly at long context. That isolated the bug to the
**MoRIIO KV transfer path**, not the model, kernels, or toolchain.

**Root cause (fix #1).** GLM-5.3-Flash uses MLA, so its KV cache is several
tensors (main attention `self_attn.attn`, the DSA `indexer.k_cache`), each paged
at its **own kernel block-size**. But vLLM's hybrid KV allocator hands out
block-ids in a padded **group** block unit (1152 tokens here). Each MLA tensor
therefore holds `num_group_blocks * kbpb` kernel blocks, where `kbpb`
("kernel-blocks-per-group-block") differs per tensor (`self_attn.attn` kbpb=18,
`indexer.k_cache` kbpb=9 for this model). The connector was transferring at the
**raw group block-id** — so the main attention KV was written to the wrong
(effectively empty) kernel blocks. Decode then read **zero** attention KV for
everything except the last group block → fluent-but-wrong output.

The fix expands each group block-id `N` into its `kbpb` contiguous kernel
sub-block-ids `[N*kbpb … N*kbpb+kbpb-1]`, for both the local and remote id lists,
for each MLA cache tensor at its own `kbpb`. `kbpb` is derived from the true
group-block count (`min` blocks over all kv_caches). Files:
`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py`
(the `_mla_kernel_blocks_per_group_block` helper + the expansion in
`compute_block_transfer_offsets`) and `.../moriio_connector.py` (compute
`_num_group_blocks` once at `register_kv_caches` and pass it in).

**Second ceiling (fix #2).** With fix #1, recall was exact up to ~one prefill
chunk (~16-28K) then broke at *every* needle depth — the tell that it is not a
"late positions" problem but "everything past chunk 1 is missing." Under chunked
prefill the connector must accumulate **each chunk's per-group blocks** across the
whole prompt and hand the full per-group block list to the write path on the final
chunk; the earlier code froze that per-group map at chunk 1, so a multi-chunk
prompt transferred only chunk-1 KV. The fix threads the accumulated per-group
block list (`_reqs_need_save` → cross-chunk accumulation in `build_connector_meta`
→ `local_block_ids[group_idx]` per layer in `_write_blocks_for_req`) so every
chunk's KV lands on decode. With this, **chunked prefill**
(`--enable-chunked-prefill --max-num-batched-tokens 16384`) recalls exactly to the
model's `max_model_len` (871K tokens verified, TP4). Chunked prefill is also
strictly better than the earlier single-chunk workaround: single-chunk
(`--max-num-batched-tokens >= max-model-len`) worked but capped ~100K on VRAM and
hit the single-chunk Triton compile wall (<~780K); chunked prefill dodges both.

**Dead ends ruled out (so nobody re-chases them):** the DSA tail_cache was a red
herring (colocated recalled without it); fp4 GEMM was not the culprit
(instrumented and cleared); it is not a kernel/toolchain/image gap (colocated
proves the model is fine on gfx950).

---

## What makes disagg correct (the fixes, in brief)

The connector / model / aiter fixes are delivered as the **11 overlays in
`./patches/`** (bind-mounted by the launcher). The two that make long-context
recall correct:
1. **MoRIIO MLA KV block-mapping + per-group WRITE routing** (`patches/moriio_connector_hma.py`
   + `moriio_engine.py` + `moriio_layout.py`) — without it decode reads zero/wrong
   KV → silently wrong long-range output.
2. **Cross-chunk KV accumulation** (same connector, `# CHUNKFIX`) — keeps every
   chunk's per-group blocks under chunked prefill. With it, the recipe uses
   **chunked prefill** (`--enable-chunked-prefill --max-num-batched-tokens 16384`)
   for exact recall to `max_model_len` (871K verified, TP4, MoRI WRITE mode).
The other overlays are load-bearing too (hybrid-KV `attn_utils.py`, DSA `indexer.py`,
`glm5next` attention, gfx950 aiter kernels, the dynamo/inductor `usercustomize.py`
guard). See `patches/README.md`.

## What's in the stack (subcomponents)

| Subcomponent | Pin | GLM-5.3 fixes delivered via |
|---|---|---|
| **base image** | `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix` (Dockerhub) | — (carries mori atomic-MR strip + aiter-tip + clr fix) |
| **vLLM** | base image's vLLM **+ `./patches/` overlays** | **the 8 vLLM overlays** (MoRIIO connector/engine/common/layout, attn_utils, indexer, glm5next attn, usercustomize) |
| **aiter** | base image's aiter **+ `./patches/` overlays** | the 3 aiter overlays (gfx950 a8w8 / fused-moe / a16wfp4 kernels) |
| **mori** | in base image (ionic atomic-MR strip) | — |
| vllm-router | `raviguptaamd/router @ 82dc9811` (= PR #223) | — |
| arch target | `gfx950` (MI355X) | — |

> A future self-contained image would bake the 11 overlays in-source (fork
> `raviguptaamd/vllm` branch `glm53-flash-disagg-v0.30` is the WIP base for that).
> Until it's verified, the base-image + overlays recipe here is the proven one.

## The 6 bring-up essentials

1. **Image + overlays**: pull base `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix`;
   the launcher bind-mounts the 11 `./patches/` overlays (`OVERLAYS=1`, default).
   Serving the base image bare (`OVERLAYS=0`) boots but does NOT recall at depth.
2. **VRAM headroom** — `GPU_MEMORY_UTILIZATION` 0.5 (TP4) / 0.40 (EP8). Too high
   starves the DSA-indexer Triton code-object load at long context
   (HSA_STATUS_ERROR_OUT_OF_RESOURCES). EP8 also needs
   `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096` (single-chunk indexer).
3. **glibc swap** (`GLIBC_SWAP=1` in the launcher) — fresh nodes ship glibc 2.39;
   the ionic RDMA driver needs `GLIBC_2.38`. The launcher bind-mounts host glibc
   at runtime; no rebuild.
4. **atomic-MR strip** — ionic rejects `REMOTE_ATOMIC` memory regions (errno
   14/22). Baked into the base image's mori. (The `MORI_PATCHED=1` launcher path
   exists only for running a host-built patched `.so` set on an image that lacks it.)
5. **The MoRIIO connector fixes** — carried by the `./patches/` overlays (essential #1).
6. **Chunked prefill** (recipe) — `--enable-chunked-prefill
   --max-num-batched-tokens 16384` + `--max-num-seqs 256` in the prefill leg's
   launch args. Correct only because of the connector's cross-chunk accumulation;
   recalls to `max_model_len` and dodges the single-chunk compile wall.

## Bring-up ORDER (the launcher is order-sensitive)

- Router (`ROLE=proxy`) with **`ROUTER_DP_LOCAL=1` for TP** (`=8` for EP/DP8).
  The wrong value makes the router return HTTP 400 for every request.
- Bring the **decode leg up first** (to "Application startup complete"), **then
  prefill**. Prefill caches the decode mori handshake; if you restart decode
  after prefill is up you must restart prefill too, else requests hang.
- **Warm** with 2–3 tiny requests (mori CreateSession is a cold-start race; the
  first request may 503).

## Gold pairs (ionic rail routing — read before picking nodes)

MoRIIO stripes the KV write across all 8 ionic rails. A rail only carries traffic
if **both** nodes' NIC on that plane sit on the same `/64` subnet. A "gold" pair
= all 8 rails match → full bandwidth, zero flush, **no IBDEV pinning**. A
mismatched rail → RDMA transport-retry (CQE status=12) → the whole KV transfer
flushes and the request hangs/500s. To find a gold pair, GID-map both candidates:

```
for i in 0 1 2 3 4 5 6 7; do printf "ionic_%s: " $i; \
  cat /sys/class/infiniband/ionic_$i/ports/1/gids/1; done
```

Compare the 3rd hextet (subnet suffix) rail-by-rail: all 8 match → gold (no
IBDEV); some match → pin `IBDEV=ionic_<matching rail>` (single-rail, reduced BW);
none → pick another node.

## Run it

`vllm_pd_launch.sh` is the per-node, per-role launcher; it bind-mounts the 11
`./patches/` overlays onto the base image (`OVERLAYS=1`, default). The two
orchestrators below drive it on both legs over `spur exec` (this cluster's
per-node exec); swap `drive()` for your own remote exec (ssh/srun) elsewhere.

TP4 1P/1D (prefill = node A, decode = node B):
```
PF_JOB=<A_handle> DC_JOB=<B_handle> PF_IP=<A_ip> DC_IP=<B_ip> \
IMG=rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix \
MODEL=<GLM-5.3-Flash weights path on the nodes> \
ROUTER_BIN=<vllm-router binary path on the nodes> \
REMOTE_DIR=<path to this recipe dir on the nodes> \
INFRA_ENV="GLIBC_SWAP=1 HOSTLIBS=<dir>" \
  bash run_flash_disagg_tp4.sh
```
(`INFRA_ENV` is optional — only for fresh nodes needing the glibc-2.39 closure;
omit if your image already carries it.) It precleans, launches router + decode +
prefill in the correct order, warms the handshake, and prints a recall smoke test
(expect `DELTA-9931` at 500w and 8000w). `orch_ep8.sh` is the EP8 variant (same
env, `MODE=ep`, `ROUTER_DP_LOCAL=8`).

Alternatively, serve via MAD's standard disagg entry point using the registry:
`MODEL_NAME=GLM-5.3-Flash` selects the recipe block in `models.yaml` (see
`scripts/vllm_dissag/README.md` for the moriio.sh / run_xPyD_models.slurm flow).

---

## Sample test plan (reproduce + gate a PR)

A reviewer with two gold-pair nodes and the built image reproduces the whole
result in ~15 min per config. **One image, both configs.**

**Gate 0 — overlays mounted.** With `OVERLAYS=1` (default), `docker inspect
vllm_prefill` shows the 11 `./patches/*.py` bind-mounted onto site-packages; the
base image is `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix`; arch is gfx950.
(sha256 of the mounted overlays == the committed `./patches/` set.)

**Gate 1 — TP4 bring-up + short recall.** Pick a gold pair (GID-map above).
Run `run_flash_disagg_tp4.sh` with the env above. Expect both legs to reach
"Application startup complete", a tiny prompt to serve, and the built-in recall
smoke to print `DELTA-9931` at 500w **and** 8000w. This is the pass/fail gate:
8000w > one attention group block, so it exercises the block-mapping fix.

**Gate 2 — TP4 long-context needle.** With **chunked prefill**
(`--enable-chunked-prefill --max-num-batched-tokens 16384`, `MAXLEN=940000`), fire
multi-chunk prompts with the needle at depths 0.1/0.5/0.9; expect `DELTA-9931` at
every depth from 30K through **871K tokens** (near `max_model_len`). Prompts over
`max_model_len` are cleanly rejected (HTTP 400), not mis-recalled. This gate
exercises fix #2 (cross-chunk KV accumulation) — the multi-chunk case that the
single-chunk recipe used to sidestep.

**Gate 3 — EP8 bring-up + recall.** On the second gold pair, `orch_ep8.sh`
(`MODE=ep`, `ROUTER_DP_LOCAL=8`, util 0.40, `SPARSE_IDX_MB=4096`, single-chunk).
Expect `DELTA-9931` at 8000w and up to ~28K (near the 32K max-model-len). Watch
for MoE all2all crashes (should be none with allgather/reducescatter) and indexer
OOM-resources (raise headroom / lower util if seen).

**Gate 4 — negative control (optional, proves the fix matters).** Run TP4 with
the fix disabled (an image whose `VLLM_REF` predates the fix, or bypass the
expansion): recall goes garbage at 8000w while short prompts still serve. This is
the "before" row in `RESULTS.md`.

**Expected results:** `RESULTS.md` — TP4 exact to **871K tokens** all depths
(chunked prefill); EP8 per RESULTS.md.

Smoke-only (no gold pair / single check): Gate 1's 8000w recall alone catches the
block-mapping regression; it is the single most valuable check.

---

## Model registry

- `scripts/vllm_dissag/models.yaml` — recipe block `GLM-5.3-Flash` (serving flags/env).
- `scripts/vllm_dissag/models.json` — CI entry `pyt_vllm_disagg_mori_glm-5.3-flash`.

## Files
- `patches/` — the 11 verified overlays (connector / model / aiter fixes) + a
  manifest README. Bind-mounted by the launcher; this is the actual fix set.
- `vllm_pd_launch.sh` — per-node, per-role launcher (proxy / prefill / decode).
  Env-driven; mounts `patches/` onto the base image (`OVERLAYS=1`, default).
- `run_flash_disagg_tp4.sh` — TP4 1P/1D orchestrator (bring-up order + chunked-prefill recipe).
- `orch_ep8.sh` — EP8 1P/1D orchestrator (DP8 + allgather/reducescatter MoE dispatch).
- `README.md` / `RESULTS.md` — this file + the verified NIAH recall matrix
  (TP4 → 871K tokens; EP8 per RESULTS.md).
