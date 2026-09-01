# GLM-5.2 MTP + EP16 disaggregation — root cause & fixes (updated 2026-08-31)

Consolidated record of the MTP (multi-token-prediction / speculative decode) work. Every claim here
is backed by a py-spy stack dump or a measured run, not inference. Fixes are env-gated so they cannot
regress the already-validated TP8 / EP8 / EP16-noMTP cases.

> **Status correction (2026-08-31):** an earlier version of this file called F2 "the breakthrough"
> and implied EP16-MTP was nearly serving. That was premature. **EP16 + MTP does NOT serve** — see
> "EP16-MTP: still broken" at the end for the fully-proven root cause (cross-node cudagraph capture
> lockstep) and the fix that remains to be written. The F1–F4 fixes below are real and necessary
> (they enable MTP on TP8/EP8 and get EP16-MTP *most* of the way through startup) but are **not
> sufficient** for EP16-MTP to serve.

## What works (measured, NIAH-clean)

| Config | FP8 | MXFP4 | MTP |
|--------|-----|-------|-----|
| TP8 1P/1D | 12/12 matrix, TPOT ~19ms | confirmed | **serves: −41% TPOT (19.2→11.3ms)** |
| EP8 1P/1D | 12/12 matrix, TPOT ~59ms | confirmed | **serves: −44% TPOT (58.9→33.2ms)** |
| EP16 2P/2D | TTFT 11.3s / TPOT 114.9ms | TTFT 10.0s / TPOT 185ms | **does NOT serve** (capture lockstep — see end) |

## The four fixes (all env-gated, PR#205 apply-scripts)

### F1. MoRIIO MTP KV-block fix — `apply_glm_dsa_moriio_mtp_blockfix.py`
- **Bug:** MTP allocates one extra trailing KV block on prefill; the MoRIIO connector asserted
  `local ≤ remote` and aborted KV transfer → decode gets no KV.
- **Fix:** truncate the surplus trailing (draft) block; decode recomputes it. NIAH 128K = 9/10 proves
  the correct block is dropped.
- **Enables:** TP8-MTP and EP8-MTP (the -41% / -38% results).
- **Safety:** only fires when `len(local) > len(remote)`, which never happens without MTP → zero effect
  on non-MTP runs.

### F2. vLLM DP profile-sync deadlock fix — `apply_glm_vllm_dp_profile_sync_fix.py`
- **Bug (py-spy proven):** during startup `profile_run`, MTP's speculator calls
  `dispatch_cg_and_sync_dp` → `sync_cudagraph_and_dp_padding`, which does a DP-group `all_reduce`
  needing ALL dp ranks. At EP16 (DP16 across 2 nodes) ranks arrive **asymmetrically** (one node lags in
  the MoE forward) → the `all_reduce` never completes → deadlock (`futex_wait`). The AITER `1tg` kernel
  is NOT the culprit — verified running single-process in ~2s.
- **Fix:** on the eager/profile path (`need_eager=True`) the `all_reduce` result is discarded anyway
  (it early-returns the all-eager NONE descriptor), so skip the collective there. Real inference
  (`need_eager=False`) still runs the DP padding sync untouched.
- **Enables:** EP16-MTP decode to boot past the deadlock (reaches KV cache + registers).
- **Safety:** gated `VLLM_SKIP_DP_SYNC_ON_PROFILE` (default 1). Only touches the profile/eager path;
  `dp_size==1` (TP8) early-returns before it; non-MTP profile runs also discard the same result, so
  behavior is identical. Cannot change inference-time correctness.

### F3. AITER 1tg→CK fmoe fix — `apply_glm_aiter_fmoe_1stage_gfx950_fix.py`  (OPTIONAL, default off recommended)
- Context: MTP raises decode M>32 → routes to the `1tg` ASM MoE kernel. Forcing CK is what AITER's own
  comment prefers ("ck has better performance"). **However, py-spy later proved the 1tg kernel is fine
  and the real block was F2's DP deadlock** — so F3 is NOT required. Kept as an optional lever
  (`AITER_FORCE_CK_FMOE`), default should be OFF now that F2 is the real fix. Set
  `AITER_FORCE_CK_FMOE=0` to keep stock 1tg.
- **Safety:** gated + gfx950-only + per_1x128-only. Off by default → no effect.

### F4. Deployment recipe (not code)
- EP16 decode `GPUUTIL=0.85` (MoRI heaps leave 254.5GiB; 0.90 fails the mem gate).
- `KV_CACHE_MEMORY_BYTES=60G`, `MAXLEN=131072`, host-local `/cache` (AITER JIT FileBaton needs local fs,
  not NFS), `MODEL` passthrough for MXFP4.

## EP16-MTP: still broken — cross-node cudagraph capture lockstep (2026-08-31, ~10 runs)

The F1–F4 fixes above (plus the `VLLM_STARTUP_DP_UNIFORM` capture patches) get EP16-MTP through
prefill and get the decode **master** rank fully through cudagraph capture. It **still does not
serve.** Root cause is fully proven (py-spy × multiple, reproduced ~10 times):

- **The wedge:** the two decode nodes (DP16 = 2×8 ranks) capture the graph-size list at different
  speeds — MTP's `M:1` draft-token shapes make one node lag. Whichever node finishes its capture
  list **first** leaves the capture phase; the other node's remaining graph contains a cross-node
  MoRI-EP all2all that now has **no partner** → that node wedges (GPU 100%, native hang at
  `seq_lens.to("cpu")` in `rocm_aiter_mla_sparse.py`, right after the orphaned all2all).
- **Where:** `vllm/v1/worker/gpu/cudagraph_utils.py` capture loop (`for desc in descs`) has **no
  cross-rank barrier between sizes**, so the two nodes desync.

**Proven characteristics (target for the fix):**
- `FULL_AND_PIECEWISE`: wedges on the **first** graph (0 captured).
- `PIECEWISE`: captures **7/8** graphs on *both* nodes — only the **last** (cross-node) graph wedges.
- **Size-independent:** trimming capture sizes to `1 2 4 8` still wedges the last graph → it is
  lockstep, **not** a size threshold. Trimming sizes does not help.

**Ruled out (do not re-chase):** DP-group timeout (raised to 7200s, no abort); post-capture
`warmup_kernels` all_reduce (skipping it did not help); sparse-MLA `synchronize()` (patched, no
effect); AITER tuned-config gaps (master lacks the same entries and passes); stale containers.

**The fix (NOT implemented in this PR):** a **cross-rank barrier inside the cudagraph capture loop**
so neither decode node leaves capture until **both** have captured every size — then the last graph's
all2all always has a partner. This is a vLLM source change (upstream-able). Scoped, not written.

**Eager is not a fallback either:** `DECODE_CUDAGRAPH_MODE=NONE` boots + smoke-tests but times out
under sustained decode. So there is no working EP16-MTP path today; it is a documented limitation.

## Regression safety summary

| Prior validated case | Affected by F1? | F2? | F3? |
|---|---|---|---|
| TP8-FP8 12/12 | no (no MTP) | no (dp_size=1 early-return) | no (default off) |
| EP8-FP8 12/12 | no (no MTP) | no (single-node dp-sync unaffected on inference path) | no |
| EP16-FP8/MXFP4 (noMTP) | no | no (profile-run result identical) | no |
| TP8/EP8-MTP (measured) | required (enables) | harmless (single-node arrives together) | off |

All fixes are additive + env-gated. Re-running any prior case with defaults reproduces prior behavior.
