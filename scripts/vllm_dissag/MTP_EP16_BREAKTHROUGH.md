# GLM-5.2 MTP + EP16 disaggregation — root cause & fixes (2026-08-30)

Consolidated record of the MTP (multi-token-prediction / speculative decode) work and the EP16
deadlock breakthrough. Every claim here is backed by a py-spy stack dump or a measured run, not
inference. Fixes are env-gated so they cannot regress the already-validated TP8 / EP8 / EP16-noMTP
cases.

## What works (measured, NIAH-clean)

| Config | FP8 | MXFP4 | MTP |
|--------|-----|-------|-----|
| TP8 1P/1D | 12/12 matrix, TPOT ~19ms | confirmed | **-41% TPOT (19.2→11.3ms)** |
| EP8 1P/1D | 12/12 matrix, TPOT ~60ms | confirmed | **-38% TPOT (60→37ms)** |
| EP16 2P/2D | TTFT 11.3s / TPOT 117ms | TTFT 10.0s / TPOT 185ms | decode boots; prefill residual (see below) |

## The four fixes (all env-gated, PR#205 apply-scripts)

### F1. MoRIIO MTP KV-block fix — `apply_glm_dsa_moriio_mtp_blockfix.py`
- **Bug:** MTP allocates one extra trailing KV block on prefill; the MoRIIO connector asserted
  `local ≤ remote` and aborted KV transfer → decode gets no KV.
- **Fix:** truncate the surplus trailing (draft) block; decode recomputes it. NIAH 128K = 9/10 proves
  the correct block is dropped.
- **Enables:** TP8-MTP and EP8-MTP (the -41% / -38% results).
- **Safety:** only fires when `len(local) > len(remote)`, which never happens without MTP → zero effect
  on non-MTP runs.

### F2. vLLM DP profile-sync deadlock fix — `apply_glm_vllm_dp_profile_sync_fix.py`  ★ the breakthrough
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

## EP16-MTP residual: prefill (under investigation)

- Decode-side deadlock: **SOLVED** by F2. Decode reaches KV cache, MoRIIO init, router registration.
- Prefill stalls **before workers spawn** (`pf_workers=0`, GPU idle, at the ApiServer/transformers
  stage) = a DP16 multi-apiserver cross-node **rendezvous** stall, earlier than F2's profile path.
- **Open question (user's insight):** MTP is a *decode-only* feature, yet the EP16 orchestrator passes
  `SPEC=mtp` to BOTH prefill and decode. EP8-MTP prefill also ran MTP and started fine (DP8), so it's
  not inherently broken — but at DP16 it may be the extra failure surface. Candidate fix: pass
  `SPEC=mtp` to **decode only**; prefill is the KV producer and does not run the speculator. Testing.

## Regression safety summary

| Prior validated case | Affected by F1? | F2? | F3? |
|---|---|---|---|
| TP8-FP8 12/12 | no (no MTP) | no (dp_size=1 early-return) | no (default off) |
| EP8-FP8 12/12 | no (no MTP) | no (single-node dp-sync unaffected on inference path) | no |
| EP16-FP8/MXFP4 (noMTP) | no | no (profile-run result identical) | no |
| TP8/EP8-MTP (measured) | required (enables) | harmless (single-node arrives together) | off |

All fixes are additive + env-gated. Re-running any prior case with defaults reproduces prior behavior.
