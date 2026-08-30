# GLM-5.2 MTP (speculative decode) on disaggregated serving — TP8 / EP8 / EP16

Dedicated, portable recipe for running GLM-5.2-FP8 disaggregated serving **with MTP
(multi-token prediction / speculative decode)** across all three parallelism modes, on
MI355X / gfx950 + AINIC ionic. This is the file to read when moving PR #205 to another
cluster. It captures **every source patch, env var, and launch option** MTP needs, plus an
honest status of what is measured vs. in-progress.

- Infra background (why EP16 needs the MoRI host-proxy, topology, image build): see
  [`GLM52_EP16_IONIC.md`](GLM52_EP16_IONIC.md) and [`GLM52_MI355X.md`](GLM52_MI355X.md).
- What MTP is and the deadlock root-causes: see
  [`MTP_EP16_BREAKTHROUGH.md`](MTP_EP16_BREAKTHROUGH.md).

---

## 0. TL;DR — apply patches, set env, launch all-together

```bash
# 1) inside the serving container/image, after pip-installing vLLM + AITER:
bash scripts/vllm_dissag/apply_all_patches.sh          # idempotent; auto-detects vllm/aiter dirs

# 2) launch ALL components together (never partial-restart a single role — see §5):
#    proxy(router) -> prefill(both ranks) -> decode(both ranks), one coherent bring-up.
```

MTP is a **decode-only** feature. Prefill is the KV producer and does **not** run the
speculator — launch prefill with `SPEC=off` and decode with `SPEC=mtp`.

---

## 1. The five MTP source patches (all in `apply_all_patches.sh`)

Every patch is anchor-based + idempotent + env-gated, so it cannot regress the already-validated
non-MTP TP8 / EP8 / EP16 runs. Gates default to the *stock* behavior unless you opt in.

| # | Apply-script | Env gate (default) | What it fixes |
|---|---|---|---|
| 1 | `apply_glm_dsa_moriio_mtp_blockfix.py` | always (only fires when `len(local)>len(remote)`) | MTP allocates one extra trailing KV block on prefill; the MoRIIO connector asserted `local ≤ remote` and aborted KV transfer. Truncates the surplus draft block; decode recomputes it. **Enables TP8-MTP + EP8-MTP.** |
| 2 | `apply_glm_vllm_dp_profile_sync_fix.py` | `VLLM_SKIP_DP_SYNC_ON_PROFILE=1` | EP16 (DP16 cross-node): MTP speculator's profile `_dummy_run` does a DP `all_reduce` (`worker/gpu/dp_utils.py`) that deadlocks when ranks arrive asymmetrically. Skips it on the eager/profile path (result is discarded there anyway). |
| 3 | `apply_glm_vllm_fwdctx_dp_sync_fix.py` | `VLLM_SKIP_FWDCTX_DP_AR=1` | Same class, deeper site: `set_forward_context` → `coordinate_batch_across_dp` DP `all_reduce`. Routes the profile case down the local-tensor branch (incl. the `torch.full((dp,),…)` shape fix `DPMetadata.make` needs). |
| 4 | `apply_glm_vllm_skip_profile_run_fix.py` | `VLLM_SKIP_PROFILE_RUN=1` | `determine_available_memory()` still runs a profile **forward** even when `kv_cache_memory_bytes` is preset; that forward's MoE all2all deadlocks cross-node. Gated so it is skipped (kernels JIT lazily on first request). |
| 5 | `apply_glm_vllm_skip_warmup_dummy_fix.py` | `VLLM_SKIP_WARMUP_DUMMY=1` | With `VLLM_SKIP_KERNEL_WARMUP=1`, `compile_or_warm_up_model` takes the `elif` branch whose `_dummy_run` drives the MTP speculator into the same DP `all_reduce`. Gates that elif too. |
| + | `apply_glm_aiter_fmoe_1stage_gfx950_fix.py` | `AITER_FORCE_CK_FMOE=1` | MTP raises decode `token>32` → AITER routes to the 1-stage ASM `fmoe_…_1tg` MoE kernel, whose `LoadKernel` wedges at EP16/DP16 on gfx950. Forces the CK 2-stage kernel (AITER-preferred for fp8 blockscale anyway). |

Patches 2–5 + the AITER one are **EP16-specific**; TP8 (`dp_size=1`) and EP8 (single-node DP8)
never reach those collectives, so leaving the gates on is harmless for them.

---

## 2. Env matrix per mode

Common to all: `SPEC=mtp` (decode) / `SPEC=off` (prefill), `SPEC_TOK=1`, `VLLM_ROCM_USE_AITER=1`.

| Env | TP8-MTP | EP8-MTP | EP16-MTP | Why |
|---|---|---|---|---|
| `VLLM_SKIP_DP_SYNC_ON_PROFILE` | 1 | 1 | **1** | startup DP profile all_reduce (no-op on TP8/EP8) |
| `VLLM_SKIP_FWDCTX_DP_AR` | 1 | 1 | **1** | fwdctx DP all_reduce (no-op on TP8/EP8) |
| `VLLM_SKIP_PROFILE_RUN` | 0 | 0 | **1** | skip deadlocking profile forward (EP16 only) |
| `VLLM_SKIP_WARMUP_DUMMY` | 0 | 0 | **1** | skip warmup dummy_run MTP-propose (EP16 only) |
| `VLLM_SKIP_KERNEL_WARMUP` | 0 | 0 | **1** | route past `warmup_kernels` (EP16 only) |
| `AITER_FORCE_CK_FMOE` | 0 | 0 | **1** | avoid 1tg LoadKernel wedge (EP16 only) |
| `VLLM_SKIP_DP_SYNC_ALL` | 0 | 0 | **0** | **must be 0** — runtime DP coord ON for serving |
| `MORI_EP_OVER_RDMA` | 0 | 0 | **1** | Tej PR#558 host-proxy for cross-node all2all |
| `DECODE_CUDAGRAPH_MODE` | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE | see §4 | decode graph capture (perf-critical) |

**Launcher passthrough:** these env vars must be forwarded into the container via `docker -e`.
`vllm_pd_ep16_launch.sh` line ~156 forwards them; if you port to a new launcher, forward the
full set or the gates silently do nothing (a real trap we hit).

**JIT cache must be host-local** (`JITCACHE_OVERRIDE=/tmp/jitcache`), not NFS — the AITER
FileBaton build-lock serializes/deadlocks the 8 ranks/node over NFS.

---

## 3. Performance — are we losing perf? (honest)

Of the EP16-MTP knobs, **only one has a steady-state cost, and it is being removed** (see §4):

- `SKIP_PROFILE_RUN` / `SKIP_WARMUP_DUMMY` / `SKIP_KERNEL_WARMUP`: **zero** steady-state cost —
  they only skip *startup* dummy-forwards; kernel JIT moves to the first real request (send one
  warmup request at deploy time to erase even that).
- `AITER_FORCE_CK_FMOE=1`: **not a loss** — AITER's own comment says CK is faster for fp8
  blockscale, and non-MTP EP16 (token=1) already used CK. MTP now matches it.
- `SKIP_DP_SYNC_ALL=0`: correct runtime setting, no loss.
- **`DECODE_CUDAGRAPH_MODE=NONE`: the one real cost** (eager decode = per-step kernel-launch
  overhead). This is a *workaround*, not the target — see §4.

**Measured MTP wins (TPOT, decode):** TP8 **−41 %** (19.2→11.3 ms), EP8 **−38 %** (60→37 ms).
EP16-MTP boots + serves; a clean sustained-decode TPOT is pending the cudagraph work in §4.

---

## 4. Decode cudagraph status (`FULL_AND_PIECEWISE`) — IN PROGRESS, perf-critical

`FULL_AND_PIECEWISE` decode cudagraphs are **required for production perf** and are the intended
end state. Current status:

- TP8-MTP, EP8-MTP: run with cudagraphs ON (validated).
- **EP16-MTP: currently requires `DECODE_CUDAGRAPH_MODE=NONE`** to boot, because cudagraph
  *capture* runs its own `_dummy_run` that hits the DP `all_reduce` / EP all2all path. With
  patches 2–5 gating the profile/warmup dummy-runs, the remaining task is to ensure the **capture
  dummy_run** is likewise coordinated (or gated) and that MoRIIO KV transfer is correct under
  captured graphs (the connector's `wait_for_layer_load` already has a "must not run during
  full-graph capture" guard — verify it fires). Tracked as the active work item; **do not ship
  EP16-MTP with `NONE` as the final config.**

Until then, EP16-MTP with `NONE` is functional-but-eager (correctness OK, perf sub-optimal).

---

## 5. Operational rule — never partial-restart one role

MoRIIO pins RDMA QPs / registered memory + the router's routable set **at handshake time**.
Restarting only decode (or only prefill) against long-lived peers leaves **stale endpoints**:
the router logs "Add Prefill" but won't route, prefill RDMA-writes KV to a dead endpoint, both
sides sit idle, requests time out (looks like a hang but **both GPUs are at 0 %** — the tell that
it is stale state, not a deadlock). **Fix: always relaunch all components together**
(`ep16_orchestrate.sh` does proxy → prefill → decode in one coherent bring-up).

---

## 6. Files

- `apply_all_patches.sh` — master, idempotent patch applier (run once in the image).
- `apply_glm_*.py` — the individual anchor-based patches (see §1).
- `vllm_pd_ep16_launch.sh` — per-node launcher (mounts patched files, forwards env).
- `ep16_orchestrate.sh` — 4-node driver (all-together bring-up).
- `MTP_EP16_BREAKTHROUGH.md` — root-cause narrative + regression-safety matrix.
