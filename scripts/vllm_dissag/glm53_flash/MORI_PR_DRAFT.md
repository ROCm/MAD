# ROCm/mori PR draft — ionic (Pensando AINIC) RDMA fixes

**Branch:** `raviguptaamd/mori:ionic-atomic-mr-strip` → `ROCm/mori:main`
**Commit:** `b271ba2` (rebased onto `v1.2.3.post1`)
**Diff:** 2 files, +48/-1 (`src/application/transport/rdma/rdma.cpp`, `src/io/rdma/backend_impl.cpp`)

---

## Title
[RDMA][ionic] Strip REMOTE_ATOMIC MR flag on ionic + restore HIP device in RDMA backend

## Summary
Two small, provider-agnostic fixes required to run MoRI-IO RDMA transfer on AMD
Pensando **ionic** (AMD AI NIC) — surfaced running GLM-5.3 disaggregated KV
transfer on MI355X (gfx950) + ionic, but neither fix is model- or app-specific.

### 1. Env-gated strip of `IBV_ACCESS_REMOTE_ATOMIC` (`rdma.cpp`)
ionic reports no atomic capability (`ibv_devinfo`: `IBV_ATOMIC_NONE`). mori's
default MR access flags include `IBV_ACCESS_REMOTE_ATOMIC`, so **`ibv_reg_mr`
returns `EINVAL` (errno 22) for every MR** on ionic, aborting all RDMA transfer.
MoRI-IO transfer is pure `batch_write`/`batch_read` and never issues NIC
remote-atomic ops, so it is safe to strip the atomic bit. Gated behind
`MORI_IO_DISABLE_ATOMIC_MR=1` (alias `MORI_NO_ATOMIC_MR=1`), applied at the single
`MaybeAddRelaxedOrderingFlag` accessFlag chokepoint (covers all 4 registration
sites). **Default off → zero change for existing providers** (mlx5, etc.).

### 2. HIP device restore around RDMA backend registration (`backend_impl.cpp`)
The RDMA backend's `RegisterMemory` / `CreateSession` path touches the HIP device
(`GetOrCreateDeviceContext`) but — unlike the **fabric** and **xgmi** backends,
which use a scoped device guard — did not restore the caller's HIP current device.
This left the caller's HIP primary context mutated. Downstream that corrupted the
context the model runs on (observed: the next Triton `load_binary` failed with
HIP-209 "no kernel image" / a spurious "0 bytes free"). Added
`MoriRdmaHipDeviceGuard` (RAII: save on entry, restore on scope exit) around
`RegisterMemory` and `CreateSession`, matching the existing fabric/xgmi pattern.
**No-op when the device is unchanged.**

## Why it's safe / general
- Fix 1 is **env-gated and default-off**; it only affects users who opt in on a
  no-atomic-capable NIC (any RoCE provider reporting `IBV_ATOMIC_NONE`, not just
  ionic). RDMA write/read is unaffected.
- Fix 2 mirrors a guard the other two backends already have; it is a strict
  correctness improvement (restores caller state) and a no-op in the common case.

## Test / reproduction
- Without fix 1 on ionic: `RegisterRdmaMemoryRegion failed! ... accessFlag:15,
  errno:22/14` on the first KV MR → transfer aborts.
- With both fixes + `MORI_IO_DISABLE_ATOMIC_MR=1`: MoRI-IO 1P/1D KV transfer over
  8 ionic rails succeeds; verified end-to-end (needle recall to 400K+ tokens,
  disaggregated GLM-5.3 on MI355X).

## Notes for reviewers
- Preferred env name is `MORI_IO_DISABLE_ATOMIC_MR`; `MORI_NO_ATOMIC_MR` kept as an
  alias for existing deployments. Happy to drop the alias or flip to default-on for
  detected `IBV_ATOMIC_NONE` devices if maintainers prefer.
