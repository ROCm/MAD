# MoRI-EP on MI300X + Broadcom Thor2 (bnxt RoCE) — Working Solution

Self-contained record of getting **native MoRI-EP internode** (GPU-initiated expert-parallel
dispatch/combine) working on the Dell XE9680 / MI300X + Broadcom BCM57608 "Thor2" 400G RoCE
cluster, plus MoRI-IO validation. Everything needed to reproduce is in this folder.

**Status: SOLVED.** Cross-node MoRI-EP dispatch/combine = **500/500 rounds, 0 errors** on a 2-node
pair. MoRI-IO cross-node = **48.4 GB/s** (host mem, near 400G line rate).

---

## TL;DR — the two fixes

Native MoRI-EP internode was blocked by **two independent** issues on this stack. Both are cleared:

| # | Blocker | Symptom | Fix |
|---|---------|---------|-----|
| 1 | **DV-CQ create** | `bnxt_re_dv_create_cq execute_ioctl() failed: 5` (EIO) → `bnxt.cpp:108 Assertion 'cq' failed` → SIGABRT | Install **bnxt 235.2.86.0** driver (driver-only, **no firmware reflash**). The 237/238 drivers added a `setup_cq_hwqs`/`cq_timer` CQ step that EIOs on our firmware; 235 does not have it. |
| 2 | **RDMA atomics into GPU VRAM** | Round-0 dispatch hangs; QP async events, `res_rx_pci_err`, QP driven to ERROR | Use the **`async_ll`** EP kernel (`--kernel-type async_ll`), which signals via RDMA WRITE+poll instead of atomics. The default `v1`/InterNodeV1 kernel posts RDMA `AMO_ADD` atomics into GPU VRAM, and the **Thor2 NIC has no PCIe atomic-completer capability** (`AtomicOpsCap: 32bit- 64bit-`). |

Why MoRI-IO always worked but MoRI-EP didn't: **MoRI-IO is pure RDMA WRITE** (no VRAM atomics);
MoRI-EP's default `v1` kernel uses VRAM atomics that this NIC cannot complete.

---

## Verified environment (both nodes)

- **Hardware:** Dell PowerEdge XE9680, 8× AMD Instinct MI300X (gfx942, `1002:74a1`), Broadcom BCM57608 Thor2 400G (`14e4:1760`)
- **OS/kernel:** Ubuntu 22.04.5, kernel `5.15.0-177-generic`, ROCm 7.2.3
- **bnxt driver:** `235.2.86.0` (bnxt_en `1.10.3-235.2.86.0`)  ← **the fix**
- **bnxt firmware:** `238.1.138.0` (UNCHANGED — driver-only fix)
- **libbnxt_re provider:** `libbnxt_re-rdmav34.so`, 539696 B (235.2.86.0 build, glibc-2.34 → OK on 22.04)
- **Container image:** `rocm/vllm-dev:vllm-wideep_06_29_2026_Shiksha_dp16_2p2d_mori_v1.2.1_aiter_v0.1.16.post3_nightlybase_mori121`
- **MoRI:** `ROCm/mori` commit `12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4` (2026-07-31, `0.1.1.dev1+g12d1bc32d`), built `BUILD_UMBP=OFF`, `MORI_GPU_ARCHS=gfx942`
- **Test pair:** `node-a` (192.0.2.10, rank0) + `node-b` (192.0.2.11, rank1)

Full detail in `SOFTWARE_MANIFEST.txt`.

### NIC naming (verified identical on both nodes)

| Role | Device | netdev | PCI | Used for |
|------|--------|--------|-----|----------|
| RDMA data rail | `rdma3` | `benic4p1` | `0000:60:00.0` | `MORI_RDMA_DEVICES=rdma3` (the RoCE fabric) |
| RDMA rails (all) | `rdma0..rdma7` | `benic1p1..benic8p1` | `3e,1c,4f,60,de,ce,be,9e :00.0` | full 8-rail fabric available |
| Mgmt / OOB | `eno8303` | — | `0000:02:00.0` | torchrun `master_addr`, `*_SOCKET_IFNAME` (gloo/NCCL/MoRI OOB) |

OOB rendezvous is deliberately on the 1G mgmt NIC (`eno8303`), keeping the RoCE rails clean for data.
`rdma1`/`benic2` carries NFS-over-RDMA — do not disturb it.

---

## How to reproduce (from scratch)

### 1. Install the 235 driver on every node, then reboot
```bash
sudo ./scripts/install_driver_235.sh ./driver-235.2.86.0   # then REBOOT
# verify: modinfo bnxt_re | grep ^version  -> 235.2.86.0 ; ibv_devinfo -d rdma3 -> PORT_ACTIVE
# verify NFS-over-RDMA (benic2/rdma1) still mounted
```

### 2. Launch the container on every node
```bash
sudo ./scripts/launch_container.sh <IMAGE>
# self-consistent v34 RDMA stack: mounts host libibverbs.so.1.14.39.0 + host
# libbnxt_re-rdmav34.so as provider, removes the image's v59 provider, sets bnxt_re.driver
```

### 3. Build MoRI inside the container on every node
```bash
sudo docker exec mori_host bash -c "$(cat scripts/build_mori.sh)"
# shallow clone + only spdlog+msgpack-c submodules (recursive clone stalls on spdk/HTTP2)
```

### 4. Run the MoRI-EP internode test  ← THE KEY RESULT
`scripts/run_ep_internode.sh` (set `node_rank=0` on node-a, `=1` on node-b):
```bash
torchrun --nnodes=2 --node_rank=<0|1> --nproc_per_node=1 \
  --master_addr=192.0.2.10 --master_port=29000 \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --cmd test --dtype bf16 --max-tokens 128 --num-qp 2 --kernel-type async_ll
#                                                        ^^^^^^^^^^^^^^^^^^^^^^ REQUIRED on Thor2
```
Key env (see the script): `MORI_RDMA_DEVICES=rdma3`, `MORI_IB_GID_INDEX=3`, `MORI_RDMA_SL=3`,
`MORI_RDMA_TC=104`, `*_SOCKET_IFNAME=eno8303`, `MORI_GPU_ARCHS=gfx942`.

**Expected (see `logs/ep_async_ll_rank{0,1}.log`):**
```
I'm pe 0 in 2 pes                 <- DV-CQ create succeeds (235 driver fix)
...
Node 0 Dispatch Pass
Node 0 Combine Pass               <- for all 500 rounds
rank:  0 error times:  0 appear round:  set()
```

### 5. (Optional) MoRI-IO validation
- **CPU/host memory** (`scripts/run_moriio_cpu.sh`): works cleanly, sweep 8B→64MiB, up to **48.4 GB/s** — see `logs/moriio_cpu_write_rank0.log`. `--host` must be **each node's own** IP.
- **GPU/dmabuf** (`scripts/run_moriio_gpu.sh`): on THIS from-source build the dmabuf MR registration fails (`errno 22`) once the pre-allocated pool exceeds the GPU dmabuf ceiling — see `logs/moriio_gpu_dmabuf_ceiling.log`. The customer's `wideep_mori123` image reaches 42–43 GB/s for GPU regions ≤8 MiB; keep GPU KV blocks ≤8 MiB. (Not a blocker for EP.)

---

## Folder contents

```
README.md                       this file
SOFTWARE_MANIFEST.txt           exact versions on both nodes (driver/fw/lib/mori/rocm/gpu)
scripts/
  install_driver_235.sh         install bnxt 235.2.86.0 (deb OR the DKMS tarball here) + gotchas
  launch_container.sh           start mori_host with a self-consistent v34 RDMA stack
  build_mori.sh                 build MoRI from source (shallow, pinned submodules)
  run_ep_internode.sh           THE MoRI-EP internode test launcher (async_ll)  <-- rank0 copy
  run_moriio_cpu.sh             MoRI-IO host-mem benchmark
  run_moriio_gpu.sh             MoRI-IO GPU-mem benchmark
driver-235.2.86.0/
  README.md                         where to download the 235.2.86.0 driver (Broadcom public repo)
  rocelib-README.TXT                Broadcom rocelib readme
  NOTE: the driver binaries (bnxt_en.ko, bnxt_re.ko, bnxt-dkms-src-*.tar.gz,
        bnxt-rocelib-*.tar.gz) are NOT redistributed here — see driver-235.2.86.0/README.md
        for the download link (packages.broadcom.com/.../ethernet-nic-debian-public/).
patches/
  atomics-evidence.txt          source grep proving v1 uses AMO_ADD atomics, async_ll uses WRITE+poll
logs/
  ep_async_ll_rank0.log         PASS 500/500 rounds, rank0
  ep_async_ll_rank1.log         PASS 500/500 rounds, rank1
  atomics_rootcause_counters.txt  hw_counters (atomics frozen, tx_write huge) + NIC/GPU AtomicOpsCap
  moriio_cpu_write_rank0.log    MoRI-IO host-mem sweep, up to 48.4 GB/s
  moriio_gpu_dmabuf_ceiling.log  MoRI-IO GPU dmabuf ceiling (errno 22) for the record
```

---

## Root-cause evidence (blocker #2, the atomics gap)

From `logs/atomics_rootcause_counters.txt` — after many async_ll runs:
```
rx_atomic_requests=2   tx_atomic_req=4          <- FROZEN (async_ll issues ZERO atomics)
res_rx_pci_err=1       unrecoverable_err=4      <- FROZEN (no new PCIe errors)
tx_write_req=4719266   rx_write_requests=472500 <- exploded (async_ll = pure WRITE)

GPU 0000:1b:00.0  AtomicOpsCap: 32bit+ 64bit+ 128bitCAS-   (GPU CAN complete atomics)
NIC 0000:60:00.0  AtomicOpsCap: 32bit- 64bit- 128bitCAS-   (Thor2 NIC CANNOT)
```
The nonzero atomic counters + `res_rx_pci_err` were produced by the earlier `v1` run; async_ll adds
none. `patches/atomics-evidence.txt` shows `internode_v1.cpp` posts `core::atomicType::AMO_ADD` at 7
sites while `low_latency_async.cpp` has those atomics commented out in favor of WRITE + `WaitUntil` polling.

---

## Open items / escalation

- **Broadcom:** does any Thor2 (BCM57608) firmware enable RDMA **atomic-completer** to a peer/GPU BAR
  (`AtomicOpsCap 32/64bit+`)? If not, `v1`/InterNodeV1 EP can never work on this NIC — only WRITE-based
  kernels (`async_ll`).
- **AMD/MoRI:** make `async_ll` (or a WRITE-based signaling path) the default / auto-selected for NICs
  without atomic-completer capability; ensure the **vLLM wide-EP** connector selects `async_ll` on Thor2.
- **Broadcom driver:** upstream fix so a ≥237 driver's CQ path (`setup_cq_hwqs`/`cq_timer`) works on this
  firmware, removing the need to pin 235.2.86.0.
- **MoRI-IO GPU dmabuf:** raise the >8 MiB GPU-region ceiling (or confirm it's image-specific to the
  from-source tip build vs the `wideep_mori123` image).

See `../LOGBOOK.md` (entries 2026-08-01) for the full investigation trail.
