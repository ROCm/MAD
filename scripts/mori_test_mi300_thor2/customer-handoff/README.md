# Customer Hand-off — Build the MoRI-EP / MoRI-IO Test Image (MI300X + Broadcom Thor2)

This folder lets you build, on **your own** MI300X + Broadcom BCM57608 "Thor2" cluster, the same
self-contained MoRI test image we validated (`...:vllm_wideEp_Mori_tests_...`). The image runs the
MoRI-EP (async_ll) and MoRI-IO benchmarks with **no runtime library bind-mounts**, because it bakes
in the host RDMA userspace.

> **The one big idea:** the base serving image ships a *newer* libibverbs (ABI v59) that **rejects the
> Broadcom bnxt kernel driver's ABI (8)** → the container sees **zero RDMA devices**. The fix is to
> bake in the **host's** v34 RDMA userspace + the bnxt 235 provider. That's what this package does.

---

## 1. What it builds — validated stack

| Component | Version |
|---|---|
| GPU | AMD Instinct MI300X (gfx942) |
| NIC | Broadcom BCM57608 Thor2 400G |
| Linux kernel | `5.15.0-177-generic` |
| ROCm | `7.2.3` (from the base image) |
| **bnxt_re/bnxt_en driver** | **`235.2.86.0`** (host prerequisite) |
| bnxt firmware | `238.1.138.0` |
| MoRI | built from source, commit `12d1bc32` (CI-green AsyncLL fix) |
| verbs ABI (baked) | v34 (`IBVERBS_PRIVATE_34`) |

---

## 2. Prerequisites on each host (hard requirements)

1. **bnxt 235.2.86.0 driver installed + rebooted.** This is what produces `libbnxt_re-rdmav34.so`.
   Use the package + script in `../driver-235.2.86.0/` (`install_driver_235.sh` or
   `standardize_node_235.sh`). Public debs also at:
   `https://packages.broadcom.com/artifactory/ethernet-nic-debian-public/pool/main/`
   Verify: `modinfo bnxt_re | grep ^version` → `235.2.86.0`, and `ibv_devinfo` shows PORT_ACTIVE NICs.
2. **rdma-core userspace on the host** (`libibverbs1`, `librdmacm1`, `libnl-3-200`, `libnl-route-3-200`).
   On Ubuntu 22.04 these come with the OS / `apt-get install -y ibverbs-providers rdma-core`.
3. **Docker Hub pull access** to the base image (section 3).
4. **Build-time internet** on the build node — the Dockerfile `git clone`s MoRI. (Offline? see §8.)

---

## 3. Base image

```
docker.io/rocm/vllm-dev:vllm-wideep_06_29_2026_Shiksha_dp16_2p2d_mori_v1.2.1_aiter_v0.1.16.post3_nightlybase_mori121
```
Provides ROCm 7.2.3 + PyTorch + Python 3.12 + gfx942 toolchain, and — importantly — a **pre-built
vLLM + AITER** (`v1.2.1` / `aiter_v0.1.16.post3`, wide-EP + MoRI connector). Override with
`--build-arg BASE=<your-mirror>` if you host it internally.

> **vLLM provenance — read this.** We do **not** build vLLM in this Dockerfile. It ships inside the base
> image; the `mori_v1.2.1_aiter_v0.1.16.post3_...mori121` tag **is** the vLLM pin. `apply_async_ll_patch.py`
> then text-patches the pre-installed `vllm/distributed/device_communicators/all2all.py`
> (`_make_all2all_kwargs`) to force the Thor2 `async_ll` EP kernel. The patch is **anchor-based and fails
> loudly** (`ERROR: anchor block not found — vLLM version differs`) if you point `--build-arg BASE=` at a
> base with a different vLLM — so a wrong base can't silently produce a broken image. If you must use a
> different vLLM, update the `OLD`/`NEW` anchor in `apply_async_ll_patch.py` to match its
> `_make_all2all_kwargs`.

---

## 4. Host libraries baked into the image (the 5 that matter)

`collect_host_libs.sh` gathers these from the build host into `./libs/`:

| Library (baked as) | Source on the host |
|---|---|
| `libibverbs.so.1.14.39.0` | distro pkg **`libibverbs1`** (rdma-core) — must be **v34 / IBVERBS_PRIVATE_34** |
| `librdmacm.so.1.3.39.0` | distro pkg **`librdmacm1`** |
| `libnl-3.so.200.26.0` | distro pkg **`libnl-3-200`** |
| `libnl-route-3.so.200.26.0` | distro pkg **`libnl-route-3-200`** |
| `libbnxt_re-rdmav34.so` (~539 KB) | the **bnxt 235.2.86.0 driver install** (`/usr/local/lib/...`) — NOT a distro pkg |

**Why baked:** see the big idea at the top. The Dockerfile removes the image's v59 libibverbs, points
`libibverbs.so.1` at the v34 file, installs the bnxt v34 provider, and writes
`/etc/libibverbs.d/bnxt_re.driver`. A build-time assertion fails the build if `libibverbs.so.1` is not v34.

> **Important:** `libbnxt_re-rdmav34.so` is host/driver-specific — collect it from a node that is **already
> on the 235 driver**, not a mismatched host. `collect_host_libs.sh` checks its size + the verbs ABI and
> errors out otherwise.

> **Your versions may differ.** The exact filenames above (`libibverbs.so.1.14.39.0`, `librdmacm.so.1.3.39.0`,
> `libnl-*.so.200.26.0`) are **our** host's rdma-core build numbers. On your site the `.so` suffixes will
> likely differ. **Do not hard-copy our filenames** — instead:
> 1. Run the **ClusterSphere recommender** (§5) on a host node to confirm *which* libraries + env your NIC needs.
> 2. Run `collect_host_libs.sh`, which resolves each soname with `readlink -f` and copies the **real
>    versioned file from your host** into `./libs/`.
> 3. The Dockerfile COPY lines reference the versioned names; if your suffixes differ, `collect_host_libs.sh`
>    prints the names it wrote — update the five `COPY libs/...` lines in the Dockerfile to match, then build.
>
> The single invariant that must hold (and the build asserts it): `libibverbs.so.1` resolves to
> **`IBVERBS_PRIVATE_34`** and the bnxt provider is the ~539 KB 235 build.

---

## 5. ClusterSphere — confirm the exact libs/env to expose (optional but recommended)

`clustersphere/cluster_rdma_env_recommender.py` is AMD's small RDMA diagnostic from
`ROCm/dist-inf-cookbook` (`cluster-sphere/cluster-rdma-env-recommender/`). It's a **single ~457-line
Python script** (+ a 102-line `html_reporter.py`) — no install. Run it on a host node:

```bash
python3 clustersphere/cluster_rdma_env_recommender.py
```
It scans each RDMA device (PCI, netdev, firmware, GID index, vendor) and prints a **"RECOMMENDED DOCKER
LAUNCH COMMAND"** listing exactly which host libraries + env to expose for your NIC (it has bnxt / mlx5 /
ionic paths). Use it to auto-confirm the §4 list matches your cluster.

---

## 6. Build

On a build node that is on the 235 driver:
```bash
cd customer-handoff
bash collect_host_libs.sh                        # fills ./libs/ from the host (verifies v34 + 235)
docker build -f Dockerfile -t <your-registry>/vllm_wideEp_Mori_tests .
# offline base image / internal mirror:
#   docker build --build-arg BASE=<your-mirror>/mori121 -f Dockerfile -t <tag> .
```
The build: installs MoRI 12d1bc32 from source, applies the async_ll vLLM patch, bakes the host RDMA libs,
and asserts the v34 ABI. ~10-15 min (MoRI compile dominates).

---

## 7. Run (device access is a docker-run concern — cannot be baked)

> **⚠️ Fabric-specific settings — change these for your cluster.** The baked defaults are **ours**:
> mgmt interface `eno8303`, RDMA device `rdma3`, GID index `3`, SL `3`, TC `104`. Your cluster will
> differ — get the right values from the **ClusterSphere recommender** (§5), then either bake them at
> build time (`docker build --build-arg SOCKET_IFNAME=<your-mgmt-nic> --build-arg GID_INDEX=<n> ...`)
> or override per-run: `docker exec -e SOCKET_IFNAME=<nic> -e MORI_IB_GID_INDEX=<n> mori_host ...`.
> **The #1 hang is a wrong `SOCKET_IFNAME`** — torchrun rendezvous silently stalls if it points at an
> interface the peer can't reach. It must be your **management/OOB NIC** (the one carrying `<master_ip>`),
> not a RoCE rail.

```bash
docker run -d --name mori_host \
  --network host --ipc host --privileged \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v /lib/modules:/lib/modules:ro \
  --ulimit memlock=-1:-1 --ulimit nproc=100000:100000 --shm-size 64g --cap-add SYS_PTRACE \
  --entrypoint sleep <your-tag> infinity
```

**MoRI-EP internode (async_ll)** — rank1 first, then rank0 (master = a node's mgmt IP):
```bash
docker exec mori_host bash /opt/mori-tests/ep_pair_test.sh 1 <master_mgmt_ip> rdma3 29100   # on node B
docker exec mori_host bash /opt/mori-tests/ep_pair_test.sh 0 <master_mgmt_ip> rdma3 29100   # on node A
# expect: "Node N Dispatch Pass" + "Node N Combine Pass" each round;
#         final "rank: N error times: 0 appear round: set()"
```

**MoRI-IO CPU write sweep**:
```bash
docker exec mori_host bash /opt/mori-tests/io_pair_test.sh 1 <master_mgmt_ip> <nodeB_mgmt_ip> rdma3 29500  # target
docker exec mori_host bash /opt/mori-tests/io_pair_test.sh 0 <master_mgmt_ip> <nodeA_mgmt_ip> rdma3 29500  # initiator
# expect: sweep table peaking ~48 GB/s @ 64 MiB (near 400G line rate)
```

> Adjust `MORI_RDMA_DEVICES`, `MORI_IB_GID_INDEX`, `MORI_RDMA_SL/TC` (baked defaults: rdma3, 3, 3, 104)
> to match your fabric if different — or take them from the ClusterSphere recommender (§5).

---

## 8. Verify — the 4-rung ladder (run in order)

Each rung tests strictly more than the last. **Stop and fix at the first failure** — a lower rung
failing explains every higher one.

### Rung 1 — libraries loaded (single node, inside the container)
```bash
strings /usr/lib/x86_64-linux-gnu/libibverbs.so.1 | grep -m1 IBVERBS_PRIVATE   # -> IBVERBS_PRIVATE_34
ibv_devinfo -d rdma3 | grep -E 'fw_ver|PORT_ACTIVE'                            # -> PORT_ACTIVE
```
If `ibv_devinfo` shows **0 devices / no PORT_ACTIVE**, your baked libs are wrong — the image's v59
libibverbs is still loading and rejecting the bnxt kernel ABI-8 provider. Re-do §4 (collect + relink).
This is the exact symptom the whole package exists to fix.

### Rung 2 — RDMA works across two nodes (the library + fabric canary)  ← `ib_write_bw`
`ib_write_bw` goes through the **same** `ibv_get_device_list` → provider-load → ABI-check path MoRI uses,
so **if the baked libraries mismatch, this fails before you ever reach MoRI.** It also validates the RoCE
fabric (GID index, SL/TC, PFC, cabling). Run inside the container on both nodes:
```bash
# node B (server):
ib_write_bw -d rdma3 -x 3 -F --report_gbits
# node A (client), point at node B's data-rail IP:
ib_write_bw -d rdma3 -x 3 -F --report_gbits <nodeB_ip>
# expect: ~370-376 Gb/s (near 400G line rate).  -x 3 = GID index 3 (RoCEv2).
```
> **What `ib_write_bw` does NOT prove:** it is pure RDMA **WRITE**, so it passes even on this NIC that
> **cannot** complete the VRAM atomics the default MoRI `v1` kernel needs. That's exactly why WRITE-based
> `async_ll` works here while `v1` hangs. **A green `ib_write_bw` confirms the libs + fabric, not EP** —
> only Rung 4 proves EP.

### Rung 3 — MoRI is importable (single node)
```bash
python3 -c "import mori; print(mori.__version__)"                              # -> 0.1.1.dev1+g12d1bc32d
```

### Rung 4 — the actual EP path (two nodes) — the real proof
Run the async_ll EP pair test from §7 (`ep_pair_test.sh`). Only this exercises the GPU-initiated
dispatch/combine + the atomics-avoidance that makes Thor2 work. Expect `Dispatch Pass` + `Combine Pass`
every round and `error times: 0`.

**Offline build** (no build-time internet): pre-vendor MoRI at commit `12d1bc32` into a `mori-src/`
folder, edit `build_mori.sh` to `cp -r` it instead of `git clone`, or bake it in a prior layer.

---

## 9. Folder contents

| Path | What |
|---|---|
| `Dockerfile` | self-contained build (FROM the public base) |
| `collect_host_libs.sh` | gathers the 5 host RDMA libs into `./libs/` (run on a 235-driver host) |
| `build_mori.sh` | builds MoRI from source, pinned to commit `12d1bc32` |
| `apply_async_ll_patch.py` | vLLM MoRI-EP kernel-selection patch (async_ll for Thor2) |
| `ep_pair_test.sh`, `io_pair_test.sh` | the EP + IO pair test runners (baked to `/opt/mori-tests/`) |
| `clustersphere/` | AMD RDMA env recommender (host-lib exposure diagnostic) |
| `libs/` | **you fill this** via `collect_host_libs.sh` (empty on delivery) |
| `../driver-235.2.86.0/` | the bnxt 235 driver package + install scripts (host prerequisite) |

## 10. Links
- Broadcom bnxt driver debs: `https://packages.broadcom.com/artifactory/ethernet-nic-debian-public/pool/main/`
- ClusterSphere / dist-inf-cookbook: `https://github.com/ROCm/dist-inf-cookbook` (`cluster-sphere/cluster-rdma-env-recommender/`)
- MoRI: `https://github.com/ROCm/mori` (commit `12d1bc32`)
