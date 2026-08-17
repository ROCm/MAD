# MoRI-IO two-node RDMA block-size sweep (AAC MI355X + Pensando AINIC)

Measures MoRI-IO transfer bandwidth and latency between **mi355-gpu-45** and
**mi355-gpu-46** across a geometric ladder of block sizes, engaging **all 8
GPU-data-plane RDMA rails** simultaneously.

## Status

| | |
|---|---|
| Test code | written, syntax-checked, **and run** (`run_moriio_sweep.sh`, `moriio_sweep.slurm`) |
| Preflight gates | **passed on the cluster** — 8/8 rails ACTIVE *inside* the container |
| Results | **MEASURED** 2026-08-17, job 5642, `exit=0`. See *Results* below. |

Every number below came out of `results/moriio_sweep_rank0_20260817_053240.log`.
Nothing here was written by hand.

## Files

| File | What it is |
|---|---|
| `run_moriio_sweep.sh` | The measurement. Runs **inside** the container, once per node. |
| `moriio_sweep.slurm` | Batch wrapper: puts the workload image on both nodes and starts rank 0/1. |
| `aggregate_sweep.py` | Folds the 8 per-rail tables into one fabric-level table + JSON. |
| `Dockerfile` | Layers the missing harness onto PR#205's workload image, pinned to the *same* mori commit as the image's libmori. Optional — see *Running it*. |
| `results/` | Per-rank logs and the aggregate JSON from the measured run. |

This folder is self-contained and is **not** registered in `models.yaml`; it is
run directly, like `scripts/kvcache_transfer_bench/`.

## Results

Job 5642, `mi355-gpu-45` (initiator) → `mi355-gpu-46` (target), RDMA write,
8 rails × 8 QP/transfer × 8 worker threads, batch 64, 10 iters.

`benchmark.py` prints **one table per rail**. The fabric number is the sum
across the 8; the per-rail min/max is what tells you whether a rail is lame.
Latency is the **max** across rails, because a batched handoff completes when
its slowest rail does.

| MsgSize | Per-rail bytes | **Agg Avg BW** | Agg Max BW | Slowest rail | Fastest rail | Max Avg Lat |
|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 0.26 MB | **11.4 GB/s** | 15.5 GB/s | 1.27 | 1.71 | 0.21 ms |
| 8 KiB | 0.52 MB | **30.9 GB/s** | 35.1 GB/s | 3.23 | 4.66 | 0.16 ms |
| 16 KiB | 1.05 MB | **58.1 GB/s** | 65.6 GB/s | 6.33 | 8.37 | 0.17 ms |
| 32 KiB | 2.10 MB | **106.7 GB/s** | 116.7 GB/s | 12.12 | 15.04 | 0.17 ms |
| 64 KiB | 4.19 MB | **176.4 GB/s** | 183.5 GB/s | 20.98 | 23.12 | 0.20 ms |
| 128 KiB | 8.39 MB | **247.0 GB/s** | 252.3 GB/s | 29.77 | 31.70 | 0.28 ms |
| 256 KiB | 16.78 MB | **301.2 GB/s** | 306.2 GB/s | 36.78 | 38.30 | 0.46 ms |
| 512 KiB | 33.55 MB | **338.0 GB/s** | 341.7 GB/s | 41.86 | 42.62 | 0.80 ms |
| 1 MiB | 67.11 MB | **360.5 GB/s** | 363.6 GB/s | 44.82 | 45.31 | 1.50 ms |
| 2 MiB | 134.22 MB | **373.2 GB/s** | 376.0 GB/s | 46.54 | 46.77 | 2.88 ms |
| **4 MiB** | 268.44 MB | **378.1 GB/s** | **381.0 GB/s** | 47.08 | 47.43 | 5.70 ms |
| 8 MiB | 536.87 MB | **353.7 GB/s** | 355.3 GB/s | 44.01 | 44.54 | 12.20 ms |
| 16 MiB | 1073.74 MB | **355.4 GB/s** | 355.6 GB/s | 44.24 | 44.67 | 24.27 ms |
| 32 MiB | 2147.48 MB | **356.4 GB/s** | 356.6 GB/s | 44.38 | 44.82 | 48.39 ms |

*(Per-rail columns are Avg BW in GB/s. Machine-readable: `results/…_aggregate.json`.)*

### Three things this table says

**1. The fabric is at line rate. 378.1 GB/s of 400 GB/s nominal = 94.5%.**
Eight 400 Gb/s rails are 8 × 50 GB/s = 400 GB/s raw. The best rail hit
47.43 GB/s, **94.9% of its own line rate** — and that is with RoCE headers,
ICRC, and MoRI's own chunk descriptors in the count. There is no meaningful
bandwidth left on this fabric to recover; MoRI-IO is not the bottleneck in a
disagg handoff on this cluster.

**2. The 4 MiB peak and the 8 MiB drop are the `max_chunks` effect, confirmed.**
The pre-run analysis predicted this exact shape from the source, before the run:
`PlanChunkGeometry` (`src/io/rdma/common.cpp:459-473`) computes
`softCount = min(ceil(total/chunkBytes), maxChunks)`, so with the build defaults
(64 KiB chunks, 64 max) the chunk **count** saturates at exactly
64 × 64 KiB = **4 MiB**. Above that the count is pinned and the chunk *size*
grows instead. The measurement lands the peak at 4 MiB and then loses 6.5%
(378.1 → 353.7) and stays flat through 32 MiB — the curve stops improving at
precisely the size the code says it should.

This is not a defect and not something to tune away here. It is the reason the
`CHUNK_SWEEP=1` mode exists, **and that mode did not run on this pass** — not
because the engine lacks the knob, but because the *harness* that was mounted
predated it. See *Which mori was measured* below.

**3. Rail spread collapses as blocks grow, which is the healthy signature.**
At 4 KiB the slowest rail does 1.27 GB/s and the fastest 1.71 — a **30.7%**
spread, because at that size you are measuring per-transfer software overhead
and scheduler jitter, not the wire. By 2 MiB the spread is **0.5%**
(46.54 vs 46.77). Rails that diverge at large blocks would mean a bad cable, a
degraded link, or a mis-mapped rail; these converge. All 8 rails are healthy.

### What this means for GLM-5.2 disagg

GLM-5.2 hands over `(512 + 64) × 1 B × 78 layers = 43.88 KiB` of KV per token.
A 28,672-token prefill is **~1.2 GiB**. At the measured 378 GB/s aggregate that
handoff costs **~3.4 ms** — against a 7 s TTFT SLO, i.e. **0.05% of the budget**.

The operative conclusion: **KV transport is not, and will not become, the TTFT
constraint on this cluster.** That is consistent with the separately-measured
finding that prefill is attention-bound (82% MLA sparse attention). Time spent
tuning MoRI-IO block sizes is time not spent on the thing that actually costs
seconds.

The one caveat worth carrying forward: real KV blocks are governed by
`KV_BLOCK_SIZE`, not by this sweep's ladder. If a future config pushes
per-transfer size *below* ~256 KiB, the table shows bandwidth falling off
steeply (301 → 176 → 107 GB/s at 256/64/32 KiB). The safe operating region is
**≥ 1 MiB per transfer**, where every point is within 5% of peak.

## What it actually measures

MoRI ships the benchmark; this is a driver for it, not a reimplementation.
`tests/python/io/benchmark.py` already emits exactly the requested table:

```
MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg BW (GB/s) | Min Lat (us) | Avg Lat (us)
```

**Why not `ib_write_bw`.** `ib_write_bw` measures the NIC. It would tell us the
link is healthy and nothing about the thing that carries KV cache: MoRI's QP
fan-out, its chunking planner, its worker-thread dispatch, and its GPU memory
registration path. A regression in any of those is invisible to `ib_write_bw`
and fatal to disagg serving. (`ib_write_bw` remains the right tool for
*isolating* a suspected link fault — that is how GID index 1 was established.
The cookbook's own `cluster-rdma-tests` suite is exactly that: a bare
`ib_write_bw -d $IBDEVICES -q 4 -a --report_gbits`, which is why this sweep is a
separate deliverable rather than a call into it.)

**Why these block sizes.** See the KV arithmetic above. Real traffic lives at
the **top** of this ladder; the small end is included to show where the
per-transfer overhead floor is, not because production sends 4 KiB blocks. The
4 KiB row earning its place is precisely the 30.7% rail-spread datapoint.

## How "all RDMA NICs" is achieved

Eight rails, one per GPU:

```
rocep9s0   rocep25s0   rocep105s0  rocep121s0
rocep137s0 rocep153s0  rocep233s0  rocep249s0     driver ionic, 400 Gb/s
```

Excluded: `rocep193s0f0/f1` (driver `bnxt_en`, 200 Gb/s) — these carry the
management/default route. Without the allowlist MoRI enumerates all 10 ibv
devices and raises QPs over the management pair, which times out.

`--num-initiator-dev 8 --num-target-dev 8` makes the harness spawn 8 processes
per node, and `gpu_index = role_rank` (benchmark.py:450) binds initiator GPU *i*
to target GPU *i* — **rail-aligned by construction**. Confirmed in the run log:
`Built RdmaConn for engine TARGET-N with topo local(N,2) remote(N,2)` for
N = 0..7. `--target-dev-offset` is deliberately *not* used: its documented
purpose is to break that alignment ("offset 5 makes GPU0 -> GPU5 … to exercise
cross-rail transfers on rail-only fabrics"), which is a different experiment.

## Exposing the host RDMA stack: the cluster-sphere pattern

The workload image ships the **mlx5** provider. These nodes are **Pensando
AINIC (ionic)**. Without intervention the container enumerates **zero** ibv
devices, and a sweep over zero devices is not a slower number — it is no number.

`moriio_sweep.slurm` follows `dist-inf-cookbook/cluster-sphere/`
`cluster-rdma-env-recommender` (`_docker_cmd_ionic`) and builds **24 bind
mounts**, verified byte-identical on both nodes. Four groups, each fixing a
distinct failure:

| Group | Mounts | What breaks without it |
|---|---|---|
| **1. sysfs** | `/sys/class/infiniband`, `/sys/class/net`, `/sys/bus/pci` (ro) | MoRI's NIC auto-detect matches device *name* prefixes (`^mlx5`, `^bnxt_re`, `^ionic`); AAC names its devices `rocep*s0`, so the name match fails and it falls back to `readlink /sys/class/infiniband/<dev>/device/driver`. With sysfs masked that readlink returns nothing and the detect **silently defaults to mlx5** — a wrong backend, not an error. |
| **2. provider plugin** | `libibverbs/` dir + `/etc/libibverbs.d/` | libibverbs dlopens the vendor `.so` *named by* `ionic.driver`. Both halves must be present. **Zero devices** otherwise. |
| **3. vendor + core + libnl** | `libionic.so*`, `libibverbs`, `librdmacm`, `libnl-3`/`libnl-route-3`, `libmlx5`, `libefa` | Providers link against libnl; a version skew is a dlopen failure that surfaces as "no devices" rather than as a link error. |
| **4. `/etc/rdma`** | `/etc/rdma` | Same reason as `/etc/libibverbs.d`. |

**Why the recommender's form and not the cookbook slurm's.** That suite's own
`rdma_perf_tests.slurm` hard-codes `libionic.so.1.0.54.0-149.g3304be71`. These
nodes carry **`libionic.so.1.1.54.0-184`**. A hard-coded version string is a
bind mount that fails container creation (exit 125) the day the driver updates,
so this script copies the *recommender's* glob-discovery instead
(`_find_lib(["libionic-rdmav*.so"])` over `LIB_SEARCH_PATHS`). Two further
details: `.a` archives are skipped (`libionic.a` is present and useless to a
running process), and the existence test is `-f` not `-e`, because a **dangling
symlink** fails the mount and the container never starts.

**Proof it worked:** the preflight reports `ACTIVE rails: 8/8` *from inside the
container*, which is the specific thing a hand-mirrored mount list could not
guarantee.

One deviation from the cookbook, kept deliberately: its `MASTER_ADDR` picker is
`hostname -I | awk 'NR==1{print $1}'`. On AAC that selects the **public** IP,
which answers ICMP but is **firewalled node-to-node**. This script picks by
`FABRIC_SUBNET` instead.

## Which mori was measured, and the flag-availability caveat

**The measured engine is the image's own pinned libmori, `MORI_REF=42e895472b08`.**
Verified by importing it in the running container:

```
mori pkg  : /usr/local/lib/python3.12/dist-packages/mori/__init__.py
MORI_REF=42e895472b08@42e895472b08e9848ef09ec458420c95e6add5ec
```

This matters, and it is the *good* case: the number characterises the stack
PR#205 actually ships for GLM-5.2 disagg. The `-v $MORI_SRC:/opt/mori` bind
mount does **not** shadow it — the repo's Python package lives at `python/mori`,
not the tree root, so `import mori` cannot resolve into the mounted source, and
the tree carries no compiled `.so` regardless. The mount exists for exactly one
reason: to supply `tests/python/io/benchmark.py`, which the wheel does not
install.

**The harness, however, was an older `6ad812c` checkout**, and *its* `benchmark.py`
predates several flags. `argparse` rejects an unknown flag with exit 2 — the
whole sweep dies before a single transfer, which is how the first run failed.
`run_moriio_sweep.sh` therefore probes `--help` once and passes only what the
harness understands, reporting omissions loudly.

Read the distinction carefully, because an earlier version of this document got
it wrong: a flag missing from the *harness* does not mean the *engine* lacks the
knob. The pinned `42e8954` `benchmark.py` **does** accept `--mem-type`,
`--max-chunks` (default `64`) and `--chunk-bytes` (default `65536`). Only
`--sweep-step` and `--target-dev-offset` are genuinely absent from it.

| Flag | On pinned `42e8954` | Missing from the `6ad812c` harness meant |
|---|---|---|
| `--mem-type` | **present** (default `gpu`) | none — GPU is what we wanted anyway |
| `--max-chunks` | **present** (default `64`) | planner ran its default 64 — which is *why* the 4 MiB knee sits where it does |
| `--chunk-bytes` | **present** (default `65536`) | planner ran its default 64 KiB; `CHUNK_SWEEP=1` was skipped loudly rather than executed, because six identical tables under six `chunk=` labels would read as "chunk size doesn't matter" when it really means the knob was never applied |
| `--sweep-step` | absent | none — it offers *linear* stepping; geometric is the ladder we want |
| `--target-dev-offset` | absent | none — its purpose is to *break* rail alignment, which we verified holds |

Those defaults are also the arithmetic behind finding 2: `65536 × 64` is exactly
**4 MiB**, the measured knee.

So the knee is **observed, and explained from the planner's own defaults**, but
not yet **isolated**. Isolating it needs a harness at `42e8954` — which is
precisely what the [`Dockerfile`](Dockerfile) in this directory bakes in. Build
that image and `CHUNK_SWEEP=1` runs.

## Running it

Two ways to supply `benchmark.py`, which the mori wheel does not install.

**(a) Bind-mount a host checkout** — what the measured run did:

```bash
export DOCKER_IMAGE_NAME=docker.io/rocmshared/vllm-disagg:glm52-gfx950-ionic-v1
export MORI_SRC=$HOME/test_vllm_40549/mori-src         # FULL checkout
sbatch -w mi355-gpu-45,mi355-gpu-46 moriio_sweep.slurm
```

**(b) Build the image in this directory**, which bakes the harness in at the
same commit as the image's libmori, so the two cannot drift and every flag the
engine supports is available:

```bash
docker build --network=host \
  --build-arg BASE_IMAGE=docker.io/rocmshared/vllm-disagg:glm52-gfx950-ionic-v1 \
  -f Dockerfile -t moriio-nic-sweep:latest .

export DOCKER_IMAGE_NAME=moriio-nic-sweep:latest
export MORI_SRC=SKIP                                   # harness baked at /opt/mori
sbatch -w mi355-gpu-45,mi355-gpu-46 moriio_sweep.slurm
```

(b) also brings `numactl`, which the workload image lacks — see caveats.

Against an **existing** allocation, skip `sbatch` and attach — this is how the
measured run was taken:

```bash
export MORIIO_JOBID=5642                # borrows the running job
export DOCKER_IMAGE_NAME=docker.io/rocmshared/vllm-disagg:glm52-gfx950-ionic-v1
export MORI_SRC=$HOME/test_vllm_40549/mori-src
export CONTAINER_CLI=podman             # compute nodes have podman, NOT docker
export ITERS=10 BATCH=64
bash moriio_sweep.slurm
```

Attach mode always pairs `--jobid` with **`--overlap`**; without it `srun` waits
for job-step resources the parent already holds and blocks forever.

Then aggregate:

```bash
python3 aggregate_sweep.py results/moriio_sweep_rank0_<stamp>.log
```

Optional at submit time: `CHUNK_SWEEP=1` (needs a `42e8954` harness — path (b),
or a matching host checkout for (a)), `BLOCK_MIN`,
`BLOCK_MAX`, `ITERS`, `BATCH`, `FABRIC_SUBNET`, `NUM_DEV`, `QP_PER_TRANSFER`,
`WORKER_THREADS`.

`MORI_SRC` must be a **full** checkout. The clone at `/tmp/mori_src` is sparse
(`core.sparseCheckout=true`): `git ls-files` lists `tests/python/io/benchmark.py`
but the file is not on disk. Both scripts preflight for exactly this.

## Configuration, and where it comes from

Every fabric value defaults to what the serving recipe
(`../vllm_dissag/connectors/moriio.env.aac`) records as measured on this
cluster, and every one is overridable:

| Setting | Value | Why |
|---|---|---|
| `MORI_IB_GID_INDEX` | `1` | Index 1 is the RoCEv2 entry on Ionic — proven by `ib_write_bw`, corroborated by MAD `61dd42c`. Not 3. |
| `MORI_DEVICE_NIC` | `ionic` | Belt-and-braces with the sysfs mounts above: the mounts make the fallback `readlink` work, the pin makes it unnecessary. |
| `IFNAME` | `enp193s0f1np1` | Control plane only. Public IPs answer ICMP but have TCP **firewalled node-to-node**; the mgmt NIC (10.2.80.x) works. |
| `PYTORCH_{,HIP_}ALLOC_CONF` | `expandable_segments:False` | ROCm cannot dmabuf-export HIP-VMM memory, so with expandable segments ON, `RegisterRdmaMemoryRegion` EFAULTs (errno 14) on the first RDMA write. |
| `RDMAV_FORK_SAFE` / `IBV_FORK_SAFE` | `1` | The harness uses `torch.multiprocessing.spawn`, which forks. A fork can COW-remap the registered pinned region while the NIC still DMAs to the old physical pages → host SIGSEGV with no GPU fault. |
| `BATCH` | `64` | `_setup_rdma` allocates `(buffer_size+1) × batch` **per GPU**, and `buffer_size` becomes `max(buffer_size, sweep_max_size)` under `--all`. At the default batch 256 and a 32 MiB max that is 8.6 GiB/GPU; 64 gives ~2.1 GiB/GPU. |

**`numactl` was requested but is absent in the workload image.** The log records
`[warn] numactl absent -- NIC/CPU affinity left to the scheduler.` It is present
on the *host*, not in the container, so `MatchCpuNics()` ordering was unpinned
for the measured run. Given the 0.5% rail spread at ≥2 MiB it cost nothing
visible — but the small-block rows are the ones that measure software overhead,
so a future run *with* numactl is not strictly comparable at that end of the
table. The `Dockerfile` here installs it, which is one reason to prefer path (b).

## Two caveats that affect how the table should be read

**1. Rank 1's table is all zeros. That is not a failure.** `_add_row`
(benchmark.py:1129-1153) always prints a row but only appends to the JSON sink
`if avg_bw > 0` — its own comment: *"Skip TARGET/no-op rows (which report all
zeros)."* The target side does not measure. Read rank 0's log; that is what
`aggregate_sweep.py` parses.

**2. The top of the curve is partly a `max_chunks` measurement, not a fabric
measurement.** Covered in *Results* finding 2 and in the mori/flags section
above. The 32 MiB row is **not** "MoRI is slower at 32 MiB" — it is
64 × 512 KiB chunks instead of 512 × 64 KiB.

## Preflight refuses to run a partial fabric

The script hard-fails unless **all 8 rails report ACTIVE**. This is deliberate:
a 7-rail number is not obviously wrong when you read it later — it looks like a
MoRI bandwidth regression when it is really a link that never came up. The
slurm wrapper applies the same reasoning to the ionic stack: it checks the
user-space library **and** the provider plugin *independently*, because either
one missing yields zero devices, and aborts rather than producing an empty sweep.
