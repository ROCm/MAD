# Reproducing the Kimi-K3 Capacity Study (MI325X + MI300X)

Everything needed to duplicate the blog's numbers. The full write-up is
[`K3_CAPACITY_MASTER_REPORT.md`](K3_CAPACITY_MASTER_REPORT.md) (rendered:
[`K3_CAPACITY_BLOG_REPORT.html`](K3_CAPACITY_BLOG_REPORT.html)).

## 0. What you get
A *computable* capacity model for 1.4 TB Kimi-K3-MXFP4 served 2P/2D disaggregated wide-EP
(TP2×DP8→EP16), validated on **two fabrics** (MI325X+Thor2/bnxt, MI300X+CX-7/mlx5) plus a real
agentic trace (AgentX). Decode obeys `wall = C0 + k·OSL` — a fixed floor + per-token slope — so
concurrency is nearly free until the KV pool saturates.

## 1. Build the image
```bash
docker build -f Dockerfile.kimik3_disagg -t kimik3-wideep-disagg:v4 .
```
One image works on both fabrics — the RDMA provider is **host-mounted**, not baked (swap the
mount block, not the image). Version pins (base image, MoRI, vLLM fork, router) are in the
Dockerfile header.

## 2. Bring up the 4-role disagg serve
Per-cluster launchers live under `study/<cluster>/launcher/`. They are **forked, not shared**,
because the two fabrics differ (see `study/README.md` for the full axis table):

| | MI325X + Thor2 | MI300X + CX-7 |
|---|---|---|
| RDMA lib (host-mounted) | `libbnxt_re-rdmav34.so` | `libmlx5` |
| RDMA HCAs | `rdma0..7` | `mlx5_0,2,3,4,5,7,8,9` (exclude mgmt 1/6) |
| Mgmt iface | `eno0` | `eth0` |
| Launch | direct ssh, shared `~/k3disagg` (WekaFS) | **SLURM** (srun/salloc), node-local NVMe |
| Model dir | `/it-share-prj2-1/models/Kimi-K3-MXFP4` | `/mnt/m2m_nobackup/models_blog/Kimi-K3-MXFP4` |

```bash
# MI325X (direct ssh):
cd study/mi325x_thor2/launcher && bash run_2p2d_launch.sh
# MI300X (SLURM):
cd study/mi300x_cx7/launcher   && bash slurm_launch.sh
```
Fixed serving contract both clusters: prefill = `mori_high_throughput` + cudagraph NONE (eager),
decode = `mori_low_latency` + cudagraph FULL_AND_PIECEWISE, `K3_WRITE_READBACK=1`, JIT cache
host-mounted, `thinking=false` for benchmarks.

**Fabric gotchas** (the ones that cost real debugging): input var is `RDMA_DEVICES` (not
`MORI_RDMA_DEVICES`), `IB_GID_INDEX` (not `NCCL_IB_GID_INDEX`). Passing Thor2's `rdma0-7` on the
CX-7 cluster gives "no transport available for peer".

## 3. Run the sweeps (identical scripts both clusters)
Shared probes under `study/shared/bench/` — every script fires ONE warmup of the measured shape
before the timed loop, so no point eats cold JIT:
```bash
bash study/shared/bench/envelope_sweep.sh   # §5 concurrency envelope (the headline)
bash study/shared/bench/osl_sweep.sh        # §6 OSL slope (k)
bash study/shared/bench/run_perf_matrix.sh  # §6 throughput classes
python study/shared/bench/niah_sweep.py     # §? NIAH accuracy grid (depth × ctx)
python study/shared/bench/perf_sweep.py     # perf points
```
`study/shared/capacity_model.py` computes the `C0 + k·OSL` prediction + the KV-pool knee.

## 4. Expected results (from the report)
- **MI325X (256 GB, ~20 GB KV):** rides the floor to **con256** — wall 614→688 s (+12%) while
  throughput 0.2→47.6 tok/s (**238×**). con512 = KV wall.
- **MI300X (192 GB, ~4 GB KV):** rides to **con64** (60×), saturates **4× earlier** (matches the
  ~4 GB vs ~20 GB KV-pool ratio). Faster per request, lower concurrency ceiling.
- **Memory frontier:** MI300X reliable-NIAH ceiling ~100K vs MI325X 300K.
- **AgentX** real-trace ITL matches the OSL slope on each platform (floor is physical, not fitted).

## 5. Layout
```
Kimik3D_DI/
├── Dockerfile.kimik3_disagg          # the image (all version pins in header)
├── run_2p2d.sh / run_2p2d_launch.sh  # single-cluster bring-up (MI300X reference)
├── niah_sweep.py / perf_sweep.py     # single-cluster probes
├── load_image.sh                     # pull/tag image on nodes
├── study/                            # cross-platform reproducibility tree
│   ├── mi325x_thor2/{launcher,env}   # bnxt fabric, ssh launch
│   ├── mi300x_cx7/{launcher,env}     # mlx5 fabric, SLURM launch
│   └── shared/{bench,capacity_model.py}
├── K3_CAPACITY_MASTER_REPORT.md      # full study (16 sections)
└── K3_CAPACITY_BLOG_REPORT.html      # rendered blog
```
Note: `secrets.env` (Docker PAT) is intentionally NOT shipped — set `DOCKER_USER`/`DOCKER_PAT`
in your environment before `load_image.sh`.
