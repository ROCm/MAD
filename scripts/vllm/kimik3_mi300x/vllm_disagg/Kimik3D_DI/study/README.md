# K3 Capacity Study — per-cluster workspace

Everything to drive the Kimi-K3 (1.5T MXFP4) 2P/2D disagg EP16 (TP2/DP8) capacity study on
both clusters, plus a copy of results here in the WSL session for analysis.

## Layout
```
study/
├── mi325x_thor2/            # MI325X + Thor2 (bnxt RoCE, 256 GiB), ssh launch, WekaFS shared home
│   ├── launcher/            # cluster-pinned run_2p2d*.sh (bnxt RDMA-lib mount block)
│   ├── env/                 # image.env (tag), fabric_env.sh, v4_env.sh
│   └── results -> ../../results/mi325x_thor2   # symlink to the clean results tree
├── mi300x_cx7/              # MI300X + CX-7 (mlx5 RoCE, 192 GiB), SLURM launch, node-local NVMe
│   ├── launcher/            # fork: mlx5 RDMA-lib mount block + SLURM per-node staging
│   ├── env/                 # image.env, fabric_env.sh (eth0 mgmt + mlx5_0,2,3,4,5,7,8,9 RDMA)
│   └── results -> ../../results/mi300x_cx7
└── shared/
    ├── capacity_model.py    # platform-agnostic (symlink)
    ├── bench/               # niah_sweep, perf_sweep, agentx invocation — shared probes
    └── secrets.env          # Docker PAT (mode 600, .gitignored — DO NOT ship to drive)
```

## Key per-cluster differences (why launchers are forked, not shared)
| Axis | MI325X + Thor2 | MI300X + CX-7 |
|---|---|---|
| RDMA provider (host-mounted) | `libbnxt_re-rdmav34.so` | `libmlx5` provider (TODO: host path) |
| RDMA HCAs | bnxt behind tw-eth0..7 | mlx5_0,2,3,4,5,7,8,9 (exclude mgmt mlx5_1/6) |
| Mgmt / SOCKET_IFNAME | eno0 | eth0 |
| Launch | direct ssh, shared `~/k3disagg` (WekaFS) | **SLURM** (srun/salloc), **node-local NVMe** — stage per-node |
| Model dir | `/it-share-prj2-1/models/Kimi-K3-MXFP4` | `/mnt/m2m_nobackup/models_blog/Kimi-K3-MXFP4` |
| Image tag | `kimik3-wideep-disagg:v4-mi325x-thor2` | `...:v4-mi300x-cx7` |

Same Dockerfile + same Docker PAT (rocmshared) both clusters; RDMA provider is host-mounted
(not baked), so one image works on both fabrics — you swap the mount block, not the image.
Cluster-specific tags are for provenance only.

## Fixed serving contract (both clusters)
2P/2D disagg, TP2/DP8/EP16 per pool.
- **Prefill:** `mori_high_throughput` + cudagraph NONE (eager), kv_producer
- **Decode:** `mori_low_latency` + cudagraph FULL_AND_PIECEWISE, kv_consumer
- JIT cache host-mounted (`/tmp/$USER/vllm_jit_cache/...` → `/opt/vllm_cache`; AITER/Triton/vLLM).
- `K3_WRITE_READBACK=1`. `thinking=false` for benchmarks.

## Results flow
Runs write to each cluster's `results/` tree (see `../results/README.md` for layout + naming).
That tree is copied back here for analysis and later shipped to the home-folder drive.

Full plan: `../K3_CAPACITY_STUDY_PLAN.md`. Fabric detail: `../results/fabric_env_both_clusters.md`.
