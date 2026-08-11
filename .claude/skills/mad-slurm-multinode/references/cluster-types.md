# Cluster archetypes and the env vars that differ

Keywords: CX7 Mellanox mlx5 RoCE, AMD AINIC Pollara ionic rdma, Broadcom
Thor2 bnxt_re RoCE, gfx942 gfx950, NCCL_IB_HCA, NCCL_SOCKET_IFNAME,
NCCL_IB_GID_INDEX, RCCL_AINIC_ROCE, RDMAV_DRIVERS, IBV_DRIVERS, eth0 eno0 fenic0,
show_gids, ibv_devices, skip_gpus_directive

Three archetypes seen in practice. The differences below decide whether the run
exercises the intended RDMA transport. Values are archetype-*typical* and are
confirmed on the actual node (`scripts/detect_cluster_env.sh`).

## Difference matrix

| Variable                     | CX7 / Mellanox-RoCE | AMD-AINIC / Pollara | Broadcom-Thor2-RoCE |
|------------------------------|--------------------------------------|----------------------------------------|----------------------------------------|
| `MAD_SYSTEM_GPU_ARCHITECTURE`| `gfx942`                             | `gfx950`                               | `gfx950` (MI355X; `gfx942` = MI300X)   |
| HCA hardware                 | Mellanox CX7 `mlx5_*`                | AMD AINIC `rdma0..7` (vendor 0x1dd8)   | Broadcom Thor2 `bnxt_re0..7`           |
| `NCCL_IB_HCA`                | `mlx5_0:1,mlx5_1:1` (or 8-rail mlx5 list) | `rdma0:1,rdma1:1,...,rdma7:1`     | `bnxt_re0:1,bnxt_re1:1,...,bnxt_re7:1` |
| `NCCL_SOCKET_IFNAME`         | `eth0`                               | `eno0`                                 | `fenic0`                               |
| `GLOO_SOCKET_IFNAME`         | `eth0`                               | `eno0`                                 | `fenic0`                               |
| `NCCL_IB_GID_INDEX`          | `3`                                  | `1`                                    | `3`                                    |
| `RDMAV_DRIVERS` / `IBV_DRIVERS` | `mlx5`                            | `ionic`                                | `bnxt_re`                              |
| `RCCL_AINIC_ROCE`            | (unset)                              | `1` (required)                         | (unset)                                |
| Shared FS paths              | shared inference FS + off-/home scratch | shared team FS (mounted on all nodes) | shared NFS work/data root on all nodes |
| SLURM selection             | `partition`+`account`+`qos`, sometimes `exclude` | usually `partition`+`reservation`+`nodelist` | `partition`, `exclude`/`nodelist`; `skip_gpus_directive` (see below) |
| AINIC driver mounts          | (not needed)                         | add to every manifest's `built_models.additional_docker_run_options` and `context.docker_mounts` — see **AINIC driver mounts** below | (not needed) |

## How to confirm each value on the node

These values live on the node, so the probe runs there before the user is
asked. The login/jump node and the compute nodes can differ, so compute-node
values come from running `scripts/detect_cluster_env.sh` through `srun` on an
allocated node (`srun -p <partition> [--reservation <res>] [--nodelist <node>]
-N1 bash scripts/detect_cluster_env.sh`); only the cluster-private selectors
(partition / account / qos / reservation / nodelist) come from the user.


- GPU arch: `rocm-smi --showhw` or `rocminfo | grep gfx` -> the `gfx9xx` target.
- IB/RDMA HCAs: `ibv_devices` (lists `mlx5_*` or `rdma*`). The `NCCL_IB_HCA`
  list should enumerate the GPU-attached HCAs (`:1` = port 1).
- Management iface: `ip -br link` / `ip -br addr` -> the routable host iface
  used for bootstrap (`eth0` vs `eno0`). This is not the data-plane device.
- RoCEv2 GID index: `show_gids` (or read
  `/sys/class/infiniband/<dev>/ports/1/gid_attrs/types/*`); pick the RoCEv2
  (v2) GID index. CX7 commonly 3, AINIC commonly 1.
- Driver: if `ibv_devices` shows `mlx5_*` -> `mlx5`; if `rdma*` (ionic) ->
  `ionic` and you are on AINIC (set `RCCL_AINIC_ROCE=1`).

## AINIC driver mounts

The ionic verbs driver lives on the host (`/usr/lib/x86_64-linux-gnu/libionic.so*`,
`/etc/libibverbs.d/`) and must be bind-mounted into every container so that
ibverbs inside the container can find and load the ionic provider.

**`built_models.additional_docker_run_options`** — add these `-v` flags:
```
-v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro
-v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro
-v /etc/libibverbs.d:/etc/libibverbs.d:ro
```

**`context.docker_mounts`** — add:
```json
"/etc/libibverbs.d": "/etc/libibverbs.d",
"/usr/lib/x86_64-linux-gnu/libibverbs": "/usr/lib/x86_64-linux-gnu/libibverbs"
```

Without these mounts `libibverbs` inside the container finds no provider, reports
0 IB devices, and RCCL silently falls back to TCP sockets.

## Broadcom-Thor2 specifics

Broadcom-Thor2 nodes carry Broadcom Thor2 NICs exposed as `bnxt_re0..7` ibverbs devices
over RoCEv2. Two things differ from the CX7/AINIC archetypes:

**1. `skip_gpus_directive` in the manifest's `deployment_config.slurm`.** This cluster's
SLURM rejects the generated `--gpus-per-node` directive (the partition does not
advertise GPU GRES the way the default sbatch template assumes), so the job is
refused before launch. Set `"skip_gpus_directive": true` so madengine emits the
sbatch script without that directive and relies on `exclusive`/`nproc_per_node`
instead. Symptom when missing: sbatch is rejected with an invalid `--gpus`/GRES
error and no job id is returned.

**2. bnxt_re transport tuning (set in BOTH manifest env blocks).** The Broadcom
provider needs a conservative QP/feature profile to run RoCEv2 reliably; the
defaults can hang or fall back. Validated set:

```json
"NCCL_IB_QPS_PER_CONNECTION": "1",
"NCCL_IB_USE_INLINE": "0",
"NCCL_IB_MERGE_NICS": "0",
"NCCL_IB_SPLIT_DATA_ON_QPS": "0",
"NCCL_IB_ADAPTIVE_ROUTING": "0",
"NCCL_GDR_FLUSH_DISABLE": "1",
"NCCL_DMABUF_ENABLE": "0",
"NCCL_IB_ROCE_VERSION_NUM": "2"
```

Also point ibverbs at the host provider inside the container (the image carries
the bnxt_re provider, but the host `/etc/libibverbs.d` is mounted to be safe):
`LIBIBVERBS_DRIVER_PATH=/usr/lib/x86_64-linux-gnu/libibverbs` and add
`"/etc/libibverbs.d": "/etc/libibverbs.d"` to `context.docker_mounts`. Confirm
the device list with `ibv_devices` (expect `bnxt_re*`) and the RoCEv2 GID with
`show_gids` (commonly index 3).

## MIOpen first-run compile time on gfx950 (AINIC)

On gfx950 (MI355X) MIOpen compiles GPU kernels on first use, which can take
**20–40 minutes**. This affects any workload that starts MIOpen (inference, training).

- Set `BARRIER_TIMEOUT_S` high enough (≥ `7200`) for the first run.
- Mount a **persistent** MIOpen cache dir so subsequent runs skip compilation:
  - env var: `MIOPEN_USER_DB_PATH` → in-container path (e.g. `/miopen_cache`)
  - mount: container `/miopen_cache` → host shared-FS dir
- Once the cache is warm, `BARRIER_TIMEOUT_S` can be reduced to a few minutes.

## Why this matters

If `NCCL_IB_HCA` names devices that do not exist on the node, RCCL/NCCL
initializes zero NICs and either aborts (`Failed to initialize any NET
plugin`) or silently falls back to TCP sockets — making the perf number a
measurement of the wrong path. On AINIC, omitting `RCCL_AINIC_ROCE=1` /
`RDMAV_DRIVERS=ionic` has the same effect (fallback to verbs/sockets). These
appear in BOTH `context.docker_env_vars` and `deployment_config.env_vars` of the
manifest (and in `mad.env` host env as a belt-and-suspenders default).
