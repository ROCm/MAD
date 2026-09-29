# Cluster profiles

One JSON file per SLURM cluster, named after the cluster (`m2m.json` is the OCI MI300X
cluster, `SLURM_CLUSTER_NAME=m2m`). A profile holds what has to be known **before** a job
is submitted: which partition to ask for, how many GPUs a node has, whether to take nodes
exclusively.

The format is madengine's `--additional-context`, so a profile is used as it is:

```bash
madengine run --manifest-file build_manifest.json \
    --additional-context-file scripts/common/clusters/m2m.json \
    --additional-context '{"slurm": {"nodes": 4, "time": "06:00:00"}}'
```

`--additional-context` is deep-merged over the file, key by key, so a run sets its own node
count and time and keeps the cluster's partition. When you submit a launcher with `sbatch`
yourself, the same `slurm` block is what you pass as options:

| profile key | sbatch option |
|---|---|
| `slurm.partition` | `--partition` |
| `slurm.gpus_per_node` | `--gpus-per-node` |
| `slurm.exclusive` | `--exclusive` |
| `slurm.nodes` (per run) | `--nodes` / `--ntasks` |
| `slurm.time` (per run) | `--time` |

What belongs in `../cluster.sh` instead: facts the job needs once it is running on the nodes
(weights locations, fabric device names, ports, timeouts). Those are `${VAR:-default}`
environment, read inside the allocation, and any of them can be overridden by exporting it.

To add a cluster: copy `m2m.json` to `<name>.json`, set its partition and GPU count, and
extend `../cluster.sh` if the nodes' filesystems or fabric differ (it detects cx7 / ainic /
thor2 adapters itself). See [../README.md](../README.md) for running a workload end to end.
