# Cluster profiles

One JSON file per SLURM cluster, named after the cluster (`m2m.json` is the OCI
MI300X cluster, `SLURM_CLUSTER_NAME=m2m`). A profile holds what has to be known
**before** a job is submitted: which partition to ask for, how many GPUs a node
has, whether to take nodes exclusively.

The format is madengine's `--additional-context`, so a profile is a fragment of
the same document a run's `ADDITIONAL_CONTEXT` is. The layers, weakest first:

1. madengine's generic SLURM presets
2. the cluster profile selected here
3. what the pipeline derives for the run: node count from the model card,
   `slurm.time` from the run's timeout, the GPU arch probed on the partition
4. the run's `ADDITIONAL_CONTEXT`

Both CI paths resolve these layers the same way, so switching a run between
MADENGINE and STANDALONE does not change its allocation.

What belongs in `../cluster.sh` instead: facts the job needs once it is running
on the nodes (weights locations, fabric device names, ports, timeouts). Those are
`${VAR:-default}` environment, read inside the allocation.

To add a cluster: copy `m2m.json`, set its partition and GPU count, and select it
by name (`"cluster": "<name>"` in the pipeline's cluster selection).
