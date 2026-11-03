# Running multinode workloads on Kubernetes

A multinode card runs on Kubernetes with its **launcher unchanged**: the same
`run_xPyD_models.slurm` or `run_multinode.slurm`, the same recipe, the same benchmarks,
the same `perf.csv`. What changes is who provides the nodes. On SLURM they are an
allocation; on Kubernetes they are pods.

This page assumes you have read [multinode-overview.md](multinode-overview.md) and know how a
card runs on SLURM ([multinode-running.md](multinode-running.md)).

## Contents

- [How a card runs on Kubernetes](#how-a-card-runs-on-kubernetes)
- [Two ways to submit](#two-ways-to-submit)
- [What the cluster must provide](#what-the-cluster-must-provide)
- [The configuration file](#the-configuration-file)
- [Submitting standalone](#submitting-standalone)
- [Submitting through madengine](#submitting-through-madengine)
- [Results and what makes a run pass](#results-and-what-makes-a-run-pass)
- [Worked example: colocated vLLM on 2 and 4 pods](#worked-example-colocated-vllm-on-2-and-4-pods)
- [Troubleshooting](#troubleshooting)
- [Checking that both paths deploy the same thing](#checking-that-both-paths-deploy-the-same-thing)

## How a card runs on Kubernetes

A launcher does two jobs. It **orchestrates**: it finds the allocation's nodes
(`scontrol show hostnames`), asks each one for its address (`srun --nodelist=<node>
hostname -I`), and starts one container per node (`srun --nodelist=<all> bash -c 'docker
run ...'`). And it **is the per-node body**: what each container runs.

Kubernetes has no `srun` and no `docker`, but a pod already is a container of the card's
image. So a card runs as:

```
Indexed Job, one pod per node of the card          headless Service <run>
  pod <run>-0   the batch host: runs the card's      (pods reach each other as
                launcher, unchanged                    <run>-0, <run>-1, ...)
  pod <run>-1   serves the launcher's srun tasks
  ...
ConfigMap       the card's script directory, its siblings (default: ../common) and the shim
PVC <run>-results   mounted at /results in every pod; LOG_PATH defaults to /results/logs
```

Pod 0 runs the launcher with the `SLURM_*` variables a batch script sees (`SLURM_JOB_ID`,
`SLURM_JOB_NODELIST`, `SLURM_NNODES`, ...) and with small stand-ins first on `PATH`
(`mad_k8s.py`):

| The launcher runs | The stand-in does |
|---|---|
| `scontrol show hostnames` | Lists the run's pods (`<run>-0,<run>-1,...`) |
| `srun [--nodes=N] [--nodelist=...] CMD` | Sends `CMD` to each named pod, one task per node, with the caller's environment and `SLURM_PROCID` / `SLURM_NODEID` set as srun would; streams stdout and stderr; returns the highest exit status |
| `docker run -e ... -v SRC:DST --entrypoint E IMAGE CMD` | Runs `CMD` **in the pod itself**, on the image's own environment plus `-e`, with `DST` made to show `SRC` |
| `docker stop / rm / ps / pull / image inspect` | Acts on those processes and on the pod's image |

Host-level `docker run` options in a launcher (`--device`, `--network host`,
`--privileged`, `--ulimit`, `--shm-size`) are ignored: on Kubernetes they are the pod
spec's job, configured as described below. Any `srun`, `docker` or `scontrol` form the
stand-ins do not implement fails with a message naming it, rather than being guessed at.

Because the launcher is unchanged, everything it does on SLURM it does here: `cluster.sh`
resolves weights and fabric, the recipe comes from `models.yaml`, servers fail fast, the
benchmark writes `perf.csv`, and the launcher exits non-zero on failure.

## Two ways to submit

| | Standalone | Through madengine |
|---|---|---|
| Command | `python3 scripts/common/k8s/submit.py --card <card> ...` | `madengine build --use-image ...` then `madengine run ...` |
| Needs | Python 3 and `kubectl` | madengine and the `kubernetes` Python package |
| SLURM counterpart | `sbatch run_xPyD_models.slurm` | `madengine run` with a `slurm` block |

They are separate implementations and they deploy the same thing: for one card and one
configuration file, the Job, Service and ConfigMap they create are identical apart from
resource names and the run id, and the pods run byte-identical stand-ins. See
[Checking that both paths deploy the same thing](#checking-that-both-paths-deploy-the-same-thing).

## What the cluster must provide

| Need | How a cluster provides it | Configuration key |
|---|---|---|
| GPUs in each pod, and only the pod's own | A GPU device plugin (for AMD: `amd.com/gpu`) | `gpu_count` (default: the card's `slurm.gpus_per_node`) |
| One results volume every pod mounts | A ReadWriteMany storage class (NFS and similar). Without one, `ReadWriteOnce` works when every pod lands on one node | `multi_node_results_storage_class`, `results_access_mode` |
| The image on every node | A registry the nodes can pull from, or the image imported into each node | `image_pull_policy` (default `Always`; `IfNotPresent` for imported images) |
| The weights | A volume holding `<dir>/<MODEL_NAME>` (PVC, NFS or host path), with `MODEL_DIR` pointing at its mount | `volumes`, `env_vars.MODEL_DIR` |
| Shared memory | A memory-backed `/dev/shm` | `shm_size` (default `64Gi`) |
| Pod-to-pod networking and DNS | Any CNI; the run's headless Service gives each pod a name | (automatic) |

**Colocated cards** (`vllm_multinode`) move data between nodes only with NCCL, which runs
over the pod network on TCP when there is no RDMA. Set `NCCL_IB_DISABLE=1`,
`NCCL_SOCKET_IFNAME=eth0` and `GLOO_SOCKET_IFNAME=eth0`. The launcher forwards the two interface
names itself; `NCCL_IB_DISABLE` goes through `COLOCATED_FORWARD_ENV`. See the
[worked example](#worked-example-colocated-vllm-on-2-and-4-pods).

**Disaggregated cards** move the KV cache between prefill and decode pods with MoRI,
Mooncake or NIXL over RDMA, and need three more things. Kubernetes provides only the first:

1. **RDMA devices in the pod.** An RDMA device plugin (for example the shared RDMA device
   plugin, the SR-IOV device plugin, or a vendor network operator) advertises a resource
   such as `rdma/hca`. Request it with `extra_resources` and add the `IPC_LOCK` capability
   with `security_context`. Do **not** use `privileged: true`: a privileged pod sees every
   GPU on the node, which defeats the GPU device plugin's isolation.
2. **A network the RDMA traffic can use.** RoCE addresses the NIC by an interface's IP and
   GID, which must exist in the pod's network namespace. Either `host_network: true` (pods
   use the node's NICs; madengine and the standalone path then place one pod per node,
   because the launchers' servers bind fixed ports), or a secondary network per pod
   (Multus with SR-IOV or macvlan), requested with `pod_annotations`.
3. **GPU memory the NIC can register.** A GPU peer-memory driver, or dma-buf support in
   the kernel (`CONFIG_DMABUF_MOVE_NOTIFY`, `CONFIG_PCI_P2PDMA`) together with a KV library
   that uses it (MoRI v1.2.3 and later falls back to dma-buf). This is the node's kernel and
   driver installation, outside Kubernetes.

The fabric settings themselves (rails, GID index, interfaces) are passed as `env_vars`
exactly as on SLURM; see [multinode-running.md](multinode-running.md#fabric).

## The configuration file

Both paths read one JSON file. Its `k8s` block uses madengine's key names, so the same file
drives either path:

```json
{
  "k8s": {
    "kubeconfig": "~/.kube/lab-config",
    "context": "lab",
    "namespace": "mad-runs",
    "gpu_count": 4,
    "cpu": "16",
    "memory": "64Gi",
    "image_pull_policy": "IfNotPresent",
    "volumes": [{"name": "weights", "pvc": "model-weights", "mount_path": "/models", "read_only": true}],
    "multi_node_results_storage_class": "local-path",
    "results_access_mode": "ReadWriteOnce",
    "node_selector": {"node.lab/gpu": "true"},
    "include_script_dirs": ["scripts/common", "scripts/vllm_dissag"]
  },
  "env_vars": {"MODEL_DIR": "/models", "GPUS_PER_NODE": "4"},
  "distributed": {"nnodes": 2}
}
```

| Key | Meaning |
|---|---|
| `kubeconfig`, `context` | The cluster. Every API call and every `kubectl` call uses exactly this file and context; a context that is not in the file is refused, never substituted |
| `namespace` | Where the run's objects go. A same-named object the tool did not create is never deleted; the run stops with the reason instead |
| `gpu_count` | GPUs per pod. Default: the card's `slurm.gpus_per_node` |
| `cpu`, `memory` | Requests per pod. Set them explicitly; see the note below |
| `volumes` | `[{name, mount_path, host_path \| pvc \| nfs: {server, path}, read_only}]` |
| `extra_resources` | Other device-plugin resources, for example `{"rdma/hca": 1}` |
| `security_context` | The container's `securityContext`, for example `{"capabilities": {"add": ["IPC_LOCK"]}}` |
| `host_network` | Pods on the node network; one pod per node |
| `pod_annotations` | Pod annotations, for example `{"k8s.v1.cni.cncf.io/networks": "rdma-net"}` |
| `node_selector`, `tolerations` | Where the pods may run |
| `shm_size` | `/dev/shm` size (default `64Gi`) |
| `image_pull_policy` | Default `Always` |
| `multi_node_results_storage_class`, `single_node_results_storage_class`, `storage_class` | The results volume's storage class |
| `results_access_mode` | Default `ReadWriteMany` for more than one pod, `ReadWriteOnce` for one |
| `include_script_dirs` | Directories shipped with the card's own. Default: its sibling `common`. The colocated launcher also needs `scripts/vllm_dissag` (recipes, benchmarks, parser) |
| `ttl_seconds_after_finished` | Delete the Job this long after it ends |
| `secrets.image_pull_secret_names` | Pull secrets for a private registry (default `["dockerhub-rocm"]`, as in madengine; `[]` for none) |

Outside the `k8s` block:

- `env_vars` override the card's `env_vars`, as on SLURM. Settings for the pod spec belong
  in `k8s`; settings for the launcher (fabric, `MODEL_DIR`, `GPUS_PER_NODE`, TP/PP) belong here.
- `distributed.nnodes` overrides the card's node count. (Through madengine, also set
  `distributed.launcher` to `slurm_multi`: madengine's multi-node preset defaults the
  launcher to `torchrun`, and a run-level `nnodes` is only read for `slurm_multi`.)

**Pod sizing comes from the card, not from presets.** madengine fills in defaults from
presets before it knows the card, and a configuration that names no `gpu_count` and no
`distributed` block is sized as single-GPU (1 GPU, 8 CPUs, 16Gi). For a multinode card those
preset values are ignored: GPUs per pod come from `gpu_count` only if you set it, else the
card, and `cpu` / `memory` are requested only if you set them. The standalone path has no
presets at all. State `cpu` and `memory` in a file both paths share and they request the same.

**The card's launcher assumes 8 GPUs per node unless told.** On pods with fewer GPUs, set
`GPUS_PER_NODE` (and for the SGLang launcher in TP mode, `GENERIC_TP_SIZE`; for the colocated
launcher, `TP_SIZE` / `PP_SIZE`) in `env_vars`. The launchers forward them into the container
when you set them.

## Submitting standalone

From the MAD repository root, with `kubectl` on `PATH`:

```bash
python3 scripts/common/k8s/submit.py --card pyt_vllm_kimi-k3_mi300x_pp2xtp8 \
    --image <registry>/<image>:<tag> --config k8s.json \
    --env MODEL_NAME=Llama-3.1-8B-Instruct --timeout 7200
```

| Option | Meaning |
|---|---|
| `--card` | A card from any `scripts/*/models.json` |
| `--image` | The image every pod runs (default: the card's `DOCKER_IMAGE_NAME`) |
| `--config` | The configuration file above |
| `--env KEY=VALUE` | Override a card env var; repeatable |
| `--nodes N` | Override the node count |
| `--timeout S` | Stop the Job after this many seconds (default: the card's `timeout`) |
| `--output-dir` | Where `perf.csv` and the logs go (default `./k8s_runs/<run>/run-<id>/`) |
| `--dry-run` | Print the manifests and exit |

What it does: deletes leftovers of the same run name that it created itself, creates the
results PVC, the ConfigMap, the Service and the Job, waits on the Job's own `Complete` /
`Failed` condition, then copies pod 0's `perf.csv` and the run's logs out through a
short-lived reader pod. Objects are named `mad-<card>` (shortened with a hash when long) and
labelled `app=mad`.

Exit status: `0` only if the Job completed and `perf.csv` has rows, none of them `FAILURE`.
Otherwise `1`, with the reason on the last line (`[mad-k8s submit] FAIL: ...`).

## Submitting through madengine

```bash
madengine build --tags pyt_vllm_kimi-k3_mi300x_pp2xtp8 --use-image <registry>/<image>:<tag> \
    --additional-context-file k8s.json --manifest-output build_manifest.json
madengine run --manifest-file build_manifest.json --additional-context-file k8s.json \
    --timeout 7200 --live-output -o perf.csv
```

The `k8s` block in the context selects Kubernetes. madengine recognises the card's
`slurm_multi` launcher and deploys it as described above. It writes the run's results into
`./k8s_results/<job>/run-<id>/`, reads exactly the `perf.csv` pod 0 published for that run,
appends its rows to `perf.csv`, and exits non-zero if the Job failed or any row is a
`FAILURE`. Objects are named `madengine-<dir>-<card>` and labelled `app=madengine`.

madengine's own reference for these keys is its
[deployment guide](https://github.com/ROCm/madengine/blob/main/docs/deployment.md).

## Results and what makes a run pass

The launchers coordinate through files under `LOG_PATH` (readiness markers, per-node server
logs, the benchmark's output), so on Kubernetes it must be a volume every pod mounts. It
defaults to `/results/logs` on the run's results volume. Per run, under
`/results/logs/<SLURM_JOB_ID>/`, you find the same files as on SLURM
([multinode-running.md](multinode-running.md#logs)).

When the launcher exits, pod 0 publishes the card's `multiple_results` file if it declares
one, else `$LOG_PATH/$SLURM_JOB_ID/perf.csv`, as `/results/<run>-0/perf.csv`, and writes its
exit code next to it. Both submission paths read that file and nothing else.

A run passes under the same rules as on SLURM
([benchmarks-and-results.md](benchmarks-and-results.md#what-makes-a-run-pass)): the Job
completed (pod 0's launcher exited 0), `perf.csv` has the rows you expect, and every row is
`SUCCESS`.

## Worked example: colocated vLLM on 2 and 4 pods

This runs the colocated launcher on one 8-GPU node split into pods, with NCCL between pods over
the pod network. It needs no RDMA, so it works on any cluster with the GPU device plugin. The
card is a Kimi-K3 colocated card, pointed at Llama-3.1-8B entirely by configuration.

```json
{
  "k8s": {"namespace": "mad-runs", "gpu_count": 4, "cpu": "16", "memory": "64Gi",
          "image_pull_policy": "IfNotPresent",
          "volumes": [{"name": "weights", "pvc": "model-weights", "mount_path": "/models", "read_only": true}],
          "multi_node_results_storage_class": "local-path", "results_access_mode": "ReadWriteOnce",
          "include_script_dirs": ["scripts/common", "scripts/vllm_dissag"]},
  "distributed": {"launcher": "slurm_multi", "nnodes": 2},
  "env_vars": {
    "MODEL_DIR": "/models", "MODEL_NAME": "Llama-3.1-8B-Instruct", "REQUIRE_LOCAL_WEIGHTS": "0",
    "GPUS_PER_NODE": "4", "TP_SIZE": "4", "PP_SIZE": "2",
    "COLOCATED_EXTRA_ARGS": "--max-model-len 8192",
    "BENCHMARK_SCRIPT": "sweep", "BENCHMARK_COMBINATIONS": "1024/1024",
    "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0",
    "COLOCATED_FORWARD_ENV": "NCCL_IB_DISABLE,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME"
  }
}
```

For 4 pods of 2 GPUs: `gpu_count` 2, `nnodes` 4, `GPUS_PER_NODE` 2, `TP_SIZE` 2, `PP_SIZE` 4.

Measured on MI300X (one sweep pass per cell, 1024/1024 tokens), every cell `SUCCESS`:

| Shape | con 8 | con 64 | con 512 |
|---|---|---|---|
| 2 pods x 4 GPUs, PP2 x TP4 | 2,093 tok/s | 15,068 tok/s | 40,754 tok/s |
| 4 pods x 2 GPUs, PP4 x TP2 | 1,443 tok/s | 5,803 tok/s | 18,956 tok/s |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The tool refuses to start: context `X` is not in the kubeconfig | It never falls back to another context. Fix `kubeconfig` / `context` |
| `... exists ... was not created by madengine` (or `by this tool`) | A same-named object belongs to someone else. Use another `namespace` |
| Pods `Pending`: `Insufficient amd.com/gpu` | Not enough free GPUs for `nnodes` x `gpu_count`, or `host_network` needs one node per pod |
| Results PVC `Pending` | No storage class can satisfy the access mode. Set `multi_node_results_storage_class`, or `results_access_mode: ReadWriteOnce` with every pod on one node |
| `Error: required file missing: /workspace/scripts/vllm_dissag/models.yaml` | The launcher reads a sibling directory that was not shipped. Add it to `include_script_dirs` |
| `RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes` | The pod sees GPUs that are not its own, usually because of `privileged: true`. Remove it; use a device plugin resource and `IPC_LOCK` for RDMA |
| Servers start with TP 8 on pods with fewer GPUs | Set `GPUS_PER_NODE` (and `GENERIC_TP_SIZE` / `TP_SIZE`) in `env_vars` |
| `No RDMA devices found` (Mooncake), `RegisterRdmaMemoryRegion failed` (MoRI) | The pod has no RDMA device, or the node cannot register GPU memory for RDMA. See [What the cluster must provide](#what-the-cluster-must-provide) |
| Every benchmark request `Not Found` (404) | The benchmark asked for a model name the server does not serve. The launchers pass `--served-model-name`; check a custom `COLOCATED_EXTRA_ARGS` or recipe does not rename the model differently |
| `[mad-k8s srun] unsupported option ...` / `[mad-k8s docker] run: unsupported option ...` | The launcher uses a form the stand-ins do not implement. The message names it |
| `metadata.annotations: Too long` | Something applied the ConfigMap with `kubectl apply`. Both paths create objects instead; do the same if you apply the dry-run output by hand |

## Checking that both paths deploy the same thing

Render both without a cluster and compare them:

```bash
python3 scripts/common/k8s/submit.py --card <card> --image <image> --config k8s.json --dry-run > standalone.json
```

madengine writes the manifests it would apply when the context sets `"debug": true`. The two
must agree on everything except resource names, the `app` label, the run id (`MAD_K8S_JOB_ID`)
and the bundle's bytes (compare the file list instead). The stand-ins are the same file in both
paths: `scripts/common/k8s/mad_k8s.py` here and `madengine/deployment/k8s_shim/mad_k8s.py` in
madengine.

Offline tests for the standalone path, including the stand-ins run end to end with two agents
on loopback:

```bash
python3 -m unittest discover -s scripts/common/k8s/tests -v
```
