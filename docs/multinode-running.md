# Running multinode inference workloads

This page shows how to run a multinode card, where its logs go, how it fails, and what
to do about it. It covers the three SLURM launchers under
[`scripts/vllm_dissag`](../scripts/vllm_dissag/) (vLLM disaggregated prefill/decode),
[`scripts/sglang_disagg`](../scripts/sglang_disagg/) (SGLang disaggregated) and
[`scripts/vllm_multinode`](../scripts/vllm_multinode/) (vLLM colocated). Read
[multinode-overview.md](multinode-overview.md) first for the concepts.

## Contents

- [Two ways to run a card](#two-ways-to-run-a-card)
- [What a run needs](#what-a-run-needs)
- [Prerequisites](#prerequisites)
- [Model weights](#model-weights)
- [GPU architecture check](#gpu-architecture-check)
- [Fabric](#fabric)
- [Through madengine](#through-madengine)
- [Site settings: a worked example](#site-settings-a-worked-example)
- [With sbatch directly](#with-sbatch-directly)
- [Inside an salloc allocation](#inside-an-salloc-allocation)
- [How the image reaches the nodes](#how-the-image-reaches-the-nodes)
- [Logs](#logs)
- [How a run fails fast](#how-a-run-fails-fast)
- [Known failure modes](#known-failure-modes)
- [Colocated launcher knobs](#colocated-launcher-knobs)
- [Checks that need no GPUs](#checks-that-need-no-gpus)

## Two ways to run a card

Each workload is a **model card** in its directory's `models.json`, backed by one batch
script (the launcher). There are two ways to run a card, and they give the job the same
allocation and the same environment:

- **Through madengine**, which builds and pushes the image, picks healthy nodes,
  submits the launcher, and collects the results.
- **With `sbatch` directly**, exporting the card's environment and submitting its
  launcher yourself.

Both work because the launchers speak only environment variables.
[`scripts/common/cluster.sh`](../scripts/common/cluster.sh) fills in whatever neither
path set, so a card runs the same either way.

## What a run needs

| Piece | Where it lives |
|---|---|
| The card: launcher, node count, image recipe, environment | `scripts/<dir>/models.json` |
| The recipe: serve flags and env per model (vLLM disagg; the colocated launcher reads its `env:` block) | [`scripts/vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml) |
| The image | `docker/<name>.ubuntu.amd.Dockerfile`, named by the card's `dockerfile` |
| The allocation defaults: partition, GPUs per node, exclusive | madengine's SLURM presets (`amd-rccl`, 8, exclusive); override in `--additional-context` |
| The cluster's in-job facts: weights, fabric, ports | [`scripts/common/cluster.sh`](../scripts/common/cluster.sh) |

The environment the launcher starts with is layered, weakest first:

1. `cluster.sh` defaults
2. the recipe in `models.yaml`
3. the card's `env_vars`
4. anything you set yourself: madengine's `env_vars`, or your exported environment
   under `sbatch`

[configuration.md](configuration.md) covers every layer in detail.

## Prerequisites

| Requirement | Detail |
|---|---|
| A SLURM cluster | `xP + yD` nodes for a disaggregated card (minimum 2: `xP=1`, `yD=1`); the card's node count for a colocated card. 8 GPUs per node. |
| Docker on the compute nodes | Not only on the login node. The launchers call `docker run` under `srun`; podman- or apptainer-only clusters will not work without editing the launcher. |
| A way to get the image onto the nodes | A registry the compute nodes can pull from, or `docker save` / `docker load` through shared storage. See [How the image reaches the nodes](#how-the-image-reaches-the-nodes). |
| An RDMA fabric between nodes | The defaults assume 8 NICs per node. See [Fabric](#fabric). |
| Model weights on every allocated node | Local NVMe or shared NFS. See [Model weights](#model-weights). |
| A shared filesystem for logs | `LOG_PATH`, mounted on every node. |
| Enough wall time | The job's time must fit your SLURM *association* limit, which can be lower than the partition's. See below. |
| An account, if your cluster needs one | See [Through madengine](#through-madengine). |

Check your association's wall-time limit before you submit:

```bash
sacctmgr show assoc user=$USER format=Account,QOS,MaxWall
```

The cards declare `24:00:00`. An association capped at `08:00:00` leaves the job
`PENDING (AssocMaxWallDurationPerJobLimit)` forever rather than failing.

For Kimi-K3 on MI300X the hardware is 16 GPUs (2 nodes) colocated or 32 GPUs (4 nodes)
disaggregated, and local NVMe is strongly recommended on every node for the ~1.5 TB
checkpoint.

## Model weights

Every rank loads from one path, `MODEL_PATH`. The launcher finds it before any
container starts, with `cluster_resolve_model_path` in `cluster.sh`.

### How the path is chosen

1. An explicit `MODEL_PATH` skips the probe entirely.
2. Otherwise the launcher walks `MODEL_DIR_CANDIDATES` in order and looks for
   `<candidate>/<MODEL_WEIGHTS_NAME>` on every node. The default list is local NVMe
   first, then shared NFS:

   | Variable | Default |
   |---|---|
   | `NVME_ROOT` | `/mnt/m2m_nobackup` |
   | `SHARED_MOUNT` | `/shared_inference` |
   | `MODEL_DIR_CANDIDATES` | `${NVME_ROOT}/models_blog ${SHARED_MOUNT}/models_blog` |
   | `MODEL_WEIGHTS_NAME` | `MODEL_NAME` |
   | `MODEL_DIR` | empty; when set, it is appended as a final candidate |

3. The first candidate that holds the same weights on every node wins, and
   `MODEL_PATH` and `MODEL_DIR` are set from it.

Order matters. Local NVMe is per node and not uniform across a cluster. Shared NFS is
one export that every node reads at once. A checkpoint too large to sit in page cache
is read at NFS speed on every node simultaneously, and for the big models that read is
the largest part of bring-up.

### What the probe reports

For each candidate, each node prints one of four outcomes, because each needs a
different fix:

| Line | Meaning | Fix |
|---|---|---|
| `- no <dir> on this node (not mounted)` | The node has no such mount at all. | A cluster fix: mount the storage. |
| `- <dir> present, <model> NOT staged` | The mount exists but the model was never copied there. | Stage the weights on that node. |
| `- <path> exists but is EMPTY` | A directory exists with nothing in it. | Re-stage; an empty or partial copy is not usable. |
| `+ Found <path> fp=<hash> real=<target>` | Present. `fp` is an md5 prefix of `config.json`; `real` is where the path resolves. | None. |

A candidate is accepted only when every node reports `Found` **and** the `fp` values
agree. The fingerprint exists because one path can resolve to different models on
different nodes. On one cluster, `Kimi-K3` was a real directory on some nodes and a
symlink to `Kimi-K3-MXFP4` on others. A split allocation would then run two
quantizations in one job and report wrong numbers rather than an error. `config.json`
is the discriminator: it is small, always present in a HuggingFace layout, and carries
`quantization_config`, so variants of one model differ in it. Differing fingerprints
reject that candidate, and the walk falls through to the next one (shared NFS, which is
uniform by construction: slower, but correct).

If nothing matches, the launcher prints every path it tried and stops.

### `REQUIRE_LOCAL_WEIGHTS`

For most models NFS is fine: a 63 GB checkpoint is served from page cache and the
difference does not show. For the very large ones it is not. A checkpoint too big to
cache is read at NFS speed by every node at once, and at that size the read alone can
outlast the partition's time limit. Falling back to NFS then does not slow the run; it
produces a job that cannot finish.

Cards for those models set `REQUIRE_LOCAL_WEIGHTS=1` (every Kimi-K3 card does). The
launcher then drops every candidate under `SHARED_MOUNT`, and a node missing the local
copy, or holding a different variant, is a hard failure that says what to fix instead
of a silent hour-long fallback. An explicit `MODEL_DIR` is appended after that filter,
so it is still considered. The default is 0: this is a property of the model, not of
the site, and the cards that need it declare it.

### `MODEL_WEIGHTS_NAME`: when the directory name differs

`MODEL_NAME` is the key into `models.yaml`. The weights directory is not always named
the same. Set `MODEL_WEIGHTS_NAME` to the directory name when they differ:

- `Kimi-K3-MXFP4-MI355X` is its own recipe but loads the `Kimi-K3-MXFP4` checkpoint,
  so its card sets `MODEL_WEIGHTS_NAME=Kimi-K3-MXFP4`.
- A card can be a configuration alias. Two cards that differ only in their all-to-all
  backend share one checkpoint, so the catalog key and the weights directory diverge.
- If your copy of the weights lives under another name (for example `Kimi-K3`), set
  `MODEL_WEIGHTS_NAME` to that name.

### Checking before you submit

[`scripts/common/preflight_weights.sh`](../scripts/common/preflight_weights.sh) checks
one directory on a list of nodes from the login node, with one `srun` task per node:

```bash
MODEL_DIR=/mnt/m2m_nobackup/models_blog MODEL_NAME=DeepSeek-V3 \
  NODELIST=<node1>,<node2> \
  bash scripts/common/preflight_weights.sh
```

It prints `PRESENT` or `MISSING` per node and fails if any node is missing the weights
or if `srun` fails. `PARTITION` defaults to `amd-rccl`.

## GPU architecture check

A recipe is built for one GPU architecture. The Kimi-K3 MI300X recipes requantise the
MoE to int4 because gfx942 has no scaled-MXFP4 MFMA, and turn off AITER MLA because
that kernel is gfx950-only; the MI355X recipes do the opposite. Run on the other GPU,
neither fails cleanly: they take the allocation, load about 1.5 TB of weights, and die
in kernel codegen or return wrong numbers.

So each launcher checks the allocated nodes before it loads anything
(`cluster_require_gpu_arch` in `cluster.sh`):

- It reads each node's architecture from the kernel's KFD topology (so no ROCm install
  is needed on the host), falling back to `rocminfo`.
- `GPU_ARCHS` lists what the recipe supports. The vLLM disagg launcher reads it from
  the model's `models.yaml` `env:` block; the colocated cards carry it in `env_vars`.
  Empty means unrestricted.
- A mismatch stops the job. A mixed allocation (two architectures) stops the job. An
  architecture that cannot be determined prints a warning and continues.
- `GPU_ARCH_CHECK=0` bypasses the check, for bringing a recipe up on a new GPU.
- `MAD_GPU_ARCH` set explicitly skips the probe.
- The detected architecture labels the results (`PERF_GPU_ARCH`). On the vLLM disagg
  path it also sets `MORI_GPU_ARCHS`, unless you set it, so MoRI builds its kernels
  for the right GPU (`connectors/moriio.env` pins gfx942).

The card's `skip_gpu_arch` says the same thing to madengine before an allocation.
`scripts/common/check_gpu_arch_declarations.py` keeps the two in step.

## Fabric

Network settings (RDMA rails, GID index, socket interface) are not part of any recipe.
They come from `cluster.sh`, which detects the fabric from the adapters present, and
from the connector `.env` files. Export any of them to override.

| `CLUSTER_ARCHETYPE` | Adapter | RDMA devices | `NCCL_IB_GID_INDEX` | `NCCL_SOCKET_IFNAME` |
|---|---|---|---|---|
| `cx7` | Mellanox CX7 / RoCE | `mlx5_*` | 3 | `eth0` |
| `ainic` | AMD AINIC / Pollara | `rdma0..7` | 1 | `eno0` |
| `thor2` | Broadcom Thor2 / RoCE | `bnxt_re0..7` | 3 | `fenic0` |

An unrecognised node gets the `cx7` values. On `ainic` and `thor2`, `NCCL_IB_HCA` and
`MORI_RDMA_DEVICES` default to the archetype's eight rails; on `cx7` each connector
keeps its own choice. `ainic` also sets `RCCL_AINIC_ROCE=1`, without which that path
falls back to verbs or sockets silently.

A wrong fabric setting does not fail loudly: RCCL initialises zero NICs and falls back
to TCP, and the run still completes and reports a number, measured over the wrong
transport. Every launcher prints the rails it will use and warns about any rail that is
absent or not `ACTIVE` (`cluster_report_rdma_rails`). Read that line in the job log.

`USE_CX7_NICS=1` selects the 8 CX7 rail NICs for KV transfer; the default 0 keeps KV on
the management NIC, which is cross-rail safe but a fraction of the bandwidth, and only
shows as a bottleneck once the input sequence is long enough to move real KV. Rail
NICs require the allocated nodes to share a rail.

## Through madengine

Install madengine:

```bash
pip install git+https://github.com/ROCm/madengine.git
```

### Build, then run

From the root of this repository, on a host with docker, build the card's image and
push it where the compute nodes can pull it:

```bash
export MAD_DOCKERHUB_USER=<user> MAD_DOCKERHUB_PASSWORD=<token>
madengine build --tags pyt_vllm_disagg_mori_deepseek-v3 --registry docker.io/<namespace> \
    --additional-context '{"slurm": {}}' --manifest-output build_manifest.json
```

Then, from the SLURM login node, run it:

```bash
madengine run --manifest-file build_manifest.json \
    --additional-context '{"slurm": {"nodes": 2, "time": "06:00:00"}}' \
    --timeout 21600 --live-output -o perf.csv
```

### `--additional-context`

`--additional-context` (inline JSON) or `--additional-context-file` (a JSON file) is
merged over madengine's SLURM presets key by key. **A `slurm` key is what makes
madengine submit to SLURM**; that is why even the build above passes `{"slurm": {}}`.
Put the run's shape and any overrides there:

- `slurm.nodes`, `slurm.time`, `slurm.nodelist`, `slurm.partition`,
  `slurm.gpus_per_node`, `slurm.exclusive`, `slurm.output_dir`, `slurm.results_dir`
- `env_vars` for benchmark knobs or the topology, for example:

  ```json
  {"env_vars": {"xP": "2", "yD": "2", "DURATION": "900"}}
  ```

The allocation defaults come from madengine's presets: partition `amd-rccl`, 8 GPUs per
node, exclusive. Values you set win over the model card; madengine only fills in a
model-card key you did not set. `slurm.nodelist` pins the job to named nodes (madengine
then skips its node-health preflight).

A context file given on `build` is written into the manifest's `deployment_config`
and merged back on `run`, so you need it only on `build`.

### Using a prebuilt image

`--use-image <image>` on `madengine build` skips the build and runs a prebuilt image:

```bash
madengine build --tags pyt_vllm_disagg_mori_kimi-k3 --use-image <registry>/<repo>:<tag> \
    --additional-context '{"slurm": {}}'
```

`slurm_multi` builds have a registry gate. Without `--registry`, `--use-image` or
`--build-on-compute`, madengine either takes the card's `DOCKER_IMAGE_NAME` as an
implicit `--use-image` (also available explicitly as `--use-image auto`) or stops with
a configuration error. The Kimi-K3 cards carry
`DOCKER_IMAGE_NAME: "<supply-your-image>"` as a fill-me-in marker that no node can
pull, so pass `--use-image` or `--registry` for those. `--build-on-compute` builds on a compute node and needs `--registry`.

### Account and QoS

Depending on your madengine version, the `slurm_multi` wrapper may not emit
`#SBATCH --account` or `--qos`. If your cluster requires an account, export them before
`madengine run`; `sbatch` honours these variables and no directive conflicts with them:

```bash
export SBATCH_ACCOUNT=<account> SBATCH_QOS=<qos>
```

### Results

`perf.csv` gets one row per result the benchmark reported; a failed run is a `FAILURE`
row with the reason. The job's own output is under `slurm_results/` (the
`slurm.output_dir`). How madengine finds the workload's CSV is in
[benchmarks-and-results.md](benchmarks-and-results.md#how-madengine-collects-the-csv).

### Agentic cards

The validated agentic configurations are `agentic`-tagged cards. List and run them by
tag:

```bash
madengine run --tags agentic
```

## Site settings: a worked example

A card cannot know everything about your cluster, but most of what a run needs already has a
source: the node count is in the card, the partition, GPUs per node and exclusivity are
madengine's SLURM presets, and the model and log paths are defaults in
`scripts/common/cluster.sh`. Pass a key in `--additional-context` only when your cluster differs
from those. MAD ships no per-site context files, because a file that restates defaults goes stale
when the defaults change.

For the Kimi-K3 cards on a cluster like the one they were written for, the one setting to pass is
the time limit, because the cards ask for 24:00:00:

```bash
IMG=<your-registry>/<repo>:<tag>

madengine build --tags pyt_vllm_disagg_mori_kimi-k3 --use-image "$IMG" \
    --additional-context '{"slurm": {"time": "06:00:00"}}'
madengine run --manifest-file build_manifest.json --live-output
```

On a cluster that differs, add only the keys that differ. For example, a different partition
and a different place for the weights and logs:

```bash
madengine build --tags pyt_vllm_disagg_mori_kimi-k3 --use-image "$IMG" \
    --additional-context '{"slurm": {"partition": "<your-gpu-partition>", "time": "06:00:00"},
                           "env_vars": {"MODEL_DIR": "/path/to/models",
                                        "LOG_PATH": "/path/to/shared/logs"}}'
```

To build and distribute the image instead of supplying one, swap `--use-image "$IMG"`
for `--registry <your-registry>`; the launcher then pulls it onto every node in
parallel.

| Field | What it is | Default comes from | How to find the value |
|---|---|---|---|
| `slurm.partition` | The GPU partition to submit to. | madengine presets: `amd-rccl` | `sinfo -o '%20P %5D %14F %10G %11l'`: pick a partition whose `GRES` column shows GPUs and whose `A/I/O/T` counts show idle nodes. |
| `slurm.nodes` | Nodes to allocate: 2 colocated, 4 disaggregated for Kimi-K3. | The card's `slurm.nodes` | Fixed by the recipe; must match the card's `distributed.nnodes` (for disagg, `xP + yD`). madengine emits this as `#SBATCH --nodes`. |
| `slurm.gpus_per_node` | GPUs per node (8 on MI300X). | madengine presets: 8 | The `GRES` column above, or `scontrol show node <node> \| grep Gres`. |
| `slurm.time` | Wall-clock limit. | The card: 24:00:00 | The `TIMELIMIT` column above is the partition's cap; your association's cap may be lower (`sacctmgr show assoc user=$USER format=Account,QOS,MaxWall`). |
| `slurm.exclusive` | Exclusive node access. | madengine presets: true | Keep `true`: the launchers use every GPU and clean up ports and containers on the node. |
| `slurm.output_dir` | Where the job's own SLURM output goes. | madengine presets: `./slurm_results` | Any writable path; `./slurm_results` is the convention. |
| `slurm.results_dir` | A fallback place madengine's `slurm_multi` collector looks for `perf*.csv`. | Unset; the collector first uses the card's `multiple_results` | The launcher's own directory: `./scripts/vllm_dissag` for disagg, `./scripts/vllm_multinode` for colocated. The launcher copies its CSV to `perf_<MODEL_NAME>.csv` there. Settable here, but not from a model card. |
| `env_vars.MODEL_DIR` | An extra directory holding the checkpoint directory (for example `Kimi-K3/`). | `cluster.sh`: local NVMe `/mnt/m2m_nobackup/models_blog`, then shared `/shared_inference/models_blog` | Wherever the weights live. Must be readable from every node. Appended as the last weight candidate (see [Model weights](#model-weights)). |
| `env_vars.LOG_PATH` | Per-job run logs and the per-job `perf.csv`. | `cluster.sh`: `/shared_inference/$USER/model_blog_logs` | Any shared, writable path mounted on every node. |

Notes:

- `MODEL_DIR` has no default of its own; the default candidates are
  `/mnt/m2m_nobackup/models_blog` then `/shared_inference/models_blog`, and weights are
  read from `<dir>/<MODEL_WEIGHTS_NAME>`. `LOG_PATH` defaults to
  `/shared_inference/$USER/model_blog_logs`. Override them only if your cluster does
  not have those paths.
- Every key under `env_vars` becomes an `export` in the generated SBATCH wrapper, so
  keep it to real variables.
- `nodes` and `gpus_per_node` already ship in the model cards; set them here only to
  override.

## With sbatch directly

The card's `env_vars` are the launcher's contract. Export them, name the image, and
submit the launcher with the cluster's allocation:

```bash
cd scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<registry>/<image>:<tag> \
       MODEL_NAME=DeepSeek-V3 xP=1 yD=1 WIDE_EP=1 RUN_MORI=1 RUN_DEEPEP=0 \
       BENCHMARK_COMBINATIONS=1024/1024
sbatch --partition=amd-rccl --nodes=2 --ntasks=2 --gpus-per-node=8 --exclusive \
       --time=06:00:00 --export=ALL run_xPyD_models.slurm
```

- **Options:** the partition, GPUs per node and `--exclusive` are madengine's SLURM
  presets; add the run's node count and time.
- **Option order:** `sbatch` options, `--export` included, go before the script.
  Anything after the script is passed to it as an argument and silently ignored.
- **`--time`:** each launcher's own `#SBATCH --time=24:00:00` exceeds most partition
  limits, so pass it.
- **Node count:** give `--nodes` the card's `distributed.nnodes` (or the `-N` in its
  `args`). A disaggregated job needs `xP + yD` nodes. If the allocation is smaller, the
  launcher stops at once with `topology needs N nodes ... but the allocation has M`
  instead of building pools from nodes it never got. If it is larger, the launcher uses
  the first `xP + yD` nodes.
- **Image access:** the launcher pulls `DOCKER_IMAGE_NAME` on every node and tolerates a
  failed pull. For a private registry, log in on the nodes first, or the job fails later
  at `docker run`.
- **Where to submit from:** `run_xPyD_models.slurm` finds its own directory (from the
  script path, or `$SLURM_SUBMIT_DIR/scripts/vllm_dissag` when submitted from the repo
  root), then checks its required files are there.
- **The launcher's `#SBATCH` defaults:** `-N 2`, one task per node, `--gres=gpu:8`,
  `--time=24:00:00`, and output to
  `/shared_inference/%u/model_blog_logs/slurm-%j.out` and `.err`. Your command-line
  options override them.

A colocated card, for example the Kimi-K3 PP2 x TP8 baseline:

```bash
cd scripts/vllm_multinode
export DOCKER_IMAGE_NAME=<registry>/<image>:<tag> MODEL_NAME=Kimi-K3 \
       TP_SIZE=8 PP_SIZE=2 ENABLE_EP=0 AITER_SITUV2_A8W4=1 GPU_ARCHS=gfx942 \
       REQUIRE_LOCAL_WEIGHTS=1 BENCHMARK_SCRIPT=niah NIAH_WORDS=10000,50000,100000,200000
export COLOCATED_EXTRA_ARGS="--reasoning-parser kimi_k3 --mm-encoder-tp-mode data --safetensors-load-strategy prefetch --max-model-len 1000000 --max-num-seqs 8 --max-num-batched-tokens 2048 --quantization-config '{\"moe\":{\"weight\":\"int4_per_group_32\"}}'"
sbatch --partition=<partition> --nodes=2 --time=06:00:00 --export=ALL run_multinode.slurm
```

These values are the card's `env_vars` from
[`scripts/vllm_multinode/models.json`](../scripts/vllm_multinode/models.json).

### Previewing the commands

`DRY_RUN=1` prints each node's server command without starting anything, and needs no
GPUs:

```bash
cd scripts/vllm_dissag
CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori DRY_RUN=1 NODE_RANK=0 \
  MODEL_NAME=DeepSeek-V3 MODEL_PATH=/models/DeepSeek-V3 NIXL_COOKBOOK_PATH=$PWD \
  xP=2 yD=2 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 bash vllm_disagg.sh
```

[`scripts/vllm_dissag/tests/argv_assert.sh`](../scripts/vllm_dissag/tests/argv_assert.sh)
has complete examples. [vllm-disagg.md](vllm-disagg.md#dry-run) explains the output.

## Inside an salloc allocation

You can hold an allocation and run into it repeatedly.

With madengine: inside `salloc`, `madengine run` detects `SLURM_JOB_ID` and runs the
wrapper synchronously with `bash` instead of nesting another `sbatch`. The node count
comes from the allocation.

```bash
salloc -N 4 --ntasks-per-node=1 --gres=gpu:8 -p <partition> -t 24:00:00
madengine run --manifest-file build_manifest.json --live-output
```

Without madengine: run the launcher with `bash`, as
[`scripts/vllm_dissag/salloc_launch.sh`](../scripts/vllm_dissag/salloc_launch.sh) shows:

```bash
salloc -N 5 --ntasks-per-node=1 --nodelist=<nodes> --gres=gpu:8 -p <partition> -t 12:00:00
cd scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<image>
export xP=2 yD=2 MODEL_NAME=DeepSeek-V3
bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
```

For live testing of single cells on an existing allocation, `tests/drive_cell.sh`
attaches to a job with `srun --overlap` and starts one container per node through
`tests/run_interactive.sh`; `tests/harvest.sh <RUN_TAG>` prints readiness, RDMA
failures, worker deaths and per-cell benchmark summaries. See
[vllm-disagg.md](vllm-disagg.md#testing).

## How the image reaches the nodes

- **madengine with a registry.** `madengine build --registry <registry>` pushes the
  image. At run time madengine runs a parallel `srun docker pull` on all nodes.
- **Every launcher pulls too.** Each node runs `docker pull $DOCKER_IMAGE_NAME` before
  `docker run`, and tolerates a failed pull. A local-only tag must therefore already be
  on every node or be in a registry the nodes can reach. For a private registry, log in
  on the nodes first.
- **No registry.** `--build-on-compute` requires `--registry`, so on a registry-less
  cluster neither madengine nor MAD can get an image onto the nodes. Build once,
  `docker save` to shared storage, then `docker load` on each node. That works and
  needs no registry.

Build instructions for each image are in [vllm-disagg.md](vllm-disagg.md#building-the-docker-image),
[sglang-disagg.md](sglang-disagg.md) and [kimi-k3.md](kimi-k3.md).

## Logs

Each job writes one directory, `$LOG_PATH/<SLURM job id>/`. Inside the containers it is
`/run_logs/<job id>/`, because `LOG_PATH` is mounted at `/run_logs`. `LOG_PATH`
defaults to `/shared_inference/$USER/model_blog_logs`, from `cluster.sh`. The directory
is shared, so every node's logs land together.

### vLLM disaggregated

| File | Written by | Content |
|---|---|---|
| `pd_vllm_bench_NODE<rank>.log` | every node | The container's whole output: the launcher's steps, `DRY_RUN` output, failure summaries, and on rank 0 the benchmark's progress |
| `prefill_NODE<rank>.log` | prefill nodes | The prefill `vllm serve` server |
| `decode_NODE<rank>.log` | decode nodes | The decode `vllm serve` server |
| `vllm_router_NODE0.log` | rank 0, `PROXY_TYPE=vllm_router` | The router |
| `proxy_NODE<rank>.log` | rank 0 toy proxy; any node on failure | The toy proxy, and the `ERROR:` line of any node that failed the job |
| `benchmark_<job>_<time>_xP<n>_yD<n>_<model>_CONCURRENCY.log` and `.csv` | rank 0, sweep | Throughput sweep output and its per-cell CSV |
| `..._STALLS.log`, `..._SHAPEWARMUP.log` | rank 0, sweep | Timed-out cells; per-shape warmup output |
| `benchmark_long_context_<...>_CONCURRENCY.log` | rank 0, long context | Long-context benchmark |
| `niah_<job>_<time>_xP<n>_yD<n>_<model>.log` | rank 0, NIAH | NIAH retrieval results |
| `agentic_<job>_xP<n>_yD<n>_<model>.json`, `suite_summary.json`, `aiperf_artifacts/`, `agentic_models_shim_NODE0.log` | rank 0, agentic | Agentic replay results |
| `perf.csv` | rank 0 | The madengine results CSV |
| `ABORTED` | any failing node | Why the job was abandoned; see below |

The SLURM job's own output, with the launcher's node selection, weight probe and GPU
check, goes to the `sbatch` output file (under madengine, to `slurm.output_dir`).

### vLLM colocated

| File | Content |
|---|---|
| `colocated_bench_NODE<rank>.log` | The container's whole output |
| `colocated_NODE<rank>.log` | The `vllm serve` process on that node |
| benchmark logs, `perf.csv` | As above; the colocated launcher uses the same benchmark scripts, with `xP1_yD0` in file names |
| `.shutdown` | Sentinel the head writes when it is done, so workers stop |

### SGLang disaggregated

| File | Content |
|---|---|
| `pd_sglang_bench_serving.sh_NODE<N>.log` | Main per-node log |
| `prefill_NODE<N>.log` | Prefill server |
| `decode_NODE<N>.log` | Decode server |
| `proxy_NODE0.log` | The `sglang_router` |
| `benchmark_*_CONCURRENCY.log` / `.csv` | Benchmark results |

## How a run fails fast

The launchers fail fast, on every node, and say why.

### A server that dies during start-up

Rank 0 waits for `Application startup complete.` in the prefill and decode master logs
(on `rixl` + TP, in every node's server log, then checks each server's port), for up to
`LOG_WAIT_TIMEOUT_SECONDS` (default 4000). While it waits, it watches each log for
lines vLLM prints only when a server has died:

| Pattern | Why it is fatal |
|---|---|
| `Engine core initialization failed`, `EngineCore failed to start` | The engine exited. |
| `RuntimeError: NCCL error`, `[FATAL ERROR]: HIP failure` | A collective failed. The engine can hang rather than exit, so the engine lines never appear; two runs failed their first all-reduce about 17 minutes in and then waited out the 4000 s. |
| `torch.OutOfMemoryError` | A worker ran out of GPU memory during start-up. One decode worker's OOM in cudagraph capture left the engine waiting about 55 minutes for vLLM's own engine-ready timeout. |

On a match, or on timeout, the launcher prints, in order:

1. **The first error lines** of the log, each distinct message once (pid and timestamp
   ignored), up to 40. vLLM's `Failed to import Triton kernels` warning is dropped,
   because healthy runs print it too and its repeats used to hide the real error. The
   first lines come first because the last lines of a dead vLLM server are the API
   server's traceback, which only says "See root cause above".
2. **The last 80 lines** of the log.
3. **A GPU memory snapshot** of that node: used and total VRAM per GPU, the host pids
   holding GPU memory (from KFD), and `rocm-smi --showpids` when available. This
   answers failures like "free memory on startup is less than desired" on a few GPUs.

### Every other node: the abort file

The failing node writes one line to `/run_logs/<job>/ABORTED` (`$LOG_PATH/<job>/` on
the host). Every barrier (`socket_barrier.py --abort-file`) and every start-up wait
watches that file, so the other nodes stop at their next barrier instead of waiting out
the time limit. Before this, a node that gave up exited alone and the nodes waiting on
it held the allocation for hours.

A node that gives up also stops everything it started: SIGTERM to its own process tree,
then SIGKILL to whatever is left after 15 seconds, because vLLM workers wedged in a
failed HIP or RCCL call ignore SIGTERM. Otherwise a still-running server holds the
container's output pipe and keeps the SLURM job alive.

### A cancelled job

`scancel` sends SIGTERM to the task, but the container belongs to the docker daemon and
its PID 1 ignores SIGTERM. So each node's task traps the signal and stops its container.
The colocated launcher also tears its containers down on exit. A cancelled job does not
leave GPU memory held on the nodes.

### A run with no results

If the benchmark produced no `perf.csv`, the two vLLM launchers exit 1. A crash then reads as a
failure rather than as a clean run with "0 successful, 0 failed".

### Colocated specifics

- The head checks that `vllm serve` is still alive while it waits for readiness. A dead
  engine prints the last 80 log lines and exits at once, instead of polling a log whose
  writer already died.
- The head writes `/run_logs/<job>/.shutdown` however it exits. Workers poll for it and
  stop their headless servers. Without it, workers waited on their servers until the
  wall clock after the results were already written.

A recipe is validated on specific shapes and GPUs, and its `models.yaml` entry says
which. Run a card at one of those shapes before suspecting the recipe.

## Known failure modes

| Symptom | Cause | What to do |
|---|---|---|
| Job stays `PENDING (AssocMaxWallDurationPerJobLimit)` | The requested time exceeds your association's limit. | Lower `slurm.time` / `--time` below the `MaxWall` from `sacctmgr`. |
| `Error: topology needs N nodes (xP=.. + yD=..) but the allocation has M` | The allocation is smaller than `xP + yD`. | Size `slurm.nodes`, `sbatch -N` or `salloc -N` to `xP + yD`. |
| `x FATAL: model '<name>' is not usable on ALL allocated nodes` | Weights missing, empty, or different on some node; with `REQUIRE_LOCAL_WEIGHTS=1`, NFS was not considered. | Read the per-node lines above it. Stage the weights, set `MODEL_WEIGHTS_NAME`, or pin `MODEL_PATH`. |
| `x <model> supports only: <arch>. These nodes are <arch>.` | The recipe is for another GPU. | Use a partition with that GPU, or the model's recipe for this GPU. `GPU_ARCH_CHECK=0` only for bring-up. |
| `Error: DOCKER_IMAGE_NAME is not set` | No image named. | Export `DOCKER_IMAGE_NAME`, or pass `--use-image` / `--registry` to madengine. |
| Fails at `docker run` on some nodes | A failed pull was tolerated and the image is not on that node. | Log in to the registry on the nodes, or `docker load` the image there. |
| `OCI runtime create failed: ... not a directory`, then the surviving node loops on `Waiting for nodes. . .` | Docker creates a *directory* at a missing bind-mount source. The first run on a node lacking an RDMA library left a directory behind, and `[ -e ]` matched it, so every later run mounted a directory over a file. Per node and self-propagating, so it looks intermittent. | Fixed: the launchers now test with `[ -f ]`. On a poisoned node, remove the empty directory the old run left under `/usr/lib/x86_64-linux-gnu/`. |
| `OSError: libibverbs.so.1: version IBVERBS_PRIVATE_57 not found (required by libmlx5.so.1)`, about ten seconds into start-up | Only the versioned host `libibverbs` was mounted, so the image copy was still loaded while the host `libmlx5` needed the host one. | Fixed: the launchers mount the SONAME files and the versioned files. |
| `RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes.` | A previous job's container still holds GPU memory. | Fixed going forward (cancelled jobs stop their containers). On an affected node, stop the leftover container. |
| `LLVM ERROR: Do not know how to expand this operator's operand!` in `determine_available_memory`, with `quantization_config=None` | gfx942 cannot codegen the a16w4 SiTUv2 heuristic kernel; the MoE must be requantised to int4. | All Kimi-K3 gfx942 cards set `AITER_SITUV2_A8W4=1` and `--quantization-config '{"moe":{"weight":"int4_per_group_32"}}'`. See [kimi-k3.md](kimi-k3.md). |
| `AssertionError: run() must precede build_extension()` building `amd_mori` | Unpinned build toolchain (setuptools 80 or later). | `PIP_CONSTRAINT` in the Dockerfile. See [kimi-k3.md](kimi-k3.md). |
| Engine init fails with a `module_gemm_common` import error, or `device_gemm ... does not support this GEMM problem` | A MoE model on an image whose AITER is not co-versioned for its MoE GEMM path. | Use the purpose-built MoRI-EP image for MoE serving. Dense models serve on the standard image. |
| Every request returns 500 after the first disagg write; MoRI `RegisterRdmaMemoryRegion` fails with EFAULT (errno 14) | ROCm 7.2.3 cannot dmabuf-export HIP-VMM memory, and `expandable_segments` was on. | `connectors/<connector>.env` sets `expandable_segments:False` on both allocators and forwards it at container start. Do not override it. |
| `Error: vllm-router not found` | The image has no router on `PATH`. | Set `ROUTER_BINARY=<path>` to a built `vllm-router` on a shared mount, or use the toy proxy (`PROXY_TYPE`). |
| NIAH: every request 404 "The model ... does not exist" | The client asked for a name the server does not serve. | The launcher now resolves `SERVED_MODEL_NAME` from the recipe. If you set `NIAH_MODEL`, it must equal the served name. See [benchmarks-and-results.md](benchmarks-and-results.md#niah-long-context-retrieval). |
| NIAH scores low only on a reasoning model, with `finish=length` / `TRUNCATED` | `max_tokens` ran out mid-trace. | Raise `NIAH_MAXTOK`. |
| Throughput far below expectation with no error | Wrong rails or GID: NCCL fell back to TCP. | Check the `RDMA rails:` line and its warnings in the job log. See [Fabric](#fabric). |
| Job reports COMPLETED with `0 successful, 0 failed` and no `perf.csv` | Older launchers returned 0 after a crash. | Fixed: the launchers exit 1 when no perf CSV is produced. Read the node logs. |
| `unbound variable` naming a variable assigned just above, or `docker run` "requires at least 1 argument" | An apostrophe inside a single-quoted `srun bash -c '...'` body, or a comment inside a backslash continuation, silently truncated the script. `bash -n` accepts both. | Run `scripts/common/check_srun_quotes.py` and `scripts/common/check_continuations.py` after editing a launcher. |
| SGLang: `TransferEncodingError: 400, message: Not enough data to satisfy transfer length header` at concurrency 512 and above on large models | A known SGLang issue; requests are dropped and throughput falls. | See [sglang-disagg.md](sglang-disagg.md). |

## Colocated launcher knobs

[`scripts/vllm_multinode/run_multinode.slurm`](../scripts/vllm_multinode/run_multinode.slurm)
and its per-node entry point
[`serve_colocated.sh`](../scripts/vllm_multinode/serve_colocated.sh). madengine passes
these through `env_vars`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | required | Key into `scripts/vllm_dissag/models.yaml`; only its `env:` block is applied. Also the served model name (`--served-model-name`). |
| `DOCKER_IMAGE_NAME` | required | Image to run. |
| `TP_SIZE` | GPUs per node | Tensor parallel within a node. |
| `PP_SIZE` | node count | Pipeline parallel across nodes. `TP_SIZE * PP_SIZE` must equal the total GPUs. |
| `ENABLE_EP` | `0` | `1` adds `--enable-expert-parallel`. |
| `ALL2ALL_BACKEND` | empty | With `ENABLE_EP=1`, passed as `--all2all-backend` (for example `allgather_reducescatter`, `mori_low_latency`). |
| `COLOCATED_EXTRA_ARGS` | empty | Extra serve flags for this variant (reasoning parser, max model length, quantization config). Word-split, so JSON values must carry their own quotes. |
| `BENCHMARK_SCRIPT` | `sweep` | `sweep`, `long_context` or `niah`. Agentic is not available on this launcher. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Unless the recipe or you set it. |
| `KV_CACHE_DTYPE`, `KV_BLOCK_SIZE`, `KV_CACHE_MEMORY_BYTES` | unset | Passed as `--kv-cache-dtype`, `--block-size`, `--kv-cache-memory-bytes` when set (usually by the recipe). |
| `SERVE_PORT` | `8000` | Head's API port; the benchmark targets it. |
| `MASTER_PORT` | `29500` | Distributed rendezvous port. |
| `DISTRIBUTED_TIMEOUT_SECONDS` | `7200` | `--distributed-timeout-seconds` (device and NCCL groups). |
| `CPU_DISTRIBUTED_TIMEOUT_SECONDS` | `7200` | `--cpu-distributed-timeout-seconds`, the gloo CPU group, where the start-up barrier waits. Unset, both fall back to PyTorch's 1800 s. |
| `LOG_WAIT_TIMEOUT_SECONDS` | `4000` | How long the head waits for readiness. |
| `CONTAINER_BARRIER_PORT` | `2223` | Container-creation barrier port. |
| `COLOCATED_FORWARD_ENV` | empty | Comma-separated names of extra variables to forward into the containers (timeouts, pre-warm), without editing the launcher. |
| `PREWARM_CHECKPOINT` | `0` | `1` reads the whole checkpoint into page cache on every node before serving. Off because it made things worse: every node reads the whole checkpoint while PP2 x TP8 loads only its own shard, about 4 times the necessary I/O against one NFS export. Bounded by `PREWARM_TIMEOUT_SECONDS` (default 900); `PREWARM_JOBS` (default 4) sets parallel readers. Useful only as a storage probe: a large spread between nodes means storage is the bottleneck. |
| `JIT_CACHE_PERSIST` / `JIT_CACHE_HOST` | `1` / `/mnt/m2m_nobackup/$USER/vllm_jit_cache/<image id>` | Persistent AITER/Triton kernel cache per image. |

The colocated cards also have a twin, `pyt_vllm_kimi-k3_mi300x_pp2xtp8_way4`, that sets
the same environment through `MAD_CONFIG: mad-config.kimi-k3.yaml`; it exists to be
compared with the original. See [configuration.md](configuration.md).

## Checks that need no GPUs

Run these after any change to a launcher, a connector or a recipe:

```bash
bash scripts/vllm_dissag/tests/run_all.sh             # gate_check + argv_assert; expect ALL OFFLINE SUITES PASSED
bash scripts/vllm_dissag/tests/argv_assert.sh         # the serve argv per connector and mode
bash scripts/vllm_dissag/tests/parse_to_csv_assert.sh # perf.csv rows and FAILURE status from a sweep log
python3 scripts/common/check_srun_quotes.py           # no apostrophe truncates an srun body
python3 scripts/common/check_continuations.py         # no comment inside a backslash continuation
python3 scripts/common/check_gpu_arch_declarations.py # cards and recipes agree on GPUs
```

Run them from the repository root. The two shell-hygiene checkers take file paths, or
scan every `scripts/**/*.slurm` and `scripts/**/*.sh` when given none. What each vLLM
test covers is in [vllm-disagg.md](vllm-disagg.md#testing).
