# Configuration

A MAD multinode workload reads its settings from several files. Each file answers a different
question, is read by a different program, and is read at a different time. This page names
every layer, says what belongs in it, and says which one wins when two layers set the same
thing.

Read this page when you want to change a setting and do not know where it goes. Putting a
value in the wrong layer is the most common configuration mistake: the value looks applied,
but a stronger layer overrides it, or it never reaches the process that needs it.

For the concepts used here (launcher, prefill and decode, connector, wideEP) see
[multinode-overview.md](multinode-overview.md). For the commands that run a card see
[multinode-running.md](multinode-running.md).

## Terms

| Term | Meaning |
|---|---|
| Model card | One entry in a `models.json` file. It names the Dockerfile, the launcher script, the node count and the environment. madengine selects cards by `--tags`. |
| Launcher | The batch script a card names in `scripts`, for example `scripts/vllm_dissag/run_xPyD_models.slurm`. It runs on the first node of the allocation and starts one container per node with `srun` and `docker run`. |
| Recipe | The per-model serving settings in `scripts/vllm_dissag/models.yaml` or `scripts/sglang_disagg/models.yaml`: serve flags per role and mode, and (vLLM only) per-model environment. |
| Allocation | What SLURM gives the job: partition, node count, GPUs per node, exclusivity, time limit. |
| Container environment | The environment variables the serving processes see inside each container. |
| Submit time | Before or while the job is submitted: on the login node (madengine, `sbatch`) and in the batch script before `docker run`. |
| In the job | Inside the container, after `docker run`. |

## The layers at a glance

| Layer | File | Holds | Read by | When |
|---|---|---|---|---|
| madengine SLURM presets | `madengine/deployment/presets/slurm/defaults.json`, `profiles/*.json` | Allocation defaults | madengine | Before submit |
| Additional context | `--additional-context`, `--additional-context-file` | Per-site and per-run overrides of anything madengine handles | madengine | Before submit |
| Model card | `scripts/<dir>/models.json`, root `models.json` | Image, launcher, node count, topology env | madengine; you, under plain `sbatch` | Before submit |
| Layered config | `mad-config.yaml` (or a file named by `MAD_CONFIG`) | A model's env and its benchmark env, by layer | madengine | Before submit |
| Site facts | `scripts/common/cluster.sh` | Weight roots, fabric, ports, timeouts, partition default for hand runs | The launcher | In the batch script, before `docker run` |
| Connector env files | `scripts/vllm_dissag/connectors/*.env` | Platform env each connector must have at container start | `scripts/vllm_dissag/run_xPyD_models.slurm` | Before `docker run`, forwarded with `-e` |
| Recipe | `scripts/vllm_dissag/models.yaml`, `scripts/sglang_disagg/models.yaml` | Serve flags per role and mode, per-model env | The in-container launcher | In the job |
| Connector and launcher defaults | `connectors/moriio.sh`, `connectors/rixl.sh`, `scripts/sglang_disagg/mori_ep_env.sh` | `${VAR:-default}` fallbacks | The in-container launcher | In the job |
| Image | `docker/*.Dockerfile` | Build-time pins, cache locations | `docker build` | At build |

Everything that reaches a serving process ends up as an environment variable or a serve flag.
The environment is the only thing that crosses both the container boundary and the choice of
launcher (madengine or plain `sbatch`), which is why most layers are expressed as environment
variables.

## 1. The model card

A model card is one JSON object in a `models.json` file. The multinode cards live in
`scripts/vllm_dissag/models.json`, `scripts/sglang_disagg/models.json` and
`scripts/vllm_multinode/models.json`. Single-node cards live in the root
[`models.json`](../models.json) and in other `scripts/*/models.json` files.

Example, from [`scripts/vllm_dissag/models.json`](../scripts/vllm_dissag/models.json):

```json
{
  "name": "pyt_vllm_disagg_mori_kimi-k3",
  "dockerfile": "../../docker/vllm_kimi_k3",
  "scripts": "run_xPyD_models.slurm",
  "multiple_results": "perf_Kimi-K3.csv",
  "tags": ["pyt", "vllm", "vllm_disagg", "mori_ep", "inference"],
  "timeout": -1,
  "skip_gpu_arch": "gfx950",
  "distributed": { "launcher": "slurm_multi", "nnodes": 4 },
  "slurm": { "nodes": 4, "gpus_per_node": 8, "time": "24:00:00" },
  "env_vars": {
    "DOCKER_IMAGE_NAME": "<supply-your-image>",
    "MODEL_NAME": "Kimi-K3",
    "xP": "2", "yD": "2", "EP_TP_SIZE": "2",
    "RUN_MORI": "1", "WIDE_EP": "1",
    "BENCHMARK_SCRIPT": "niah",
    "NIAH_WORDS": "10000,50000,100000,200000",
    "REQUIRE_LOCAL_WEIGHTS": "1"
  },
  "args": ""
}
```

### Fields that matter for configuration

| Field | What it holds | Who reads it |
|---|---|---|
| `dockerfile` | The Dockerfile prefix. `../../docker/vllm_kimi_k3` means `docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`. | madengine build |
| `scripts` | The launcher, relative to the card's directory. | madengine; you under `sbatch` |
| `env_vars` | The launcher's contract: model name, topology, benchmark selection, model properties such as `REQUIRE_LOCAL_WEIGHTS`. | madengine exports them into the generated batch script; under plain `sbatch` you export them yourself |
| `slurm` | Allocation for this card: `nodes`, `gpus_per_node`, `time`. madengine copies `partition`, `nodes`, `gpus_per_node`, `time`, `exclusive`, `reservation`, `output_dir` and `nodelist` from it, but only for keys you did not set in `--additional-context`. | madengine |
| `distributed` | `launcher` (`slurm_multi` for every multinode card here) and `nnodes`. `slurm_multi` means the card's own script manages its per-node containers. | madengine |
| `args` | Arguments. madengine passes them to the launcher script, not to `sbatch`. See the note below. | madengine; you |
| `skip_gpu_arch` | GPU architectures the card must not run on. madengine enforces it before submission. | madengine |
| `docker_build_arg` | Build arguments for this card's image. The additional context overrides the card for the same key. | madengine build |
| `timeout` | madengine's per-model timeout in seconds. `0` or `-1` means no timeout. | madengine |
| `multiple_results` | The CSV name the launcher publishes for madengine. | madengine |

`DOCKER_IMAGE_NAME: "<supply-your-image>"` is a fill-me-in marker. madengine replaces it with
the image it built and pushed (or the one you pass with `--use-image`) and refuses the marker
itself. Under plain `sbatch` you export a real image name.

**The `args` trap.** `args` means different things to the two ways of running a card. madengine
passes it to the launcher as `bash <script> <args>`. When you submit the card script with plain
`sbatch`, the same text is what you would give `sbatch`. The SGLang cards carry
`"args": "-N 2 -n 2"`, which madengine warns about. The node count for a distributed card comes
from `distributed.nnodes` and `slurm.nodes`; keep those two equal.

**Node count.** madengine sizes the allocation from `slurm.nodes` (default 1). It emits it as
`#SBATCH --nodes`. `distributed.nnodes` carries the same number for launcher detection but does
not size the allocation.

**Override.** Anything in `env_vars` is overridden by `env_vars` in `--additional-context`
(madengine) or by your own export (plain `sbatch`). The card's `slurm` keys are overridden by
the same keys in `--additional-context`.

## 2. The recipe (`models.yaml`)

A recipe holds how one model is served: the serve flags for each role (prefill, decode) in each
parallelism mode, and, for vLLM, the model's environment. It is keyed by `MODEL_NAME`. The
launcher reads it inside the container, so a change to `models.yaml` needs no image rebuild.

There are two recipe files:

- [`scripts/vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml): vLLM disaggregated
  (and, for `env:`, the vLLM colocated launcher, which reads the same file).
- [`scripts/sglang_disagg/models.yaml`](../scripts/sglang_disagg/models.yaml): SGLang
  disaggregated.

### Schema

| Key | Applies to | Meaning |
|---|---|---|
| `base_flags` | Both roles, both modes | Always applied. |
| `tp_flags` / `dp_flags` | Both roles, one mode | Mode-level flags. Omit when empty. |
| `prefill.tp`, `prefill.dp`, `decode.tp`, `decode.dp` | One role, one mode | Role and mode specific flags. |
| `experimental_flags` | Both roles, both modes | Optional extra flags appended last. Use for try-outs independent of the mode. |
| `env:` | vLLM only | Per-model environment. See below. |

The mode is `PARALLEL_MODE`, which each launcher derives from one switch:

| Launcher | Switch | `tp` mode | `dp` mode |
|---|---|---|---|
| vLLM disagg (`vllm_disagg.sh`) | `WIDE_EP` | `WIDE_EP=0` | `WIDE_EP=1` (wide expert parallel) |
| SGLang disagg (`sglang_disagg_mori_io_ep.sh`) | `DP_MODE` | `DP_MODE=0` (default) | `DP_MODE=1` (MoRI EP with DP attention) |

The composed flag string for a role is:

```
base_flags + <mode>_flags + <role>.<mode> + experimental_flags
```

vLLM composes it in [`vllm_disagg.sh`](../scripts/vllm_dissag/vllm_disagg.sh) into
`MODEL_CONFIG_PREFILL` and `MODEL_CONFIG_DECODE`. SGLang composes it into
`PREFILL_MODEL_CONFIG` and `DECODE_MODEL_CONFIG`, then appends
`--disaggregation-transfer-backend <mori|mooncake>`.

### What does not go in a recipe

The launcher and the connector own the parallelism degree and the transport. Do not put these
in `models.yaml`:

- vLLM: `--data-parallel-size`, `--enable-expert-parallel`, `--all2all-backend`,
  `--kv-transfer-config`, `--port`, and the transfer backend. The connector emits them.
- SGLang: `--tp-size` and `--disaggregation-transfer-backend`. The launcher manages them.

One exception: vLLM `tp:` blocks do carry `--tensor-parallel-size`, because the rixl+TP path
takes the degree from the recipe. The moriio+TP path emits its own `--tensor-parallel-size` and
strips the duplicate from the recipe string.

Fabric settings (RDMA rail list, GID index, socket interface) do not belong in a recipe either.
They are facts about the cluster, not the model. A recipe that pinned them would break on a
different fabric. They come from `cluster.sh`, the connector env file, and the connector
script.

### YAML anchors

Several models share one recipe through YAML anchors. An anchor (`&name`) labels a value; an
alias (`*name`) reuses it. For example the DeepSeek family:

```yaml
_deepseek_recipe_env: &deepseek_recipe_env
  VLLM_USE_V1: "1"
  KV_BLOCK_SIZE: "16"
  KV_CACHE_MEMORY_BYTES: "20000000000"
  # ...

DeepSeek-V3:
  env: *deepseek_recipe_env
  prefill:
    dp: ""
  decode:
    dp: ""
```

Keys that start with `_` (such as `_deepseek_recipe_env`, `_kimi_k3_recipe_env`,
`_kimi_k3_dp_flags`) exist only to hold anchors. They are not model names.

A flag string can also be a folded block scalar (`>-`), which YAML joins into one line:

```yaml
_kimi_k3_dp_flags: &kimi_k3_dp_flags >-
  --reasoning-parser kimi_k3
  --max-model-len 1000000
  --quantization-config '{"moe":{"weight":"int4_per_group_32"}}'
```

Flag strings are split with shell-style quoting (Python `shlex`). A value that contains JSON must
carry its own quotes, as above. Without them, quote removal strips the JSON's double quotes and
vLLM receives invalid JSON. vLLM tokenizes the strings without `bash eval` for the same reason.

### How the vLLM launcher applies `env:`

[`vllm_disagg.sh`](../scripts/vllm_dissag/vllm_disagg.sh) does this, in order:

1. Reads `EP_TP_SIZE` from the recipe early, unless it is already set, so the topology math can
   size the per-node DP ranks.
2. Exports every key of the model's `env:` block **that is not already in the environment**.
   A name that is already set (by `docker run -e`, or by the image) keeps its value.
3. Applies the `GPU_MEMORY_UTILIZATION` fallback if neither the environment nor the recipe set it:
   0.70 when the decode EP width (`yD` times `GPUS_PER_NODE`) is 32 or more, else 0.8. The 0.70
   leaves room for larger MoRI dispatch and combine buffers at high EP width. Models that need a
   topology-dependent value should leave the key out of the recipe.
4. Composes the role flag strings for the active mode.
5. Sets `SERVED_MODEL_NAME`, then sources `parallelism.sh` and the connector script. The
   connector's own `${VAR:-default}` fallbacks now yield to everything above.

The batch script, `run_xPyD_models.slurm`, also reads two recipe keys on the batch node, before
any container starts, because it needs them to size and gate the job: `EP_TP_SIZE` and
`GPU_ARCHS`. An exported value of either wins over the recipe.

The colocated launcher ([`scripts/vllm_multinode/serve_colocated.sh`](../scripts/vllm_multinode/serve_colocated.sh))
applies the same `env:` block with the same rule, from the same `scripts/vllm_dissag/models.yaml`.
It does not use the recipe's flag strings; colocated cards pass serve flags in
`COLOCATED_EXTRA_ARGS`.

The `env:` block is the single home for a model's serving recipe: `KV_BLOCK_SIZE`,
`KV_CACHE_DTYPE`, `PREFILL_CUDAGRAPH_MODE`, `DECODE_CUDAGRAPH_MODE`, `*_MORI_BACKEND`,
`KV_CACHE_MEMORY_BYTES`, AITER switches and so on. The Docker images ship none of these, so the
same image serves any model on any cluster.

### The served model name

A recipe's `--served-model-name` sets the name clients must request. The vLLM launcher reads it
from the composed flags and exports it as `SERVED_MODEL_NAME`. If the flags do not set one, it is
`MODEL_PATH`, which is vLLM's default. The NIAH benchmark sends requests to
`SERVED_MODEL_NAME`. For example, the `Kimi-K3-MXFP4` recipe serves `kimi-k3`; requests for the
weights path would get a 404.

The launcher reads the name from the recipe rather than from the router's `/v1/models`, because
that endpoint returns 503 under MoRIIO service discovery.

### SGLang recipes

The SGLang recipe has no `env:` block. SGLang's environment comes from
[`mori_ep_env.sh`](../scripts/sglang_disagg/mori_ep_env.sh) inside the container. The SGLang
`base_flags` carry `--disable-radix-cache` for clean throughput sweeps; the launcher strips it
when `BENCHMARK_SCRIPT=agentic` or `ENABLE_RADIX_CACHE=1`. See
[sglang-disagg.md](sglang-disagg.md#recipes-modelsyaml) for every SGLang recipe.

### Per-model ENV layering (vLLM disaggregated)

This is the layering the vLLM README and ARCHITECTURE file describe, lowest to highest:

1. Connector default: `export VAR=${VAR:-default}` in `connectors/<connector>.sh`.
2. `models.yaml` `env:`, exported before the connector's `setup_env`.
3. Submit-time `-e VAR=...`: a site or per-run override. It wins.

So a model can override a connector default (for example a `MORI_*` knob) without editing any
script, and a per-run or per-cluster value still wins.

Step 3 is broader than it looks. Everything the batch script forwards with `docker run -e` is
already in the container when the recipe is read, so it beats the recipe. That includes:

- every `KEY=VALUE` in `connectors/<connector>.env` (with your export of the same name winning
  over the file);
- the `cluster.sh` defaults the batch script forwards, for example
  `DISTRIBUTED_TIMEOUT_SECONDS`, `LOG_WAIT_TIMEOUT_SECONDS`, `GPUS_PER_NODE`,
  `NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`;
- names the batch script always forwards with its own default, for example
  `MORI_SOCKET_IFNAME` (`eth0`), `VLLM_RPC_TIMEOUT` and `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`.

A recipe value for one of those names does not take effect on the `run_xPyD_models.slurm` path.
To change one, export it at submit time.

The batch script forwards a fixed list of names, not the whole environment. If you export a
variable that is not on the list, it does not reach the container. Check the `docker run`
block in the batch script for the list.

## 3. Connector env files (`connectors/*.env`)

[`scripts/vllm_dissag/connectors/moriio.env`](../scripts/vllm_dissag/connectors/moriio.env) and
[`rixl.env`](../scripts/vllm_dissag/connectors/rixl.env) hold environment that every run of one
connector needs, for every model and both modes.

They exist because these values must reach the container at process start (PID 1). PyTorch
reads its allocator configuration at import, so a later `export` inside the launcher is too late.

**How they are read.** After it resolves `CONNECTOR`, `run_xPyD_models.slurm` reads
`connectors/${CONNECTOR}.env`, skips comments and blank lines, and turns each line into
`-e KEY=${KEY:-VALUE}` on the `docker run` command. A submit-time export of the same name wins
over the file.

**Why they exist.** ROCm 7.2.3 cannot dmabuf-export HIP virtual-memory allocations. With
`expandable_segments` on, the KV cache is VMM-backed and MoRI RDMA registration fails with
`EFAULT` (errno 14) on the first disaggregated write, so every request returns 500. With NIXL,
UCX or DeepEP the same registration fails and transfers fail.

Both files set:

| Variable | Value | Why |
|---|---|---|
| `PYTORCH_ALLOC_CONF` | `expandable_segments:False` | See above; must be off on both allocators |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:False` | Same |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `0` | Legacy HSA IPC off, so the dmabuf path is used |
| `MORI_GPU_ARCHS` | `gfx942` | Arch pin. The batch script replaces it with the detected arch when you did not set it, because on MI355X the pinned value builds MoRI kernels for the wrong GPU |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | |

`moriio.env` adds MoRI and RDMA fabric tuning. These are cluster-specific (validated on one
MI300X RoCEv2 fabric) and are kept out of the image so a different cluster needs no rebuild:

| Variable | Value |
|---|---|
| `MORI_RDMA_TC` | `41` |
| `MORI_RDMA_SL` | `0` |
| `MORI_IO_SL` | `1` |
| `MORI_IB_ENABLE_RELAXED_ORDERING` | `1` |
| `MORI_IB_GID_INDEX` | `1` |
| `MORI_NUM_QP_PER_PE` | `8` |
| `VLLM_MORIIO_QP_PER_TRANSFER` | `2` |
| `VLLM_MORIIO_NUM_WORKERS` | `4` |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` |
| `HSA_ENABLE_SDMA` | `1` |

**Override.** Export the same name before `sbatch`, or put it in `env_vars` under madengine.

The interactive test driver `tests/drive_cell.sh` does not source these files. That is why the
`Kimi-K3-MXFP4` recipe repeats some of them in its `env:` block.

## 4. Connector and launcher defaults

Inside the container, the connector scripts fill every name nobody else set, with
`${VAR:-default}`. They are the weakest layer. Examples from
[`connectors/moriio.sh`](../scripts/vllm_dissag/connectors/moriio.sh):

| Variable | Default |
|---|---|
| `MORI_SERVE_PORT` (vLLM server) | `20005` |
| `MORI_PROXY_PORT` (toy proxy) | `10001` |
| `ROUTER_PORT` | `30000` |
| `PROXY_TYPE` | `vllm_router` |
| `PREFILL_MORI_BACKEND` | `VLLM_ALL2ALL_BACKEND`, else `mori_high_throughput` |
| `DECODE_MORI_BACKEND` | `mori_low_latency` |
| `VLLM_ROCM_USE_AITER_MLA` | `1` |
| `NCCL_IB_HCA`, `MORI_RDMA_DEVICES` | `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9` |
| `NCCL_IB_GID_INDEX`, `MORI_IB_GID_INDEX` | `3` |
| `MORI_NUM_QP_PER_PE`, `VLLM_MORIIO_QP_PER_TRANSFER` | `4` |
| `MORI_SHMEM_HEAP_SIZE` | `17179869184` (16 GiB) |
| `KV_BLOCK_SIZE` (wideEP) | `1` |
| `KV_CACHE_DTYPE` (wideEP) | `fp8` |
| `GPU_MEMORY_UTILIZATION` | `0.8` |
| `DISTRIBUTED_TIMEOUT_SECONDS` | `7200` |
| `CUDAGRAPH_CAPTURE_SIZES` | `1 2 4 8 16 32 64 128 256` |

The SGLang equivalent is [`mori_ep_env.sh`](../scripts/sglang_disagg/mori_ep_env.sh); its
variables are listed in [sglang-disagg.md](sglang-disagg.md#rdma-and-nccl-environment).

## 5. Site facts (`scripts/common/cluster.sh`)

[`cluster.sh`](../scripts/common/cluster.sh) is the one generic site configuration for the
multinode launchers. It is sourced by:

- `scripts/sglang_disagg/run_xPyD_models.slurm`
- `scripts/vllm_dissag/run_xPyD_models.slurm`
- `scripts/vllm_multinode/run_multinode.slurm`

It runs on the batch node, inside the job, before any container starts. Every value is
`${VAR:-default}`, so the rule is always: **the environment beats these defaults**. That one
property lets the same card run identically under madengine and under plain `sbatch`: both only
set environment variables, and `cluster.sh` fills whatever neither set.

**What belongs here:** facts about the cluster. Filesystem roots, fabric device names, port
numbers, timeouts.

**What does not:** model performance flags (those are the recipe's), and what must be known
before the job is submitted: the partition to ask for, GPUs per node, exclusivity. Those are the
submitter's, from madengine's SLURM presets or your `sbatch` options.

There is no longer a `scripts/common/clusters/` profile file. The allocation defaults it used to
hold are madengine's SLURM presets (section 6).

### Weights

| Variable | Default | Meaning |
|---|---|---|
| `NVME_ROOT` | `/mnt/m2m_nobackup` | Per-node local NVMe root. Not uniform across nodes. |
| `SHARED_MOUNT` | `/shared_inference` | One shared NFS export every node reads. |
| `MODEL_DIR_CANDIDATES` | `${NVME_ROOT}/models_blog ${SHARED_MOUNT}/models_blog` | Roots probed in order: local NVMe first, shared NFS second. |
| `MODEL_NAME` | `None` | The catalog key. |
| `MODEL_WEIGHTS_NAME` | `${MODEL_NAME}` | The directory name the weights live in, when it differs from `MODEL_NAME`. |
| `MODEL_DIR` | empty | When set, appended as a final candidate (it does not replace the list). |
| `MODEL_PATH` | empty | When set, skips the probe entirely. |
| `REQUIRE_LOCAL_WEIGHTS` | `0` | `1` drops every candidate under `SHARED_MOUNT`. |

Why the order matters: a checkpoint too large for page cache is read at NFS speed by every node
at once, which is the largest part of bring-up for big models.

`cluster_resolve_model_path` walks the candidates. For each one it checks
`<root>/<MODEL_WEIGHTS_NAME>` on every allocated node, and reports one of four outcomes per node:
root not mounted, root present but model not staged, directory empty, or found. A found
directory is fingerprinted by the md5 of its `config.json`, because one path can resolve to
different weights on different nodes (a real directory on some, a symlink to another variant on
others). A candidate is accepted only if it exists, is non-empty, and has the same fingerprint on
every node. Differing fingerprints reject that candidate and the walk moves on to shared NFS,
which is uniform by construction.

`MODEL_WEIGHTS_NAME` exists because a card can be a configuration alias. Two cards that differ
only in their all-to-all backend share one checkpoint, so the catalog key and the directory
diverge. The same checkpoint may be staged as `Kimi-K3` on some nodes and `Kimi-K3-MXFP4` on
others; set `MODEL_WEIGHTS_NAME` to pick the directory name.

`REQUIRE_LOCAL_WEIGHTS=1` is a property of the model, not the site, so cards set it. For a very
large checkpoint the NFS read alone can outlast the partition's time limit, so falling back to
NFS would produce a job that cannot finish. With the flag set, a node missing local weights, or
holding a different variant, is a hard failure with a message saying what to fix.

### Paths, SLURM, topology

| Variable | Default | Meaning |
|---|---|---|
| `LOG_PATH` | `${SHARED_MOUNT}/${USER}/model_blog_logs` | Per-job logs, mounted at `/run_logs` in the container |
| `SBATCH_PARTITION` | `amd-rccl` | Only a default for running a launcher by hand outside `sbatch` |
| `SLURM_JOB_PARTITION`, `SLURM_JOB_ACCOUNT`, `SLURM_JOB_QOS`, `SLURM_CLUSTER_NAME`, `SLURM_CONF` | site values | Mirror what the scheduler tells a running job, for hand runs where the `SLURM_*` names are unset |
| `SLURM_SUBMIT_DIR` | `$PWD` | The invoking user's working directory |
| `GPUS_PER_NODE` | `8` | |
| `NNODES` | `${SLURM_NNODES:-1}` | |
| `xP`, `yD` | `1`, `1` | Prefill and decode node counts |
| `TP_SIZE` | `${GPUS_PER_NODE}` | For the colocated launcher. The vLLM wideEP path uses `EP_TP_SIZE` instead, so this default does not turn a TP1/DP16 recipe into TP8/DP2 |
| `PP_SIZE` | `1` | |

### Fabric

The fabric family, `CLUSTER_ARCHETYPE`, is detected from the RDMA adapters present
(`/sys/class/infiniband`, else `ibv_devices`). Set it to override detection, or set any single
variable below; an explicit value always wins over the archetype.

| Archetype | Adapter | Devices | `NCCL_IB_GID_INDEX` | `NCCL_SOCKET_IFNAME` | `RDMAV_DRIVERS` | `KV_IB_DEVICE` |
|---|---|---|---|---|---|---|
| `cx7` | Mellanox CX7 / RoCE | `mlx5_*` | `3` | `eth0` | `mlx5` | `mlx5_1` |
| `ainic` | AMD AINIC / Pollara | `rdma0..7` | `1` | `eno0` | `ionic` | `rdma0` |
| `thor2` | Broadcom Thor2 / RoCE | `bnxt_re0..7` | `3` | `fenic0` | `bnxt_re` | `bnxt_re0` |
| `unknown` | anything else | | same as `cx7` | | | |

Why detect rather than hardcode: getting the GID index, driver or interface wrong does not fail
loudly. RCCL initializes zero NICs, falls back to TCP, and the run reports a number measured over
the wrong transport.

Other fabric variables:

| Variable | Default | Meaning |
|---|---|---|
| `GLOO_SOCKET_IFNAME` | `${NCCL_SOCKET_IFNAME}` | |
| `IBV_DRIVERS` | archetype driver | |
| `RCCL_AINIC_ROCE` | `1` on `ainic` | Without it the AINIC path falls back to verbs or sockets silently |
| `NCCL_IB_HCA`, `MORI_RDMA_DEVICES` | `rdma0..rdma7` on `ainic`, `bnxt_re0..7` on `thor2`, unset on `cx7` | The RDMA rails. On `cx7` each connector keeps its own choice (`rixl.sh` probes `ibstat`, which a preset `NCCL_IB_HCA` would switch off) |
| `USE_CX7_NICS` | `0` | `1` selects the 8 CX7 rail NICs for KV transfer; `0` keeps KV on the management NIC, which is cross-rail safe but a fraction of the bandwidth. Rail NICs require nodes on one rail. Meaningful on `cx7` only |
| `FABRIC_SUBNET_PREFIX` | `10.158.` | Preferred subnet when the SGLang launcher picks each node's IP from `hostname -I`. If no address matches, it uses the first address |

`cluster_report_rdma_rails` prints the rails a job will use and warns about any that is absent or
not `ACTIVE`. It changes nothing; it makes a wrong rail list visible in the log.

### Ports, timeouts, container

| Variable | Default |
|---|---|
| `SERVE_PORT` | `8000` |
| `MASTER_PORT` | `29500` |
| `PROXY_PORT` | `10001` |
| `ROUTER_PORT` | `30000` |
| `LOG_WAIT_TIMEOUT_SECONDS` | `4000` |
| `DISTRIBUTED_TIMEOUT_SECONDS` | `7200` |
| `CPU_DISTRIBUTED_TIMEOUT_SECONDS` | `7200` |
| `DOCKER_SHM_SIZE` | `256G` |

### GPU architecture check

A recipe is written for one GPU architecture. The MI300X Kimi-K3 recipes requantize the MoE to
int4 and turn off AITER MLA; the MI355X recipe does the opposite. On the wrong GPU neither fails
cleanly: they load about 1.5 TB of weights and then die in kernel codegen or return wrong numbers.

madengine enforces the card's `skip_gpu_arch`, but on SLURM it runs on a login node with no GPU.
So the launcher checks for itself, on the nodes it was given, identically under madengine and
plain `sbatch`:

- `cluster_detect_gpu_arch` reads each node's `gfx` name from the KFD topology
  (`gfx_target_version`, for example `90402` is `gfx942`, `90500` is `gfx950`), falling back to
  `rocminfo`. It sets `MAD_GPU_ARCH`. An explicit `MAD_GPU_ARCH` skips the probe. A mixed
  allocation is an error.
- `cluster_require_gpu_arch <model> <GPU_ARCHS>` fails if the recipe declares architectures
  (`GPU_ARCHS` in its `env:` block) and the allocation is none of them. It runs before the weights
  probe, so the wrong GPU costs seconds. If the arch cannot be determined it warns and continues.
  `GPU_ARCH_CHECK=0` bypasses the check, for bringing up a recipe on a new GPU.
- It also sets `PERF_GPU_ARCH` to the detected arch, so results are labelled with the GPU the run
  actually used.

[`scripts/common/check_gpu_arch_declarations.py`](../scripts/common/check_gpu_arch_declarations.py)
keeps each card's `skip_gpu_arch` consistent with its recipe's `GPU_ARCHS`.

## 6. madengine's SLURM presets

madengine fills the allocation from presets before it applies anything you pass. They live in the
madengine repository at `src/madengine/deployment/presets/slurm/`.

`defaults.json`:

| Key | Value |
|---|---|
| `slurm.partition` | `amd-rccl` |
| `slurm.nodes` | `1` |
| `slurm.gpus_per_node` | `8` |
| `slurm.time` | `24:00:00` |
| `slurm.output_dir` | `./slurm_results` |
| `slurm.exclusive` | `true` |
| `slurm.modules` | `[]` |
| `distributed.backend`, `distributed.port` | `nccl`, `29500` |
| `env_vars` | `OMP_NUM_THREADS=8`, `MIOPEN_FIND_MODE=1`, `MIOPEN_USER_DB_PATH=/tmp/.miopen` |

Then one profile, chosen by node count after a trial merge with your settings:

| Profile | When | `slurm` | `env_vars` |
|---|---|---|---|
| `profiles/single-node.json` | `nodes` is 1 | `nodes: 1`, `gpus_per_node: 8`, `time: 12:00:00` | `NCCL_DEBUG=WARN` |
| `profiles/multi-node.json` | `nodes` > 1 | `nodes: 2`, `gpus_per_node: 8`, `time: 24:00:00` | `NCCL_DEBUG=WARN`, `NCCL_DEBUG_SUBSYS=INIT,NET`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=eth0`, `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=600`, `HSA_ENABLE_SDMA=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, `RCCL_ENABLE_HIPGRAPH=0` |

**Preset `env_vars` do not reach `slurm_multi` cards.** madengine records which `env_vars` keys
came only from a preset and drops them for self-managed launchers. These cards run their own
script, which owns its environment, and the same card under plain `sbatch` never receives the
preset values. Passing them anyway made the madengine run a different experiment: the
multi-node profile's `HSA_ENABLE_SDMA=0` beat the connector's `HSA_ENABLE_SDMA=1`, its
`NCCL_SOCKET_IFNAME=eth0` would preempt fabric detection, and `NCCL_IB_DISABLE=1` would push NCCL
onto TCP. If you set one of those names yourself, you get your value.

The allocation presets (partition, GPUs per node, exclusive, time) do apply.

## 7. `--additional-context` and `--additional-context-file`

The additional context is a JSON object you pass to `madengine build` and `madengine run`. It
overrides madengine's defaults and the card. Pass it inline, from a file, or both:

```bash
madengine run --tags pyt_vllm_disagg_mori_kimi-k3 \
  --additional-context '{"slurm": {"time": "06:00:00"}}'
```

**Deep merge.** Layers merge key by key. A nested object in a stronger layer overrides only the
keys it names; lists and scalars are replaced. So one key in the additional context changes one
value and keeps the rest of the presets and the card:

```bash
madengine run --tags pyt_vllm_disagg_mori_kimi-k3 \
    --additional-context '{"slurm": {"time": "06:00:00"}}'
# -> slurm: partition amd-rccl, gpus_per_node 8, exclusive (presets), nodes 4 (the card),
#    time 06:00:00 (the additional context)
```

Pass only what differs from the presets, the card and `cluster.sh`. MAD ships no per-site
context files: a file that restates defaults goes stale when the defaults change. When both are
given, the inline string wins over `--additional-context-file`.

**A `slurm` key makes madengine target SLURM.** The deployment type is inferred from the presence
of the `slurm` field. Even `{"slurm": {}}` is enough; the presets fill the rest.

On the build-then-run path, the context given to `madengine build` is written into the
manifest's `deployment_config` and merged back on `madengine run`.

### Keys used by the multinode cards

| Key | Meaning |
|---|---|
| `slurm.partition` | Partition. Preset `amd-rccl`. |
| `slurm.nodes` | Node count. Sizes the allocation. Wins over the card's `slurm.nodes`. |
| `slurm.gpus_per_node` | Preset `8`. |
| `slurm.time` | Wall time `HH:MM:SS`. Preset `24:00:00`. |
| `slurm.exclusive` | Preset `true`. |
| `slurm.nodelist` | Run only on these nodes. Skips node health preflight. |
| `slurm.exclude` | Nodes to avoid. |
| `slurm.reservation`, `slurm.constraint`, `slurm.account`, `slurm.qos` | Passed to SLURM. The `slurm_multi` header may omit account and QOS on older madengine; export `SBATCH_ACCOUNT` / `SBATCH_QOS` instead. |
| `slurm.output_dir` | Where SLURM `.out` / `.err` go. Preset `./slurm_results`. |
| `slurm.results_dir` | Where the `slurm_multi` collector globs `perf*.csv`. Set it to the launcher's directory (`./scripts/vllm_dissag` or `./scripts/vllm_multinode`) so the declared `multiple_results` file is found. Not settable from a card. |
| `slurm.gpu_arch` | The compute nodes' GPU arch, for enforcing `skip_gpu_arch`. When unset, madengine probes one node of the partition with `srun ... rocminfo`. |
| `slurm.skip_gpus_directive` | Omit `#SBATCH --gpus-per-node` on clusters with no GPU GRES. |
| `env_vars` | Exported into the generated batch script. Wins over the card's `env_vars`. Use for topology and benchmark knobs, and site paths such as `MODEL_DIR`, `LOG_PATH`. |
| `docker_env_vars` | Also applied on the `slurm_multi` path, below the card's `env_vars`. |
| `docker_build_arg` | Build arguments. Wins over the card's `docker_build_arg` for the same key. |
| `MAD_CONTAINER_IMAGE` | Run an existing image and skip the build. |

Every key under `env_vars` becomes an `export` in the generated batch wrapper, so keep it to real
variables.

Pass only what differs from the presets, the card and `cluster.sh`. For a 4-node disaggregated card
on a cluster that has a different partition and different model and log paths:

```bash
--additional-context '{"slurm": {"partition": "<your-partition>", "time": "06:00:00"},
                       "env_vars": {"MODEL_DIR": "<directory holding the weights>",
                                    "LOG_PATH": "<shared writable path>"}}'
```

Where each of these values comes from when you do not pass it is in the
[site settings table](multinode-running.md#site-settings-a-worked-example).

## 8. `MAD_SYSTEM_GPU_ARCHITECTURE` for images built per GPU

Some Dockerfiles compile for exactly one GPU and have no default. The Kimi-K3 image,
[`docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`](../docker/vllm_kimi_k3.ubuntu.amd.Dockerfile),
declares `ARG MAD_SYSTEM_GPU_ARCHITECTURE` and refuses anything but `gfx942` or `gfx950`. MoRI's
JIT target, the vLLM compile, and (with `WITH_NIXL=1`) rocSHMEM and DeepEP follow it. An image
built for the wrong arch fails at run time, so the build refuses to guess.

Where the value comes from:

- A full local `madengine run` (build and run on one GPU node) auto-detects the local GPU and
  injects it into `docker_build_arg`, unless you set it.
- A `madengine build` on a host without the target GPU (for example a SLURM login node) cannot
  detect it. madengine warns when a Dockerfile requires the argument and none is resolved. Pass
  it yourself:

```bash
madengine build --tags pyt_vllm_disagg_mori_kimi-k3 --registry <your-registry> \
  --additional-context '{"slurm": {}, "docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
```

- By hand:

```bash
docker build -f docker/vllm_kimi_k3.ubuntu.amd.Dockerfile \
  --build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx942 -t <registry>/vllm-kimi-k3:gfx942 .
```

Runtime differences between the arches (AITER MLA on gfx950 only, the int4 MoE requant on gfx942)
are recipe knobs in `models.yaml`, not build steps.

## 9. `mad-config.yaml`: a model's env and its measurement, by layer

`mad-config.yaml` is a madengine format. It holds the two things that travel with a model: how it
is served (`model:`) and what is measured (`benchmark:`). madengine resolves it into exactly the
`env_vars` a card already carries; no launcher needs to know it exists. The card suffix `way4`
refers to it (it was once the fourth of several numbered configuration methods).

Where it is found: a sibling `mad-config.yaml` in the card's script directory, or an explicit path
in the card's `env_vars.MAD_CONFIG`. No file means nothing changes. It is only available on the
madengine path; plain `sbatch` does not read it.

A file named `mad-config.yaml` is picked up by every card in that directory. A file meant for one
card must use another name and be named by `MAD_CONFIG`. That is why the Kimi-K3 file is
[`scripts/vllm_multinode/mad-config.kimi-k3.yaml`](../scripts/vllm_multinode/mad-config.kimi-k3.yaml).

That file carries exactly the values the `pyt_vllm_kimi-k3_mi300x_pp2xtp8` card carries in
`env_vars`, split by who owns them. The twin card `pyt_vllm_kimi-k3_mi300x_pp2xtp8_way4` points at
it; a run of each must produce the same environment and the same `perf_Kimi-K3.csv`.

```yaml
version: 1
model:
  id: moonshotai/Kimi-K3
  local_name: Kimi-K3          # the on-disk directory name that cluster.sh resolves
  env:
    REQUIRE_LOCAL_WEIGHTS: '1'
    MODEL_NAME: Kimi-K3
    TP_SIZE: '8'
    PP_SIZE: '2'
    ENABLE_EP: '0'
    AITER_SITUV2_A8W4: '1'
    COLOCATED_EXTRA_ARGS: >-
      --reasoning-parser kimi_k3 --mm-encoder-tp-mode data
      --safetensors-load-strategy prefetch --max-model-len 1000000
      --max-num-seqs 8 --max-num-batched-tokens 2048
      --quantization-config '{"moe":{"weight":"int4_per_group_32"}}'
benchmark:
  - kind: niah
    env:
      BENCHMARK_SCRIPT: niah
      NIAH_WORDS: '10000,50000,100000,200000'
```

The `_way4` card itself only carries `GPU_ARCHS`, `DOCKER_IMAGE_NAME`, `MAD_CONFIG` and
`BENCHMARK_SCRIPT`.

A `benchmark:` entry without `kind` applies to every benchmark. A `model.serve` section can hold
serve flags as `base`, `modes` and `roles`; within it a flag string is parsed into a map, so a
later layer can override a single flag (`--tp 1` in `base` and `--tp 8` in `modes.tp` gives
`--tp 8` and keeps the rest). Multi-value flags such as `--cuda-graph-bs 8 16 32` stay whole.

A `site:` block is refused with an error. Cluster facts belong in `cluster.sh` or in a per-run
override, not in a model file.

Precedence, lowest first:

```
model                                   from mad-config.yaml
benchmark  (task-level, then by kind)   from mad-config.yaml
model_info.env_vars                     the card's own env_vars
additional_context.env_vars             --additional-context (highest)
```

Layers merge; a key set only in `model` survives a `benchmark` section that does not mention it.
The madengine document is `docs/distributed-config.md` in the madengine repository.

## Precedence

### (a) The allocation

Weakest first. The strongest layer that sets a key wins.

Through madengine:

1. `#SBATCH` lines in the launcher script. Inert under madengine: it generates its own wrapper.
2. madengine `presets/slurm/defaults.json`.
3. The profile: `single-node.json` or `multi-node.json`, by node count.
4. The card's `slurm` block (`partition`, `nodes`, `gpus_per_node`, `time`, `exclusive`,
   `reservation`, `output_dir`, `nodelist`), for keys you did not set yourself.
5. `--additional-context-file`.
6. `--additional-context`.

Under plain `sbatch`:

1. `#SBATCH` lines in the launcher script (for example `--time=24:00:00`, `-N 2`, `--gres=gpu:8`).
2. Options on the `sbatch` command line, which override the script's directives.

The `#SBATCH --time=24:00:00` in each launcher exceeds many partition limits, so pass `--time`.
The partition, `--gpus-per-node=8` and `--exclusive` that madengine would use are its presets;
pass them yourself. `cluster.sh`'s `SBATCH_PARTITION` is only for running a launcher by hand
outside `sbatch`.

### (b) The container environment (vLLM disaggregated)

Weakest first:

1. Image `ENV`. The Kimi-K3 and shared disagg images ship no recipe or platform env, only cache
   locations such as `AITER_JIT_DIR`.
2. Connector script defaults: `${VAR:-default}` in `connectors/<connector>.sh`, and the launcher's
   `GPU_MEMORY_UTILIZATION` fallback.
3. The recipe's `env:` block in `models.yaml`. It sets only names not already set.
4. What the batch script forwards with `docker run -e`. For each forwarded name the value is
   whatever the batch script's environment holds, so within this layer, weakest first:
   1. `connectors/<connector>.env` values (used only when the environment has no value);
   2. `cluster.sh` defaults (filled only when the name is unset);
   3. the submit-time environment, which is, weakest first:
      - `mad-config.yaml` `model:`, then `benchmark:` (madengine only);
      - `docker_env_vars` from the additional context (madengine only);
      - the card's `env_vars`;
      - `env_vars` from `--additional-context` (madengine), or your own `export` (plain `sbatch`).

   Only names on the batch script's forward list reach the container. madengine preset
   `env_vars` are not passed to `slurm_multi` cards.

The short form in [`scripts/common/README.md`](../scripts/common/README.md) is: `cluster.sh`
defaults, then the recipe, then the card's `env_vars`, then anything you set yourself. The long
form above adds that a name `cluster.sh` sets and the batch script forwards is already present
when the recipe is read.

The colocated launcher follows the same order for the recipe's `env:`. SGLang has no recipe
`env:`; its container environment is the forwarded names over the `${VAR:-default}` values in
`mori_ep_env.sh`.

## Where do I change X

| I want to change | Where | Notes |
|---|---|---|
| A model's serve flags | `models.yaml`: `base_flags`, `<mode>_flags`, `<role>.<mode>`, `experimental_flags` | vLLM: `scripts/vllm_dissag/models.yaml`; SGLang: `scripts/sglang_disagg/models.yaml`; vLLM colocated: `COLOCATED_EXTRA_ARGS` in the card |
| The name clients request | `--served-model-name` in the vLLM recipe flags | Resolved as `SERVED_MODEL_NAME` |
| KV cache size | `KV_CACHE_MEMORY_BYTES` in the recipe `env:` | Also `GPU_MEMORY_UTILIZATION`; SGLang: `--mem-fraction-static` in the flags |
| KV block size, KV dtype | `KV_BLOCK_SIZE`, `KV_CACHE_DTYPE` in the recipe `env:` | |
| Cudagraph mode | `PREFILL_CUDAGRAPH_MODE`, `DECODE_CUDAGRAPH_MODE`, `VLLM_CUDAGRAPH_MODE`, `CUDAGRAPH_CAPTURE_SIZES` in the recipe `env:` | SGLang: `--cuda-graph-bs` / `--disable-cuda-graph` in the flags |
| AITER knobs | `VLLM_ROCM_USE_AITER*`, `AITER_*` in the recipe `env:` | SGLang: `SGLANG_USE_AITER` in `mori_ep_env.sh` |
| MoRI all-to-all backend per role | `PREFILL_MORI_BACKEND`, `DECODE_MORI_BACKEND`, `VLLM_ALL2ALL_BACKEND` in the recipe `env:` | |
| TP inside each EP pool | `EP_TP_SIZE` in the recipe `env:` or the card | Must divide `GPUS_PER_NODE`; needs equal pools when above 1 |
| Node count | `slurm.nodes` in `--additional-context`, and `xP` / `yD` in `env_vars` | Keep the card's `slurm.nodes` and `distributed.nnodes` equal; plain `sbatch`: `--nodes` |
| Time limit | `slurm.time` in `--additional-context` | Plain `sbatch`: `--time` |
| Partition | `slurm.partition` in `--additional-context` | Plain `sbatch`: `--partition` |
| NIC / RDMA devices | Export `NCCL_IB_HCA`, `MORI_RDMA_DEVICES`, `NCCL_IB_GID_INDEX`, `MORI_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, or `CLUSTER_ARCHETYPE` | SGLang: `USE_CX7_NICS` (its batch script does not forward `IB_DEVICES` or `NCCL_IB_HCA`). Never in a recipe |
| MoRI fabric tuning | `connectors/moriio.env`, or export the same name | |
| Model weights location | `MODEL_DIR` (added candidate), `MODEL_DIR_CANDIDATES`, `NVME_ROOT`, `SHARED_MOUNT`, or `MODEL_PATH` (skips the probe) | Directory name: `MODEL_WEIGHTS_NAME`; NVMe only: `REQUIRE_LOCAL_WEIGHTS=1` |
| Log location | `LOG_PATH` | |
| Benchmark knobs | Card or `--additional-context` `env_vars`: `BENCHMARK_SCRIPT`, `BENCHMARK_COMBINATIONS`, `BENCHMARK_CON`, `BENCHMARK_ITR`, `NIAH_WORDS`, `DURATION`, `AGENTIC_*` | Or `benchmark:` in `mad-config.yaml` |
| Image | `--use-image` on `madengine build`, or `MAD_CONTAINER_IMAGE`; plain `sbatch`: `DOCKER_IMAGE_NAME` | The card's `dockerfile` for what madengine builds |
| Image base or build pins | `docker_build_arg` in the card or the additional context | e.g. `BASE_DOCKER`, `WITH_NIXL` |
| GPU arch for a build | `docker_build_arg.MAD_SYSTEM_GPU_ARCHITECTURE` | `--build-arg` by hand |
| GPU arch check | `GPU_ARCHS` in the recipe; card `skip_gpu_arch`; `GPU_ARCH_CHECK=0` to bypass | |
| One-off override for a single run | `--additional-context '{"env_vars": {...}, "slurm": {...}}'`; plain `sbatch`: `export` before `sbatch` | Only forwarded names reach the container |
| Ports | `ROUTER_PORT`, `MORI_SERVE_PORT`, `MORI_PROXY_PORT`, `SERVE_PORT`, `MASTER_PORT` | |
| Timeouts | `DISTRIBUTED_TIMEOUT_SECONDS`, `LOG_WAIT_TIMEOUT_SECONDS`: export them (they are always forwarded, so a recipe value does not apply). `VLLM_ENGINE_READY_TIMEOUT_S`: recipe `env:` or export | |
| madengine's own timeout | card `timeout`, or `madengine run --timeout` | `0` means none |

## See also

- [multinode-overview.md](multinode-overview.md): concepts, launchers, connectors, topology.
- [multinode-running.md](multinode-running.md): running through madengine, `sbatch`, `salloc`.
- [vllm-disagg.md](vllm-disagg.md): the vLLM disaggregated launcher in full.
- [sglang-disagg.md](sglang-disagg.md): the SGLang disaggregated launcher in full.
- [kimi-k3.md](kimi-k3.md): Kimi-K3 end to end.
- [adding-a-model.md](adding-a-model.md): adding a card and a recipe.
