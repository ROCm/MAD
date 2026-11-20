# Adding a model

This page shows how to add a workload to MAD. A workload is three things: a **model card** that
describes it, a **Dockerfile** that builds its image, and a **script** that runs it and prints its
performance. The page explains every card field, how madengine turns a card into a Docker build,
how your script must report results, and how to add a multinode card.

Read [Getting started](getting-started.md) first. It explains discovery, tags and what happens
during a run.

## Contents

- [Where cards live](#where-cards-live)
- [Step 1: choose a name](#step-1-choose-a-name)
- [Step 2: write the card](#step-2-write-the-card)
- [Card field reference](#card-field-reference)
- [Step 3: write the Dockerfile](#step-3-write-the-dockerfile)
- [How the Dockerfile is found](#how-the-dockerfile-is-found)
- [The MAD_SYSTEM_GPU_ARCHITECTURE build argument](#the-mad_system_gpu_architecture-build-argument)
- [Step 4: write the script](#step-4-write-the-script)
- [Report performance](#report-performance)
- [Generating cards with get_models_json.py](#generating-cards-with-get_models_jsonpy)
- [Restricting a card to some GPUs](#restricting-a-card-to-some-gpus)
- [Adding a multinode card](#adding-a-multinode-card)
- [Checklist](#checklist)

## Where cards live

Cards used to live in the root `models.json`. They now live next to their scripts, in
`scripts/<dir>/models.json`. The root `models.json` is an empty list, kept because madengine
requires the file to exist. [`tools/migrate_to_dir_models.py`](../tools/migrate_to_dir_models.py)
did the move; its path rules show how a root card maps to a directory card:

| Field | Root `models.json` | `scripts/<dir>/models.json` |
|---|---|---|
| `dockerfile` | `docker/X` | `../../docker/X` |
| `scripts` | `scripts/<dir>` or `scripts/<dir>/` | `.` |
| `scripts` | `scripts/<dir>/f.sh` | `f.sh` |
| `scripts` | `scripts/<other>/...` | `../../scripts/<other>/...` |
| `dockercontext` | kept as is, relative to the repository root | kept as is; madengine does not rewrite it |

Put a new card in the `models.json` of the directory that holds its script. Create the directory
if it is a new family. Remember the discovery rules from
[Getting started](getting-started.md#how-madengine-discovers-models): the card's name gets the
directory as a prefix, its paths are relative to the directory, and a directory cannot have both
`models.json` and `get_models_json.py`.

## Step 1: choose a name

Name the workload `{framework}_{project}_{workload}`. Examples:

- `tf2_huggingface_gpt2`
- `pyt_torchvision_resnet50`
- `ort_onnx_bert`

MAD's current cards follow the same shape, for example `pyt_vllm_deepseek-r1`,
`pyt_sglang_disagg_mori_io_llama-3.1-8b` and `pyt_ncf_training`. The name must be unique.

## Step 2: write the card

The root README gives this example of a card, written for the root `models.json`:

```json
{
  "name": "tf2_bert_large",
  "url": "https://github.com/ROCmSoftwarePlatform/bert",
  "dockerfile": "docker/tf2_bert_large",
  "scripts": "scripts/tf2_bert_large",
  "n_gpus": "4",
  "owner": "john.doe@amd.com",
  "training_precision": "fp32",
  "tags": [
    "per_commit",
    "tf2",
    "bert",
    "fp32"
  ],
  "args": ""
}
```

The same card in `scripts/tf2_bert_large/models.json` uses paths relative to that directory:

```json
[
  {
    "name": "tf2_bert_large",
    "url": "https://github.com/ROCmSoftwarePlatform/bert",
    "dockerfile": "../../docker/tf2_bert_large",
    "scripts": "run.sh",
    "n_gpus": "4",
    "owner": "john.doe@amd.com",
    "training_precision": "fp32",
    "tags": ["per_commit", "tf2", "bert", "fp32"],
    "args": ""
  }
]
```

A real single-node card, from [`scripts/ncf/models.json`](../scripts/ncf/models.json):

```json
{
  "name": "pyt_ncf_training",
  "dockerfile": "../../docker/pyt_ncf_training",
  "scripts": "run.sh",
  "url": "https://github.com/ROCm/DeepLearningExamples",
  "data": "",
  "n_gpus": "1",
  "owner": "mad.support@amd.com",
  "training_precision": "fp32",
  "multiple_results": "results_ncf.csv",
  "tags": ["pyt", "training", "recommendation", "ncf"],
  "timeout": -1,
  "args": ""
}
```

Check the card with `madengine discover --tags <name> --verbose`. It prints the card as madengine
resolved it, with the directory prefix on the name and the paths rewritten.

## Card field reference

These are all the fields that appear in MAD's cards, plus the fields madengine accepts that MAD
cards do not use yet. "Required" follows the root README; madengine itself fills defaults for most
fields.

### Core fields

| Field | Required | Type | Description |
|---|---|---|---|
| `name` | Yes | string | Unique model identifier. For a directory card madengine prefixes it with the directory, for example `vllm/pyt_vllm_deepseek-r1`. It sets the image name, the log file names and `MAD_MODEL_NAME`. |
| `url` | Yes | string | Repository to clone into the container before the script runs. madengine runs `git clone <url>`, updates its submodules, and records the commit. The model directory is named after the last part of the URL, which should contain only letters, digits, `-` and `_`. Many MAD cards set `""` or omit it; the script then runs in an empty `run_directory`. |
| `dockerfile` | Yes | string | Path prefix of the Dockerfile, without the `.Dockerfile` suffix. See [How the Dockerfile is found](#how-the-dockerfile-is-found). |
| `scripts` | Yes | string | What to run. A `.sh` or `.slurm` file runs with `bash`, a `.py` file with `python3`, and a directory runs its `run.sh`. The file's directory is copied into the model directory before the run. |
| `n_gpus` | Yes | string | Number of GPUs. `"-1"` means all available; madengine converts it to the system GPU count. MAD cards use `"-1"`, `"1"` and `"8"`. |
| `owner` | Yes | string | Contact email. MAD cards use `mad.support@amd.com`. |
| `training_precision` | Yes | string | Precision label such as `fp16` or `fp32`. Inference cards leave it `""`. |
| `tags` | Yes | list of strings | Labels for selection with `--tags`. Include the framework (`pyt`), the family (`vllm`, `sglang_disagg`) and the kind (`inference`, `training`). |
| `args` | No | string | Arguments appended to the script. Extra arguments from a tag (`--tags <card>:key=value`) are appended after these. Example: `"--model_repo deepseek-ai/DeepSeek-R1-0528 --config configs/default.yaml"`. |
| `data` | No | string | Name of a data provider entry in the root [`data.json`](../data.json). MAD defines one entry, `huggingface`, which most inference cards use. `""` means no data step. |
| `timeout` | No | integer | Timeout in seconds for this card, overriding the 7200 s default. `--timeout` on the command line overrides it. `0` or any negative value means no timeout; most MAD cards set `-1`. |
| `multiple_results` | No | string | Name of a CSV file the script writes with one row per result. See [Report performance](#report-performance). |
| `skip_gpu_arch` | No | string | Comma-separated GPU architectures the card must not run on, for example `"gfx942"`. See [Restricting a card to some GPUs](#restricting-a-card-to-some-gpus). |

### Build fields

| Field | Description |
|---|---|
| `dockercontext` | Docker build context. The default is `./docker`, so a Dockerfile can only `COPY` files from `docker/`. madengine uses `.` (the repository root) when the `dockerfile` path contains `primus`. Set `dockercontext` to `.` when the Dockerfile copies from elsewhere in the repository. The JAX and Primus card generators set it. madengine does not rewrite it relative to the card's directory. |
| `docker_build_arg` | A JSON object of Docker build arguments for this card only, for example pins like `{"VLLM_REF": "<sha>"}`. `docker_build_arg` in `--additional-context` wins for the same key. No MAD card uses it yet. |
| `cred` | Name of a credential entry in `credential.json`. madengine uses it to clone a private `url` and passes its values as build arguments. No MAD card uses it. |

### Multinode fields

Only multinode cards use these. See [Adding a multinode card](#adding-a-multinode-card).

| Field | Description |
|---|---|
| `distributed.launcher` | How madengine starts the workload. Every MAD multinode card uses `slurm_multi`. |
| `distributed.nnodes` | Number of nodes. Must equal `slurm.nodes`. |
| `slurm.nodes` | Number of nodes madengine allocates (`#SBATCH --nodes`). |
| `slurm.gpus_per_node` | GPUs per node. MAD cards use `8`. |
| `slurm.time` | Wall-clock limit. MAD cards use `"24:00:00"`; override it for your partition. |
| `env_vars` | Environment the launcher script starts with: the image, the model, the topology and the benchmark knobs. madengine exports each one in the job script it generates. |

## Step 3: write the Dockerfile

Create the Dockerfile in `docker/`. The root README's example:

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
FROM rocm/tensorflow:latest

# Install system dependencies
RUN apt update && apt install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy

# Download model data
RUN URL=https://example.com/model-data.zip && \
    wget --directory-prefix=/data -c $URL && \
    ZIP_NAME=$(basename $URL) && \
    unzip /data/$ZIP_NAME -d /data && \
    rm /data/$ZIP_NAME

# Set working directory
WORKDIR /workspace
```

Conventions MAD's Dockerfiles follow:

- **Name it `<name>.ubuntu.amd.Dockerfile`**, for example `docker/pyt_vllm.ubuntu.amd.Dockerfile`,
  and point the card's `dockerfile` at `docker/<name>` (or `../../docker/<name>` from a
  directory card).
- **Start with a `# CONTEXT` line** in the first five lines. madengine reads it to decide whether
  the Dockerfile fits the run. See below.
- **Take the base image from `BASE_DOCKER`**, so users can change it without editing the file:

  ```dockerfile
  # CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
  ARG BASE_DOCKER=rocm/pytorch
  FROM $BASE_DOCKER
  ```

  That is the whole of [`docker/dummy.ubuntu.amd.Dockerfile`](../docker/dummy.ubuntu.amd.Dockerfile).
  To build on a different base:

  ```bash
  madengine build --tags <card> \
    --additional-context '{"docker_build_arg": {"BASE_DOCKER": "rocm/pytorch:rocm6.1_ubuntu22.04_py3.10"}}'
  ```

- **Pin sources to commits.** [`docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`](../docker/vllm_kimi_k3.ubuntu.amd.Dockerfile)
  pins every source to an immutable commit SHA so the image can be rebuilt to the same bits later.
  It also sets `PIP_CONSTRAINT` globally, because pip resolves build dependencies such as
  setuptools from PyPI in an isolated environment, and a plain `pip install setuptools==X` does
  not reach that environment. A newer setuptools (80 and later) broke the MoRI wheel build that
  way while every pinned source was unchanged.

## How the Dockerfile is found

The card's `dockerfile` value is a **prefix**, not a file name. madengine lists the files that
start with it and keeps only these two forms:

- `<prefix>.Dockerfile`
- `<prefix>.<os>.<vendor>.Dockerfile`, where `<os>` and `<vendor>` are single words of letters,
  digits, `_` or `-`

The match is exact. Other files that merely share the prefix are ignored. For example, with
`"dockerfile": "../../docker/vllm_disagg_inference"`:

| File | Built for this card? |
|---|---|
| `docker/vllm_disagg_inference.ubuntu.amd.Dockerfile` | Yes |
| `docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile` | No: `glmv5.1.ubuntu.amd` is not `<os>.<vendor>` |

The GLM-5.1 cards select that second file with their own prefix,
`../../docker/vllm_disagg_inference.glmv5.1`. To give a card its own image, give the Dockerfile
its own prefix and point the card at it.

madengine then reads the `# CONTEXT` line from the first five lines of each file and keeps the
files whose context matches the run. A key in the header must equal the same key in the run
context (`gpu_vendor` defaults to `AMD`, `guest_os` to `UBUNTU`). A Dockerfile without the header
cannot be read this way, and madengine finds no Dockerfile for the card. Each kept file is built
as its own image.

The image is named `ci-<card name>_<Dockerfile name without .Dockerfile>`, lowercased, with `/`
replaced by `_`. The build log is `<card name>_<Dockerfile name>.build.live.log`.

## The MAD_SYSTEM_GPU_ARCHITECTURE build argument

`MAD_SYSTEM_GPU_ARCHITECTURE` is the GPU architecture, such as `gfx942` (MI300X) or `gfx950`
(MI355X). madengine sets it in two places:

- **Inside the container**, as an environment variable, so scripts can adapt. For example
  [`scripts/huggingface_bert/run.sh`](../scripts/huggingface_bert/run.sh) picks a batch size
  from it.
- **At build time**, as a Docker build argument, so a Dockerfile can compile for one
  architecture. A Dockerfile opts in by declaring it:

  ```dockerfile
  ARG MAD_SYSTEM_GPU_ARCHITECTURE
  ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
  ```

  [`docker/pyt_clip_inference.ubuntu.amd.Dockerfile`](../docker/pyt_clip_inference.ubuntu.amd.Dockerfile),
  `pyt_janus_pro_inference` and `pyt_wan2.1_inference` do this.

Where the build value comes from, first match wins:

1. `docker_build_arg.MAD_SYSTEM_GPU_ARCHITECTURE` in `--additional-context`.
2. `--target-archs` on `madengine build`, which builds one image per listed architecture and
   passes each one in turn. The image name gets an `_<arch>` suffix.
3. Local detection. `madengine run --tags ...` (build and run together) detects the local GPU and
   passes its architecture. A standalone `madengine build` does not detect, because a build host
   may have no GPU.

**A declaration with no default makes the argument required.** When a Dockerfile declares
`ARG MAD_SYSTEM_GPU_ARCHITECTURE` with no value (or an empty one), and none of the three sources
gives one, madengine prints a warning that names the cards and Dockerfiles affected and suggests:

```bash
--additional-context '{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
```

A declaration with a non-empty default (`ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942`) does not need
it.

**Example: an image built for exactly one architecture.**
[`docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`](../docker/vllm_kimi_k3.ubuntu.amd.Dockerfile) is the
image for the Kimi-K3 vLLM multinode cards, on MI300X (`gfx942`) and MI355X (`gfx950`), colocated
and disaggregated. It declares the argument with no default, on purpose: an image built for the
wrong architecture fails at runtime, so the build refuses to guess.

```dockerfile
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV K3_GFX_ARCH=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN case "${K3_GFX_ARCH}" in \
      gfx942|gfx950) echo "Kimi-K3 image for ${K3_GFX_ARCH}" ;; \
      *) echo "MAD_SYSTEM_GPU_ARCHITECTURE must be gfx942 or gfx950, got '${K3_GFX_ARCH}'" >&2; exit 1 ;; \
    esac
```

Everything architecture-specific in the build follows that value: MoRI's JIT target, the vLLM
compile, and, with `WITH_NIXL=1`, rocSHMEM and DeepEP. The value is copied into `K3_GFX_ARCH` so
that madengine's Dockerfile architecture check, which parses `GFX_COMPILATION_ARCH`,
`PYTORCH_ROCM_ARCH` and `GPU_ARCHS`, does not read it as a fixed architecture. Build it through
madengine with the architecture of the nodes it will run on:

```bash
madengine build --tags pyt_vllm_kimi-k3_mi300x_pp2xtp8 --registry <registry> \
  --additional-context '{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
```

or by hand, from the repository root:

```bash
docker build -f docker/vllm_kimi_k3.ubuntu.amd.Dockerfile \
  --build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx942 -t <registry>/vllm-kimi-k3:gfx942 .
```

Details of that image are in [kimi-k3.md](kimi-k3.md).

## Step 4: write the script

Create the script in your `scripts/<dir>/`. The root README's example `run.sh`:

```bash
#!/bin/bash
set -e

# Model configuration
MODEL_CONFIG_DIR=/data/model_config
BATCH_SIZE=2
SEQUENCE_LENGTH=512
TRAIN_STEPS=100
WARMUP_STEPS=10
LEARNING_RATE=1e-4

# Prepare data
echo "Preparing training data..."
python3 prepare_data.py \
    --config_dir=$MODEL_CONFIG_DIR \
    --batch_size=$BATCH_SIZE \
    --seq_length=$SEQUENCE_LENGTH

# Train model
echo "Starting model training..."
python3 train_model.py \
    --config_dir=$MODEL_CONFIG_DIR \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$SEQUENCE_LENGTH \
    --num_train_steps=$TRAIN_STEPS \
    --num_warmup_steps=$WARMUP_STEPS \
    --learning_rate=$LEARNING_RATE \
    2>&1 | tee training.log

# Report performance
echo "Generating performance metrics..."
python3 report_metrics.py
```

What the script can rely on at run time:

- **Working directory.** The script runs inside the model directory: the cloned repository if the
  card has a `url`, or `run_directory` otherwise. The files from the card's script directory are
  copied into it, so helpers next to `run.sh` are in the current directory.
- **Paths.** The MAD checkout is mounted at `/myworkspace` and is the container's working
  directory, so the model directory is `/myworkspace/<model dir>` and `../` is the MAD root.
- **Arguments.** `$@` holds the card's `args` plus any extra arguments from the tag.
- **Environment.** `MAD_SYSTEM_GPU_ARCHITECTURE`, `MAD_RUNTIME_NGPUS`, `MAD_SYSTEM_NGPUS`,
  `MAD_MODEL_NAME`, `MAD_OUTPUT_CSV` and anything passed with `docker_env_vars`. See
  [Getting started](getting-started.md#environment-variables-inside-the-container).

## Report performance

madengine does not know what your model measures. Your script tells it, in one of two ways.

### Single result

Print one line of this form to standard output:

```python
print(f"performance: {throughput} examples/sec")
```

madengine searches the run log for `performance:`, then a number, then a metric name. The number
can be an integer, a decimal or scientific notation (`1.23e+4`). A unit suffix such as `/s` and a
comma may sit between the number and the metric, in either order, so all of these parse:

```text
performance: 14164 samples_per_second
performance: 14164/s, samples_per_second
performance: 14164, /s samples_per_second
```

If there is no such line, madengine falls back to the Hugging Face Trainer's
`train_samples_per_second` value, with the metric `samples_per_second`. If neither is found, the
run has no performance and its status is `FAILURE`.
[`scripts/huggingface_bert/run.sh`](../scripts/huggingface_bert/run.sh) ends this way:

```bash
set +x
echo "performance: $performance samples_per_second"
```

The script turns off shell tracing (`set +x`) before it prints. With tracing on, bash also logs
the `echo` command itself (`+ echo 'performance: ...'`), and madengine takes the first match in
the log.

### Multiple results

When one run produces several results, for example one per batch size or per precision, write a
CSV file and name it in the card's `multiple_results` field. madengine also passes the name to the
container as `MAD_OUTPUT_CSV`.

The CSV must have the columns `model`, `performance` and `metric`:

```csv
model,performance,metric
model_1,156.7,examples/sec
model_2,89.3,tokens/sec
```

The root README names the first column `models`. madengine's results code requires `model`, and
MAD's scripts write `model` (see [`scripts/dummy/run_multi.sh`](../scripts/dummy/run_multi.sh) and
[`scripts/ncf/get_ncf_model_metrics.py`](../scripts/ncf/get_ncf_model_metrics.py)). Use `model`.

How madengine reads the file:

- It looks for the file in the MAD root first, then in the model directory, and copies it out of
  the model directory when it is there.
- A missing `performance` column, or a `performance` column empty in every row, means the run has
  no performance.
- Each row becomes one row in `perf.csv`. The row's model name is `<card name>_<model>`.
- Extra columns in your CSV (for example batch size, input length, tensor-parallel size) are
  carried into `perf.csv`.
- Each row gets its own status: `SUCCESS` when its `performance` is set, `FAILURE` when it is
  empty.

The dummy card is the smallest working example. Its script writes four rows and copies the file
to the MAD root:

```bash
echo "model,performance,metric
1,$RANDOM,samples_per_sec
2,$RANDOM,samples_per_sec
3,$RANDOM,samples_per_sec
4,$RANDOM,samples_per_sec" >>perf_dummy.csv

cp perf_dummy.csv ../
```

The full `perf.csv` schema and status rules are in
[Benchmarks and results](benchmarks-and-results.md).

## Generating cards with get_models_json.py

When a family has many similar cards, generate them in Python instead of listing them. Put a
`get_models_json.py` in `scripts/<dir>/` (and no `models.json` in the same directory). It must
define `list_models()`, which returns a list of `CustomModel` objects:

```python
from madengine.utils.discover_models import CustomModel

def list_models():
    return [
        CustomModel(
            name="default",
            dockerfile="../../docker/primus",
            dockercontext=".",
            scripts="run.sh",
            n_gpus="-1",
            owner="mad.support@amd.com",
            tags=["training", "primus", "megatron", "pretrain"],
            args="",
        )
    ]
```

`CustomModel` has the core fields (`name`, `dockerfile`, `dockercontext`, `scripts`, `url`,
`cred`, `owner`, `data`, `n_gpus`, `timeout`, `training_precision`, `tags`, `args`,
`multiple_results`, `skip_gpu_arch`). Its defaults are `n_gpus="-1"` and `timeout=7200`. As with
`models.json`, madengine prefixes the name with the directory and resolves `dockerfile` and
`scripts` relative to it.

MAD has three generators. [`scripts/primus_train/get_models_json.py`](../scripts/primus_train/get_models_json.py)
makes one card per Primus example config and passes the config path in `args`.
`scripts/jax-maxtext/get_models_json.py` and `scripts/jax-maxdiffusion/get_models_json.py` do the
same for JAX.

## Restricting a card to some GPUs

`skip_gpu_arch` lists the architectures a card must not run on, separated by commas
(`"gfx942"`, or `"gfx942,gfx950"`). madengine applies it before running:

- **Locally**, it compares the list with the host GPU.
- **On SLURM**, it probes the compute nodes' architecture before submitting, and drops the card
  from the job. If the probe cannot tell, madengine submits anyway with a warning; set
  `slurm.gpu_arch` in `--additional-context` to enforce the check.

A skipped card gets a `SKIPPED` row in `perf.csv`. `--disable-skip-gpu-arch` turns the check off.

Multinode cards can also declare `GPU_ARCHS`, the architectures they support. The launcher reads it
on the allocation and refuses the wrong nodes. For `scripts/vllm_dissag` it lives in
`models.yaml` under the card's `MODEL_NAME`; elsewhere it is in the card's `env_vars`, and a card's
own value wins. The two fields must agree, or one path skips a card that the other runs.
[`scripts/common/check_gpu_arch_declarations.py`](../scripts/common/check_gpu_arch_declarations.py)
checks, for every card that declares `GPU_ARCHS`, that:

1. no architecture it supports is also in `skip_gpu_arch`;
2. every known architecture (`gfx942`, `gfx950`) it does not support is in `skip_gpu_arch`, so
   madengine skips it without spending an allocation to find out.

```bash
python3 scripts/common/check_gpu_arch_declarations.py
```

## Adding a multinode card

A multinode card runs one workload across several nodes of a SLURM cluster. Read
[Multinode overview](multinode-overview.md) first for the concepts, and
[Running multinode workloads](multinode-running.md) for how to run one.

### What is different

Every MAD multinode card uses the `slurm_multi` launcher. With it, madengine does not start a
container itself. Instead it:

1. writes a wrapper SBATCH script (`slurm_results/madengine_<name>.sh`) that exports the card's
   `env_vars`;
2. pulls the image on every node in parallel when it is a registry image;
3. submits the wrapper with `sbatch`, or runs it with `bash` when you are already inside a
   `salloc` allocation;
4. runs the card's own `.slurm` script on the head node, with the card's `args`. That script
   starts the per-node containers itself with `srun`;
5. collects the `perf.csv` the script writes.

The card's `.slurm` script is the **launcher**. The card's `env_vars` are the launcher's contract:
the same variables work when you export them and submit the launcher with `sbatch` yourself.

### An example card

From [`scripts/vllm_dissag/models.json`](../scripts/vllm_dissag/models.json):

```json
{
  "name": "pyt_vllm_disagg_nixl_deepseek-v3",
  "dockerfile": "../../docker/vllm_disagg_inference",
  "scripts": "run_xPyD_models.slurm",
  "url": "",
  "data": "huggingface",
  "n_gpus": "-1",
  "owner": "mad.support@amd.com",
  "training_precision": "",
  "tags": ["pyt", "vllm", "vllm_disagg", "nixl", "inference"],
  "timeout": -1,
  "distributed": {
    "launcher": "slurm_multi",
    "nnodes": 2
  },
  "env_vars": {
    "DOCKER_IMAGE_NAME": "<supply-your-image>",
    "MODEL_NAME": "DeepSeek-V3",
    "xP": "1",
    "yD": "1",
    "WIDE_EP": "1",
    "RUN_MORI": "0",
    "RUN_DEEPEP": "0",
    "BENCHMARK_COMBINATIONS": "1024/1024"
  },
  "args": "-N 2 -n 2",
  "slurm": {
    "nodes": 2,
    "gpus_per_node": 8,
    "time": "24:00:00"
  }
}
```

### Rules for a multinode card

- **`slurm.nodes` and `distributed.nnodes` must agree.** madengine sizes the allocation from
  `slurm.nodes` only (it becomes `#SBATCH --nodes`, default 1), and the launcher then reads
  `SLURM_NNODES` to find its nodes. `distributed.nnodes` carries the same number for launcher
  detection but does not size the allocation. If they differ, the job gets the wrong number of
  nodes. `--additional-context '{"slurm": {"nodes": N}}'` overrides the card.
- **Put the node count in `args` too.** MAD cards set `"args": "-N <nodes> -n <nodes>"`, the same
  count you give `sbatch` when you submit the launcher by hand.
- **For disaggregated cards, nodes = `xP + yD`.** `xP` is the number of prefill nodes and `yD` the
  number of decode nodes, so `xP=1, yD=1` needs 2 nodes and `xP=2, yD=2` needs 4.
- **Use `"n_gpus": "-1"`** and set GPUs per node in `slurm.gpus_per_node`.
- **Leave allocation defaults out of the card.** Partition, GPUs per node and exclusivity come from
  madengine's SLURM presets: partition `amd-rccl`, 8 GPUs per node, exclusive. Users override them
  in `--additional-context`. The card's `slurm.time` of `24:00:00` exceeds many partitions'
  limits, so users usually pass their own `slurm.time`.
- **Declare GPU support** with `skip_gpu_arch`, and with `GPU_ARCHS` if the launcher should refuse
  wrong nodes. See [Restricting a card to some GPUs](#restricting-a-card-to-some-gpus).

### The `DOCKER_IMAGE_NAME` placeholder

A multinode launcher runs the image named by `DOCKER_IMAGE_NAME` on every node, so each node must be
able to pull it. MAD cannot know your registry, so every multinode card ships with:

```json
"DOCKER_IMAGE_NAME": "<supply-your-image>"
```

This is a fill-me-in marker, not an image. Supply the real image in one of these ways:

- `madengine build --tags <card> --registry <registry>` builds the image, pushes it, and records
  the pushed name for the run.
- `madengine build --tags <card> --use-image <registry>/<image>:<tag>` uses an image you already
  pushed.
- Under plain `sbatch`, `export DOCKER_IMAGE_NAME=<registry>/<image>:<tag>` before submitting.

A `slurm_multi` build with neither `--registry` nor `--use-image` (nor `--build-on-compute`) takes
the card's `DOCKER_IMAGE_NAME` as the image, and would take the placeholder. When no card image is
set, it stops with `slurm_multi launcher requires --registry or --use-image`. madengine's SLURM
deployment does not accept a name that starts with `<` as an image, and it refuses a local
`ci-...` image for a run of more than one node, because the other nodes cannot pull it.

Keep the placeholder in cards you contribute. Do not commit a private image name.

### Recipes and environment

Keep the card small. Model-specific serve flags belong in the **recipe**, not in the card:

- vLLM disaggregated: an entry in [`scripts/vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml)
  keyed by `MODEL_NAME`, plus the name in `VALID_MODELS` in `run_xPyD_models.slurm`. See
  [vllm-disagg.md](vllm-disagg.md).
- SGLang disaggregated: [`scripts/sglang_disagg/models.yaml`](../scripts/sglang_disagg/models.yaml).
  See [sglang-disagg.md](sglang-disagg.md).
- vLLM colocated multinode: the card's `env_vars` (for example `TP_SIZE`, `PP_SIZE`,
  `COLOCATED_EXTRA_ARGS`) and [`scripts/vllm_multinode/mad-config.kimi-k3.yaml`](../scripts/vllm_multinode/mad-config.kimi-k3.yaml).
  See [kimi-k3.md](kimi-k3.md).

The launcher's environment is layered, weakest first: `scripts/common/cluster.sh` defaults, the
recipe in `models.yaml`, the card's `env_vars`, then what the user sets (madengine `env_vars` in
`--additional-context`, or the exported environment under `sbatch`). The full rules and every knob
are in [configuration.md](configuration.md).

### Offline checks

These need no GPUs. Run them before you open a pull request:

```bash
bash scripts/vllm_dissag/tests/argv_assert.sh           # the serve argv per connector and mode
python3 scripts/common/check_srun_quotes.py             # no apostrophe truncates an srun body
python3 scripts/common/check_gpu_arch_declarations.py   # cards and recipes agree on GPUs
```

A launcher can also be previewed without starting servers: `DRY_RUN=1` prints each node's server
command. See [Running multinode workloads](multinode-running.md).

## Checklist

Before you open a pull request:

1. The card is in `scripts/<dir>/models.json`, with a unique name and useful tags.
2. `madengine discover --tags <name> --verbose` shows the card with the paths you expect.
3. The Dockerfile is `docker/<name>.ubuntu.amd.Dockerfile`, starts with a `# CONTEXT` line, and
   is found for the card (only `<prefix>.Dockerfile` and `<prefix>.<os>.<vendor>.Dockerfile`
   match).
4. If the Dockerfile declares `ARG MAD_SYSTEM_GPU_ARCHITECTURE` with no default, you built it with
   the argument set.
5. The script prints `performance: <number> <metric>`, or writes the `multiple_results` CSV with
   `model`, `performance` and `metric` columns.
6. `madengine run --tags <name> --live-output` finishes with `SUCCESS` and a row in `perf.csv`.
7. For a multinode card: `slurm.nodes` equals `distributed.nnodes`, `DOCKER_IMAGE_NAME` is
   `<supply-your-image>`, `skip_gpu_arch` and `GPU_ARCHS` agree, and the offline checks pass.
