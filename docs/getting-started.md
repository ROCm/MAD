# Getting started

This page takes you from an empty machine to a finished benchmark run on one GPU host. It explains
what happens during a run, how madengine finds models, how building and running relate, and where
the results go. When you finish, continue with [Adding a model](adding-a-model.md).

## Contents

- [The pieces](#the-pieces)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Run your first model](#run-your-first-model)
- [What happens during a run](#what-happens-during-a-run)
- [How madengine discovers models](#how-madengine-discovers-models)
- [Selecting models with tags](#selecting-models-with-tags)
- [Options for `madengine run`](#options-for-madengine-run)
- [Build and run as separate steps](#build-and-run-as-separate-steps)
- [Timeouts](#timeouts)
- [Debugging](#debugging)
- [Where results go](#where-results-go)
- [Environment variables inside the container](#environment-variables-inside-the-container)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

## The pieces

Four terms come up on every page. Learn them first.

- **madengine** is the command-line tool that does the work. It lives in its own repository,
  [ROCm/madengine](https://github.com/ROCm/madengine). MAD holds the model definitions; madengine
  reads them and runs them.
- A **model card** is one JSON object that describes one workload: which Dockerfile builds its
  image, which script runs it, how many GPUs it needs, and which tags it has.
- A **tag** is a label on a card. You select what to run by tag or by card name.
- **perf.csv** is the results table madengine writes after a run.

The repository is laid out like this:

| Path | What it holds |
|---|---|
| `models.json` | The root card list. madengine requires this file to exist. In MAD it is an empty list (`[]`); every card lives under `scripts/`. |
| `scripts/<dir>/models.json` | The cards for one workload family, for example [`scripts/vllm/models.json`](../scripts/vllm/models.json). |
| `scripts/<dir>/get_models_json.py` | Python that generates cards at discovery time, for example [`scripts/primus_train/get_models_json.py`](../scripts/primus_train/get_models_json.py). |
| `scripts/<dir>/` | The run scripts, helper scripts and configuration for that family. |
| `docker/` | The Dockerfiles, named `<name>.ubuntu.amd.Dockerfile`. |
| `data.json` | Data provider definitions that a card's `data` field refers to. MAD defines one entry, `huggingface`. |
| `benchmark/` | Per-blueprint documentation, for example [`benchmark/vllm/README.md`](../benchmark/vllm/README.md). |
| `scripts/common/` | Shared scripts and checks used by the multinode workloads. |
| `tools/` | Legacy tooling. `tools/run_models.py` is no longer recommended and will be removed; use madengine. |
| `requirements.txt` | Installs madengine from `git+https://github.com/ROCm/madengine.git@main`. |

## Prerequisites

You need:

- Docker, installed and running.
- Python 3.9 or higher. (madengine's own installation guide lists Python 3.8+; MAD asks for 3.9.)
- GPU drivers: AMD ROCm or NVIDIA CUDA.
- Git.

Your user must be able to run Docker. On Linux, add yourself to the `docker` group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Check that containers can see the GPUs. On AMD:

```bash
rocm-smi
ls -la /dev/kfd /dev/dri
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest rocm-smi
```

On NVIDIA:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:latest nvidia-smi
```

If ROCm is not installed under `/opt/rocm` on the host (for example a TheRock or pip install), set
`ROCM_PATH` on the host, or pass `MAD_ROCM_PATH` in `--additional-context`, so madengine's host GPU
checks find `rocminfo` and `amd-smi`. The ROCm path inside the container is resolved separately;
see madengine's [configuration guide](https://github.com/ROCm/madengine/blob/main/docs/configuration.md#rocm-path-run-only).

## Install

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/ROCm/MAD.git
   cd MAD
   ```

2. Install the dependencies. This installs madengine:

   ```bash
   pip install -r requirements.txt
   ```

   A virtual environment (`python3 -m venv venv && source venv/bin/activate`) keeps it separate
   from system packages. To install madengine by hand instead, run
   `pip install git+https://github.com/ROCm/madengine.git`.

3. Check the installation:

   ```bash
   madengine --version
   madengine discover
   ```

   `madengine discover` with no tags prints every card it can find. If it prints nothing or fails
   with `models.json file not found`, you are not in the MAD root. See
   [Troubleshooting](#troubleshooting).

## Run your first model

Run every command from the root of the MAD checkout.

```bash
madengine run --tags pyt_huggingface_bert
```

This builds the image for the `pyt_huggingface_bert` card, starts a container, clones the
Hugging Face transformers repository into it, runs [`scripts/huggingface_bert/run.sh`](../scripts/huggingface_bert/run.sh),
reads the throughput from the log, and appends a row to `perf.csv`.

For a faster smoke test, run the dummy card. It writes four random results and needs no model
download:

```bash
madengine run --tags dummies --live-output
```

`dummies` is the tag on the `dummy_multi` card in [`scripts/dummy/models.json`](../scripts/dummy/models.json).

madengine assumes an AMD GPU and an Ubuntu image unless told otherwise. You can state that
explicitly, which the madengine guide recommends outside AMD/Ubuntu setups:

```bash
madengine run --tags dummies \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

The supported values are `gpu_vendor`: `AMD` or `NVIDIA`, and `guest_os`: `UBUNTU` or `CENTOS`.

## What happens during a run

`madengine run --tags ...` without a manifest runs the full workflow: discover, build, run, report.
The root README summarises it in five steps:

1. **Build**: create a Docker image named `ci-<model name>...`.
2. **Start**: launch a container named `container_<image name>`.
3. **Clone**: download the model repository from the card's `url`.
4. **Execute**: run the model script.
5. **Report**: write `perf.csv`.

In more detail, for each selected card:

1. **Discover.** madengine reads every card and keeps the ones your tags select. See
   [How madengine discovers models](#how-madengine-discovers-models).
2. **Find the Dockerfile.** The card's `dockerfile` field is a prefix. madengine picks
   `<prefix>.Dockerfile` or `<prefix>.<os>.<vendor>.Dockerfile`, then keeps the ones whose
   `# CONTEXT` header matches the run (for example `{'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}`).
   Details are in [Adding a model](adding-a-model.md#how-the-dockerfile-is-found).
3. **Build the image.** The image is named
   `ci-<card name>_<Dockerfile name without .Dockerfile>`, with `/` in the card name replaced by
   `_` and the name lowercased. For example the `pyt_huggingface_bert` card in
   `scripts/huggingface_bert` builds `ci-huggingface_bert_pyt_huggingface_bert_pyt_huggingface.ubuntu.amd`.
   The build log goes to `<card name>_<Dockerfile name>.build.live.log`. On a full workflow run,
   madengine detects the local GPU architecture and passes it to the build as
   `MAD_SYSTEM_GPU_ARCHITECTURE`.
4. **Start the container.** The container is named `container_<image name>`. Your current
   directory (the MAD root) is mounted at `/myworkspace` and is the working directory.
5. **Prepare the model directory.** If the card has a `url`, madengine clones it with
   `git clone`, updates its submodules, and records the commit. The directory is named after the
   last part of the URL. With no `url`, madengine creates an empty `run_directory`. It then copies
   the card's script directory into the model directory, prepares data if the card has a `data`
   field, and runs any pre-scripts.
6. **Run the script.** madengine runs the script inside the model directory, with the card's
   `args` appended. A `.sh` or `.slurm` script runs with `bash`, a `.py` script with `python3`,
   and a directory runs its `run.sh`. The run log goes to `<card name>_<Dockerfile name>.run.live.log`.
   After a full workflow run the build and run logs are also joined into `<card name>_<Dockerfile name>.live.log`.
7. **Collect the result.** madengine reads the performance from the log (a line of the form
   `performance: <number> <metric>`), or from the CSV named by the card's `multiple_results`, and
   writes the row to `perf.csv`. See [Adding a model](adding-a-model.md#report-performance).
8. **Clean up.** madengine runs any post-scripts, removes the model directory, and stops and
   removes the container, unless you asked it to keep them (`--keep-alive` keeps both,
   `--keep-model-dir` keeps the directory).

A run's status is `SUCCESS` when a valid performance value was found. If no performance was found,
or an error pattern matched in the log with no performance to contradict it, the status is
`FAILURE`. [Benchmarks and results](benchmarks-and-results.md) covers the status rules in full.

## How madengine discovers models

madengine looks for cards in three places, all relative to the current directory:

| Source | What madengine does with it |
|---|---|
| `models.json` in the MAD root | Reads it as a list of cards. The file must exist, or discovery fails with `models.json file not found`. MAD keeps it as an empty list. |
| `scripts/<dir>/models.json` | Reads every card in it. madengine walks `scripts/` recursively, so nested directories work too. |
| `scripts/<dir>/get_models_json.py` | Imports it and calls its `list_models()` function, which returns `CustomModel` objects. This is how a family generates many cards from files, for example one card per Primus config. |

Three rules apply to directory cards:

- **The name gets the directory as a prefix.** A card named `pyt_vllm_deepseek-r1` in
  `scripts/vllm/models.json` is known as `vllm/pyt_vllm_deepseek-r1`. You can still select it by
  its short name.
- **Paths are relative to the card's directory.** In `scripts/vllm/models.json`,
  `"dockerfile": "../../docker/pyt_vllm"` means `docker/pyt_vllm`, and `"scripts": "run.sh"` means
  `scripts/vllm/run.sh`.
- **A directory has one registration file.** A directory that contains both `models.json` and
  `get_models_json.py` stops discovery with an error.

Tags are not prefixed; they stay as written in the card.

madengine expects to run from the MAD root. If you run it from somewhere else, set `MODEL_DIR` to
the MAD checkout. madengine then copies `docker/`, `scripts/`, `models.json`, `credential.json` and
`data.json` from there into the current directory before discovering:

```bash
export MODEL_DIR=/path/to/MAD
madengine discover
```

Use `madengine discover` to check what a set of tags will select before you run anything:

```bash
madengine discover --tags vllm --verbose
```

## Selecting models with tags

Tags let you run a subset of models by their characteristics. The root README groups them like
this:

| Kind | Examples |
|---|---|
| Framework tags | `pyt`, `tf2`, `ort` |
| Model tags | `bert`, `gpt2`, `resnet50` |
| Precision tags | `fp16`, `fp32` |
| Custom tags | Any tag defined in a card, for example `inference`, `vllm_disagg`, `mori_io` |

Each value you pass to `--tags` is matched on its own. A card is selected when the value is:

- the card's full name, such as `vllm/pyt_vllm_deepseek-r1`;
- the card's short name, such as `pyt_vllm_deepseek-r1`;
- one of the card's tags, such as `vllm`;
- `all`, which selects every card.

The cards selected by all the values are run together, so more values select more cards, not
fewer. If a value matches nothing, madengine stops with `No models found corresponding to the given tag`.

```bash
# Run a specific model
madengine run --tags pyt_huggingface_bert

# Run all PyTorch models
madengine run --tags pyt

# Run several tags: repeat the option, or separate the values with commas
madengine run --tags tf2 --tags bert --tags fp32
madengine run --tags tf2,bert,fp32
```

The root README writes the last example as `madengine run --tags tf2 bert fp32`. madengine's own
guide documents the repeated and comma-separated forms shown above.

**Scoped tags.** A value of the form `<dir>/<filter>` (one `/`, no `:`) only looks at cards from
`scripts/<dir>/`. The filter can be a tag, a card name, or `all`:

```bash
madengine discover --tags vllm_dissag/all       # every card in scripts/vllm_dissag
madengine discover --tags sglang_disagg/mori_io # cards in scripts/sglang_disagg tagged mori_io
```

**Extra script arguments.** Anything after a `:` in a tag value is passed to the script. Each
`key=value` becomes `--key value` and is appended to the card's `args`:

```bash
madengine run --tags <card>:batch_size=512:in=32
# appends " --batch_size 512 --in 32" to the card's args
```

The script must accept those arguments; madengine only appends them.

## Options for `madengine run`

The basic form is:

```bash
madengine run [OPTIONS]
```

The options you will use most:

| Option | What it does | Default |
|---|---|---|
| `--tags TAGS`, `-t` | Select cards by tag or name. Repeat or comma-separate for several. | none |
| `--timeout TIMEOUT` | Timeout in seconds. See [Timeouts](#timeouts). | `-1`, meaning not set, which falls through to the card or 7200 s (2 hours) |
| `--live-output`, `-l` | Print the build and run output in real time. | off |
| `--clean-docker-cache` | Rebuild the images without the Docker cache. | off |
| `--keep-alive` | Keep the container running after the run. Local Docker only. | off |
| `--keep-model-dir` | Keep the model directory after the run. Local Docker only. | off |
| `--skip-model-run` | Start the container and run the pre-scripts, but do not run the model script. The status is `SKIPPED`. | off |
| `-o OUTPUT`, `--output OUTPUT` | The performance CSV to write. | `perf.csv` |
| `--verbose`, `-v` | Verbose logging. | off |
| `--additional-context`, `-c` | JSON that adds or overrides configuration. | `{}` |
| `--additional-context-file`, `-f` | The same JSON, read from a file. `--additional-context` is merged over it key by key. | none |
| `--manifest-file`, `-m` | Run from an existing build manifest instead of building. | none |
| `--registry`, `-r` | Docker registry to push to or pull from. | none |
| `--disable-skip-gpu-arch` | Run cards even when their `skip_gpu_arch` names this GPU. | off |
| `--data-config` | The data provider file. | `data.json` |
| `--cleanup-perf` | Remove the intermediate `perf_entry` files after the run. | off |

The root README also lists `--log-level LOG_LEVEL` (default `INFO`). The current madengine CLI
reference does not list that option; use `--verbose` for debug logging. The full option list is in
madengine's [CLI reference](https://github.com/ROCm/madengine/blob/main/docs/cli-reference.md#run---execute-models).

## Build and run as separate steps

`madengine run --tags X` builds and runs in one call. You can split the two:

```bash
# Build the images and write build_manifest.json
madengine build --tags pyt_huggingface_bert

# Run from the manifest, without building
madengine run --manifest-file build_manifest.json
```

Splitting helps when you build on one machine and run on another, or when you want to build once
and run many times. The build manifest, `build_manifest.json`, records for each image the card it
was built for, the Dockerfile, the build time, the registry image if it was pushed, and the
context. Change its name with `--manifest-output`.

Things that differ between the two modes:

- **GPU architecture.** In the full workflow (`madengine run --tags`), madengine detects the local
  GPU architecture and passes it to the build as `MAD_SYSTEM_GPU_ARCHITECTURE`. A standalone
  `madengine build` does not detect it, because it may run on a machine with no GPU. For a
  Dockerfile that needs it, pass it yourself:

  ```bash
  madengine build --tags <card> \
    --additional-context '{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
  ```

  Or build one image per architecture with `--target-archs`, which adds the architecture to each
  image name:

  ```bash
  madengine build --tags <card> --target-archs gfx942,gfx950 --registry <registry>
  ```

- **Registries.** `--registry` pushes each image after building it, so other machines can pull
  it. Credentials come from an existing `docker login`, from `credential.json` in the MAD root, or
  from `MAD_DOCKERHUB_USER`, `MAD_DOCKERHUB_PASSWORD` and `MAD_DOCKERHUB_REPO`.
- **Prebuilt images.** `madengine build --use-image <image>` skips the build and writes a manifest
  for an image you already have. `--use-image auto` takes the image from the card's
  `env_vars.DOCKER_IMAGE_NAME`. For a local run, the `MAD_CONTAINER_IMAGE` key in
  `--additional-context` does the same thing: it skips the build and runs the card in that image.

  ```bash
  madengine run --tags pyt_huggingface_bert \
    --additional-context '{"MAD_CONTAINER_IMAGE": "rocm/pytorch:custom-tag"}'
  ```

- **Target.** A `slurm` key in `--additional-context` makes madengine submit to SLURM, and a `k8s`
  key sends the run to Kubernetes. Without either, the run is local Docker. Multinode runs are
  covered in [Running multinode workloads](multinode-running.md).

## Timeouts

A timeout stops a model that runs too long. It is set at three levels, lowest priority first:

1. **Default:** 7200 seconds (2 hours).
2. **Card:** the `timeout` field in the card.
3. **Command line:** `--timeout`.

```bash
# Override the timeout for this run
madengine run --tags pyt_huggingface_bert --timeout 3600

# No timeout
madengine run --tags pyt_huggingface_bert --timeout 0
```

Rules:

- A timeout of `0` disables the timeout. Any resolved value of 0 or less means no timeout. That
  includes a card with `"timeout": 0` or `"timeout": -1`, which many MAD cards use.
- `--timeout -1` is the CLI default and means "not passed", so the card's value (or 7200) applies.
  An explicit `--timeout 7200` still overrides a card, even though it equals the default.
- On SLURM the same precedence applies. The submitting process caps its own wait at the resolved
  timeout, and the card's value still wins inside the job.

## Debugging

```bash
# See real-time logs
madengine run --tags pyt_huggingface_bert --live-output

# Keep the container running for inspection
madengine run --tags pyt_huggingface_bert --keep-alive

# Rebuild the Docker images from scratch
madengine run --tags pyt_huggingface_bert --clean-docker-cache

# Verbose logging and live output together
madengine run --tags pyt_huggingface_bert --verbose --live-output

# Set up the container but do not run the model; then run it by hand
madengine run --tags pyt_huggingface_bert --skip-model-run --keep-alive
docker exec -it <container name> bash
```

With `--skip-model-run --keep-alive`, madengine prints the command to run the model by hand
(`cd <model dir> && <script> <args>`).

**Warning:** the root README says that with `--keep-alive` you must stop and remove the container
yourself before you run the same model again. Current madengine also removes an existing container
with the same name before it starts a new one. Either way, a kept container keeps running until
you remove it. When madengine keeps a container, it prints the `docker exec`, `docker stop` and
`docker rm` commands for it. To find and remove kept containers:

```bash
docker ps -a --filter name=container_
docker stop -t 1 <container name>
docker rm -f <container name>
```

`--keep-alive`, `--keep-model-dir` and `--skip-model-run` apply to local Docker runs only.
madengine ignores them, with a warning, on SLURM and Kubernetes.

Other things to check when a run fails:

- **The logs.** `*.build.live.log` holds the Docker build; `*.run.live.log` holds the container
  run. They are written to the directory you ran madengine from.
- **What was selected.** `madengine discover --tags <tags> --verbose` prints the selected cards
  with their resolved fields.
- **False failures.** If a run is marked `FAILURE` only because its log contains a harmless string
  such as `RuntimeError:`, configure madengine's log error pattern scan with the
  `log_error_pattern_scan` and `log_error_benign_patterns` keys. See madengine's
  [configuration guide](https://github.com/ROCm/madengine/blob/main/docs/configuration.md#run-phase-log-error-pattern-scan).

## Where results go

Everything is written to the directory you run madengine from.

| File | What it is |
|---|---|
| `perf.csv` | The results table. One row per model, or one row per result line for a card with `multiple_results`. Failed runs are `FAILURE` rows, and cards skipped by `skip_gpu_arch` are `SKIPPED` rows. The file is created if it is missing and appended to otherwise. Change the name with `-o`. |
| `perf_entry.csv`, `perf_entry.json` | The row or rows from the latest model. `--cleanup-perf` removes these after the run. |
| `perf_super.json`, `perf_super.csv` | An extended results record that `--cleanup-perf` keeps. |
| `build_manifest.json` | The build manifest. |
| `*.build.live.log`, `*.run.live.log`, `*.live.log` | Build and run logs per card. |
| `slurm_results/` | The job output of a SLURM run. See [Running multinode workloads](multinode-running.md). |

The root README lists a `perf.html` report as part of a run. With current madengine, create HTML
reports with the `report` command:

```bash
# One CSV to HTML; writes perf.html next to perf.csv
madengine report to-html --csv-file-path perf.csv

# All CSV files in a directory to one email-ready HTML file
madengine report to-email --directory . --output run_results.html
```

madengine exits with `0` on success, `2` when an image build failed, `3` when a model run failed,
and `4` for invalid arguments or configuration, so scripts and CI can test the result. The column
schema of `perf.csv` and the status rules are in [Benchmarks and results](benchmarks-and-results.md).

## Environment variables inside the container

madengine passes system information into the container as environment variables. Model scripts
read them to adapt to the host.

| Variable | Description |
|---|---|
| `MAD_SYSTEM_GPU_ARCHITECTURE` | Host GPU architecture, for example `gfx942`. |
| `MAD_SYSTEM_NGPUS` | Number of GPUs in the system. |
| `MAD_RUNTIME_NGPUS` | Number of GPUs this run was given. |
| `MAD_SYSTEM_HIP_VERSION` | Host HIP version. |
| `MAD_SYSTEM_GPU_PRODUCT_NAME` | Host GPU product name. |
| `MAD_MODEL_NAME` | The card's name. |
| `MAD_OUTPUT_CSV` | The card's `multiple_results` file name, when the card has one. |

The root README also lists `MAD_MODEL_NUM_EPOCHS` (training epochs) and `MAD_MODEL_BATCH_SIZE`
(batch size) as model variables. madengine does not set these itself. Scripts read them if you set
them, and fall back to their own defaults otherwise. For example
[`scripts/huggingface_bert/run.sh`](../scripts/huggingface_bert/run.sh) picks a batch size per GPU
architecture unless `MAD_MODEL_BATCH_SIZE` is set. Set your own container variables with
`docker_env_vars`:

```bash
madengine run --tags pyt_huggingface_bert \
  --additional-context '{"docker_env_vars": {"MAD_MODEL_BATCH_SIZE": "8"}}'
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `models.json file not found`, or a model you expect is missing | Run from the MAD root (`cd /path/to/MAD`), or set `MODEL_DIR`. Then check with `madengine discover --tags <tag>`. |
| `No models found corresponding to the given tag` | The tag matches no card name or tag. List everything with `madengine discover`. |
| Docker permission denied | `sudo usermod -aG docker $USER` then `newgrp docker`. |
| GPU not detected | Run `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA), and the Docker GPU test in [Prerequisites](#prerequisites). |
| Build fails | Check the daemon with `docker ps`, read `*.build.live.log`, and rebuild with `madengine build --tags <card> --clean-docker-cache --verbose`. |
| `import` errors from madengine | Activate the virtual environment you installed into and check `pip list | grep madengine`. |
| A card is skipped with `SKIPPED` | Its `skip_gpu_arch` names your GPU. Use `--disable-skip-gpu-arch` only if you know the card works there. |

## Next steps

- Add your own workload: [Adding a model](adding-a-model.md).
- Move to multiple nodes: [Multinode overview](multinode-overview.md), then
  [Running multinode workloads](multinode-running.md).
- See every setting and its precedence: [Configuration](configuration.md).
- madengine's own [usage guide](https://github.com/ROCm/madengine/blob/main/docs/usage.md) and
  [CLI reference](https://github.com/ROCm/madengine/blob/main/docs/cli-reference.md).
