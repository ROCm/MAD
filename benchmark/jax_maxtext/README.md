# Training Performance Validation with ROCm Maxtext-jax Training Docker on the AMD Instinct Accelerators

## Overview

MaxText framework for ROCm is a specialized fork from upstream MaxText, designed to enable training of large language model (LLM) on AMD GPUs. By leveraging AMD Instinct™ MI300X and MI355X GPUs, MaxText delivers great scalability, performance, and resource utilization for AI workload. See the GitHub repository at [ROCm/maxtext](https://github.com/ROCm/maxtext/).

AMD provides a ready-to-use Docker image for AMD Instinct MI300X and MI355X GPUs containing essential components, including Jax, XLA, ROCm libraries, and MaxText utilities.

> **Canonical reference:** For the full Primus JAX MaxText training guide — including detailed environment setup, all supported models, multi-node networking, and the complete `primus-cli` reference — see the [Primus JAX MaxText training documentation](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md). This README focuses on the MAD integration layer and quick-start workflows.

> [!NOTE]
> Shardy is the partitioning system in JAX. The v26.6 Docker image ships JAX 0.11.0, so you now have to set `shardy=True` during the training run. You might get related errors if it's not configured correctly. See the [migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html) for more details.

> [!NOTE]
> There is a discrepancy in loss curve if you set `packing=false`. It converges at a slightly higher value than previous docker images. We can achieve the same convergence as past docker images if you set `NVTE_CK_USES_FWD_V3=0`. (i.e. using FAv2 for forward instead of FAv3). This is being tracked and will be addressed in a future release.

> [!NOTE]
> On MI355X (gfx950), RCCL's WarpSpeed feature (`RCCL_WARP_SPEED_AUTO`) — a gfx950-only optimization that is enabled by default in gfx950 builds — can cause NaN losses during training. To avoid this, set `RCCL_WARP_SPEED_AUTO=0`. For the MAD-integrated benchmarking system, this is applied automatically by the Primus MaxText backend when a gfx950 (MI355X) device is detected, so the benchmark scripts handle it for you. If you launch training manually on MI355X, export `RCCL_WARP_SPEED_AUTO=0` yourself. This variable is a no-op on MI300X (gfx942).


| Software component | Version                   |
| ------------------ | ------------------------- |
| ROCm               | 7.14.0                    |
| Jax                | 0.11.0                    |
| Python             | 3.12.3                    |
| Transformer Engine | 2.17.0+rocm7.14.0.50a84ad |
| hipBLASLt          | 1.4.1+cd957402            |


## Supported features and models

MaxText supports the following key features to train large language models efficiently:

- Transformer Engine (TE)
- Flash Attention (FA) 3, with or without input sequence packing
- GEMM tuning
- Multi-node Support
- NANOO FP8 (for MI300X) or FP8 (for MI355X)

The following models are pre-optimized for performance on the AMD Instinct MI300X and MI355X accelerators.

- Llama 2 7B
- Llama 2 70B
- Llama 3/3.1 8B
- Llama 3/3.1 70B
- Llama 3.3 70B
- DeepSeek-V2-lite (16B)
- Gemma4 26B
- Gemma4 31B
- Mixtral-8x7B
- Qwen3 14B
- Qwen3 30B-A3B

Note: Some models, such as Llama 3, require an external license agreement through a third party (for example, Meta).

## System validation

If you have already validated your system, skip this step. Otherwise, please complete the following [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html#train-a-model-system-validation) to set up your system before starting training.

## Environment setup

This Docker image is optimized for specific model configurations outlined below. Performance can vary for other training workloads, as AMD doesn't validate configurations and run conditions outside those described.

For multi-node training, `primus-cli` handles node discovery, RDMA interface
selection, and environment variable propagation automatically via its Slurm mode.
If you need to customize networking (for example, selecting specific RDMA devices
or overriding the socket interface), see the
[Primus multi-node networking guide](https://github.com/AMD-AGI/Primus/blob/main/docs/04-technical-guides/multi-node-networking.md).

> [!NOTE]
> The only models supported in this workflow are those listed in the above section.
> This container is optimized for the model configurations described below; other
> configurations and run conditions are not validated by AMD.

### Quick start (single-node, MAD-integrated)

```sh
# 1. Clone MAD and install dependencies
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt

# 2. Fetch Primus (required before discovery or docker build)
bash tools/fetch_primus.sh

# 3. Discover available models (auto-filters by your GPU arch)
madengine discover --tags maxtext        # MaxText models
madengine discover --tags maxdiffusion   # MaxDiffusion models

# 4. Run all MaxText models
export MAD_SECRETS_HFTOKEN="<your Hugging Face token>"
madengine run --tags maxtext --live-output --timeout 14400
```

For standalone (no MAD) or multi-node usage, see the sections below.

Users have three paths to reproduce the benchmark results:

- [MAD-integrated benchmarking](#mad-integrated-benchmarking) — recommended, auto-discovers models and manages Docker
- [Standalone benchmarking](#standalone-benchmarking) — run training manually inside a Docker container
- [Primus benchmarking](#using-primus-cli-to-run-training-jobs-with-jax-maxtext-backend) — use primus-cli directly

Jax MaxText has also been integrated into [Primus](https://github.com/AMD-AGI/Primus), which supports multiple backends including Megatron-LM, TorchTitan, and JAX MaxText, alongside ROCm-optimized components. MAD launches all JAX MaxText training through Primus: the MAD-integrated path uses the `scripts/jax-maxtext/run.sh` wrapper around Primus (`examples/run_pretrain.sh` with `BACKEND=MaxText`), and you can also drive `primus-cli` directly (see [Using primus-cli](#using-primus-cli-to-run-training-jobs-with-jax-maxtext-backend)).

> [!NOTE]
> There is no single Primus image that covers every backend. `rocm/primus:*` ships the torch/megatron/torchtitan stack and does **not** include JAX. JAX MaxText runs on the dedicated `rocm/jax-training:maxtext-v26.6` image, which is what MAD builds from `docker/primus_maxtext.ubuntu.amd.Dockerfile`. Primus is available as a git submodule (`git submodule update --init scripts/Primus`) or via `tools/fetch_primus.sh`, which clones the pinned branch into the same gitignored `scripts/Primus` path — see below. The `scripts/jax-maxtext/` launcher and metric parser are MaxText-only (no Megatron/TorchTitan logic).

## MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository and install the required
packages on the host machine. Primus must be checked out into `scripts/Primus` **before
discovery or build**, since the JAX models are discovered from its example configs and both
`primus_`* images bake the repo into the image. You can either initialize the git submodule
(`git submodule update --init scripts/Primus`) or use `tools/fetch_primus.sh`.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt

# Check Primus out into scripts/Primus. Idempotent, so it is safe to re-run.
bash tools/fetch_primus.sh
```

`tools/fetch_primus.sh` clones the pinned branch (`main`). Override `PRIMUS_URL`,
`PRIMUS_REF`, or `PRIMUS_DIR` for a fork, a different branch or commit, or another location.

> [!NOTE]
> Cloning with `--recursive` is **not required** — `tools/fetch_primus.sh` or
> `git submodule update --init scripts/Primus` are sufficient. Both Docker images take their
> framework from the base image — `/workspace/maxtext` and `/workspace/maxdiffusion`, each at
> the same commit as Primus's pin — and pin `MAXTEXT_PATH` / `MAXDIFFUSION_PATH` to it.
> Primus's own `third_party/` submodules are not needed for MAD builds; the base image's
> pre-patched copies are used instead.

This step is **not** automatic: the checkout has to exist in the docker build context
before any image is built, so neither the dockerfiles nor `scripts/jax-*/run.sh` (which
runs inside the container) can do it for you. If `scripts/Primus` is missing, discovery
finds **zero** JAX models and prints a warning naming this script, rather than failing
in a way `madengine run` reports.

In CI pipelines, set `MAD_AUTO_FETCH_PRIMUS=1` to have discovery fetch Primus
automatically when the checkout is absent (off by default):

```sh
MAD_AUTO_FETCH_PRIMUS=1 madengine run --tags maxtext --live-output --timeout 14400
```

JAX MaxText models are **auto-discovered** from the Primus MaxText experiment configs
(`scripts/Primus/examples/maxtext/configs/<DEVICE>/<config>.yaml`). madengine walks every
`scripts/<dir>/` directory at discovery time and, for any directory containing a
`get_models_json.py`, calls its `list_models()` to get one virtual model per config — this
happens for `scripts/jax-maxtext/get_models_json.py` unconditionally, with no entry needed in
the root `models.json` (there is none). A `jax-maxtext/default` model is also always
registered, pointing at the same fallback config `scripts/jax-maxtext/run.sh` uses when no
`--config_path` is given — a stable name that doesn't require knowing a specific config.

Discovered tags follow the pattern `jax-maxtext/maxtext_<DEVICE>_<config>` (the `jax-maxtext/`
prefix is the `scripts/jax-maxtext` directory), e.g.
`jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain` or
`jax-maxtext/maxtext_MI355X_llama2_7B-fp8-pretrain`. All of them build the
`docker/primus_maxtext` image and run through `scripts/jax-maxtext/run.sh`. Each model also
carries tags (`maxtext`, `jax`, `<DEVICE>`, `<config>`, `<precision>`) so you can select a
single model by its full name or a group by a shared tag.

List the available models with madengine discovery:

```sh
madengine discover --tags maxtext          # all MaxText models
madengine discover --tags maxdiffusion     # all MaxDiffusion models
madengine discover --tags jax              # all JAX models (MaxText + MaxDiffusion)
madengine discover --tags nanoo_fp8        # all nanoo_fp8 (MI300X quantized) models
```

> [!NOTE]
> On MI300X machines, madengine automatically skips MI355X models (and vice versa) via
> the `skip_gpu_arch` field — you do not need to add `MI300X` or `MI355X` to the tags.

Run all MaxText models, all MaxDiffusion models, or both at once:

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
madengine run --tags maxtext --live-output --timeout 14400        # all MaxText models
madengine run --tags maxdiffusion --live-output --timeout 14400   # all MaxDiffusion models
madengine run --tags jax --live-output --timeout 14400            # all JAX models (MaxText + MaxDiffusion)
```

Run a single model by its full discovered name:

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
madengine run --tags jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain --keep-model-dir --live-output --timeout 28800
```

Or the nanoo_fp8 quantized Llama 2 7B on MI300X:

```sh
madengine run --tags jax-maxtext/maxtext_MI300X_llama2_7B-nanoo_fp8-pretrain --keep-model-dir --live-output --timeout 28800
```

> [!NOTE]
> `tools/run_models.py` remains available as a drop-in alternative to `madengine run` for the same `--tags`.

MAD launches a Docker container named `container_ci-<mad_model>`. Performance metrics
(`tokens_per_second`, `tflops`) are parsed from the training log by
`scripts/jax-maxtext/extract_maxtext_perf.py` and collected in:

```sh
~/MAD/perf.csv
```

#### Available models

Model tags are generated from the Primus MaxText configs for each device, so the exact
list tracks whatever configs ship in your `scripts/Primus` checkout. List the live
set via madengine discovery (`scripts/jax-maxtext/get_models_json.py`) or by
browsing `scripts/Primus/examples/maxtext/configs/`.

Every listed model has a bf16 variant (`jax-maxtext/maxtext_<DEVICE>_<model>-bf16-pretrain`). Quantization is
**device-specific**: MI300X uses **NANOO FP8** (`-nanoo_fp8`) and MI355X uses **FP8** (`-fp8`)
— there is no plain-fp8 on MI300X and no nanoo_fp8 on MI355X. Every model has a bf16
variant; the columns below show which quantized variant is also available:


| Model            | MI300X (bf16 + …) | MI355X (bf16 + …) |
| ---------------- | ----------------- | ----------------- |
| Llama 2 7B       | `-nanoo_fp8`      | `-fp8`            |
| Llama 2 70B      | `-nanoo_fp8`      | `-fp8`            |
| Llama 3/3.1 8B   | `-nanoo_fp8`      | `-fp8`            |
| Llama 3/3.1 70B  | bf16 only         | `-fp8`            |
| Llama 3.3 70B    | bf16 only         | `-fp8`            |
| DeepSeek-V2-lite | `-nanoo_fp8`      | `-fp8`            |
| Gemma4 26B       | `-nanoo_fp8`      | `-fp8`            |
| Gemma4 31B       | `-nanoo_fp8`      | `-fp8`            |
| Mixtral-8x7B     | `-nanoo_fp8`      | `-fp8`            |
| Qwen3 14B        | `-nanoo_fp8`      | `-fp8`            |
| Qwen3 30B-A3B    | `-nanoo_fp8`      | `-fp8`            |


Example tags: `jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain`, `jax-maxtext/maxtext_MI300X_llama2_7B-nanoo_fp8-pretrain`, `jax-maxtext/maxtext_MI355X_llama2_7B-fp8-pretrain`.
(MI350X/MI325X map to MI355X/MI300X configs respectively.)

> [!NOTE]
> Multi-node-only models (Llama 3.1 405B, Grok-1, Mixtral-8x22B) are excluded from
> single-node `jax-maxtext/` discovery. Set `JAX_MAXTEXT_INCLUDE_MULTINODE=1` to include them.

#### MaxDiffusion models

MaxDiffusion models are discovered the same way as MaxText — auto-discovered from
`scripts/Primus/examples/maxdiffusion/configs/<DEVICE>/<config>.yaml` by
`scripts/jax-maxdiffusion/get_models_json.py`, tagged
`jax-maxdiffusion/maxdiffusion_<DEVICE>_<config>` — and run through
`scripts/jax-maxdiffusion/run.sh` with the `docker/primus_maxdiffusion` image. A
`jax-maxdiffusion/default` model is also always registered, mirroring
`jax-maxtext/default` above. Use `--tags maxdiffusion` to run all of them:

```sh
madengine run --tags maxdiffusion --live-output --timeout 14400
```


| Model        | MI300X | MI355X |
| ------------ | ------ | ------ |
| FLUX.1-dev   | bf16   | bf16   |
| WAN 2.1 1.3B | bf16   | bf16   |
| WAN 2.1 14B  | bf16   | bf16   |


Example tags: `jax-maxdiffusion/maxdiffusion_MI300X_flux_dev-pretrain`, `jax-maxdiffusion/maxdiffusion_MI300X_wan2.1_1.3b-pretrain`.

## Standalone benchmarking

Download and launch the Docker image

Use the following command to pull the Docker image from Docker Hub.

```
docker pull rocm/jax-training:maxtext-v26.6
```

### Single Node Training examples

#### Setup

> [!NOTE]
> Please adjust the following variables based on your environment.

Export variables

- MAD_SECRETS_HFTOKEN is your HuggingFace token to access models, tokenizers, data. See this [page](https://huggingface.co/docs/hub/en/security-tokens) for more info.
- HF_HOME is where huggingface_hub will store local data, please refer to [Huggingface cli Document](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-download) on how to download the data. If you already have downloaded/cached huggingface artifacts, set this variable to that path. Downloaded files typically get cached to a place like this: `~/.cache/huggingface`.

```
export MAD_SECRETS_HFTOKEN=<Your HuggingFace token>
export HF_HOME=<Location of saved/cached HuggingFace models>
```

Launch the Docker container.

```bash
docker run -it \
  --device /dev/dri --device /dev/kfd \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh \
  -v $HF_HOME:/hf_cache -e HF_HOME=/hf_cache \
  -e MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
  --shm-size 64G --name training_env \
  rocm/jax-training:maxtext-v26.6
```

Execute the training_env container (optional if not already in the container)

```
docker start training_env
docker exec -it training_env bash
```

Inside the container, the Primus repository (with the MaxText backend) is available at
`/workspace/Primus`. Run training with `primus-cli`; **direct** mode runs in the current
container. Configs live under `examples/maxtext/configs/<DEVICE>/` where `<DEVICE>` is
`MI300X` or `MI355X`.

```bash
cd /workspace/Primus

# Unquantized (bf16), e.g. Llama 2 7B on MI300X
# Note: RCCL_WARP_SPEED_AUTO=0 is auto-set by Primus on MI355X (gfx950).
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-bf16-pretrain.yaml
```

For quantized training, replace `-bf16-` in the config name with `-fp8-` (MI355X)
or `-nanoo_fp8-` (MI300X):

```bash
# nanoo_fp8 on MI300X
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-nanoo_fp8-pretrain.yaml

# fp8 on MI355X
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama2_7B-fp8-pretrain.yaml
```

The same pattern applies to every supported model (`llama2_70B`, `llama3_8B`, `llama3_70B`,
`llama3.3_70B`, `deepseek_v2_16B`, `gemma4_26B`, `gemma4_31B`,
`mixtral_8x7B`, `qwen3_14B`, `qwen3_30B_A3B`). See the
[Using primus-cli](#using-primus-cli-to-run-training-jobs-with-jax-maxtext-backend)
section for container and Slurm modes.

### Multi-Node Training examples

Multi-node training is launched through the unified `primus-cli` in Slurm mode.
The standalone MAD multinode launcher and the per-model `env_scripts/*.yml`
configuration files have been retired; model/precision/parallelism settings now
live in the Primus MaxText experiment configs under
`examples/maxtext/configs/<DEVICE>/<model>-<precision>-pretrain.yaml`
(bundled in the `rocm/jax-training` image at `/workspace/Primus`).

See the [Using primus-cli](#using-primus-cli-to-run-training-jobs-with-jax-maxtext-backend)
section below for direct, container, and Slurm examples. The general form for a
multi-node run is:

```bash
# From /workspace/Primus (or a cloned Primus checkout)
# RCCL_WARP_SPEED_AUTO=0 is auto-set by Primus on MI355X (gfx950).
./primus-cli --config my_maxtext_config.yaml slurm srun -N <NUM_NODES> \
  -- train pretrain --config examples/maxtext/configs/<DEVICE>/<model>-<precision>-pretrain.yaml
```

where `<DEVICE>` is `MI300X` or `MI355X`, `<model>` is one of the MaxText
configs (e.g. `llama2_7B`, `llama2_70B`, `llama3_8B`, `llama3_70B`,
`gemma4_26B`, `gemma4_31B`, `mixtral_8x7B`,
`qwen3_14B`, `qwen3_30B_A3B`), and `<precision>` is `bf16`,
`fp8` (MI355X), or `nanoo_fp8` (MI300X), e.g. `llama2_7B-bf16-pretrain.yaml`.

## Using primus-cli to run training jobs with Jax MaxText backend

**Clone the Primus repository**

```
git clone https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout main
git submodule update --init third_party/maxtext/
```

**Run the training job with primus-cli**  

For detailed usage of primus-cli, please refer to [Primus CLI User Guide](https://github.com/AMD-AGI/Primus/blob/main/docs/cli/PRIMUS-CLI-GUIDE.md).

Here are some examples of using primus-cli to run training jobs with Jax MaxText backend.

Direct Mode: Running the training directly on current host or within an existing docker container.

```bash
# RCCL_WARP_SPEED_AUTO=0 is auto-set by Primus on MI355X (gfx950); no action needed.
./primus-cli direct -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-bf16-pretrain.yaml
```

Container Mode: execute in Docker/Podman containers. You **must** pass `--image` because
the default Primus image (`rocm/primus`) is the PyTorch stack and does not include JAX.

```bash
./primus-cli container --image rocm/jax-training:maxtext-v26.6 \
  -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-bf16-pretrain.yaml
```

Slurm Mode: execute distributed training on a Slurm cluster

```bash
# Use a custom config file, where you can specify the docker image and set environment variables.
./primus-cli --config my_maxtext_config.yaml slurm srun -N 8 \
  -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-bf16-pretrain.yaml
```

## Profiling with JAX XPlane Profiler

MaxText has built-in XPlane profiling support via JAX's profiler. Traces capture GPU kernel timelines, RCCL collectives, HLO graphs, and more. The output can be viewed in TensorBoard's Trace Viewer or analyzed with TraceLens.

### Key MaxText Profiler Flags

The following MaxText config keys control profiling:

```
profiler=xplane                    # Use xplane format (produces .xplane.pb files)
skip_first_n_steps_for_profiler=2  # Skip compilation/warmup steps
profiler_steps=5                   # Number of steps to profile
upload_all_profiler_results=True   # Save all GPU profiles (not just GPU0)
```

**Choosing step counts:**

- `steps` should be > `skip_first_n_steps_for_profiler` + `profiler_steps` (e.g., `steps=12` with skip=2, profile=5 gives 5 warmup + 5 profiled + 2 cooldown)
- `skip_first_n_steps_for_profiler=2` skips step 0 (compilation) and step 1 (warmup)
- `profiler_steps=5` is typically enough; more steps = larger `.xplane.pb` files

### Profiling with MAD/madengine

The Primus MaxText experiment configs (`examples/maxtext/configs/<DEVICE>/<model>-<precision>-pretrain.yaml` in `/workspace/Primus`) already include a `profiler` key under `overrides` (set to `""` by default). To enable profiling when running through MAD or madengine, edit the `overrides` block of the config for your model and set the profiler fields:

```yaml
profiler: "xplane"
skip_first_n_steps_for_profiler: 2
profiler_steps: 5
upload_all_profiler_results: True
steps: 12
```

Then run the benchmark as usual:

```bash
# Via madengine
madengine run --tags jax-maxtext/maxtext_MI300X_llama3_8B-bf16-pretrain --keep-model-dir --live-output --timeout 28800

# Or via run_models.py
python3 tools/run_models.py --tags jax-maxtext/maxtext_MI300X_llama3_8B-bf16-pretrain --keep-model-dir --live-output --timeout 28800
```

Profile output will be written under the `base_output_directory` specified in the YAML (see [Output Structure](#output-structure) below). Use `--keep-model-dir` so the container's output directory is preserved after the run.

### Example: Profile a Model Standalone in Docker

```bash
#!/bin/bash
set -e

IMAGE="$1"       # Docker image, e.g. rocm/jax-training:maxtext-v26.6
TAG="$2"         # Short tag for output folder, e.g. v26.6_llama2_7b
PROFILE_DIR="/path/to/profiles/${TAG}"

mkdir -p "${PROFILE_DIR}"

docker run --rm --privileged --network=host \
  --device=/dev/dri --device=/dev/kfd --ipc=host \
  -v "${PROFILE_DIR}:/mnt/profile" \
  "${IMAGE}" bash -c '
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_enable_command_buffer= <your other XLA flags>"
export GPU_MAX_HW_QUEUES=2
# On MI355X (gfx950), disable RCCL WarpSpeed to avoid NaN losses (no-op on MI300X)
export RCCL_WARP_SPEED_AUTO=0

cd /workspace/maxtext

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=profile \
  base_output_directory=/mnt/profile \
  hardware=gpu \
  steps=12 \
  model_name=<your-model> \
  dataset_type=synthetic \
  enable_checkpointing=False \
  enable_goodput_recording=False \
  monitor_goodput=False \
  <your model-specific flags> \
  profiler=xplane \
  skip_first_n_steps_for_profiler=2 \
  profiler_steps=5 \
  upload_all_profiler_results=True
' 2>&1 | tee "${PROFILE_DIR}/run.log"

echo "Profile files:"
find "${PROFILE_DIR}" -name "*.xplane.pb" -o -name "*.trace.json.gz" 2>/dev/null
```

### Output Structure

MaxText writes profiles in TensorBoard format:

```
<base_output_directory>/
└── profile/
    └── tensorboard/
        └── plugins/
            └── profile/
                └── <YYYY_MM_DD_HH_MM_SS>/
                    ├── <hostname>.xplane.pb          # Raw XPlane proto (GPU timelines)
                    ├── <hostname>.trace.json.gz       # Trace viewer data
                    └── *.hlo_proto.pb                 # HLO graphs for each compiled module
```

### Viewing Traces in TensorBoard

```bash
pip install tensorboard tensorboard-plugin-profile

# Point --logdir at the directory containing the tensorboard/ folder
tensorboard --logdir /path/to/profiles/<TAG>/profile --port 6006
```

Navigate to **Profile > Trace Viewer** in the TensorBoard UI.

**Tips:**

- Zoom into a single training step (skip the first profiled step as it may have residual warmup)
- Look at individual GPU streams to see compute/RCCL overlap

### Keeping Profile Files Small

- Use `profiler_steps=5` (not more) to keep `.xplane.pb` under ~100MB
- Too many steps can produce files >500MB that TensorBoard struggles to load
- `enable_checkpointing=False` avoids checkpoint I/O noise in the trace
- `dataset_type=synthetic` eliminates data loading variability

## Profiling with rocprofv3

If you need to collect a trace and the JAX profiler isn't working then you can use rocprofv3 as a temporary workaround like this:

```
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace --output-format pftrace -d ./v3_traces -- python3 app.py
```

- Just replace `python3 app.py` with any command line command that you want to run such as `./primus-cli direct -- train pretrain --config examples/maxtext/configs/MI300X/llama2_7B-bf16-pretrain.yaml` (run from `/workspace/Primus`).
- You can set the directory where you want the .json traces to be saved using `-d <TRACE_DIRECTORY>`
- The resulting traces can be opened in perfetto: [https://ui.perfetto.dev/](https://ui.perfetto.dev/)

## Related documentation

- [Primus JAX MaxText training guide](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md) — canonical reference for environment setup, models, and training options
- [Primus CLI reference](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/cli-reference.md) — full `primus-cli` command-line reference
- [End-to-end training recipes](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/end-to-end-training-recipes.md) — complete config inventory and step-by-step recipes
- [MaxText parameters](https://github.com/AMD-AGI/Primus/blob/main/docs/03-configuration-reference/maxtext-parameters.md) — YAML config field reference
- [Multi-node networking](https://github.com/AMD-AGI/Primus/blob/main/docs/04-technical-guides/multi-node-networking.md) — RDMA, NCCL, and Slurm networking setup
- [Release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md) — full software stack details for each image tag
