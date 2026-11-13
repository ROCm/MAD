# MAD - Model Automation and Dashboarding

## Overview

MAD is a platform that consists of curated list of AI models that allow us to run on various GPU architectures seamlessly while tracking performance and generating dashboards for insights.

## Blueprints

This repository provides state-of-the-art deep learning recipes for training, inference and easy deployment on AMD Instinct GPUs.
Below are blueprints of supported models along with their documentation.

| Blueprint | Description | Models |
|-----------|-------------|--------|
| [**Kimi-K3 inference (vLLM / SGLang / ATOM)**](benchmark/kimi_k3/README.md) | Kimi-K3 (2.8T) day-0 inference on MI350X/MI355X across three frameworks | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) |
| [**xDiT diffusion inference**](benchmark/xdit/README.md) | Diffusion Transformer inference using xDiT | FLUX.1, FLUX.1 Kontext, FLUX.2, FLUX.2 Klein, HunyuanVideo, HunyuanVideo 1.5, LTX-2, Stable Diffusion 3.5, Wan 2.1, Wan 2.2, Z-Image Turbo |
| [**JAX training (MaxText + MaxDiffusion)**](benchmark/jax_maxtext/README.md) | Train LLMs and diffusion models on AMD Instinct GPUs using JAX, via Primus | Llama 2 7B/70B, Llama 3/3.1 8B/70B, Llama 3.1 405B, Llama 3.3 70B, DeepSeek-V2-lite 16B, Gemma4 26B/31B, Mixtral-8x7B, Qwen3 14B/30B-A3B, FLUX.1-dev, WAN 2.1 1.3B/14B |
| [**vLLM inference**](benchmark/vllm/README.md) | LLM Inference with vLLM on AMD Instinct GPUs | DeepSeek-R1, gpt-oss-20b/120b, Kimi-K3, Llama-2-70b, Llama-3.1-8b/405b, Llama-3.3-70b, Llama-4-Scout/Maverick, Mixtral-8x7b/8x22b, Phi-4, Qwen3-8b/32b/30b-a3b/235b-a22b |
| [**SGLang inference**](benchmark/sglang/README.md) | LLM Inference with SGLang on AMD Instinct GPUs | DeepSeek-R1-Distill-Qwen-32B, Kimi-K3 |
| [**PyTorch training**](benchmark/pytorch_train/README.md) | Train LLMs on AMD Instinct GPUs using AMD's Primus | Llama 2/3/3.1/3.2/3.3/4, GPT-OSS 20B/120B, Qwen2/2.5/3, Flux, SDXL, DLRM, and others |
| [**PyTorch inference**](benchmark/pytorch_inference/README.md) | Inference recipes for Multimodal, video and vision transformer models | Mochi video, Chai-1, CLIP (ViT-B-32), Wan2.1, HunyuanVideo |
| [**Megatron-LM training**](benchmark/megatron_lm/README.md) | Train LLMs on AMD Instinct GPUs using ROCm Megatron-LM | Llama 2 7B/70B, Llama 3/3.1 8B/70B, Llama 3.3 70B, DeepSeek-V2-lite, DeepSeek-V3, Mixtral 8x7B/8x22B, Qwen 2.5 7B/72B |
| [**MPT-30B training (llm-foundry)**](benchmark/llm-foundry/mpt-30b/README.md) | LLM Training for Mosaic Pretrained Transformer (MPT) models using llm-foundry | MPT-30B |
| [**PyTorch PEFT/FSDP fine-tuning**](scripts/pytorch_train/HF_PEFT_FSDP/README.md) | Finetuning a HF model with LoRA approach & FSDP strategy | Llama-2-70b-chat-hf |
| [**Large EP microbenchmark**](scripts/large-ep-benchmark/README.md) | MoE Large Expert Paralellism with MoRI-EP & DeepEP communication microbenchmarks | no specific models |
| [**vLLM disaggregated P/D inference**](scripts/vllm_dissag/README.MD) | Distributed Inference P/D disaggregation with vLLM (Default, MoRI EP, DeepEP) | DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-5layer, amd-Llama-3.3-70B-Instruct-FP8-KV, Llama-3.1-405B-Instruct-FP8-KV, gpt-oss-120b |
| [**SGLang disaggregated P/D inference**](scripts/sglang_disagg/README.MD) | Distributed Inference P/D disaggregation with SGLang (MoRI IO, Mooncake) | Llama-3.1-8B, Qwen3-32B, Llama-3.3-70B-FP8, Llama-3.1-405B-FP8, Mixtral-8x7B, DeepSeek-V3, DeepSeek-R1 |
| [**SGLang disaggregated P/D inference with WideEP/LargeEP**](scripts/sglang_disagg/README.MD) | Distributed Inference P/D disaggregation with SGLang with WideEP/LargeEP | DeepSeek-V3, DeepSeek-R1 |
| [**KVCache Transfer Bench**](scripts/kvcache_transfer_bench/README.md) | Inter-node Transfer Benchmark | no specific models |

## Recipes and status dashboard

[MAD Recipes](https://rocm.github.io/MAD/) is a site generated from this repository by `.github/workflows/dashboard.yml` on every push to `develop`. The top navigation has one page per question:

- **Overview** (home): this release at a glance, and what changed since the last one.
- **Models**: every supported model with its GPU support (MI300X, MI325X, MI350X, MI355X, and MI455X where a config names it) and scale (nodes per run, and whether it needs SLURM), filterable and exportable as CSV. Each model opens a recipe with the commands to run it, either `madengine run` or plain `docker build` / `docker run` (`sbatch` for multi-node models) with no madengine, for each cluster type the model supports: single node, SLURM, or Kubernetes where a manual per-node run is documented. A model that also runs on more nodes under another framework (e.g. vLLM and vLLM disaggregated P/D) links to it. Multi-node-only JAX MaxText configs, which madengine hides unless `JAX_MAXTEXT_INCLUDE_MULTINODE=1`, are listed as opt-in with an experimental recipe.
- **ROCm**, **Roadmap**: the ROCm support matrix and the oncoming models.
- **Releases**: every frozen release keeps the whole site under `/<version>/`, with recipes pinned to that release's commit.

How it is maintained:

- Models, recipes, base images and ROCm versions are read from `scripts/*/models.json`, `scripts/*/get_models_json.py` and `docker/`, so adding a model adds its recipe page.
- **Oncoming models** come from GitHub: open issues labeled `new-model` (use the "Model onboarding" issue form) are the plan, and the PRs that close them (`Closes #<issue>`) or carry the label themselves set the status. Merged work leaves the Roadmap and appears under Models. `python3 tools/dashboard/build.py --roadmap-report` lists model PRs missing the label and issues whose model already landed.
- Everything else (each registry's display name and guide, entries to leave out, ROCm versions that are not in an image tag, roadmap items with no issue yet, which registries get one recipe page (`one_page`) or a manual Kubernetes recipe (`k8s: manual`)) lives in [`dashboard/status.yaml`](dashboard/status.yaml). A new `scripts/<dir>` registry needs an entry there, or its models are listed under the bare directory name. The order of model pages within each framework is set in [`dashboard/priority.yaml`](dashboard/priority.yaml) (`build.py --list-pages` prints the slugs). Both files are public: only add what is cleared for disclosure.
- Hosting: set the repository variable `DASHBOARD_PAGES=true` and Settings > Pages > Source to GitHub Actions. The site rebuilds on pushes to `develop` and every 3 hours; if GitHub issues cannot be read, the run fails and the live site is left as it was, and build warnings appear in the run's job summary. A fork can publish upstream's state by also setting `DASHBOARD_SOURCE_REPO=ROCm/MAD`.
- To cut a release, run `python3 tools/dashboard/build.py --freeze v26.10` on a clean checkout of a commit that is on `ROCm/MAD` `develop` (fetched into one of your remotes), with `scripts/Primus` checked out, and commit `dashboard/releases/v26.10.json`. The frozen recipes check out that commit, so the build refuses a dirty tree or an unpublished commit.
- **Logos and icons**: the header carries the AMD logo (from AMD's ROCm documentation theme, linked to amd.com) and the footer AMD's copyright, legal links and trademark attribution, as on rocm.docs.amd.com. Model-provider logos are Hugging Face organization avatars (the same ones recipes.vllm.ai shows) and framework marks come from Simple Icons (CC0); both are vendored in `dashboard/logos/` (see its `NOTICE.md`), so the site loads no third-party images. Edit `dashboard/icons.yaml` and run `python3 tools/dashboard/fetch_logos.py` to add one. Set `logos: neutral` there (or build with `--logos neutral`) to replace every brand mark with a neutral monogram.
- To preview locally: `pip install -r requirements.txt pyyaml jinja2`, then `python3 tools/dashboard/build.py --out _site` and open `_site/index.html`. Add `--offline` to skip the GitHub and image-registry lookups. The build replaces the `--out` directory and refuses one it did not create. Tests: `python3 -m pytest tools/dashboard`.
- More detail for contributors and coding agents: [AGENTS.md](AGENTS.md#mad-recipes-site-dashboard).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Running Models](#running-models)
  - [Tag Functionality](#tag-functionality)
  - [Timeout Configuration](#timeout-configuration)
  - [Debugging Options](#debugging-options)
- [Recipes and status dashboard](#recipes-and-status-dashboard)
- [Claude Code Integration](#claude-code-integration)
- [Contributing](#contributing)
  - [Adding New Models](#adding-new-models)
  - [Model Configuration](#model-configuration)
  - [Docker Setup](#docker-setup)
  - [Script Implementation](#script-implementation)
- [Environment Variables](#environment-variables)
- [License](#license)

## Prerequisites

- Docker installed and running
- Python 3.9 or higher
- GPU drivers (AMD ROCm or NVIDIA CUDA)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MAD
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a model**:
   ```bash
   madengine run --tags pyt_huggingface_bert
   ```



## Usage Guide

### Running Models

The madengine CLI [ROCm/madengine](https://github.com/ROCm/madengine/) provides a simple interface for running models locally. All models registered in the `scripts/**/models.json` registries can be executed on a Docker host to collect performance results.

**Please note that support of running models using tools/run_models.py is no longer recommended, and tools/run_models.py will be removed from MAD repo soon.**

#### Basic Usage

```bash
madengine run [OPTIONS]
```

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tags TAGS` | Tags to filter models (space-separated) | - |
| `--timeout TIMEOUT` | Timeout in seconds | 7200 (2 hours) |
| `--live-output` | Show real-time output | False |
| `--clean-docker-cache` | Rebuild Docker images without cache | False |
| `--keep-alive` | Keep container running after completion | False |
| `--keep-model-dir` | Preserve model directory after run | False |
| `-o OUTPUT, --output OUTPUT` | Output file for results | - |
| `--log-level LOG_LEVEL` | Set logging level | INFO |

#### Execution Process

For each model, MAD performs the following steps:

1. 🔨 **Build**: Creates Docker image named `ci-$(model_name)`
2. 🚀 **Start**: Launches container named `container_$(model_name)`
3. 📥 **Clone**: Downloads model repository from specified URL
4. ▶️ **Execute**: Runs the model script
5. 📊 **Report**: Generates `perf.csv` and `perf.html`

### Tag Functionality

Tags allow you to run specific subsets of models based on their characteristics:

- **Framework tags**: `pyt`, `tf2`, `ort`
- **Model tags**: `bert`, `gpt2`, `resnet50`
- **Precision tags**: `fp16`, `fp32`
- **Custom tags**: Any tag defined in `models.json`

#### Examples

```bash
# Run a specific model
madengine run --tags pyt_huggingface_bert

# Run all PyTorch models
madengine run --tags pyt

# Run multiple tag combinations
madengine run --tags tf2 bert fp32
```

### Timeout Configuration

Configure execution timeouts at multiple levels:

1. **Default**: 2 hours (7200 seconds)
2. **Model-specific**: Set `timeout` field in `models.json`
3. **Runtime override**: Use `--timeout` command line option

> **Note**: Setting timeout to `-1` disables the timeout entirely.

### Debugging Options

For troubleshooting and development:

```bash
# See real-time logs
madengine run --tags model_name --live-output

# Keep container running for inspection
madengine run --tags model_name --keep-alive

# Rebuild Docker images from scratch
madengine run --tags model_name --clean-docker-cache
```

> ⚠️ **Warning**: When using `--keep-alive`, you must manually stop and remove the container before running the same model again.

## Claude Code Integration

MAD ships agent skills for [Claude Code](https://claude.ai/code), Cursor and Codex CLI that cover benchmarking, profiling, tuning, adding models, reporting, and validation. `AGENTS.md` holds the shared conventions; `CLAUDE.md` adds the Claude Code specifics; [`.claude/README.md`](.claude/README.md) explains how the skills are shared across the three tools.

| Skill | Invocation | What it does |
|-------|-----------|--------------|
| `/mad-benchmark <tag>[,<tag>...]` | Explicit | Build and run a madengine benchmark on AMD GPUs |
| `/mad-profile <tag> [operator\|dispatch\|kernel\|assembly\|pc_sampling\|thread_trace]` | Explicit | Profile at the operator (TraceLens), dispatch (rocm-trace-lite), kernel (rocprofv3 counters and a kernel card), assembly (code-object summary), PC-sampling, or thread-trace (rocprofv3 `--att`) depth |
| `/mad-tune <tag>` | Explicit | Measure-change-measure tuning loop for training or inference |
| `/primus-projection <tag>` | Explicit | Projected memory fit and tokens/s for a Primus training tag |
| `/mad-add-model` | Explicit | Guided model addition: per-directory `models.json` entry, Dockerfile, run.sh, validation |
| `/mad-generate-dockerfile` | Explicit | Render a MAD Dockerfile from a framework template |
| `/mad-report [csv]` | Auto | Summarize or compare benchmark results |
| `/mad-validate-model [model]` | Auto | GPU-free lint of every model registry, its files, and the output contract |

Explicit skills run only when you invoke them (`/name` in Claude Code and Cursor, `$name` in Codex). Long runs go through `.claude/skills/mad-common/mad-run.sh`, which detaches each run and serializes runs on one host with a GPU lock.

Claude Code also has three workflows for larger fan-out jobs (Cursor and Codex users can pass several tags to `/mad-benchmark` instead): `mad-benchmark-sweep` (a matrix across tags), `mad-tune-search` (operator → dispatch → kernel → assembly, then clean measurements repeated `--repeats` times, 3 by default), and `mad-optimize` (one model, a dispatch profile and a kernel card, then GEAK `kernel_workflow`, kept only when the latency gate passes; see `docs/mad-optimize.md`). `mad-tune-search` and `mad-optimize` stop a counter run that stalls with the GPU idle and retry it once with `--serialize`. All three plan by default and only run on GPUs with `--execute`. Pass `--expect` on `mad-tune-search` to attach Primus projection numbers for a Primus training model; that step stays off otherwise.

## Contributing

### Adding New Models

The fastest way to add a model is the `mad-add-model` agent skill (works in
Claude Code, Cursor and Codex CLI) — see
[`.claude/README.md`](.claude/README.md) for the available skills, or follow
the manual steps below.

Follow these steps to add a new model to the MAD repository:

#### Step 1: Create Workload Name

Follow the naming convention: `{framework}_{project}_{workload}`

**Examples**:
- `tf2_huggingface_gpt2`
- `pyt_torchvision_resnet50`
- `ort_onnx_bert`

#### Step 2: Model Configuration

Models are registered **per directory**: add the entry to the `models.json` in
the model's script directory, `scripts/<dir>/models.json` (create
`scripts/<name>/models.json` for a model with its own run script, or add to an
existing one such as `scripts/vllm/models.json` when reusing its runner). The
root `models.json` stays `[]`; madengine only requires it to exist. Paths in an
entry are relative to the directory holding that `models.json`:

```json
{
  "name": "pyt_huggingface_bert",
  "url": "https://github.com/huggingface/transformers",
  "dockerfile": "../../docker/pyt_huggingface",
  "scripts": "run.sh",
  "n_gpus": "-1",
  "owner": "mad.support@amd.com",
  "training_precision": "",
  "tags": [
    "pyt",
    "bert"
  ],
  "args": ""
}
```

madengine names it `<dir>/<name>` (e.g. `huggingface_bert/pyt_huggingface_bert`),
which is also the `model` value in `perf.csv`. This is the real entry in
`scripts/huggingface_bert/models.json`. Names must be unique across all
directories.

#### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Unique model identifier, `{framework}_{project}_{workload}` |
| `dockerfile` | ✅ | Dockerfile path prefix relative to the registry directory; `.ubuntu.amd.Dockerfile` is appended |
| `scripts` | ✅ | Run script relative to the registry directory (e.g. `run.sh`) |
| `n_gpus` | ✅ | Number of GPUs as a string (`"-1"` for all available) |
| `owner` | ✅ | Contact email |
| `tags` | ✅ | List of tags for categorization |
| `args` | ✅ | Additional script arguments (may be `""`) |
| `url` | ❌ | Repository URL cloned into the container (`""` if none) |
| `training_precision` | ❌ | Precision level (fp16, bf16, fp32, etc.) |
| `data` | ❌ | Optional data path |
| `timeout` | ❌ | Model-specific timeout override in seconds |
| `multiple_results` | ❌ | CSV file for multiple results |

#### Step 3: Docker Setup

Create `docker/<name>.ubuntu.amd.Dockerfile` (reuse an existing Dockerfile of
the same stack when you can). The `# CONTEXT` header must be the first line:

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

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

#### Step 4: Script Implementation

Create `run.sh` in the model's script directory (`scripts/<dir>/`). Only that
directory is copied into the container, so everything the script calls must
live in it:

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

#### Performance Reporting

**Single Result Format**:
```python
print(f"performance: {throughput} examples/sec")
```

**Multiple Results Format**:
Create a CSV file with columns: `models,performance,metric`

```csv
models,performance,metric
model_1,156.7,examples/sec
model_2,89.3,tokens/sec
```

#### Step 5: Validate

GPU-free checks, then a real run on a GPU host:

```bash
python3 tools/validation/model_config_validator.py --model <name>
madengine discover --tags <name>          # must list <dir>/<name>
madengine run --tags <name> --live-output
```

#### Step 6: Recipes site

The model appears on [MAD Recipes](#recipes-and-status-dashboard) once it
merges. If you created a new `scripts/<dir>/` registry, add it under
`frameworks:` in [`dashboard/status.yaml`](dashboard/status.yaml) (display
name, `workload`, `readme`); otherwise the site lists its models under the
bare directory name. If a "Model onboarding" issue tracks the model, put
`Closes #<issue>` in the PR description so the Roadmap shows its progress.

## Environment Variables

### System Variables

MAD provides system information through environment variables:

| Variable | Description |
|----------|-------------|
| `MAD_SYSTEM_GPU_ARCHITECTURE` | Host GPU architecture |
| `MAD_RUNTIME_NGPUS` | Available GPU count |

### Model Variables

Runtime model configuration:

| Variable | Description |
|----------|-------------|
| `MAD_MODEL_NAME` | Model name from `models.json` |
| `MAD_MODEL_NUM_EPOCHS` | Training epochs |
| `MAD_MODEL_BATCH_SIZE` | Batch size |

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2025 Advanced Micro Devices, Inc. All Rights Reserved.

