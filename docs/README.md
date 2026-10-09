# MAD documentation

MAD (Model Automation and Dashboarding) is a curated list of AI models that run on various GPU
architectures while tracking performance and generating dashboards for insights. It provides deep
learning recipes for training, inference and deployment on AMD Instinct GPUs.

You do not run MAD directly. You run it with **madengine**, a command-line tool from
[ROCm/madengine](https://github.com/ROCm/madengine). madengine reads the model definitions in this
repository, builds a Docker image for each model, runs the model's script inside a container (or
submits it to a SLURM cluster), and writes the results to a CSV file.

This folder is a guide from first contact to writing your own workloads. Start at step 1 of the
learning path and stop when you have what you need.

## Who should read what

| You want to | Read |
|---|---|
| Run one model on one GPU host | [Getting started](getting-started.md) |
| Add a model, a Dockerfile or a script | [Adding a model](adding-a-model.md) |
| Understand multinode inference (disaggregated prefill/decode, colocated) | [Multinode overview](multinode-overview.md) |
| Run a multinode card on a SLURM cluster | [Running multinode workloads](multinode-running.md) |
| Change a setting and know which layer wins | [Configuration](configuration.md) |
| Read or compare benchmark results | [Benchmarks and results](benchmarks-and-results.md) |
| Look up every knob of one launcher or one model | [vLLM disaggregated](vllm-disagg.md), [SGLang disaggregated](sglang-disagg.md), [Kimi-K3](kimi-k3.md) |

## Learning path

1. **Run a single-node model.** Install madengine, run a model with one command, and learn what
   happens during a run and where the results go.
   [getting-started.md](getting-started.md)
2. **Add a model.** Write a model card, a Dockerfile and a run script, and report performance in
   the format madengine expects. [adding-a-model.md](adding-a-model.md)
3. **Learn the multinode concepts.** Disaggregated prefill/decode versus colocated multinode,
   launchers, KV connectors, EP backends and node topology.
   [multinode-overview.md](multinode-overview.md)
4. **Run a multinode workload.** Prerequisites, running a card through madengine, `sbatch` or
   `salloc`, reading logs and results, and fixing failures.
   [multinode-running.md](multinode-running.md)
5. **Configure.** Every configuration layer, its precedence, and where to change a given setting.
   [configuration.md](configuration.md)
6. **Benchmark and read results.** Throughput sweeps, needle-in-a-haystack (NIAH), agentic replay,
   and the `perf.csv` schema. [benchmarks-and-results.md](benchmarks-and-results.md)
7. **Use the references.** Full per-launcher and per-model references:
   [vllm-disagg.md](vllm-disagg.md), [sglang-disagg.md](sglang-disagg.md),
   [kimi-k3.md](kimi-k3.md).

## Page index

| Page | What it covers |
|---|---|
| [README.md](README.md) | This page: what MAD is, the learning path, the glossary and the index. |
| [getting-started.md](getting-started.md) | Prerequisites, installation, running models, tags, timeouts, debugging, model discovery, build versus run, and where results go. |
| [adding-a-model.md](adding-a-model.md) | Every model-card field, Dockerfile resolution, the GPU architecture build argument, run scripts, the performance reporting contract, and multinode cards. |
| [multinode-overview.md](multinode-overview.md) | Concepts: disaggregated prefill/decode versus colocated multinode, launchers, KV connectors, EP backends, topology, architecture diagrams. |
| [multinode-running.md](multinode-running.md) | Running a multinode card through madengine, `sbatch` or `salloc`; logs, results, failure modes, troubleshooting and offline checks. |
| [configuration.md](configuration.md) | Every configuration layer and its precedence, `models.yaml` recipes, connector environment, `cluster.sh`, madengine presets and `--additional-context` keys, `mad-config.yaml`. |
| [benchmarks-and-results.md](benchmarks-and-results.md) | Throughput sweep, NIAH, agentic replay, the `perf.csv` schema and status semantics. |
| [vllm-disagg.md](vllm-disagg.md) | Full reference for [`scripts/vllm_dissag`](../scripts/vllm_dissag). |
| [sglang-disagg.md](sglang-disagg.md) | Full reference for [`scripts/sglang_disagg`](../scripts/sglang_disagg). |
| [kimi-k3.md](kimi-k3.md) | Kimi-K3 on MI300X and MI355X, colocated and disaggregated. |

## Blueprints

These are the supported model families, with the documentation that ships next to each one.

| Blueprint | Description | Models |
|-----------|-------------|--------|
| [Kimi-K3 inference (vLLM / SGLang / ATOM)](../benchmark/kimi_k3/README.md) | Kimi-K3 (2.8T) day-0 inference on MI350X/MI355X across three frameworks. See also [kimi-k3.md](kimi-k3.md). | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) |
| [xDiT diffusion inference](../benchmark/xdit/README.md) | Diffusion Transformer inference using xDiT | FLUX.1, FLUX.1 Kontext, FLUX.2, FLUX.2 Klein, HunyuanVideo, HunyuanVideo 1.5, LTX-2, Stable Diffusion 3.5, Wan 2.1, Wan 2.2, Z-Image Turbo |
| [JAX MaxText training](../benchmark/jax_maxtext/README.md) | Train LLMs on AMD Instinct GPUs using JAX MaxText | Llama 2 7B/70B, Llama 3/3.1 8B/70B, Llama 3.1 405B, Llama 3.3 70B, DeepSeek-V2-lite 16B, Mixtral-8x7B |
| [vLLM inference](../benchmark/vllm/README.md) | LLM inference with vLLM on AMD Instinct GPUs | DeepSeek-R1, gpt-oss-20b/120b, Kimi-K3, Llama-2-70b, Llama-3.1-8b/405b, Llama-3.3-70b, Llama-4-Scout/Maverick, Mixtral-8x7b/8x22b, Phi-4, Qwen3-8b/32b/30b-a3b/235b-a22b |
| [SGLang inference](../benchmark/sglang/README.md) | LLM inference with SGLang on AMD Instinct GPUs | DeepSeek-R1-Distill-Qwen-32B, Kimi-K3 |
| PyTorch training | Train LLMs on AMD Instinct GPUs using AMD's Primus. Primus notes are in [benchmark/primus/README.md](../benchmark/primus/README.md). | Llama 2/3/3.1/3.2/3.3/4, GPT-OSS 20B/120B, Qwen2/2.5/3, Flux, SDXL, DLRM, and others |
| [PyTorch inference](../benchmark/pytorch_inference/README.md) | Inference recipes for multimodal, video and vision transformer models | Mochi video, Chai-1, CLIP (ViT-B-32), Wan2.1, Janus-Pro-7B, HunyuanVideo |
| Megatron-LM training | Train LLMs on AMD Instinct GPUs using ROCm Megatron-LM | Llama 2 7B/70B, Llama 3/3.1 8B/70B, Llama 3.3 70B, DeepSeek-V2-lite, DeepSeek-V3, Mixtral 8x7B/8x22B, Qwen 2.5 7B/72B |
| MPT-30B training (llm-foundry) | LLM training for Mosaic Pretrained Transformer (MPT) models using llm-foundry | MPT-30B |
| PyTorch PEFT/FSDP fine-tuning | Fine-tuning a Hugging Face model with the LoRA approach and the FSDP strategy | Llama-2-70b-chat-hf |
| [Large EP microbenchmark](../scripts/large-ep-benchmark/README.md) | MoE large expert parallelism with MoRI-EP and DeepEP communication microbenchmarks | No specific models |
| [vLLM disaggregated P/D inference](../scripts/vllm_dissag/README.MD) | Distributed inference with prefill/decode disaggregation in vLLM (Default, MoRI EP, DeepEP). See also [vllm-disagg.md](vllm-disagg.md). | DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-5layer, amd-Llama-3.3-70B-Instruct-FP8-KV, Llama-3.1-405B-Instruct-FP8-KV, gpt-oss-120b |
| [SGLang disaggregated P/D inference](../scripts/sglang_disagg/README.MD) | Distributed inference with prefill/decode disaggregation in SGLang (MoRI IO, Mooncake). See also [sglang-disagg.md](sglang-disagg.md). | Llama-3.1-8B, Qwen3-32B, Llama-3.3-70B-FP8, Llama-3.1-405B-FP8, Mixtral-8x7B, DeepSeek-V3, DeepSeek-R1 |
| [SGLang disaggregated P/D inference with WideEP/LargeEP](../scripts/sglang_disagg/README.MD) | Distributed inference with prefill/decode disaggregation in SGLang with WideEP/LargeEP | DeepSeek-V3, DeepSeek-R1 |
| [KVCache Transfer Bench](../scripts/kvcache_transfer_bench/README.md) | Inter-node transfer benchmark | No specific models |

The root [README](../README.md) links the PyTorch training, Megatron-LM, MPT-30B and PEFT/FSDP
blueprints to files that are not in this checkout, so those rows have no link here.

## Glossary

Terms are listed in the order you meet them.

| Term | Meaning |
|---|---|
| **madengine** | The CLI from [ROCm/madengine](https://github.com/ROCm/madengine) that discovers, builds, runs and reports MAD models. `pip install -r requirements.txt` in this repository installs it. Its main commands are `discover`, `build`, `run`, `report` and `database`. |
| **Model card** (or **card**) | One JSON object in a `models.json` file that describes one workload: its name, Dockerfile, script, GPU count, tags, arguments and, for multinode work, its launcher, node count and environment. Cards live in `scripts/<dir>/models.json`. A directory can also generate cards in Python with `get_models_json.py`. See [adding-a-model.md](adding-a-model.md). |
| **Tag** | A label in a card's `tags` list, such as `pyt`, `vllm` or `inference`. `madengine run --tags X` selects every card whose name or tags match `X`. See [getting-started.md](getting-started.md#selecting-models-with-tags). |
| **Recipe** | The per-model serving flags and environment for a multinode workload, kept apart from the card. For vLLM disaggregated serving it is the model's entry in [`scripts/vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml); SGLang has [`scripts/sglang_disagg/models.yaml`](../scripts/sglang_disagg/models.yaml). See [configuration.md](configuration.md). |
| **Launcher** | Two related meanings. (1) The batch script a multinode card runs, for example `run_xPyD_models.slurm` or `run_multinode.slurm`. (2) The value of a card's `distributed.launcher`, which tells madengine how to start the workload (`torchrun`, `vllm`, `sglang`, `slurm_multi` and others). |
| **slurm_multi** | The madengine launcher used by every MAD multinode inference card. madengine writes a wrapper SBATCH script that exports the card's `env_vars`, then runs the card's own `.slurm` script on the head node. That script starts the per-node Docker containers itself with `srun`. The hyphenated `slurm-multi` is accepted as an alias. |
| **Connector** | In disaggregated serving, the component that moves the KV cache from the prefill server to the decode server. vLLM uses `rixl` (NixlConnector) or `moriio` (MoRIIOConnector). SGLang uses MoRI IO or Mooncake as its transfer backend. See [multinode-overview.md](multinode-overview.md). |
| **EP backend** | The all-to-all communication library used for wide expert parallelism (wideEP) in mixture-of-experts models: `mori` (MoRI-EP) or `deepep` (DeepEP). In vLLM each connector pairs with its own backend: `moriio` with `mori`, `rixl` with `deepep`. |
| **xP/yD** | The shape of a disaggregated run: `xP` prefill nodes and `yD` decode nodes. The job needs `xP + yD` nodes. For example `1P/1D` is 2 nodes. |
| **`--additional-context`** | A JSON string passed to `madengine build` or `madengine run` that adds or overrides configuration: `gpu_vendor`, `guest_os`, `docker_env_vars`, `docker_build_arg`, `env_vars`, a `slurm` block, and more. `--additional-context-file` reads the same JSON from a file, and `--additional-context` is merged over it key by key. |
| **Build manifest** | `build_manifest.json`, written by `madengine build` (and by the build half of `madengine run`). It records each built image, the card it was built for, the registry image if one was pushed, and the context. `madengine run --manifest-file build_manifest.json` runs from it without building again. |
| **perf.csv** | The results table. `madengine run` appends one row per model (or one row per result for cards with `multiple_results`), including failed runs as `FAILURE` rows. Change the file name with `-o`. See [benchmarks-and-results.md](benchmarks-and-results.md). |
| **skip_gpu_arch** | A card field listing GPU architectures (comma-separated, for example `gfx942`) the card must not run on. madengine checks it before running, locally or on SLURM, and writes a `SKIPPED` row instead. `--disable-skip-gpu-arch` turns the check off. |
| **MAD_SYSTEM_GPU_ARCHITECTURE** | The host GPU architecture, for example `gfx942` (MI300X) or `gfx950` (MI355X). madengine sets it as an environment variable inside the container, and passes it as a Docker build argument so a Dockerfile can build for one architecture. See [adding-a-model.md](adding-a-model.md#the-mad_system_gpu_architecture-build-argument). |

## madengine documentation

madengine has its own documentation in the
[`docs/` folder of ROCm/madengine](https://github.com/ROCm/madengine/tree/main/docs). These pages
are the ones MAD users need most.

| Page | What it covers |
|---|---|
| [installation.md](https://github.com/ROCm/madengine/blob/main/docs/installation.md) | Installing madengine, setting up the MAD package, testing Docker GPU access for ROCm and CUDA, and fixing import, permission and ROCm-path problems. |
| [usage.md](https://github.com/ROCm/madengine/blob/main/docs/usage.md) | The five commands, the three model discovery methods, the build and run workflows, timeouts, debugging, profiling, reporting and MongoDB upload. |
| [cli-reference.md](https://github.com/ROCm/madengine/blob/main/docs/cli-reference.md) | Every option of `discover`, `build`, `run`, `report` and `database`, with defaults, examples and exit codes. |
| [configuration.md](https://github.com/ROCm/madengine/blob/main/docs/configuration.md) | Every `--additional-context` key: defaults, log error pattern scan, pinned image digests, Docker environment, build arguments, mounts, timeouts, Kubernetes and SLURM blocks, profiling, pre/post scripts, credentials and configuration priority. |
| [launchers.md](https://github.com/ROCm/madengine/blob/main/docs/launchers.md) | Each distributed launcher (`torchrun`, DeepSpeed, Megatron-LM, TorchTitan, Primus, vLLM, SGLang, SGLang disaggregated, `slurm_multi`), with a comparison matrix and troubleshooting. |
| [distributed-config.md](https://github.com/ROCm/madengine/blob/main/docs/distributed-config.md) | How distributed inference configuration splits by scope (the run, the site, the model, the measurement) across madengine `--config`, `cluster.sh`, `models.yaml`, `configs/*.yaml` and `mad-config.yaml`. |
| [deployment.md](https://github.com/ROCm/madengine/blob/main/docs/deployment.md) | Deploying workloads to Kubernetes and SLURM: workflow, configuration examples and troubleshooting. |
