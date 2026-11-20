# End-to-end examples

Nine workloads, five model families, two engines, every serving type MAD runs and the next GPU
generation, each written as a few lines on shared layers. Each is either a **reproduction** -- it
runs exactly the engine commands of an existing card, and a test holds them equal -- or a **demo**
of a mechanism, with no invented setting presented as tuned. Their tests run each one through the
real launchers on each GPU it targets. Every example builds its image from an image spec over the
compatibility matrix, and the last section follows one card from its build to the generalized
configuration and matrix additions the build writes.

How to configure a model this way, step by step: [configuring-a-model.md](configuring-a-model.md).

## Contents

- [The workload library](#the-workload-library)
- [The examples](#the-examples)
- [What each example proves](#what-each-example-proves)
- [Running them](#running-them)
- [The images](#the-images)
- [From a build to the matrix](#from-a-build-to-the-matrix)
- [Maintaining workloads with inheritance](#maintaining-workloads-with-inheritance)
- [How they are tested](#how-they-are-tested)

## The workload library

A workload file says what to run and what to measure. It is madengine's layered config
(`MAD_CONFIG`, version 2), whose `defaults:` list inherits other files with OmegaConf: mappings
merge in order, the file itself on top, and `${...}` interpolations resolve after the merge
(`${model.env.TP_SIZE}`, `${oc.env:NAME,default}`). madengine resolves it at submit; without
madengine the launcher resolves it itself (`scripts/common/config/workload.py`, standard library
and PyYAML), so MAD runs every example on its own. A test holds the two readers to the same
result, and anything passed through config at submit time wins over every layer.

```
scripts/common/workloads/
  base.yaml                     RECIPE_CHECK=1, MAD_BEST_CONFIG=1 -- the root of every workload
  serve/vllm-disagg-wide-ep.yaml   MoRI-IO disaggregation, wide EP
  serve/vllm-disagg-nixl.yaml      rixl connector (NIXL), tensor parallel
  serve/sglang-disagg-mori-io.yaml SGLang disaggregation, tensor parallel, KV over MoRI-IO
  serve/replicas.yaml              independent servers per node behind a router (PERF_TAGS by interpolation)
  serve/sglang-disagg-dp-ep.yaml   SGLang disaggregation, DP attention + EP
  bench/{sweep,niah,accuracy,embeddings,lora}.yaml   what is measured
  family/kimi-k3.yaml              facts true of every Kimi-K3 workload (local weights only)
  family/deepseek-v3.yaml          DeepSeek-V3 family
```

Site facts are not here: a run names its site (`MAD_CLUSTER` / `MAD_CLUSTER_FILE`, a pipeline
context, `MAD_PLATFORM`), so one workload runs on every site.

## The examples

| Card | Model, type | Workload (inherits) | Runs on | Status |
|---|---|---|---|---|
| `pyt_e2e_kimi-k3_disagg_niah` | Kimi-K3, MoE MXFP4, 2P2D | family/kimi-k3 + bench/niah | gfx942, gfx950 | reproduction of `pyt_vllm_disagg_mori_kimi-k3-mxfp4` (and its MI355X twin) |
| `pyt_e2e_llama-3.1-405b_disagg` | Llama-3.1-405B, dense FP8, 1P1D | serve/vllm-disagg-nixl + bench/sweep | gfx942 | reproduction of `pyt_vllm_disagg_nixl_llama-3.1-405b-fp8` |
| `pyt_e2e_deepseek-r1_sglang` | DeepSeek-R1 on SGLang, DP + EP | serve/sglang-disagg-dp-ep + bench/sweep | gfx942, gfx950 | reproduction of `pyt_sglang_disagg_mori_dp_deepseek-r1` |
| `pyt_e2e_qwen3-32b_sglang` | Qwen3-32B on SGLang, dense BF16, TP | serve/sglang-disagg-mori-io + bench/sweep | gfx942, gfx950 | reproduction of `pyt_sglang_disagg_mori_io_qwen3-32b` |
| `pyt_e2e_deepseek-v3_disagg_mtp` | DeepSeek-V3, MoE FP8, 1P1D | family/deepseek-v3 + bench/accuracy | gfx942, gfx950 | demo (validated base recipe; MTP not run yet) |
| `pyt_e2e_deepseek-v3_dq` | DeepSeek-V3, disaggregated quantization | family/deepseek-v3 + bench/accuracy | gfx950 (refused on gfx942) | demo (placeholder checkpoints) |
| `pyt_e2e_kimi-k3_helios` | Kimi-K3 on Helios, 72-GPU pools | family/kimi-k3 + bench/sweep | CDNA 5 once brought up (refused today) | demo |
| `pyt_e2e_qwen3-32b_replicas_lora` | Qwen3-32B, dense BF16, replicas + LoRA | serve/replicas + bench/lora | gfx942, gfx950 | demo (placeholder adapters) |
| `pyt_e2e_qwen3-embedding_replicas` | Qwen3-Embedding-8B, embeddings | serve/replicas + bench/embeddings | gfx942, gfx950 | demo |

The cards (tags `e2e_example`, `reproduction` or `demo`, and `any_gpu` where they run on any GPU) hold only what registers a run -- name, image, node
count, `MAD_CONFIG`; no GPU name or GPU fact. The example recipes in
`scripts/vllm_dissag/models.yaml` (`DeepSeek-V3-MTP`, `DeepSeek-V3-DQ`, `Qwen3-32B-LoRA`,
`Qwen3-Embedding-8B`) are aliases: each inherits a validated recipe and changes only what it shows.

## What each example proves

**Kimi-K3, disaggregated.** One card on any GPU: the recipe's gfx942 variant converts the MXFP4
experts to int4 (the roles add `--quantization-config`), its gfx950 variant runs MXFP4 natively.
Its workload points `MAD_BEST_REGISTRY` at an example registry, so on MI300X the best-known record
is laid over the recipe by default -- the launcher says so and `config.resolved.json` names it --
and `MAD_BEST_CONFIG=0` runs the recipe as written. NIAH measures long-context retrieval.

**DeepSeek-V3, MTP.** Speculative decoding as a recipe-role need: only the decode role gets
`--speculative-config {"method": "deepseek_mtp", ...}`. The KV budget is a tenth of one GPU's
memory, so it is 19.2 GB on MI300X and 28.8 GB on MI355X from one recipe. Accuracy harness.

**DeepSeek-V3, disaggregated quantization.** Prefill in FP4 W4A4, decode weight-only with 16-bit
activations, each from its own checkpoint. On gfx942 the launcher refuses before anything runs,
naming the FP4 activations gfx942 cannot compute; on gfx950 each role serves its own checkpoint
under the model's served name.

**Kimi-K3 on Helios.** The topology spec sizes pools in GPUs on the Helios platform (4 GPUs per
tray, 18 trays per 72-GPU rack), places each pool in its own rack (`--segment=18` on SLURM, pod
affinity on Kubernetes) and verifies the nodes are CDNA 5. Until an image spec and engine build exist
for CDNA 5 the run is refused with what is missing (`recipe.py bringup --arch cdna5`).

**Llama-3.1-405B, disaggregated.** The existing NIXL card's run, written as a serving layer
(`serve/vllm-disagg-nixl.yaml`) and the sweep: the same engine commands as
`pyt_vllm_disagg_nixl_llama-3.1-405b-fp8`.

**Qwen3-32B on SGLang.** The existing MoRI-IO card's run on the SGLang layers: the same engine
commands as `pyt_sglang_disagg_mori_io_qwen3-32b`.

**Qwen3-32B, replicas with LoRA.** Eight TP-1 servers per node behind a router, each serving two
adapters beside the base model; the LoRA benchmark spreads requests over the adapters.

**Qwen3-Embedding-8B.** Pooling servers as replicas; embeddings throughput and latency per
concurrency.

**DeepSeek-R1 on SGLang.** The same layers, the second engine: DP attention + MoRI EP.

## Running them

Without madengine -- the launcher reads the card's `MAD_CONFIG` itself:

```bash
export MAD_CONFIG=workloads/e2e/deepseek-v3-mtp.yaml MODEL_NAME=... DOCKER_IMAGE_NAME=...
sbatch -N 2 scripts/vllm_dissag/run_xPyD_models.slurm          # or scripts/common/k8s/submit.py --card ...
```

Through madengine:

```bash
madengine build --tags pyt_e2e_deepseek-v3_disagg_mtp --additional-context-file <site context>
madengine run --manifest-file build_manifest.json --additional-context-file <site context>
```

A CI pipeline can run either way, per site, and gets the same run: the launcher resolves the
workload itself when madengine did not.

Anything can be overridden for one run without editing a file -- submit-time values win:

```bash
--additional-context '{"env_vars": {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0", "ACC_TASKS": "/data/gsm8k.jsonl"}}'
```

## The images

Every example card names its image as a spec, not a Dockerfile:

| Image spec | Examples | What it is |
|---|---|---|
| `compat/images/vllm-kimi-k3.yaml` | Kimi-K3, DeepSeek-V3 MTP and DQ, Helios, Qwen3 replicas | extends `di-vllm.yaml`; a base per GPU (gfx942, gfx950); MoRI, AITER + flydsl, vLLM with the Kimi-K3 patch, vllm-router, the rixl transport stack |
| `compat/images/vllm-mori121.yaml` | Llama-3.1-405B | extends `di-vllm.yaml`; gfx942; the AITER release wheel, vLLM with the wide-EP patch, the same router and transports |
| `compat/images/sglang-disagg.yaml` | DeepSeek-R1, Qwen3-32B on SGLang | the SGLang release image per GPU, its pip packages, MoRI on gfx942 only (a per-GPU variant) |

The specs inherit: `compat/images/di-vllm.yaml` holds what every vLLM stack shares (build
prerequisites, env, cache locations, the smoke check), and each stack adds only its bases and the
builds it selects. The builds themselves are authorized once in `compat/rocm.yaml` and shared --
the three vLLM stacks use one vllm-router build, two of them one transport stack. `madengine build
--tags <card> --explain` prints the plan and the Dockerfile madengine generates from it.

## From a build to the matrix

A build manifest is the specific instance: for every card, how its image was built and the stack it
holds. What the build writes beside it (`madengine build --emit-config DIR`; the pipeline's
STANDALONE mode the same) is the generalized form. For `pyt_e2e_llama-3.1-405b_disagg`:

- **Built from its spec** (as the card says): the card's configuration fits the shared layers, its
  `image.spec` is the spec itself, and nothing is added to the matrix -- the stack is already general.
- **Built by hand** (a Dockerfile someone wrote, recorded with `image_probe`): the stack is the
  frameworks vLLM stack with a newer vLLM than the matrix lists and a private build. The build writes:

```yaml
# DIR/vllm_dissag/pyt_e2e_llama-3.1-405b_disagg.yaml
image:
  spec:                           # the image, generalized over the matrix
    base: rocm/base:10.1
    python: '3.14'
    rocm: '10.1'
    components:
      amd-aiter: 0.1.22.post2.dev0+gb4d9154d1.d20261002@frameworks-vllm
      flash-attn: 2.8.3@frameworks-vllm
      torch: 2.13.0+rocm10.1.0
      torchvision: 0.28.0+rocm10.1.0
      vllm: 0.30.0.dev0+rocm10.1.0.g1234abcd.d20261101@frameworks_vllm   # new: added to the matrix below
  unresolved:                     # what the matrix cannot express yet; not invented
  - component: my-kernels
    version: 0.1+g0000001
    why: no wheel in the matrix's indexes -- a source build or a local wheel needs its build entry
defaults:                         # the shared layers whose every value the card has
- ../../../scripts/common/workloads/base.yaml
- ../../../scripts/common/workloads/serve/vllm-disagg-nixl.yaml
- ../../../scripts/common/workloads/bench/sweep.yaml
model:
  env:                            # the card's own part
    BENCHMARK_SCRIPT: sweep
    MODEL_NAME: Llama-3.1-405B-Instruct-FP8-KV
    xP: '1'
    yD: '1'
    BENCHMARK_COMBINATIONS: 1024/1024
```

```diff
# DIR/compat/rocm.yaml: the checkout's matrix, comments kept, with what this build established
+    0.30.0.dev0+rocm10.1.0.g1234abcd.d20261101@frameworks_vllm:
+      url: https://rocm.frameworks.amd.com/whl-multi-arch/vllm/vllm/vllm-0.30.0.dev0-cp314-cp314-linux_x86_64.whl
+      requires: [torch==2.13.0+rocm10.1.0, amd-aiter, flash-attn==2.8.3]
+      python: ['3.14']
+      note: generalized from a working build manifest
```

The requirements come from the wheel's own metadata, narrowed to components the matrix knows
(`transformers` is pip's job, not the matrix's). When the run passes, the engineer puts both files
in their PR. The test `FromABuildToItsGeneralizedForm` in `scripts/common/tests/test_e2e_examples.py`
runs exactly this (with the index stood in for).

## Maintaining workloads with inheritance

- A fact lives in one layer: a model family's (`family/kimi-k3.yaml`), a serving shape's
  (`serve/replicas.yaml`), a benchmark's (`bench/niah.yaml`). Changing it there changes every
  workload that inherits it, and the tests show what moved.
- A new workload is a few lines: its `defaults:` and what differs.
- A new GPU needs no workload change: the recipe gains a variant, the catalog an entry, the image
  spec a base for it.
- A new stack is a few lines too: a spec in `compat/images/` that extends `di-vllm.yaml` and selects
  builds; a build that is new to the matrix comes in through a build manifest (above).
- A run-level change is config, not a file edit: submit-time env, a site profile, `--config` groups.

## How they are tested

`scripts/common/tests/test_e2e_examples.py` holds every example to its label -- a reproduction's
engine commands equal its source card's on every rank, every example is a reproduction or a demo --
and needs no madengine: it resolves every workload with
MAD's reader (and, when madengine is importable, checks madengine's reader agrees), and runs it
through the real launchers and role scripts (the snapshot harness's stand-ins for SLURM, Docker and
the GPUs), on each GPU it targets, asserting what it claims: the int4 conversion on gfx942 only,
the best-known record applied and switched off, MTP on decode only, the KV budget per GPU, the DQ
refusal and per-role checkpoints, the Helios topology and refusal, the replicas and their LoRA or
pooling flags, the SGLang DP flags. Their goldens are in the snapshot suite, and the pipeline's
path-equivalence check runs every example card in STANDALONE and through madengine and requires
the same environment, directory and allocation.
