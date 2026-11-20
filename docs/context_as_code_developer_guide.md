# Context as code: developer guide

A run's whole context -- what to run, how the engine is configured on each GPU, what it runs in,
what works together, and where it runs -- is code: small files, each owned by whoever knows it,
each inheriting what it does not change. This guide shows how the pieces fit, how to add a model
card that inherits what works and changes its recipe and its stack, how a run's cluster, recipe and
stack are switched, what each change does to the run, and how what runs feeds back into the
compatibility matrix. Every result shown is produced by the real launchers and held by a test. The
high-level version is [context_as_code_user_guide.md](context_as_code_user_guide.md).

The examples: nine workloads, five model families, two engines, every serving type MAD runs and the next GPU
generation, each written as a few lines on shared layers. Each is either a **reproduction** -- it
runs exactly the engine commands of an existing card, and a test holds them equal -- or a **demo**
of a mechanism, with no invented setting presented as tuned. Their tests run each one through the
real launchers on each GPU it targets. Every example builds its image from an image spec over the
compatibility matrix, on each GPU with the stack validated there. The last sections follow the
Kimi-K3 example both ways: from its build to the generalized configuration and matrix additions the
build writes, and from the existing card to the example.

How to configure a model this way, step by step: [configuring-a-model.md](configuring-a-model.md).

## Contents

- [Follow along](#follow-along)
- [How it fits together: Kimi-K3, end to end](#how-it-fits-together-kimi-k3-end-to-end)
- [How a run puts them together](#how-a-run-puts-them-together)
- [Add a model card that inherits, and give it the stack it needs](#add-a-model-card-that-inherits-and-give-it-the-stack-it-needs)
- [Change the cluster, the recipe and the stack of a run](#change-the-cluster-the-recipe-and-the-stack-of-a-run)
- [The workload library](#the-workload-library)
- [The examples](#the-examples)
- [What each example proves](#what-each-example-proves)
- [Running them](#running-them)
- [The images](#the-images)
- [From a build to the matrix](#from-a-build-to-the-matrix)
- [From a card to its example](#from-a-card-to-its-example)
- [Maintaining workloads with inheritance](#maintaining-workloads-with-inheritance)
- [How they are tested](#how-they-are-tested)

## Follow along

Every file this guide names is linked; open it beside the guide. Every command below runs from the
MAD root on a laptop -- no cluster, no GPU, no madengine needed (except the last).

| To see | Run |
|---|---|
| what a workload file resolves to (its layers merged, interpolations resolved) | `python3 scripts/common/config/workload.py show scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml` |
| the environment it gives | `python3 scripts/common/config/workload.py env <file> \| tr '\0' '\n'` |
| a recipe as one GPU uses it (alias followed, variant applied) | `python3 scripts/common/recipe/models.py scripts/vllm_dissag/models.yaml Kimi-K3-MXFP4 --arch gfx950` |
| whether a recipe runs on a GPU with an image, and why | `python3 scripts/common/recipe/recipe.py check --recipes scripts/vllm_dissag/models.yaml --model Kimi-K3-MXFP4 --arch gfx942 --image vllm-kimi-k3` (`explain` for the detail) |
| **what every node of a run does** -- the real launchers, with SLURM, Docker and the GPUs stood in for | `python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950` (`--env NAME=VALUE` as at submit, `--nodes N`, `--print-env NAME,...` for container env) |
| the recorded result of every card | [`scripts/common/tests/snapshot/golden/`](../scripts/common/tests/snapshot/golden/) (per node: the container's env and the engine command) |
| an existing card as a workload file on the shared layers | `python3 scripts/common/config/emit.py vllm_dissag pyt_vllm_disagg_mori_kimi-k3-mxfp4` |
| the whole catalog (`compat/` as one document: matrix, hardware, sites, named specs) | `python3 scripts/common/catalog.py dump` |
| the Dockerfile an image spec generates (madengine) | `madengine build --tags pyt_e2e_kimi-k3_disagg_niah --explain` |

## How it fits together: Kimi-K3, end to end

One run -- the card [`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/models.json) -- is
described by these files. Each is owned by whoever knows that part, and each inherits what it does
not change; nothing is copied, so a fix in any of them reaches every run built on it.

**The card** registers the run: its name, launcher, node count, image and workload. Nothing about the
GPU, the model's tuning or the site is here.

```json
{
  "name": "pyt_e2e_kimi-k3_disagg_niah",
  "image_spec": "${images.vllm-kimi-k3}",
  "scripts": "run_xPyD_models.slurm",
  "distributed": {"launcher": "slurm_multi", "nnodes": 4},
  "env_vars": {"MAD_CONFIG": "workloads/e2e/kimi-k3-disagg-niah.yaml", "DOCKER_IMAGE_NAME": "<supply-your-image>"}
}
```

The inheritance, at a glance -- each arrow is "inherits", and a value set lower overrides one set
higher:

```
THE CARD (scripts/vllm_dissag/models.json: pyt_e2e_kimi-k3_disagg_niah)
  MAD_CONFIG ──────────────▶ WHAT TO RUN
  image_spec ──────────────▶ WHAT IT RUNS IN
  scripts ─────────────────▶ the launcher (run_xPyD_models.slurm)

WHAT TO RUN (workload, scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml)
  defaults:
    common/workloads/base.yaml ............ RECIPE_CHECK=1, MAD_BEST_CONFIG=1
    common/workloads/family/kimi-k3.yaml .. REQUIRE_LOCAL_WEIGHTS=1
      └─ serve/vllm-disagg-wide-ep.yaml ... RUN_MORI=1, WIDE_EP=1
    common/workloads/bench/niah.yaml ...... BENCHMARK_SCRIPT=niah, NIAH_WORDS=10000..200000
  model.env (its own) ..................... MODEL_NAME=Kimi-K3-MXFP4, xP=2, yD=2, its best-known registry
          │ MODEL_NAME selects
          ▼
HOW THE ENGINE RUNS ON EACH GPU (recipe, scripts/vllm_dissag/models.yaml: Kimi-K3-MXFP4)
  roles ..................... MXFP4 weights, FP8 activations and KV cache, on every GPU
  variants.gfx942 ........... EP_TP_SIZE=2, AITER MLA off, MoE converted to int4 (--quantization-config)
  variants.gfx950 ........... EP_TP_SIZE=1, AITER MLA on, native MXFP4, --max-num-batched-tokens 4096
  + the best-known record for this model, platform and shape (MAD_BEST_REGISTRY), laid over it
  + the connector env (connectors/moriio.env), under it

WHAT IT RUNS IN (image spec, compat/images/vllm-kimi-k3.yaml; the card: "image_spec": "${images.vllm-kimi-k3}")
  extends: di-vllm.yaml ..... build prerequisites, runtime env, cache locations, smoke check
  base ...................... one per GPU (gfx942, gfx950), by digest
  components ................ mori, amd-aiter, flydsl, vllm, vllm-router, ucx, googletest, rixl, rocshmem, deepep
          │ each component selects a build in
          ▼
WHAT WORKS TOGETHER (matrix, compat/rocm.yaml)
  each build once: its source (repo, commit, patches) or wheel, how it installs, what it requires,
  and for an engine what it runs on each GPU (capabilities) -- shared by every stack that selects it

WHERE IT RUNS (the run names it: MAD_CLUSTER=<site>; the GPU is detected on the nodes)
  compat/sites/<site>.yaml .. scheduler, platform, fabric, mounts
    └─ compat/hardware.yaml . platforms (GPU, GPUs per node), accelerators (memory, formats it
                              computes), fabrics (rails, GID index, interfaces)

THE RECIPE CHECK meets them: the recipe's roles × the GPU's formats × what the image's vLLM build
runs on that GPU -- resolved, converted, or refused with the reason, before any engine starts.
```

Each of those files, linked:

| Scope | File | What it holds for this run |
|---|---|---|
| **What to run** | [`workloads/e2e/kimi-k3-disagg-niah.yaml`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml) | `MODEL_NAME=Kimi-K3-MXFP4`, `xP=2`, `yD=2`, its best-known registry; inherits the layers below |
| ↳ root layer | [`common/workloads/base.yaml`](../scripts/common/workloads/base.yaml) | `RECIPE_CHECK=1` (check before running), `MAD_BEST_CONFIG=1` (use the best-known configuration) |
| ↳ model family | [`common/workloads/family/kimi-k3.yaml`](../scripts/common/workloads/family/kimi-k3.yaml) | `REQUIRE_LOCAL_WEIGHTS=1`; inherits the serving method |
| ↳ ↳ serving method | [`common/workloads/serve/vllm-disagg-wide-ep.yaml`](../scripts/common/workloads/serve/vllm-disagg-wide-ep.yaml) | `RUN_MORI=1`, `WIDE_EP=1`: disaggregation over MoRI-IO, wide expert parallelism |
| ↳ benchmark | [`common/workloads/bench/niah.yaml`](../scripts/common/workloads/bench/niah.yaml) | `BENCHMARK_SCRIPT=niah`, its context lengths |
| **How the engine runs the model** | [`vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml) → `Kimi-K3-MXFP4` | `roles`: MXFP4 weights, FP8 activations and KV cache; `variants.gfx942`: `EP_TP_SIZE=2`, AITER MLA off, int4 conversion; `variants.gfx950`: `EP_TP_SIZE=1`, AITER MLA on, native MXFP4 |
| ↳ best-known configuration | [`workloads/e2e/best_known.example.jsonl`](../scripts/vllm_dissag/workloads/e2e/best_known.example.jsonl) | a measured configuration for this model, platform and shape, laid over the recipe |
| ↳ connector env | [`vllm_dissag/connectors/moriio.env`](../scripts/vllm_dissag/connectors/moriio.env) | how MoRI-IO moves the KV cache on this fabric |
| **What it runs in** | [`compat/images/vllm-kimi-k3.yaml`](../compat/images/vllm-kimi-k3.yaml) | a base per GPU and ten builds: MoRI, AITER, FlyDSL, vLLM with the Kimi-K3 patch, vllm-router, the transport stack |
| ↳ what every vLLM stack shares | [`compat/images/di-vllm.yaml`](../compat/images/di-vllm.yaml) | build prerequisites, runtime env, cache locations, the smoke check |
| **What works together** | [`compat/rocm.yaml`](../compat/rocm.yaml) | each build once: its source (repo, commit, patches) or wheel, how it installs, what it requires; for vLLM, what it runs on each GPU (`capabilities`) |
| **Where it runs** | a site profile, e.g. [`compat/sites/example-slurm-mi355x-ainic.yaml`](../compat/sites/example-slurm-mi355x-ainic.yaml) | its platform and fabric, mounts, scheduler |
| ↳ platforms, GPUs, fabrics | [`compat/hardware.yaml`](../compat/hardware.yaml) | `platforms.mi355x-ubb-8`: gfx950, 8 GPUs per node; `accelerators.gfx950`: memory, the formats it computes; `fabrics.ainic`: rails, GID index, interfaces |

### The full system it resolves to

The run's GPU picks the recipe variant and the image's base; everything else is the same file.
Try it: `python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942`
(then `gfx950`), and `python3 scripts/common/recipe/recipe.py check --recipes scripts/vllm_dissag/models.yaml --model Kimi-K3-MXFP4 --arch gfx942 --image vllm-kimi-k3`.

<table>
<tr><th></th><th>on gfx942 (MI300X)</th><th>on gfx950 (MI355X)</th></tr>
<tr><td>workload</td><td colspan="2"><code>RECIPE_CHECK=1 MAD_BEST_CONFIG=1 RUN_MORI=1 WIDE_EP=1 REQUIRE_LOCAL_WEIGHTS=1 BENCHMARK_SCRIPT=niah NIAH_WORDS=10000,50000,100000,200000 MODEL_NAME=Kimi-K3-MXFP4 xP=2 yD=2</code></td></tr>
<tr><td>recipe</td>
<td><pre>EP_TP_SIZE=2  GPU_ARCHS=gfx942
VLLM_ROCM_USE_AITER_MLA=0
--quantization-config
  '{"moe":{"weight":"int4_per_group_32"}}'
best-known record laid over it</pre></td>
<td><pre>EP_TP_SIZE=1  GPU_ARCHS=gfx950
VLLM_ROCM_USE_AITER_MLA=1
native MXFP4 (no conversion)
--max-num-batched-tokens 4096</pre></td></tr>
<tr><td>image</td><td colspan="2"><pre>base  rocm/vllm-dev:ci_base-0fcd9b99… (the GPU's digest)
mori fe12a11a@di · amd-aiter 68e42f5f@di · flydsl 0.2.4@di · vllm 9a7ae4b8+cd0d2656@di (Kimi-K3 patch)
vllm-router 1d10e71f+0cc1c443@di · ucx da3fac2a@di · googletest 1.14.0@di · rixl f33a5599@di
rocshmem 9c43b232@di · deepep 0f63d3e9@di-cx7</pre></td></tr>
<tr><td>checked before it runs</td><td colspan="2">the recipe's roles against the GPU and the image's vLLM build (what that build runs on that GPU), and the nodes' GPU against the recipe's</td></tr>
</table>

### A change, side by side

An engineer builds Kimi-K3 their own way: their own Dockerfile on the same base, with AITER at a
newer commit and a newer vLLM wheel from the frameworks index. They run it, and it works. The build
writes the generalized form of what they ran
([details below](#from-a-build-to-the-matrix)):

<table>
<tr><th>The stack the card names</th><th>The stack they ran, generalized by the build</th></tr>
<tr><td><pre>image_spec: ${images.vllm-kimi-k3}</pre></td>
<td><pre>image_spec:
  extends: ${images.vllm-kimi-k3}
  components:
    amd-aiter: 1a2b3c4d@di
    vllm: 0.12.1.dev0+rocm7.1.0.g5e3f9a01@frameworks_vllm</pre></td></tr>
<tr><th colspan="2">The full image each resolves to (madengine composes it, on either GPU)</th></tr>
<tr><td><pre>mori        fe12a11a@di
<b>amd-aiter   68e42f5f@di</b>
flydsl      0.2.4@di
<b>vllm        9a7ae4b8+cd0d2656@di</b>
vllm-router 1d10e71f+0cc1c443@di
ucx         da3fac2a@di
googletest  1.14.0@di
rixl        f33a5599@di
rocshmem    9c43b232@di
deepep      0f63d3e9@di-cx7</pre></td>
<td><pre>mori        fe12a11a@di
<b>amd-aiter   1a2b3c4d@di</b>
flydsl      0.2.4@di
<b>vllm        0.12.1.dev0+rocm7.1.0.g5e3f9a01@frameworks_vllm</b>
vllm-router 1d10e71f+0cc1c443@di
ucx         da3fac2a@di
googletest  1.14.0@di
rixl        f33a5599@di
rocshmem    9c43b232@di
deepep      0f63d3e9@di-cx7</pre></td></tr>
</table>

Two lines differ, and only those two are written: the base, MoRI, FlyDSL, the router and the native
transport stack are inherited from the stack, though the engineer's Dockerfile never mentioned
the stack. The workload, the recipe and the checks are untouched: the same run, on a different
image.

### How compat gets more accurate over time

<table>
<tr><th></th><th>What happens</th><th>What the matrix (<a href="../compat/rocm.yaml"><code>compat/rocm.yaml</code></a>) learns</th></tr>
<tr><td><b>Run 1</b><br>(the engineer, on their machine)</td>
<td>Their image is matched to the matrix: MoRI by the commit in its version, FlyDSL by its version;
torch and triton are the base's. Two builds are new.</td>
<td><ul>
<li>the vLLM wheel: a build entry with its url, python, and requirements from its own metadata</li>
<li>the wheel asks for <code>amd-aiter&gt;=0.1.18</code>, but the run worked with 0.1.17: reported, and left out of the entry</li>
<li>AITER at 1a2b3c4d: a draft entry from the matrix's AITER build, for the engineer to confirm</li></ul></td></tr>
<tr><td><b>The PR</b></td>
<td>The engineer commits the generated matrix and confirms the AITER draft, alongside their change.</td>
<td>two builds, each recorded once, true to what ran</td></tr>
<tr><td><b>Run 2</b><br>(anyone, the same stack, either GPU)</td>
<td>Everything matches: the build writes nothing new, and nothing is left for a person. The
generalized spec now composes into the whole ten-build image on gfx942 and gfx950, and can be a
card's image spec as it stands.</td>
<td>nothing: it is already true</td></tr>
<tr><td><b>Run N</b></td>
<td>Each run proves the stack it ran on its hardware.</td>
<td>only what is new: a version it had not seen, a requirement a working run contradicts, a
build a person confirms. The matrix stays general (one entry per build, no per-run record) and
grows as engineers work.</td></tr>
</table>

The test `CompatOverTime` in [`test_e2e_examples.py`](../scripts/common/tests/test_e2e_examples.py) runs this timeline: run 1,
the PR, run 2 adding nothing, and its spec composing on both GPUs into the stack with exactly those
two builds changed.

## How a run puts them together

What happens from submit to the engine command, and the file behind each step. Try it with
`snapshot.py --show` (above): its `[config]` and `[recipe]` lines are these steps.

1. **Build.** `madengine build` composes the card's image spec
   ([`compat/images/vllm-kimi-k3.yaml`](../compat/images/vllm-kimi-k3.yaml), extending
   [`di-vllm.yaml`](../compat/images/di-vllm.yaml)) over the matrix
   ([`compat/rocm.yaml`](../compat/rocm.yaml)) for the target GPU, generates the Dockerfile and builds it.
   The build manifest records the plan; `--emit-config` also writes the run's configuration in
   generalized form ([below](#from-a-build-to-the-matrix)).
2. **Submit.** `madengine run` (or `sbatch`) starts the card's launcher,
   [`run_xPyD_models.slurm`](../scripts/vllm_dissag/run_xPyD_models.slurm), with the card's
   `env_vars` and anything given at submit time.
3. **Submitted values first.** [`layers.sh`](../scripts/common/config/layers.sh) `config_begin`
   records every name already set; those win over every layer below.
4. **The workload file.** `config_apply_workload` reads `MAD_CONFIG` with
   [`workload.py`](../scripts/common/config/workload.py): the file merged over its `defaults:`,
   interpolations resolved (with submitted values), its values set where nothing was submitted; its
   `recipes:`/`connectors:` laid over the launcher's.
5. **The site.** [`cluster.sh`](../scripts/common/cluster.sh) reads the site profile
   (`MAD_CLUSTER`, [`compat/sites/`](../compat/sites/)) and its platform and fabric
   ([`compat/hardware.yaml`](../compat/hardware.yaml)): GPUs per node, mounts, the RDMA rails, NCCL and
   MoRI network settings.
6. **The recipe.** [`models.py`](../scripts/common/recipe/models.py) reads the model's recipe from
   [`models.yaml`](../scripts/vllm_dissag/models.yaml) -- alias followed, the GPU's variant applied,
   memory sizes derived from the GPU, the best-known record laid over it
   ([`scripts/common/perf/`](../scripts/common/perf/)). [`recipe.py`](../scripts/common/recipe/recipe.py)
   checks every role against the GPU's formats ([`compat/hardware.yaml`](../compat/hardware.yaml)
   `accelerators`) and what the image's vLLM build runs there ([`compat/rocm.yaml`](../compat/rocm.yaml)
   `capabilities`): it resolves, converts (`mxfp4 → int4_g32` on gfx942), or refuses naming the need
   nothing meets. The recipe's env is applied over the site's and the connector's
   ([`connectors/moriio.env`](../scripts/vllm_dissag/connectors/moriio.env)).
7. **The shape.** `xP`, `yD` (or a topology spec, `MAD_TOPOLOGY`,
   [`common/topology/`](../scripts/common/topology/)) give each node its role; the allocation must match.
8. **The record.** Every value and the layer that set it is written to
   `<logs>/<job>/config.resolved.json`, and the recipe as used to `recipe.yaml`.
9. **The nodes.** [`node.sh`](../scripts/vllm_dissag/node.sh) starts each node's container with that
   environment; [`vllm_disagg.sh`](../scripts/vllm_dissag/vllm_disagg.sh) builds the role's engine
   command (and the router on the first prefill node); the benchmark
   ([`benchmark_niah.sh`](../scripts/vllm_dissag/benchmark_niah.sh)) measures it.

Which value wins, lowest first ([`layers.sh`](../scripts/common/config/layers.sh)):

```
site (cluster.sh, the site profile)  <  connector .env  <  recipe  <  topology spec
                                     <  workload file  <  submit time (the card's env_vars, --additional-context, export)
```

## Add a model card that inherits, and give it the stack it needs

A new model card is a few lines of config on top of what already works: a workload that inherits
the shared layers, a recipe that inherits a validated one and changes only what the model needs, and
an image that inherits existing stacks and changes only what differs. The real example:
**DeepSeek-V3 with multi-token-prediction speculative decoding**, the card
[`pyt_e2e_deepseek-v3_disagg_mtp`](../scripts/vllm_dissag/models.json). Every file below is in the
repository; open them as you read.

### 1. The card registers the run

In [`scripts/vllm_dissag/models.json`](../scripts/vllm_dissag/models.json) -- a name, the launcher,
the nodes, the image and the workload; nothing about the model's tuning, the GPU or the site:

```json
{
  "name": "pyt_e2e_deepseek-v3_disagg_mtp",
  "image_spec": "${images.vllm-per-gpu}",
  "scripts": "run_xPyD_models.slurm",
  "tags": ["pyt", "e2e_example", "demo", "any_gpu", "vllm_disagg"],
  "distributed": {"launcher": "slurm_multi", "nnodes": 2},
  "slurm": {"nodes": 2, "time": "06:00:00"},
  "env_vars": {"DOCKER_IMAGE_NAME": "<supply-your-image>", "MAD_CONFIG": "workloads/e2e/deepseek-v3-mtp.yaml"}
}
```

### 2. The workload inherits what it shares

[`workloads/e2e/deepseek-v3-mtp.yaml`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-mtp.yaml):

```yaml
version: 2
defaults: [../../../common/workloads/base.yaml, ../../../common/workloads/family/deepseek-v3.yaml, ../../../common/workloads/bench/accuracy.yaml]
model:
  env: {MODEL_NAME: DeepSeek-V3-MTP, MODEL_WEIGHTS_NAME: DeepSeek-V3}
```

| Inherited from | It gives |
|---|---|
| [`base.yaml`](../scripts/common/workloads/base.yaml) | the recipe check and best-known configurations |
| [`family/deepseek-v3.yaml`](../scripts/common/workloads/family/deepseek-v3.yaml) → [`serve/vllm-disagg-wide-ep.yaml`](../scripts/common/workloads/serve/vllm-disagg-wide-ep.yaml) | the DeepSeek-V3 family, served over MoRI-IO with wide expert parallelism |
| [`bench/accuracy.yaml`](../scripts/common/workloads/bench/accuracy.yaml) | the accuracy harness |
| its own | which recipe (`MODEL_NAME: DeepSeek-V3-MTP`), and the weights it serves (DeepSeek-V3's) |

### 3. The recipe inherits a validated one and changes what the model needs

In [`models.yaml`](../scripts/vllm_dissag/models.yaml), `DeepSeek-V3-MTP` is the validated
`DeepSeek-V3` recipe (its roles: FP8 weights, activations and KV cache; its env and flags) with two
changes:

```yaml
DeepSeek-V3-MTP:
  alias: DeepSeek-V3                                   # everything DeepSeek-V3's recipe says, then:
  env: {KV_CACHE_MEMORY_BYTES: null}                    # not DeepSeek-V3's fixed 20 GB KV budget...
  memory: {KV_CACHE_MEMORY_BYTES: {fraction: 0.1}}       # ...a tenth of the GPU's memory, measured on the node
  roles:
    decode: {speculative: {method: deepseek_mtp, num_speculative_tokens: 1}}   # MTP on decode only
```

A recipe can change anything a model needs this way, each in one place:

| To change | Recipe key | Real example in [`models.yaml`](../scripts/vllm_dissag/models.yaml) |
|---|---|---|
| what every role needs (weight, activation, KV formats) | `roles.default` | `Kimi-K3-MXFP4`: MXFP4 weights, FP8 activations and KV |
| what one role needs differently | `roles.prefill`, `roles.decode` | `DeepSeek-V3-MTP` (speculative decoding on decode); `DeepSeek-V3-DQ` (a checkpoint and formats per role) |
| adapters served beside the model | `roles.default.lora` | `Qwen3-32B-LoRA` |
| the environment | `env` | `DeepSeek-V3`: AITER switches, KV block size, CUDA-graph modes |
| engine flags per role and parallel mode | `prefill: {dp, tp}`, `decode: {dp, tp}` | `Kimi-K3-MXFP4`: context length, sequences, MoE conversion |
| sizes from the GPU's memory | `memory` | `DeepSeek-V3-MTP`: KV as a fraction |
| behaviour per GPU | `variants: {gfx942, gfx950}` | `Kimi-K3-MXFP4`: int4 conversion and `EP_TP_SIZE=2` on gfx942, native MXFP4 and `EP_TP_SIZE=1` on gfx950 |
| the recipe it starts from | `alias` | every example recipe |

### 4. The stack inherits existing stacks and changes what differs

The card's image, [`compat/images/vllm-per-gpu.yaml`](../compat/images/vllm-per-gpu.yaml), is
two existing stacks composed by config, with no build of its own:

```yaml
extends: ${images.vllm-mori121}        # the validated MI300X stack the DeepSeek-V3 cards run
base:
  gfx950: rocm/vllm-dev:ci_base-...    # and a base for MI355X
variants:
  gfx950:                              # on MI355X, four builds differ: those validated there
    components:
      mori: fe12a11a@di
      amd-aiter: 68e42f5f@di
      flydsl: 0.2.4@di
      vllm: 9a7ae4b8+cd0d2656@di
    env: {NVCC_THREADS: '8'}
    setup: [...]                       # and that stack's build setup and checks
    finish: [...]
```

`extends` inherits [`vllm-mori121`](../compat/images/vllm-mori121.yaml), which inherits
[`di-vllm`](../compat/images/di-vllm.yaml) (what every vLLM stack shares). Every build it names is
recorded once in [`compat/rocm.yaml`](../compat/rocm.yaml): its source or wheel, how it installs,
and, for vLLM, what it runs on each GPU. Composed for gfx942 it is exactly `vllm-mori121`; for gfx950,
exactly the Kimi-K3 stack's builds (a test holds both equal).

Other real ways to change a stack through config:

| Change | Real example |
|---|---|
| a stack with different builds, inheriting the shared one | [`vllm-glm51`](../compat/images/vllm-glm51.yaml): extends `di-vllm`, selects AITER with a fix for GLM-5.1 and its own vLLM |
| a base per GPU | [`vllm-kimi-k3`](../compat/images/vllm-kimi-k3.yaml), [`sglang-disagg`](../compat/images/sglang-disagg.yaml) |
| a build on one GPU only | [`sglang-disagg`](../compat/images/sglang-disagg.yaml): MoRI on gfx942 only (`variants.gfx942`) |
| a card's own change to a stack | `"image_spec": {"extends": "${image_spec}", "gpu_arch": "gfx942"}` -- [`pyt_kvcache_transfer_bench_all_image_spec`](../scripts/kvcache_transfer_bench/models.json) |

### 5. What the run does

| | on MI300X (gfx942) | on MI355X (gfx950) |
|---|---|---|
| the recipe check | `DeepSeek-V3-MTP resolves on gfx942 with vllm@vllm-mori121`: FP8 as `fp8_e4m3fnuz` | `… on gfx950 with vllm@vllm-kimi-k3`: FP8 as `fp8_e4m3` |
| the image | the `vllm-mori121` builds | the gfx950 builds |
| prefill | `--kv-cache-memory-bytes 19200000000` | `--kv-cache-memory-bytes 28800000000` |
| decode | the same, plus `--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}` | the same, plus MTP |
| switch back to the base recipe (`MODEL_NAME=DeepSeek-V3` at submit) | no MTP, the fixed `--kv-cache-memory-bytes 20000000000` | the same |

Try it:

```bash
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_deepseek-v3_disagg_mtp --env MAD_GPU_ARCH=gfx942
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_deepseek-v3_disagg_mtp --env MAD_GPU_ARCH=gfx950
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_deepseek-v3_disagg_mtp --env MAD_GPU_ARCH=gfx942 --env MODEL_NAME=DeepSeek-V3
```

### The other examples, built the same way

| Card | Inherits | Changes |
|---|---|---|
| [`pyt_e2e_deepseek-v3_dq`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-dq.yaml) | the DeepSeek-V3 workload layers; recipe `alias: DeepSeek-V3`; the per-GPU stack | each role its own checkpoint and formats (FP4 W4A4 prefill, W4A16 decode): runs on gfx950, refused on gfx942 naming the FP4 it cannot compute |
| [`pyt_e2e_qwen3-32b_replicas_lora`](../scripts/vllm_multinode/workloads/e2e/qwen3-32b-replicas-lora.yaml) | [`serve/replicas.yaml`](../scripts/common/workloads/serve/replicas.yaml), [`bench/lora.yaml`](../scripts/common/workloads/bench/lora.yaml); recipe `alias: Qwen3-32B` | two LoRA adapters beside the base model on every replica |
| [`pyt_e2e_qwen3-embedding_replicas`](../scripts/vllm_multinode/workloads/e2e/qwen3-embedding-replicas.yaml) | the replicas layer, [`bench/embeddings.yaml`](../scripts/common/workloads/bench/embeddings.yaml) | a pooling server |
| [`pyt_e2e_kimi-k3_helios`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-helios.yaml) | the Kimi-K3 family and recipe | a topology in GPUs on the Helios platform (a 72-GPU pool per rack), refused until the platform is brought up |
| [`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml) | the Kimi-K3 family, NIAH; recipe `Kimi-K3-MXFP4` with its variants | nothing new: one card replaces the GPU-named MI300X and MI355X cards and runs exactly what each ran |

## Change the cluster, the recipe and the stack of a run

The same card, [`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/models.json), with one
thing switched each time, at submit time -- no file edited. What changes is everything that depends
on it, and nothing else:

| Switch | What the run does |
|---|---|
| **cluster**: `MAD_CLUSTER=example-slurm-mi300x-cx7` ([site](../compat/sites/example-slurm-mi300x-cx7.yaml)) | gfx942 recipe variant: `EP_TP_SIZE=2`, MoE converted to int4, `--data-parallel-size 8`; CX7 fabric (`NCCL_IB_GID_INDEX=3`, `eth0`); the best-known record for `mi300x-oam-8` applied |
| **cluster**: `MAD_CLUSTER=example-slurm-mi355x-ainic` ([site](../compat/sites/example-slurm-mi355x-ainic.yaml)) | gfx950 variant: `EP_TP_SIZE=1`, native MXFP4, `--data-parallel-size 16`, `--max-num-batched-tokens 4096`; AINIC fabric (rails `rdma0…7`, GID index 1, `eno0`); no best-known record for this platform |
| **cluster**: `MAD_CLUSTER_FILE=<your site file>` (platform `mi325x-oam-8`, fabric `thor2` from [`compat/hardware.yaml`](../compat/hardware.yaml)) | the Thor2 rails (`bnxt_re0…7`, `fenic0`); the MI300X best-known record is not applied on another platform |
| **recipe**: `MODEL_NAME=Kimi-K3` (the other Kimi-K3 recipe in [`models.yaml`](../scripts/vllm_dissag/models.yaml)) | `--kv-cache-memory-bytes 8000000000 → 40000000000`, `--max-model-len 320000 → 1000000`; the same stack, the same checks |
| **recipe value**: `EP_TP_SIZE=2` on MI355X (the recipe's documented fallback) | `[config] EP_TP_SIZE=2 (submit) over the recipe's 1`; `--data-parallel-size 16 → 8` |
| **best-known off**: `MAD_BEST_CONFIG=0` | the recipe as written, without the record laid over it |
| **stack**: the card's `image_spec` | every build, per GPU -- [above](#4-the-stack-inherits-existing-stacks-and-changes-what-differs) |

Try them:

```bash
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942 --env MAD_CLUSTER=example-slurm-mi300x-cx7 --print-env EP_TP_SIZE,NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950 --env MAD_CLUSTER=example-slurm-mi355x-ainic --print-env EP_TP_SIZE,NCCL_IB_HCA,NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942 --env MODEL_NAME=Kimi-K3
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950 --env EP_TP_SIZE=2
```

### Who changes what

| To change | Edit | Owned by |
|---|---|---|
| the serving shape, benchmark, a run's values | the workload file ([`workloads/`](../scripts/vllm_dissag/workloads/)) or submit-time config | whoever runs the workload |
| how a serving method or benchmark works for everyone | a shared layer ([`scripts/common/workloads/`](../scripts/common/workloads/)) | the layer's owner |
| how the engine runs a model on each GPU | its recipe ([`models.yaml`](../scripts/vllm_dissag/models.yaml), or the file's `recipes:`) | the model's developer |
| how a KV connector behaves | its env ([`connectors/`](../scripts/vllm_dissag/connectors/)) | the connector's owner |
| what an image holds | its image spec ([`compat/images/`](../compat/images/)) | the stack's owner |
| what works together | the matrix ([`compat/rocm.yaml`](../compat/rocm.yaml)), from builds that ran | everyone, through their runs' PRs |
| where it runs | a site profile ([`compat/sites/`](../compat/sites/)), platforms and fabrics ([`compat/hardware.yaml`](../compat/hardware.yaml)) | the site's owner |

## The workload library

A workload file says what to run and what to measure. It is madengine's layered config
(`MAD_CONFIG`, version 2), whose `defaults:` list inherits other files with OmegaConf: mappings
merge in order, the file itself on top, and `${...}` interpolations resolve after the merge
(`${model.env.TP_SIZE}`, `${oc.env:NAME,default}`). madengine resolves it at submit; without
madengine the launcher resolves it itself ([`workload.py`](../scripts/common/config/workload.py), standard library
and PyYAML), so MAD runs every example on its own. A test holds the two readers to the same
result, and anything passed through config at submit time wins over every layer.

```
scripts/common/workloads/
  base.yaml                        the root of every workload
  serve/                           how it is served (a workload inherits one)
    vllm-disagg-wide-ep.yaml         vLLM, MoRI-IO disaggregation, wide EP
    vllm-disagg-nixl.yaml            vLLM, the NIXL connector
    sglang-disagg-mori-io.yaml       SGLang, tensor parallel, KV over MoRI-IO
    sglang-disagg-dp-ep.yaml         SGLang, DP attention + EP
    replicas.yaml                    independent servers per node behind a router
  bench/                           what is measured
    sweep.yaml niah.yaml accuracy.yaml embeddings.yaml lora.yaml
  family/                          what is true of every workload of a model family (inherits its serve/)
    kimi-k3.yaml deepseek-v3.yaml
```

| Layer | What it gives |
|---|---|
| [`base.yaml`](../scripts/common/workloads/base.yaml) | `RECIPE_CHECK=1`, `MAD_BEST_CONFIG=1` -- the root of every workload |
| [`serve/vllm-disagg-wide-ep.yaml`](../scripts/common/workloads/serve/vllm-disagg-wide-ep.yaml) | MoRI-IO disaggregation, wide EP |
| [`serve/vllm-disagg-nixl.yaml`](../scripts/common/workloads/serve/vllm-disagg-nixl.yaml) | the rixl connector (NIXL) |
| [`serve/sglang-disagg-mori-io.yaml`](../scripts/common/workloads/serve/sglang-disagg-mori-io.yaml) | SGLang disaggregation, tensor parallel, KV over MoRI-IO |
| [`serve/sglang-disagg-dp-ep.yaml`](../scripts/common/workloads/serve/sglang-disagg-dp-ep.yaml) | SGLang disaggregation, DP attention + EP |
| [`serve/replicas.yaml`](../scripts/common/workloads/serve/replicas.yaml) | independent servers per node behind a router (`PERF_TAGS` by interpolation) |
| [`bench/`](../scripts/common/workloads/bench/) (`sweep`, `niah`, `accuracy`, `embeddings`, `lora`) | what is measured |
| [`family/kimi-k3.yaml`](../scripts/common/workloads/family/kimi-k3.yaml) | facts true of every Kimi-K3 workload (local weights only) |
| [`family/deepseek-v3.yaml`](../scripts/common/workloads/family/deepseek-v3.yaml) | the DeepSeek-V3 family |

Site facts are not here: a run names its site (`MAD_CLUSTER` / `MAD_CLUSTER_FILE`, a pipeline
context, `MAD_PLATFORM`), so one workload runs on every site.

## The examples

| Card | Model, type | Workload (inherits) | Runs on | Status |
|---|---|---|---|---|
| [`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml) | Kimi-K3, MoE MXFP4, 2P2D | family/kimi-k3 + bench/niah | gfx942, gfx950 | reproduction of `pyt_vllm_disagg_mori_kimi-k3-mxfp4` (and its MI355X twin) |
| [`pyt_e2e_llama-3.1-405b_disagg`](../scripts/vllm_dissag/workloads/e2e/llama-405b-disagg.yaml) | Llama-3.1-405B, dense FP8, 1P1D | serve/vllm-disagg-nixl + bench/sweep | gfx942 | reproduction of `pyt_vllm_disagg_nixl_llama-3.1-405b-fp8` |
| [`pyt_e2e_deepseek-r1_sglang`](../scripts/sglang_disagg/workloads/e2e/deepseek-r1-sglang.yaml) | DeepSeek-R1 on SGLang, DP + EP | serve/sglang-disagg-dp-ep + bench/sweep | gfx942, gfx950 | reproduction of `pyt_sglang_disagg_mori_dp_deepseek-r1` |
| [`pyt_e2e_qwen3-32b_sglang`](../scripts/sglang_disagg/workloads/e2e/qwen3-32b-sglang.yaml) | Qwen3-32B on SGLang, dense BF16, TP | serve/sglang-disagg-mori-io + bench/sweep | gfx942, gfx950 | reproduction of `pyt_sglang_disagg_mori_io_qwen3-32b` |
| [`pyt_e2e_deepseek-v3_disagg_mtp`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-mtp.yaml) | DeepSeek-V3, MoE FP8, 1P1D | family/deepseek-v3 + bench/accuracy | gfx942, gfx950 | demo |
| [`pyt_e2e_deepseek-v3_dq`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-dq.yaml) | DeepSeek-V3, disaggregated quantization | family/deepseek-v3 + bench/accuracy | gfx950 (refused on gfx942) | demo |
| [`pyt_e2e_kimi-k3_helios`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-helios.yaml) | Kimi-K3 on Helios, 72-GPU pools | family/kimi-k3 + bench/sweep | CDNA 5 once brought up (refused today) | demo |
| [`pyt_e2e_qwen3-32b_replicas_lora`](../scripts/vllm_multinode/workloads/e2e/qwen3-32b-replicas-lora.yaml) | Qwen3-32B, dense BF16, replicas + LoRA | serve/replicas + bench/lora | gfx942, gfx950 | demo |
| [`pyt_e2e_qwen3-embedding_replicas`](../scripts/vllm_multinode/workloads/e2e/qwen3-embedding-replicas.yaml) | Qwen3-Embedding-8B, embeddings | serve/replicas + bench/embeddings | gfx942, gfx950 | demo |

The cards (tags `e2e_example`, `reproduction` or `demo`, and `any_gpu` where they run on any GPU) hold only what registers a run -- name, image, node
count, `MAD_CONFIG` (in [`vllm_dissag/models.json`](../scripts/vllm_dissag/models.json),
[`vllm_multinode/models.json`](../scripts/vllm_multinode/models.json),
[`sglang_disagg/models.json`](../scripts/sglang_disagg/models.json)); no GPU name or GPU fact. Each card
name links to its workload file. The example recipes in
[`vllm_dissag/models.yaml`](../scripts/vllm_dissag/models.yaml) (`DeepSeek-V3-MTP`, `DeepSeek-V3-DQ`, `Qwen3-32B-LoRA`,
`Qwen3-Embedding-8B`) are aliases: each inherits a validated recipe and changes only what it shows.

## What each example proves

**Kimi-K3, disaggregated.** One card on any GPU: the recipe's gfx942 variant converts the MXFP4
experts to int4 (the roles add `--quantization-config`), its gfx950 variant runs MXFP4 natively.
Its workload points `MAD_BEST_REGISTRY` at an example registry, so on MI300X the best-known record
is laid over the recipe by default -- the launcher says so and `config.resolved.json` names it --
and `MAD_BEST_CONFIG=0` runs the recipe as written. NIAH measures long-context retrieval.

**DeepSeek-V3, MTP.** Speculative decoding as a recipe-role need, on the validated DeepSeek-V3
base recipe: only the decode role gets `--speculative-config {"method": "deepseek_mtp", ...}`. The KV budget is a tenth of one GPU's
memory, so it is 19.2 GB on MI300X and 28.8 GB on MI355X from one recipe. Accuracy harness.

**DeepSeek-V3, disaggregated quantization.** Prefill in FP4 W4A4, decode weight-only with 16-bit
activations, each from its own checkpoint (found under `MODEL_DIR_CANDIDATES`). On gfx942 the launcher refuses before anything runs,
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
adapters beside the base model (the adapters a run names); the LoRA benchmark spreads requests over
them.

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
madengine run --manifest-file build_manifest.json \
    --additional-context '{"slurm": {"partition": "<partition>"}, "env_vars": {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0", "ACC_TASKS": "/data/gsm8k.jsonl"}}'
```

## The images

Every example card names its image as a spec, not a Dockerfile:

| Image spec | Examples | What it is |
|---|---|---|
| [`compat/images/vllm-kimi-k3.yaml`](../compat/images/vllm-kimi-k3.yaml) | Kimi-K3, Kimi-K3 on Helios | extends `di-vllm.yaml`; a base per GPU (gfx942, gfx950); MoRI, AITER + FlyDSL, vLLM with the Kimi-K3 patch, vllm-router, the rixl transport stack |
| [`compat/images/vllm-mori121.yaml`](../compat/images/vllm-mori121.yaml) | Llama-3.1-405B | extends `di-vllm.yaml`; gfx942; the validated mori121 stack the DeepSeek-V3 and Llama cards run: the AITER release wheel, vLLM with the wide-EP patch, the same router and transports |
| [`compat/images/vllm-per-gpu.yaml`](../compat/images/vllm-per-gpu.yaml) | DeepSeek-V3 MTP and DQ, Qwen3 replicas | composed from the two above, one per GPU: extends `vllm-mori121.yaml`, and on gfx950 (`variants.gfx950`) selects the builds the matrix's gfx950 vLLM is validated with. No build of its own |
| [`compat/images/sglang-disagg.yaml`](../compat/images/sglang-disagg.yaml) | DeepSeek-R1, Qwen3-32B on SGLang | the SGLang release image per GPU, its pip packages, MoRI on gfx942 only (a per-GPU variant) |

The specs inherit: [`compat/images/di-vllm.yaml`](../compat/images/di-vllm.yaml) holds what every vLLM stack shares (build
prerequisites, env, cache locations, the smoke check), and each stack adds only its bases and the
builds it selects. The builds themselves are authorized once in [`compat/rocm.yaml`](../compat/rocm.yaml) and shared --
the vLLM stacks use one vllm-router build and one transport stack. `vllm-per-gpu` shows how far that
goes: a stack that is two existing ones, chosen by the GPU the image is built for. Composed for
gfx942 it is exactly `vllm-mori121`, for gfx950 exactly `vllm-kimi-k3` (same builds, base and steps;
a test holds them equal), so the recipe check knows what each image runs by its builds.
`madengine build --tags <card> --explain` prints the plan and the Dockerfile madengine generates.

## From a build to the matrix

A build manifest is the specific instance: for every card, how its image was built and the stack it
holds. What the build writes beside it (`madengine build --emit-config DIR`; the pipeline's
STANDALONE mode the same) is the generalized form: the card on the shared layers, and its image
placed under the named stack it is closest to (`extends: ${images.<stack>}` and only what differs),
so it inherits everything that stack holds. For `pyt_e2e_kimi-k3_disagg_niah`:

- **Built from its spec** (as the card says), on either GPU: the configuration is the example's own
  file, the image is `extends: ${images.vllm-kimi-k3}` with nothing different, and nothing is added
  to the matrix -- the stack is already general.
- **Built by hand** (someone's own Dockerfile on the Kimi-K3 base; the build probes the image and its
  base for the Python packages each holds, `image_probe`): the stack's MoRI from source at its
  commit, its FlyDSL, AITER from source at a newer commit, and a newer vLLM wheel from the frameworks
  index in place of the patched source build. The build writes:

```yaml
# DIR/vllm_dissag/pyt_e2e_kimi-k3_disagg_niah.yaml
image:
  image_spec: \${images.vllm-kimi-k3}
  spec:                                 # the image, generalized: the stack it is, and what differs
    extends: \${images.vllm-kimi-k3}    # MoRI (matched by the commit in its version), FlyDSL, the base,
                                        # vllm-router and the native transport stack are the stack's
    components:
      amd-aiter: 1a2b3c4d@di            # drafted below
      vllm: 0.12.1.dev0+rocm7.1.0.g5e3f9a01@frameworks_vllm   # new: added to the matrix below
  unresolved:                           # what needs a person: listed, not invented
  - component: amd-aiter
    version: 0.1.17.dev40+g1a2b3c4d
    why: built from source at a commit the matrix has no build for
    commit: 1a2b3c4d
    draft:                              # the matrix's AITER build at the new commit
      1a2b3c4d@di:
        source:
          repo: https://github.com/ROCm/aiter.git
          commit: 1a2b3c4d
          path: /tmp/aiter-src
          steps: [...]                  # the existing build's steps
        note: drafted from amd-aiter 68e42f5f@di at the commit a working build used; confirm the steps
  - component: vllm                     # the wheel asks for more than the working run had
    build: 0.12.1.dev0+rocm7.1.0.g5e3f9a01@frameworks_vllm
    requires: amd-aiter>=0.1.18
    ran_with: amd-aiter==0.1.17.dev40+g1a2b3c4d
defaults:                               # the shared layers whose every value the card has
- ../../../scripts/common/workloads/base.yaml
- ../../../scripts/common/workloads/family/kimi-k3.yaml
- ../../../scripts/common/workloads/bench/niah.yaml
model:
  env:                                  # the card's own part
    MODEL_NAME: Kimi-K3-MXFP4
    xP: '2'
    yD: '2'
    MAD_BEST_REGISTRY: vllm_dissag/workloads/e2e/best_known.example.jsonl
```

```diff
# DIR/compat/rocm.yaml: the checkout's matrix, comments kept, with what this build established
+    0.12.1.dev0+rocm7.1.0.g5e3f9a01@frameworks_vllm:
+      url: https://rocm.frameworks.amd.com/whl-multi-arch/vllm/vllm/vllm-0.12.1.dev0-cp312-cp312-linux_x86_64.whl
+      requires: [mori]                 # torch is the base image's, as for the matrix's own builds
+      python: ['3.12']
+      note: generalized from a working build manifest
```

What each part of the image becomes:

| In the image | Becomes |
|---|---|
| a build the matrix has (by version, or by the commit a source build carries) | the stack's, inherited |
| a package the base image already has (torch, triton) | nothing: it is the base's, and no build's requirement |
| a plain index dependency (msgspec) | nothing: pip installs it |
| a version an authorized index serves | a build entry in the matrix: url, python, and its requirements on components the matrix knows (from the wheel's metadata) |
| a requirement of that wheel the working run contradicts | reported, and left out of the entry: the run shows the wheel works without it |
| a source build at a commit the matrix lacks | a draft entry, from the matrix's build of that repository, for a person to confirm |

When the run passes, the engineer puts both files in their PR; the next run on a stack the matrix
now covers adds nothing. The test `FromABuildToItsGeneralizedForm` in
[`test_e2e_examples.py`](../scripts/common/tests/test_e2e_examples.py) runs exactly this (with the index stood in for).

## From a card to its example

The same generation, from the existing card: [`emit.py`](../scripts/common/config/emit.py)
`vllm_dissag pyt_vllm_disagg_mori_kimi-k3-mxfp4` gives the example's layers and its values --

```yaml
defaults:
- ../../../common/workloads/family/kimi-k3.yaml   # wide EP, MoRI, local weights
- ../../../common/workloads/bench/niah.yaml       # NIAH and its context lengths
model:
  env: {MODEL_NAME: Kimi-K3-MXFP4, xP: '2', yD: '2'}
```

-- everything else the card set is given by the layers or by the recipe on the GPU the card runs on
(`EP_TP_SIZE=2` is the gfx942 recipe's), and its image name stays on the card. The example adds the
root layer (the recipe check, best-known configurations) and its best-known registry, and drops the
card's `skip_gpu_arch`, so one card runs on both GPUs: the gfx950 recipe gives that GPU its own
`EP_TP_SIZE=1`, which is what the separate MI355X card was for. `FromACardToItsExample` holds the two
equal; `snapshot.py --from-emitted` runs every card from its generated file on each GPU it runs on
and requires the same engine commands and container environment.

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

[`test_e2e_examples.py`](../scripts/common/tests/test_e2e_examples.py) holds every example to its label -- a reproduction's
engine commands equal its source card's on every rank, every example is a reproduction or a demo --
and needs no madengine: it resolves every workload with
MAD's reader (and, when madengine is importable, checks madengine's reader agrees), and runs it
through the real launchers and role scripts (the snapshot harness's stand-ins for SLURM, Docker and
the GPUs), on each GPU it targets, asserting what it claims: the int4 conversion on gfx942 only,
the best-known record applied and switched off, MTP on decode only, the KV budget per GPU, the DQ
refusal and per-role checkpoints, the Helios topology and refusal, the replicas and their LoRA or
pooling flags, the SGLang DP flags. Their goldens are in the snapshot suite, and the pipeline's
path-equivalence check runs every example card in STANDALONE and through madengine and requires
the same environment, directory and allocation. `PerGpuStack` holds `vllm-per-gpu` equal to the
stack of each GPU it is composed from.
