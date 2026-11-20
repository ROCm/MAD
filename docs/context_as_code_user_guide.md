# Context as code: user guide

A distributed-inference run depends on a lot of context: the model and how its engine is tuned on
each GPU, the serving shape, the benchmark, the image and everything in it, which versions work
together, the cluster's fabric and storage. In MAD all of it is code -- small files, each owned by
whoever knows that part, each inheriting what it does not change. The more context is written down,
the more the system can check, reproduce and reuse, because each part can be changed on its own,
by the person whose scope it is, without touching anyone else's.

This guide is complete on its own: what the pieces are, how a run uses them, and how to run, change,
read, extend and improve them, with the commands and the real results. The
[developer guide](context_as_code_developer_guide.md) goes through the same system file by file and
line by line, with every change held by a test.

## Contents

- [One run, five scopes](#one-run-five-scopes)
- [How the pieces connect](#how-the-pieces-connect)
- [What you get](#what-you-get)
- [Run a card](#run-a-card)
- [Switch the cluster](#switch-the-cluster)
- [Switch the recipe](#switch-the-recipe)
- [Change a run without editing anything](#change-a-run-without-editing-anything)
- [Read what happened](#read-what-happened)
- [When a run is refused](#when-a-run-is-refused)
- [Write your own workload](#write-your-own-workload)
- [Add your model](#add-your-model)
- [Images: pick a stack, or bring your own](#images-pick-a-stack-or-bring-your-own)
- [How the matrix gets more accurate over time](#how-the-matrix-gets-more-accurate-over-time)
- [See it yourself](#see-it-yourself)
- [Glossary](#glossary)
- [Where to go next](#where-to-go-next)

## One run, five scopes

| Scope | The file | Who owns it | Example |
|---|---|---|---|
| **What to run** | a workload file on shared layers ([`workloads/e2e/kimi-k3-disagg-niah.yaml`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml), layers in [`scripts/common/workloads/`](../scripts/common/workloads/)) | whoever runs the workload | `xP: "2", yD: "2"`, the NIAH benchmark |
| **How the engine runs a model** | its recipe, with a variant per GPU ([`models.yaml`](../scripts/vllm_dissag/models.yaml)) | the model's developer | Kimi-K3: int4 conversion on MI300X, native MXFP4 on MI355X |
| **What it runs in** | an image spec: a base and a selection of builds ([`compat/images/`](../compat/images/)) | the stack's owner | [`vllm-kimi-k3`](../compat/images/vllm-kimi-k3.yaml): MoRI, AITER, vLLM with its patch, the transport stack |
| **What works together** | the compatibility matrix ([`compat/rocm.yaml`](../compat/rocm.yaml)): each build once, what it needs, what it runs | everyone, through the runs they make | the Kimi-K3 vLLM build runs FP8 and MXFP4 on gfx950 |
| **Where it runs** | a site profile ([`compat/sites/`](../compat/sites/)), platforms and fabrics ([`compat/hardware.yaml`](../compat/hardware.yaml)) | the site's owner | AINIC rails `rdma0…7`, shared storage at `/shared` |

The card ([`models.json`](../scripts/vllm_dissag/models.json)) only registers a run: its name,
launcher, node count, image spec and workload file. A run puts the five together. Each scope's file
is small because it inherits: a workload is a few lines on shared layers (`base` → serving method →
model family → benchmark), a stack extends the one every vLLM stack shares, a new model's recipe
extends a validated one.

## How the pieces connect

Kimi-K3 on two prefill and two decode nodes, measuring long-context retrieval -- the card
[`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/models.json):

```
the card ─── MAD_CONFIG ──▶ the workload (what to run)
   │                          base ─ RECIPE_CHECK, MAD_BEST_CONFIG
   │                          family/kimi-k3 ─ local weights ── serve/vllm-disagg-wide-ep ─ MoRI-IO, wide EP
   │                          bench/niah ─ the benchmark and its context lengths
   │                          its own: MODEL_NAME=Kimi-K3-MXFP4, xP=2, yD=2
   │                              │
   │                              │ MODEL_NAME
   │                              ▼
   │                        the recipe (how the engine runs it): roles + a variant per GPU
   │
   └── image_spec ────────▶ the image spec (what it runs in): vllm-kimi-k3 ── extends di-vllm
                                  │ selects builds from
                                  ▼
                            the matrix (what works together): each build, what it needs, what it runs

the run names its site ──▶ the site profile (where it runs) ── platform, fabric (compat/hardware.yaml)
the nodes report their GPU ──▶ which recipe variant, which image base, which formats
```

On each GPU this resolves to one complete system:

| | on MI300X (gfx942) | on MI355X (gfx950) |
|---|---|---|
| workload | 2 prefill + 2 decode nodes, MoRI-IO, wide EP, NIAH | the same |
| recipe | `EP_TP_SIZE=2`, MoE converted to int4, AITER MLA off, its best-known configuration laid over it | `EP_TP_SIZE=1`, native MXFP4, AITER MLA on |
| image | the `vllm-kimi-k3` stack, built on the gfx942 base | the same stack, on the gfx950 base |
| checked | MXFP4 weights converted to int4 by this vLLM build on gfx942 | MXFP4 computed natively |

One file, both GPUs; nothing in the workload names a GPU.

## What you get

- **Change one thing, in one place.** Move a run to another cluster, run another recipe, change the
  shape, the serving method or the benchmark, add a model on a validated recipe: each is one value or
  one small file, and nothing else moves.
- **Values that follow each other.** A value built from another follows it -- the replicas layer
  names its runs from the tensor-parallel size; set that at submit time and the name follows.
- **One file, every GPU and every site.** The GPU picks its recipe variant and its image; the site
  picks its fabric. The workload does not change.
- **Checked before anything starts.** The recipe's needs are checked against the GPU and against
  what the image's engine build runs, the allocation against the shape, the nodes' GPUs against the
  recipe. A run that cannot work is refused with the reason, before any engine starts.
- **Reproducible from its record.** Every run records every value and where it came from; every
  build writes the run's configuration as a workload file, and running any card from its generated
  file gives the same engine commands and environment (checked for every card on every GPU it runs on).
- **The same run everywhere it is launched.** SLURM, Kubernetes, madengine or the pipeline's
  standalone mode give the same environment, checked card by card.
- **A matrix that gets more accurate as people work.** Build your image your own way. When it runs,
  the build writes what it proved: the image as the stack it is closest to plus what differs, and
  the facts the matrix lacks. Put them in your PR; the next run on that stack adds nothing.

## Run a card

Nothing changes in how you run: a card with a workload file (`MAD_CONFIG`) reads it; a card without
one runs as it always has.

Through madengine:

```bash
madengine build --tags pyt_e2e_kimi-k3_disagg_niah --additional-context-file <site context>
madengine run --manifest-file build_manifest.json --additional-context-file <site context>
```

Without madengine -- the launcher reads the workload file itself:

```bash
cd scripts/vllm_dissag
export MAD_CONFIG=workloads/e2e/kimi-k3-disagg-niah.yaml DOCKER_IMAGE_NAME=<your image>
sbatch -N 4 run_xPyD_models.slurm
```

On Kubernetes: `python3 scripts/common/k8s/submit.py --card pyt_e2e_kimi-k3_disagg_niah ...`
([multinode-kubernetes.md](multinode-kubernetes.md)). Every way gives the same run.

## Switch the cluster

A run names its cluster; the workload, the recipe and the image stay. The GPU is detected on the
nodes and picks the recipe's variant, the image's base and the formats; the site picks the fabric,
the storage and the platform.

| To run on | Give the run | Defined in |
|---|---|---|
| an example MI300X site (CX7 fabric) | `MAD_CLUSTER=example-slurm-mi300x-cx7` | [`compat/sites/example-slurm-mi300x-cx7.yaml`](../compat/sites/example-slurm-mi300x-cx7.yaml) |
| an example MI355X site (AINIC fabric) | `MAD_CLUSTER=example-slurm-mi355x-ainic` | [`compat/sites/example-slurm-mi355x-ainic.yaml`](../compat/sites/example-slurm-mi355x-ainic.yaml) |
| your own site | `MAD_CLUSTER_FILE=<your file>` | a file of the same shape, kept with your site |

A site file is a few lines naming a platform and a fabric from
[`compat/hardware.yaml`](../compat/hardware.yaml) (platforms: `mi300x-oam-8`, `mi325x-oam-8`,
`mi355x-ubb-8`, `mi455x-helios-tray`; fabrics: `cx7`, `ainic`, `thor2`) and your storage:

```yaml
# compat/sites/example-slurm-mi355x-ainic.yaml
scheduler: slurm
platform: mi355x-ubb-8      # gfx950, 8 GPUs per node
fabric: ainic               # rails rdma0..7, GID index 1, interface eno0
env:
  SHARED_MOUNT: /shared
  NVME_ROOT: /mnt/nvme
```

Through madengine, pass it like any value: `--additional-context '{"env_vars": {"MAD_CLUSTER":
"example-slurm-mi355x-ainic"}}'`; with `sbatch`, `export MAD_CLUSTER=...` first.

The Kimi-K3 card ([`pyt_e2e_kimi-k3_disagg_niah`](../scripts/vllm_dissag/models.json)) on three
clusters -- the same card, nothing edited:

| | MI300X, CX7 | MI355X, AINIC | your MI325X site, Thor2 |
|---|---|---|---|
| recipe variant | gfx942: `EP_TP_SIZE=2`, MoE converted to int4 | gfx950: `EP_TP_SIZE=1`, native MXFP4 | gfx942 |
| engine | `--data-parallel-size 8`, `--max-num-batched-tokens 2048` | `--data-parallel-size 16`, `--max-num-batched-tokens 4096` | as on MI300X |
| fabric | GID index 3, `eth0`; rails chosen by the connector | rails `rdma0…7`, GID index 1, `eno0` | rails `bnxt_re0…7`, GID index 3, `fenic0` |
| best-known configuration | the one recorded for `mi300x-oam-8`, applied | none recorded for this platform | none: records are per platform |

Try it:

```bash
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942 \
    --env MAD_CLUSTER=example-slurm-mi300x-cx7 --print-env EP_TP_SIZE,NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950 \
    --env MAD_CLUSTER=example-slurm-mi355x-ainic --print-env EP_TP_SIZE,NCCL_IB_HCA,NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME
```

## Switch the recipe

The recipe -- how the engine runs a model on each GPU -- is chosen by `MODEL_NAME` and can be tuned
for one run, all at submit time. The recipes are in [`models.yaml`](../scripts/vllm_dissag/models.yaml).

| To | Give the run | What the run does (real cards) |
|---|---|---|
| run another recipe of the same model | `MODEL_NAME=Kimi-K3` on the Kimi-K3 card | the other Kimi-K3 recipe: `--kv-cache-memory-bytes 8000000000 → 40000000000`, `--max-model-len 320000 → 1000000`; the same stack and checks |
| run the base recipe instead of a variant of it | `MODEL_NAME=DeepSeek-V3` on the DeepSeek-V3 MTP card | no speculative decoding; the fixed 20 GB KV budget instead of a tenth of GPU memory |
| change one recipe value | `EP_TP_SIZE=2` on MI355X (the fallback the recipe documents) | `[config] EP_TP_SIZE=2 (submit) over the recipe's 1`; `--data-parallel-size 16 → 8` |
| run the recipe without its best-known configuration | `MAD_BEST_CONFIG=0` | the recipe as written |
| run a recipe file of your own | `MAD_RECIPE_FILE=<file>` | your file is used instead of the job's written recipe |

Every switch is checked like the original: a recipe the GPU and image cannot run is refused with the
reason ([below](#when-a-run-is-refused)). Try it:

```bash
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942 --env MODEL_NAME=Kimi-K3
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_deepseek-v3_disagg_mtp --env MAD_GPU_ARCH=gfx942 --env MODEL_NAME=DeepSeek-V3
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950 --env EP_TP_SIZE=2
```

## Change a run without editing anything

Anything given at submit time wins over every file. With madengine:

```bash
--additional-context '{"env_vars": {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0"}}'
```

or `export NAME=value` before `sbatch`. Which value wins, lowest first:

```
the site  <  the connector's env  <  the recipe  <  a topology spec  <  the workload file  <  submit time
```

A value given at submit time is also used when the workload file is resolved, so values built from
it follow. The replicas layer ([`serve/replicas.yaml`](../scripts/common/workloads/serve/replicas.yaml))
names its runs from the tensor-parallel size:

```yaml
model:
  env:
    TP_SIZE: "1"
    PERF_TAGS: "vllm_multinode,replicas,tp${model.env.TP_SIZE}"
```

so on the Qwen3 replicas card, `TP_SIZE=2` at submit reaches the servers and the run's tags follow:
`vllm_multinode,replicas,tp2`.

## Read what happened

The launcher says what it did, as it does it:

```
[config] workload workloads/e2e/kimi-k3-disagg-niah.yaml: 11 value(s) (submitted ones kept)
[recipe] Kimi-K3-MXFP4 prefill: mxfp4 weights are converted to int4_g32 by vllm@vllm-kimi-k3 on gfx942
[recipe] Kimi-K3-MXFP4 resolves on gfx942 with vllm@vllm-kimi-k3: prefill weights=int4_g32,activations=fp8_e4m3fnuz,kv_cache=fp8_e4m3fnuz; ...
[config] resolved environment: <logs>/<job>/config.resolved.json
```

In the job's log directory (`$LOG_PATH/<job>/`, `/run_logs/<job>/` in the containers):

| File | What it holds |
|---|---|
| `config.resolved.json` | every value the run used, and the layer that set it: `{"env": {"EP_TP_SIZE": {"value": "2", "source": "recipe"}, "RUN_MORI": {"value": "1", "source": "workload"}, ...}}` |
| `recipe.yaml` | the recipe exactly as this job used it: alias followed, the GPU's variant applied, memory sizes computed, the best-known record laid over it |
| `perf.csv` | the results, one row per measurement ([benchmarks-and-results.md](benchmarks-and-results.md)); copied to `perf_<MODEL_NAME>.csv` for madengine |
| the per-node logs | each node's engine and benchmark output |

A build with `--emit-config DIR` also writes the run's configuration as a workload file (on the
shared layers, with its image generalized) -- the record you can rerun from.

## When a run is refused

A run that cannot work stops before any engine starts, and says why. Real messages:

| Situation | The launcher says |
|---|---|
| the shape needs more nodes than were allocated | `topology needs 4 nodes but the allocation has 2. Submit with --nodes=4 (madengine: the card's slurm.nodes)` |
| the model needs a format the GPU and image cannot give | `DeepSeek-V3-DQ on gfx942 with vllm@vllm-mori121: prefill: activations 'mxfp4' -- gfx942 with vllm@vllm-mori121 offers ['bf16', 'fp8_e4m3fnuz']` |
| the platform the run names is not what the nodes are | `MAD_PLATFORM=mi455x-helios-tray is cdna5, but these nodes are gfx942.` |
| a model is served a way it does not support | `MODEL_NAME 'DeepSeek-V3' is wideEP-only (set WIDE_EP=1). TP mode is not supported for it.` |

Ask the same question offline: `python3 scripts/common/recipe/recipe.py check --recipes
scripts/vllm_dissag/models.yaml --model <model> --arch <gfx> --image <stack>`.

## Write your own workload

A workload is a few lines: the layers your run shares, and what is yours. The shared layers:

| Layer | What it gives |
|---|---|
| [`base.yaml`](../scripts/common/workloads/base.yaml) | the recipe check and best-known configurations, on |
| [`serve/`](../scripts/common/workloads/serve/) | how it is served: vLLM over MoRI-IO with wide EP, vLLM over NIXL, SGLang (TP, or DP attention + EP), replicas behind a router |
| [`family/`](../scripts/common/workloads/family/) | what is true of every workload of a model family (Kimi-K3, DeepSeek-V3) |
| [`bench/`](../scripts/common/workloads/bench/) | what is measured: throughput sweep, NIAH, accuracy, embeddings, LoRA |

The real workloads, each a few lines on those layers:

| Workload | Inherits | Its own |
|---|---|---|
| [`kimi-k3-disagg-niah.yaml`](../scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml) | base, family/kimi-k3, bench/niah | `MODEL_NAME=Kimi-K3-MXFP4`, `xP=2`, `yD=2`, its best-known registry |
| [`llama-405b-disagg.yaml`](../scripts/vllm_dissag/workloads/e2e/llama-405b-disagg.yaml) | base, serve/vllm-disagg-nixl, bench/sweep | the model, 1P1D, its input/output lengths |
| [`deepseek-v3-mtp.yaml`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-mtp.yaml) | base, family/deepseek-v3, bench/accuracy | `MODEL_NAME=DeepSeek-V3-MTP`, the DeepSeek-V3 weights |
| [`deepseek-r1-sglang.yaml`](../scripts/sglang_disagg/workloads/e2e/deepseek-r1-sglang.yaml) | base, serve/sglang-disagg-dp-ep, bench/sweep | the model, its input/output lengths |
| [`qwen3-32b-replicas-lora.yaml`](../scripts/vllm_multinode/workloads/e2e/qwen3-32b-replicas-lora.yaml) | base, serve/replicas, bench/lora | `MODEL_NAME=Qwen3-32B-LoRA`, the context length |

To change what a run does, change one line or one layer: the shape (`xP`, `yD`), the serving
method (another `serve/` layer), the benchmark (another `bench/` layer), the model (`MODEL_NAME`).

Already have a card? `python3 scripts/common/config/emit.py <catalog> <card>` prints it as a
workload file on the shared layers -- its own part is only what no layer or recipe already gives.
For the existing MI300X Kimi-K3 card it prints exactly the Kimi-K3 example's layers and values.
`emit.py --all -o DIR --suggest` does every card and proposes new shared layers from values many
cards repeat.

## Add your model

A new model is a card, a workload and a recipe -- each inheriting what already works. The real
example is DeepSeek-V3 with multi-token-prediction speculative decoding
([`pyt_e2e_deepseek-v3_disagg_mtp`](../scripts/vllm_dissag/models.json)):

| | File | What it says |
|---|---|---|
| the card | [`models.json`](../scripts/vllm_dissag/models.json) | its name, launcher, 2 nodes, `"image_spec": "${images.vllm-per-gpu}"`, `MAD_CONFIG` |
| the workload | [`deepseek-v3-mtp.yaml`](../scripts/vllm_dissag/workloads/e2e/deepseek-v3-mtp.yaml) | inherits base, the DeepSeek-V3 family, accuracy; `MODEL_NAME=DeepSeek-V3-MTP` |
| the recipe | `DeepSeek-V3-MTP` in [`models.yaml`](../scripts/vllm_dissag/models.yaml) | `alias: DeepSeek-V3` -- the validated recipe -- plus MTP on decode and a KV budget of a tenth of GPU memory |
| the image | [`vllm-per-gpu`](../compat/images/vllm-per-gpu.yaml) | each GPU's validated stack, composed from two existing ones |

```yaml
DeepSeek-V3-MTP:
  alias: DeepSeek-V3
  env: {KV_CACHE_MEMORY_BYTES: null}
  memory: {KV_CACHE_MEMORY_BYTES: {fraction: 0.1}}
  roles:
    decode: {speculative: {method: deepseek_mtp, num_speculative_tokens: 1}}
```

What the run does:

| | MI300X | MI355X |
|---|---|---|
| the recipe check | `DeepSeek-V3-MTP resolves on gfx942 with vllm@vllm-mori121` | `… on gfx950 with vllm@vllm-kimi-k3` |
| decode | `--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}` | the same |
| prefill | no speculative decoding | the same |
| `--kv-cache-memory-bytes` | 19200000000 (a tenth of 192 GB) | 28800000000 (a tenth of 288 GB) |

A recipe can say, each in one place: what every role needs (formats), what one role needs
differently (speculative decoding, its own checkpoint, LoRA adapters), its environment, its flags
per role and parallel mode, sizes from the GPU's memory, a variant per GPU, and the recipe it starts
from. [`models.yaml`](../scripts/vllm_dissag/models.yaml) has a real example of each; the
[developer guide](context_as_code_developer_guide.md#add-a-model-card-that-inherits-and-give-it-the-stack-it-needs)
goes through them. A recipe can also be kept in your workload file (`recipes:`), laid over
`models.yaml` by name, until you share it.

## Images: pick a stack, or bring your own

A card names its image as a stack, not a Dockerfile: `"image_spec": "${images.<stack>}"`. madengine
composes the stack over the matrix for the GPU the image is built for and generates the Dockerfile.

| Stack | For |
|---|---|
| [`vllm-mori121`](../compat/images/vllm-mori121.yaml) | the validated vLLM disaggregation stack on MI300X (DeepSeek-V3, Llama, most vLLM cards) |
| [`vllm-kimi-k3`](../compat/images/vllm-kimi-k3.yaml) | Kimi-K3, on MI300X and MI355X |
| [`vllm-glm51`](../compat/images/vllm-glm51.yaml) | GLM-5.1 |
| [`vllm-per-gpu`](../compat/images/vllm-per-gpu.yaml) | each GPU's stack: `vllm-mori121` on MI300X, the Kimi-K3 stack's builds on MI355X |
| [`sglang-disagg`](../compat/images/sglang-disagg.yaml) | SGLang disaggregation, per GPU |

Every vLLM stack extends [`di-vllm`](../compat/images/di-vllm.yaml), so it lists only its base and
its builds. `madengine build --tags <card> --explain` prints the plan and the Dockerfile without
building.

**Bring your own image.** Keep your Dockerfile. Build with `madengine build --emit-config DIR` and
`"image_probe": true` in the context. When the run passes, DIR holds:

- `DIR/<catalog>/<card>.yaml` -- the run's configuration, with your image written as the stack it
  is closest to plus only what differs (for example `extends: ${images.vllm-kimi-k3}` and a newer
  vLLM), so it inherits everything else that stack holds;
- `DIR/compat/rocm.yaml` -- the matrix with what your run proved added (a new version from an
  authorized index, with its requirements);
- under `image.unresolved`, what needs you: a source build at a new commit comes with a draft entry
  to confirm; a requirement your working run contradicts is reported.

Commit both with your change.

## How the matrix gets more accurate over time

| | What happens | What the matrix learns |
|---|---|---|
| **Run 1** (you, your stack, your machine) | your image is matched to the matrix's builds; the base image's own packages are left out | each new version an index serves (a build entry); a requirement your run shows is not needed (reported, left out); a new source commit (a draft for you to confirm) |
| **Your PR** | you commit the generated files with your change | those builds, each recorded once |
| **Run 2** (anyone, the same stack, any GPU) | everything matches: nothing new, nothing for a person | nothing: it is already true |
| **Run N** | each run proves the stack it ran on its hardware | only what is new |

The matrix stays general -- one entry per build, no per-run record -- and becomes the true picture
of what works together as people work. The [developer guide](context_as_code_developer_guide.md#how-compat-gets-more-accurate-over-time)
walks a Kimi-K3 image through this, with the files each step writes.

## See it yourself

From the MAD root, on a laptop -- the real launchers run with the cluster, Docker and the GPUs stood
in for:

```bash
# what every node of the Kimi-K3 run does on MI300X, then on MI355X
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx942
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950

# the same card on the AINIC site: the fabric follows the cluster
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_kimi-k3_disagg_niah --env MAD_GPU_ARCH=gfx950 \
    --env MAD_CLUSTER=example-slurm-mi355x-ainic --print-env NCCL_IB_HCA,NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME

# the new model (DeepSeek-V3 MTP): its recipe change on each GPU
python3 scripts/common/tests/snapshot/snapshot.py --show --card pyt_e2e_deepseek-v3_disagg_mtp --env MAD_GPU_ARCH=gfx950

# why a model can or cannot run on a GPU with an image
python3 scripts/common/recipe/recipe.py check --recipes scripts/vllm_dissag/models.yaml \
    --model DeepSeek-V3-DQ --arch gfx942 --image vllm-mori121

# what a workload file resolves to, and a recipe as one GPU uses it
python3 scripts/common/config/workload.py show scripts/vllm_dissag/workloads/e2e/kimi-k3-disagg-niah.yaml
python3 scripts/common/recipe/models.py scripts/vllm_dissag/models.yaml Kimi-K3-MXFP4 --arch gfx950
```

## Glossary

| Term | Meaning |
|---|---|
| card | an entry in a `models.json`: a run's name, launcher, node count, image spec and workload file |
| workload file | what to run and measure (`MAD_CONFIG`): a few lines on shared layers |
| layer | a shared workload file others inherit (`defaults:`): a serving method, a model family, a benchmark |
| recipe | how the engine runs a model: its roles' needs, env, flags, sizes, a variant per GPU |
| role | prefill, decode (or one server): a recipe says what each needs |
| image spec, stack | what an image is built from: a base per GPU and a selection of builds |
| matrix | [`compat/rocm.yaml`](../compat/rocm.yaml): every authorized build, what it needs, what an engine build runs |
| site profile | where a run goes: scheduler, platform, fabric, mounts |
| platform, fabric | the node's GPUs and their interconnect; the network between nodes ([`compat/hardware.yaml`](../compat/hardware.yaml)) |
| recipe check | before a run starts: the recipe's needs against the GPU and the image's engine build |
| best-known configuration | a measured configuration for a model, platform and shape, laid over its recipe by default |
| xP, yD | prefill and decode node counts |

## Where to go next

- The [developer guide](context_as_code_developer_guide.md):
  - [Follow along](context_as_code_developer_guide.md#follow-along): every inspection command.
  - [How it fits together](context_as_code_developer_guide.md#how-it-fits-together-kimi-k3-end-to-end):
    the files behind one Kimi-K3 run, the full system they resolve to on each GPU.
  - [How a run puts them together](context_as_code_developer_guide.md#how-a-run-puts-them-together):
    from submit to the engine command, the file behind each step, and which value wins.
  - [Change anything, see what the run does](context_as_code_developer_guide.md#change-anything-see-what-the-run-does):
    seven changes, each with its effect on the run.
  - [From a build to the matrix](context_as_code_developer_guide.md#from-a-build-to-the-matrix):
    exactly what a build writes for a hand-built image.
- [configuring-a-model.md](configuring-a-model.md): adding or tuning a model, step by step.
- [benchmarks-and-results.md](benchmarks-and-results.md): every benchmark and `perf.csv`.
