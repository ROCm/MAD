# End-to-end examples

Nine workloads, five model families, two engines, every serving type MAD runs and the next GPU
generation, each written as a few lines on shared layers. Each is either a **reproduction** -- it
runs exactly the engine commands of an existing card, and a test holds them equal -- or a **demo**
of a mechanism, with no invented setting presented as tuned. Their tests run each one through the
real launchers on each GPU it targets. Every example builds its image from an image spec over the
compatibility matrix, on each GPU with the stack validated there. The last sections follow the
Kimi-K3 example both ways: from its build to the generalized configuration and matrix additions the
build writes, and from the existing card to the example.

How to configure a model this way, step by step: [configuring-a-model.md](configuring-a-model.md).

## Contents

- [The workload library](#the-workload-library)
- [The examples](#the-examples)
- [What each example proves](#what-each-example-proves)
- [Running them](#running-them)
- [The images](#the-images)
- [From a build to the matrix](#from-a-build-to-the-matrix)
- [From a card to its example](#from-a-card-to-its-example)
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
| `pyt_e2e_deepseek-v3_disagg_mtp` | DeepSeek-V3, MoE FP8, 1P1D | family/deepseek-v3 + bench/accuracy | gfx942, gfx950 | demo |
| `pyt_e2e_deepseek-v3_dq` | DeepSeek-V3, disaggregated quantization | family/deepseek-v3 + bench/accuracy | gfx950 (refused on gfx942) | demo |
| `pyt_e2e_kimi-k3_helios` | Kimi-K3 on Helios, 72-GPU pools | family/kimi-k3 + bench/sweep | CDNA 5 once brought up (refused today) | demo |
| `pyt_e2e_qwen3-32b_replicas_lora` | Qwen3-32B, dense BF16, replicas + LoRA | serve/replicas + bench/lora | gfx942, gfx950 | demo |
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
--additional-context '{"env_vars": {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0", "ACC_TASKS": "/data/gsm8k.jsonl"}}'
```

## The images

Every example card names its image as a spec, not a Dockerfile:

| Image spec | Examples | What it is |
|---|---|---|
| `compat/images/vllm-kimi-k3.yaml` | Kimi-K3, Kimi-K3 on Helios | extends `di-vllm.yaml`; a base per GPU (gfx942, gfx950); MoRI, AITER + FlyDSL, vLLM with the Kimi-K3 patch, vllm-router, the rixl transport stack |
| `compat/images/vllm-mori121.yaml` | Llama-3.1-405B | extends `di-vllm.yaml`; gfx942; the validated mori121 stack the DeepSeek-V3 and Llama cards run: the AITER release wheel, vLLM with the wide-EP patch, the same router and transports |
| `compat/images/vllm-per-gpu.yaml` | DeepSeek-V3 MTP and DQ, Qwen3 replicas | composed from the two above, one per GPU: extends `vllm-mori121.yaml`, and on gfx950 (`variants.gfx950`) selects the builds the matrix's gfx950 vLLM is validated with. No build of its own |
| `compat/images/sglang-disagg.yaml` | DeepSeek-R1, Qwen3-32B on SGLang | the SGLang release image per GPU, its pip packages, MoRI on gfx942 only (a per-GPU variant) |

The specs inherit: `compat/images/di-vllm.yaml` holds what every vLLM stack shares (build
prerequisites, env, cache locations, the smoke check), and each stack adds only its bases and the
builds it selects. The builds themselves are authorized once in `compat/rocm.yaml` and shared --
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
+      requires: [torch==2.9.0+rocm7.1.0, mori]
+      python: ['3.12']
+      note: generalized from a working build manifest
```

What each part of the image becomes:

| In the image | Becomes |
|---|---|
| a build the matrix has (by version, or by the commit a source build carries) | the stack's, inherited |
| a package the base image already has (torch, triton) | nothing: it is the base's |
| a plain index dependency (msgspec) | nothing: pip installs it |
| a version an authorized index serves | a build entry in the matrix: url, python, and its requirements on components the matrix knows (from the wheel's metadata) |
| a requirement of that wheel the working run contradicts | reported, and left out of the entry: the run shows the wheel works without it |
| a source build at a commit the matrix lacks | a draft entry, from the matrix's build of that repository, for a person to confirm |

When the run passes, the engineer puts both files in their PR; the next run on a stack the matrix
now covers adds nothing. The test `FromABuildToItsGeneralizedForm` in
`scripts/common/tests/test_e2e_examples.py` runs exactly this (with the index stood in for).

## From a card to its example

The same generation, from the existing card: `emit.py vllm_dissag pyt_vllm_disagg_mori_kimi-k3-mxfp4`
gives the example's layers and its values --

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
the same environment, directory and allocation. `PerGpuStack` holds `vllm-per-gpu` equal to the
stack of each GPU it is composed from.
