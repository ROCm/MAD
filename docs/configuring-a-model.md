# Configuring a distributed-inference model

How to add or tune a multinode model (vLLM disaggregated, vLLM colocated, SGLang disaggregated),
the way that is familiar first, then the way that keeps one card working on every GPU and lets a
fleet of cards be maintained in one place. Both are supported; nothing below requires the second.

## Contents

- [The familiar way: a card and a recipe](#the-familiar-way-a-card-and-a-recipe)
- [What a run will do, and why](#what-a-run-will-do-and-why)
- [Using the generated configuration](#using-the-generated-configuration)
- [When you need more](#when-you-need-more)
- [Where everything lives](#where-everything-lives)

## The familiar way: a card and a recipe

A card in `scripts/<launcher>/models.json` names the launcher, the node count, the image and the
run's settings; the model's recipe in that launcher's `models.yaml` holds its engine flags and env.
This is unchanged:

```json
{"name": "pyt_vllm_disagg_mori_deepseek-v3", "scripts": "run_xPyD_models.slurm",
 "dockerfile": "../../docker/di_vllm", "docker_build_arg": {"PINS": "vllm-mori121"},
 "env_vars": {"MODEL_NAME": "DeepSeek-V3", "xP": "1", "yD": "1"}, ...}
```

```yaml
# scripts/vllm_dissag/models.yaml
DeepSeek-V3:
  env: {...}
  prefill: {tp: "...", dp: "..."}     # engine flags per role and parallel mode
  decode: {tp: "...", dp: "..."}
```

Anything can be overridden for one run without editing either file: submit-time env
(`--additional-context '{"env_vars": {...}}'` through madengine, or an `export` under plain
`sbatch`) wins over every layer.

## What a run will do, and why

Every run writes `config.resolved.json` into its log directory: each value and the layer that set
it (site, connector env, recipe, topology, workload, submit). Before a run:

```bash
python3 scripts/common/config/emit.py vllm_dissag pyt_vllm_disagg_mori_deepseek-v3
```

prints the card's configuration as a workload file that fits the shared layers: the layers
(`scripts/common/workloads/`) whose values the card has, and the part that is the card's own. Its
recipe and connector env stay where they live (`models.yaml`, `connectors/`), so a fix there
reaches it. `madengine build --emit-config DIR` writes it for every card it builds, and the
pipeline records it with each run (`run_configs/`, the build manifest).

## Using the generated configuration

That file is also a workload file the launcher runs. Point the card at it and the run is the same
-- CI checks this for every card (`snapshot.py --from-emitted`: identical engine commands and
container environment on each GPU the recipe covers):

```json
"env_vars": {"MAD_CONFIG": "workloads/deepseek-v3.yaml"}
```

From then on the card's own part is in one place you can edit, and the shared parts stay
shared: a serving shape, a benchmark or a recipe changed in its layer changes every card on it. A
card that needs its own recipe or connector values can add them to its file (`recipes:`,
`connectors:`), over the shared ones. `emit.py --all -o DIR --suggest` lists values many cards keep
that no layer holds yet: candidates for a new shared layer.

## When you need more

**One card for every GPU.** Give the recipe what differs per GPU as a variant, and its memory as a
fraction of the GPU's, instead of a card per GPU:

```yaml
Kimi-K3-MXFP4:
  default_arch: gfx942
  variants:
    gfx942: {env: {...}, prefill: {dp: "... --quantization-config ..."}}
    gfx950: {env: {...}, prefill: {dp: "..."}}
  memory: {KV_CACHE_MEMORY_BYTES: {fraction: 0.1}}
```

The launcher detects the GPU and runs its variant (`pyt_vllm_disagg_mori_kimi-k3-mxfp4_auto` runs
on MI300X and MI355X).

**Refused before queueing, with the reason.** Say what each serving role needs; the launcher (and
madengine, before the job waits in a queue, when the target GPU is known) checks it against the GPU and the image's engine build,
refuses what cannot run with the reason, and adds the arguments a GPU needs (an int4 conversion on
MI300X, a KV-cache dtype):

```yaml
  roles:
    default: {quantization: {weights: mxfp4, activations: fp8}, kv_cache: fp8}
```

**Many cards from shared layers.** A workload file inherits layers with `defaults:`: a serving
shape, a benchmark, a model family (`scripts/common/workloads/`). Changing NIAH settings, or a fact
of a model family, is then one edit for every card that uses it:

```yaml
version: 2
defaults: [../../../common/workloads/family/kimi-k3.yaml, ../../../common/workloads/bench/niah.yaml]
model:
  env: {MODEL_NAME: Kimi-K3-MXFP4, xP: "2", yD: "2"}
```

[e2e-examples.md](e2e-examples.md) has nine worked examples; the ones marked *reproduction* run exactly what
an existing card runs, the ones marked *demo* show a mechanism.

**An image described, not written.** Keep building your image your way (a Dockerfile). Once a
run with it works, the stack it proved is generalized into `compat/rocm.yaml`, the dependency state
of what works together. A card can then select an image spec composed from those authorized builds
instead of a Dockerfile (a folder's `image_spec.yaml`, or a shared one in `compat/images/`):

```json
"image_spec": {"extends": "${image_spec}", "gpu_arch": "gfx950"}
```

**The best-known configuration.** A recorded best configuration for the model, platform and
shape (`scripts/common/perf/best_known.jsonl`) is laid over the recipe by default;
`MAD_BEST_CONFIG=0` runs the recipe as written.

## Where everything lives

What a model developer edits:

| What | Where |
|---|---|
| The card | `scripts/<launcher>/models.json` |
| The recipe | `scripts/<launcher>/models.yaml` (the colocated launcher reads `vllm_dissag`'s) |
| A card's own part, on the shared layers (optional) | the workload file its `MAD_CONFIG` names |
| Shared layers (optional) | `scripts/common/workloads/` |

What platform and CI owners maintain, and runs read on their own:

| What | Where |
|---|---|
| Software matrix, engine builds and what each runs per GPU | `compat/rocm.yaml` |
| GPUs, fabrics, node platforms | `compat/hardware.yaml` |
| Shared image specs | `compat/images/` |
| Example sites | `compat/sites/` (a site keeps its own outside the repository: `MAD_CLUSTER_FILE`) |
| Topology specs | `scripts/common/topology/specs/` |
| Best-known configurations | `scripts/common/perf/best_known.jsonl` |
| Connector defaults | `scripts/<launcher>/connectors/*.env` |

The layer order, the record of where each value came from, and every variable:
[configuration.md](configuration.md).
