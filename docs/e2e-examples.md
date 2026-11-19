# End-to-end examples

Eight workloads, five model families, two engines, every serving type MAD runs and the next GPU
generation, each written as a few lines on shared layers. They show the features working together,
and their tests run each one through the real launchers on each GPU it targets.

## Contents

- [The workload library](#the-workload-library)
- [The examples](#the-examples)
- [What each example proves](#what-each-example-proves)
- [Running them](#running-them)
- [Maintaining workloads with inheritance](#maintaining-workloads-with-inheritance)
- [How they are tested](#how-they-are-tested)

## The workload library

A workload file says what to run and what to measure. It is madengine's layered config
(`MAD_CONFIG`, version 2), whose `defaults:` list inherits other files with OmegaConf: mappings
merge in order, the file itself on top, and `${...}` interpolations resolve after the merge
(`${model.env.TP_SIZE}`, `${oc.env:NAME,default}`). Both Jenkins modes resolve it with the same
madengine code, and anything passed through config at submit time wins over every layer.

```
scripts/common/workloads/
  base.yaml                     RECIPE_CHECK=1, MAD_BEST_CONFIG=1 -- the root of every workload
  serve/vllm-disagg-wide-ep.yaml   MoRI-IO disaggregation, wide EP
  serve/vllm-colocated.yaml        one TP x PP server over several nodes
  serve/replicas.yaml              independent servers per node behind a router (PERF_TAGS by interpolation)
  serve/sglang-disagg-dp-ep.yaml   SGLang disaggregation, DP attention + EP
  bench/{sweep,niah,accuracy,embeddings,lora}.yaml   what is measured
  family/kimi-k3.yaml              facts true of every Kimi-K3 workload (local weights only)
  family/deepseek-v3.yaml          DeepSeek-V3 family
```

Site facts are not here: a run names its site (`MAD_CLUSTER` / `MAD_CLUSTER_FILE`, a pipeline
context, `MAD_PLATFORM`), so one workload runs on every site.

## The examples

| Card | Model, type | Workload (inherits) | Runs on |
|---|---|---|---|
| `pyt_e2e_kimi-k3_disagg_niah` | Kimi-K3, MoE MXFP4, 2P2D | family/kimi-k3 + bench/niah | gfx942, gfx950 |
| `pyt_e2e_deepseek-v3_disagg_mtp` | DeepSeek-V3, MoE FP8, 1P1D | family/deepseek-v3 + bench/accuracy | gfx942, gfx950 |
| `pyt_e2e_deepseek-v3_dq` | DeepSeek-V3, disaggregated quantization | family/deepseek-v3 + bench/accuracy | gfx950 (refused on gfx942) |
| `pyt_e2e_kimi-k3_helios` | Kimi-K3 on Helios, 72-GPU pools | family/kimi-k3 + bench/sweep | CDNA 5 once brought up (refused today) |
| `pyt_e2e_llama-3.1-405b_colocated` | Llama-3.1-405B, dense FP8, TP8 x PP2 | serve/vllm-colocated + bench/accuracy | gfx942, gfx950 |
| `pyt_e2e_qwen3-32b_replicas_lora` | Qwen3-32B, dense BF16, replicas + LoRA | serve/replicas + bench/lora | gfx942, gfx950 |
| `pyt_e2e_qwen3-embedding_replicas` | Qwen3-Embedding-8B, embeddings | serve/replicas + bench/embeddings | gfx942, gfx950 |
| `pyt_e2e_deepseek-r1_sglang` | DeepSeek-R1 on SGLang, DP + EP | serve/sglang-disagg-dp-ep + bench/sweep | gfx942, gfx950 |

The cards (tags `e2e_example`, `any_gpu`) hold only what registers a run -- name, image, node
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
affinity on Kubernetes) and verifies the nodes are CDNA 5. Until a pin set and engine build exist
for CDNA 5 the run is refused with what is missing (`recipe.py bringup --arch cdna5`).

**Llama-3.1-405B, colocated.** A dense model over two nodes as one TP8 x PP2 server; the recipe's
roles add the FP8 KV-cache argument its server flags lack.

**Qwen3-32B, replicas with LoRA.** Eight TP-1 servers per node behind a router, each serving two
adapters beside the base model; the LoRA benchmark spreads requests over the adapters.

**Qwen3-Embedding-8B.** Pooling servers as replicas; embeddings throughput and latency per
concurrency.

**DeepSeek-R1 on SGLang.** The same layers, the second engine: DP attention + MoRI EP.

## Running them

Through madengine (either Jenkins mode resolves the same workload):

```bash
madengine build --tags pyt_e2e_deepseek-v3_disagg_mtp --additional-context-file <site context>
madengine run --manifest-file build_manifest.json --additional-context-file <site context>
```

From the pipeline, per site (`rocAutomation/.../ci_trigger.sh`, `SITES="oci-mi300x crs-spur"`), or
`EXECUTION_MODE=STANDALONE` for the same run without madengine at run time.

Anything can be overridden for one run without editing a file -- submit-time values win:

```bash
--additional-context '{"env_vars": {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0", "ACC_TASKS": "/data/gsm8k.jsonl"}}'
```

## Maintaining workloads with inheritance

- A fact lives in one layer: a model family's (`family/kimi-k3.yaml`), a serving shape's
  (`serve/replicas.yaml`), a benchmark's (`bench/niah.yaml`). Changing it there changes every
  workload that inherits it, and the tests show what moved.
- A new workload is a few lines: its `defaults:` and what differs.
- A new GPU needs no workload change: the recipe gains a variant, the catalog an entry.
- A run-level change is config, not a file edit: submit-time env, a site profile, `--config` groups.

## How they are tested

`scripts/common/tests/test_e2e_examples.py` resolves every workload with madengine and runs it
through the real launchers and role scripts (the snapshot harness's stand-ins for SLURM, Docker and
the GPUs), on each GPU it targets, asserting what it claims: the int4 conversion on gfx942 only,
the best-known record applied and switched off, MTP on decode only, the KV budget per GPU, the DQ
refusal and per-role checkpoints, the Helios topology and refusal, the replicas and their LoRA or
pooling flags, the SGLang DP flags. Their goldens are in the snapshot suite, and the pipeline's
path-equivalence check runs every example card in STANDALONE and through madengine and requires
the same environment, directory and allocation.
