# Context as code: user guide

A distributed-inference run depends on a lot of context: the model and how its engine is tuned on
each GPU, the serving shape, the benchmark, the image and everything in it, which versions work
together, the cluster's fabric and storage. In MAD all of it is code -- small files, each owned by
whoever knows that part, each inheriting what it does not change. The more context is written down,
the more the system can check, reproduce and reuse, because each part can be changed on its own,
by the person whose scope it is, without touching anyone else's.

This page is the overview. [context_as_code_developer_guide.md](context_as_code_developer_guide.md)
has the details, worked end to end.

## One run, five scopes

| Scope | The file | Who owns it | Example |
|---|---|---|---|
| **What to run** | a workload file: shape, benchmark, the model | whoever runs the workload | `xP: "2", yD: "2"`, the NIAH benchmark |
| **How the engine runs a model** | its recipe, with a variant per GPU | the model's developer | Kimi-K3: int4 conversion on MI300X, native MXFP4 on MI355X |
| **What it runs in** | an image spec: a base and a selection of builds | the stack's owner | `vllm-kimi-k3`: MoRI, AITER, vLLM with its patch, the transport stack |
| **What works together** | the compatibility matrix: each build once, what it needs, what it runs | everyone, through the runs they make | vLLM build X runs FP8 and MXFP4 on gfx950 |
| **Where it runs** | a site profile, platforms and fabrics | the site's owner | AINIC rails `rdma0…7`, shared storage at `/shared` |

A run puts the five together. Each scope's file is small because it inherits: a workload is a few
lines on shared layers (`base` → serving method → model family → benchmark), a stack extends the
one every vLLM stack shares, a new model's recipe extends a validated one.

## What you get

- **Change one thing, in one place.** Scale 1P1D to 2P2D (on the 4 nodes it then asks for), swap
  MoRI-IO for NIXL, add a model with a longer context, move to another site: each is one small file
  or one value, and nothing else moves.
  The developer guide shows each change next to what the run then does.
- **Values that follow each other.** One knob can size both pools and name the run; set it at
  submit time and everything built from it follows.
- **One file, every GPU and every site.** The GPU picks its recipe variant and its image; the site
  picks its fabric. The workload does not change.
- **Checked before anything starts.** The recipe's needs are checked against the GPU and against
  what the image's engine build runs, the allocation against the shape, the nodes' GPUs against the
  recipe. A run that cannot work is refused with the reason, before any engine starts.
- **Reproducible from its record.** Every build writes the run's configuration in this form; running
  any card from its generated file gives the same engine commands and environment (checked for every
  card on every GPU it runs on).
- **The same run everywhere it is launched.** SLURM, Kubernetes, madengine or the pipeline's
  standalone mode give the same environment, checked card by card.
- **A matrix that gets more accurate as people work.** Build your image your own way. When it runs,
  the build writes what it proved: the image as the stack it is closest to plus what differs, and
  the facts the matrix lacks. Put them in your PR; the next run on that stack adds nothing.

## How to use it

**Run a card.** Nothing changes: `madengine build` and `madengine run` (or `sbatch` with the
launcher) as before. A card with a workload file (`MAD_CONFIG`) reads it; without one it runs as it
always has.

**Change one run.** Pass the value at submit time; it wins over every layer:

```bash
--additional-context '{"env_vars": {"BENCHMARK_COMBINATIONS": "8192/1024"}}'
```

**Add your model.** Write a workload file that inherits the layers your run shares, and a recipe that
aliases the closest validated one and changes what yours needs -- in the same file if you like:

```yaml
version: 2
defaults: [deepseek-v3.yaml]
model:
  env: {MODEL_NAME: DeepSeek-V3-LongCtx, MODEL_WEIGHTS_NAME: DeepSeek-V3}
recipes:
  DeepSeek-V3-LongCtx:
    alias: DeepSeek-V3
    memory: {KV_CACHE_MEMORY_BYTES: {fraction: 0.25}}
    decode: {dp: "--max-model-len 131072 --max-num-seqs 64"}
```

**Run somewhere else.** Name the site (`MAD_CLUSTER=<site>`); the GPU is detected. The workload stays.

**Bring your own image.** Keep your Dockerfile. Build with `madengine build --emit-config DIR` (and
`"image_probe": true`): DIR gets your run's configuration and, under `compat/`, what your stack adds
to the matrix. When the run passes, commit both with your change.

**Start from what exists.** `scripts/common/config/emit.py <catalog> <card>` prints any existing
card as a workload file on the shared layers; `--all --suggest` does every card and proposes new
shared layers.

## Where to go next

- [How it fits together](context_as_code_developer_guide.md#how-it-fits-together-kimi-k3-end-to-end):
  the files behind one Kimi-K3 run, the full system they resolve to on each GPU.
- [Change anything, see what the run does](context_as_code_developer_guide.md#change-anything-see-what-the-run-does):
  seven changes, each with its effect on the run.
- [How compat gets more accurate over time](context_as_code_developer_guide.md#how-compat-gets-more-accurate-over-time).
- [configuring-a-model.md](configuring-a-model.md): adding or tuning a model, step by step.
