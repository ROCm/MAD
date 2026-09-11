# Kimi-K3 inference on AMD Instinct MI300X (gfx942)

## Overview

[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) is Moonshot AI's 2.8T-parameter
Mixture-of-Experts model (896 experts, natively MXFP4/QAT, hybrid MLA +
Kimi-Delta-Attention, 1M-token context).

The single-node recipes in [`../README.md`](../README.md) target **MI350X / MI355X
(gfx950)** at TP8 and are skipped on gfx942. MI300X has 192 GB/GPU, so the ~1.5 TB
checkpoint does **not** fit one 8-GPU node — every MI300X recipe here is
**multi-node**. That is the difference: same model, different hardware, different
sharding.

## Models

| MAD tag | Nodes | Parallelism | Expert all2all | Use when |
|---------|-------|-------------|----------------|----------|
| `pyt_vllm_kimi-k3_mi300x_pp2xtp8` | 2 | PP2 × TP8, no EP | — | Simplest baseline; lowest single-user latency |
| `pyt_vllm_kimi-k3_mi300x_wideep_allgather` | 2 | PP2 × TP8, EP8/node | `allgather_reducescatter` | Expert-parallel without MoRI kernels |
| `pyt_vllm_kimi-k3_mi300x_wideep_moriep` | 2 | PP2 × TP8, EP8/node | `mori_low_latency` (MoRI-EP) | MoRI-EP expert dispatch |
| `pyt_vllm_disagg_mori_kimi-k3` | 4 | 2P/2D, TP2 × DP8 → EP16 per pool | MoRI-EP + MoRIIO KV transfer | Highest concurrent throughput |

The first three are **colocated** — one instance spanning 2 nodes, no
prefill/decode split, one request uses all 16 GPUs. The fourth is
**disaggregated**: 2 prefill + 2 decode nodes joined by the MoRIIO connector.

**Pick by workload.** Colocated gives the lowest single-request latency. Disagg
gives **5.7× throughput at concurrency 8 (7.3× at 16)** plus decode-latency
isolation, at roughly 4× higher single-stream latency — one request runs on 2 GPUs
instead of all 16. That trade is architectural, not a tuning defect.

> **EP scope (colocated).** The 896 experts split 8-way across each node's 8 GPUs
> (112 experts/GPU), and that EP8 group is replicated on each of the 2 pipeline
> stages. The expert all2all runs **intra-node**; the only cross-node traffic is
> the PP activation hand-off over NCCL. "16" is the GPU count, not the EP width.

## Quick start

Run from a SLURM login node. Both steps need your image — the model cards carry
`DOCKER_IMAGE_NAME: "<supply-your-image>"` as a fill-me-in marker, and madengine
rejects it rather than trying to pull it (see [Image](#image)).

First fill in your cluster's partition and paths in the matching site config —
[`slurm-config.colocated.json`](slurm-config.colocated.json) (2 node) or
[`slurm-config.disagg.json`](slurm-config.disagg.json) (4 node). Every field marked
`CHANGE-ME` is site-specific; see [Site configuration](#site-configuration).

```sh
IMG=<your-registry>/<repo>:<tag>
CFG=benchmark/kimi_k3/mi300x

# colocated, 2 nodes
madengine build --tags pyt_vllm_kimi-k3_mi300x_pp2xtp8 --use-image "$IMG" \
  --additional-context-file $CFG/slurm-config.colocated.json
madengine run --manifest-file build_manifest.json --live-output

# prefill/decode disaggregated, 4 nodes
madengine build --tags pyt_vllm_disagg_mori_kimi-k3 --use-image "$IMG" \
  --additional-context-file $CFG/slurm-config.disagg.json
madengine run --manifest-file build_manifest.json --live-output
```

The context file is only needed on `build` — its `slurm` and `env_vars` blocks are
written into the manifest's `deployment_config` and merged back on `run`.

To build and distribute the image instead of supplying a pre-built one, swap
`--use-image "$IMG"` for `--registry <your-registry>`; the launcher then pulls it
onto every node in parallel.

Both are `slurm_multi` models: madengine generates a wrapper SBATCH script that runs
the model's `.slurm` script on the head node with `bash`, and that script manages its
own per-node containers via `srun`. The `#SBATCH` header inside the launcher is
therefore inert — every allocation knob comes from madengine. Nothing is passed to the
`.slurm` script on the command line; the topology travels entirely through `env_vars`.

The allocation is sized from each entry's **`slurm.nodes`** (2 or 4), which madengine
emits as `#SBATCH --nodes`; the launcher then reads `SLURM_NNODES` for node discovery.
`distributed.nnodes` carries the same number for the launcher-detection path but is
**not** what sizes the allocation — madengine reads only `slurm.nodes`, defaulting to
1. Keep the two in sync when editing a model card, or override `nodes` in the site
config, where it wins over the model card.

Alternatively, run inside an allocation you already hold, in which case madengine
executes the launcher synchronously and the node count comes from the allocation:

```sh
salloc -N 4 --ntasks-per-node=1 --gres=gpu:8 -p <partition> -t 24:00:00
madengine run --manifest-file build_manifest.json --live-output
```

## Hardware requirements

- **16× MI300X** (2 nodes) colocated, or **32× MI300X** (4 nodes) disaggregated
- RDMA fabric between nodes; the defaults assume 8 NICs per node
- Checkpoint is ~1.5 TB — local NVMe strongly recommended, on every node
- **Docker** on the compute nodes. The launchers call `docker run` under `srun`;
  podman/apptainer-only clusters will not work without editing the launcher.

## Site configuration

Everything cluster-specific lives in the two `slurm-config.*.json` files next to this
README, passed with `--additional-context-file`. Values set there override the model
cards (`build_orchestrator.py` only fills in a model-card key you did *not* set).

| Field | What it is | How to find it |
|-------|-----------|----------------|
| `slurm.partition` | GPU partition to submit to | `sinfo -o '%20P %5D %14F %10G %11l'` — pick a partition whose `GRES` column shows GPUs and whose `A/I/O/T` counts show idle nodes |
| `slurm.gpus_per_node` | GPUs per node (8 on MI300X) | `GRES` column above, or `scontrol show node <node> \| grep Gres` |
| `slurm.nodes` | 2 colocated / 4 disaggregated | Fixed by the recipe; must match `distributed.nnodes` |
| `slurm.time` | Wall clock | `TIMELIMIT` column above is the partition's cap |
| `env_vars.MODEL_DIR` | Directory holding `Kimi-K3/` | Wherever the ~1.5 TB checkpoint lives; must be readable from every node |
| `env_vars.LOG_PATH` | Run logs + per-job `perf.csv` | Any shared, writable path |

Two knobs are **not** settable through the config file on this launcher:

- **`--account` / `--qos`.** madengine emits these only for its templated launchers;
  the hand-built `slurm_multi` header omits them. If your cluster requires an
  account, export `SBATCH_ACCOUNT` (and `SBATCH_QOS`) before `madengine run` —
  `sbatch` honors those environment variables and no directive conflicts with them.
- **`slurm.results_dir`.** Settable in the config file, but *not* from a model card —
  it is absent from the key list madengine copies out of `models.json`.

## Image

`docker/pyt_vllm_kimi_k3_mi300x.ubuntu.amd.Dockerfile` builds the whole stack from
public sources on the open ROCm vLLM CI base: MoRI 1.2.2, AITER 0.1.19 +
flydsl 0.2.4, vLLM with the K3/MoRIIO fixes, and the DP-rank/KV-notify vllm-router.
Every source is pinned to an immutable commit SHA.

It also grafts a K3-aware AITER from the public `amdsiloai/vllm:kimi-k3-mi325x-release-v2`
image. Without that graft the K3 MoE profiling shape finds no tuned FlyDSL config,
falls back to a heuristic kernel, and aborts LLVM inside
`determine_available_memory` — the worker dies natively with no Python traceback.

### The build pins its toolchain, not just its sources

SHA-pinning every source is not sufficient for a reproducible build. pip resolves
build dependencies in an **isolated environment** from PyPI at build time, so a
newer setuptools can break a build whose sources have not moved. That is not
hypothetical: setuptools ≥ 80 added `assert isinstance(self.compiler, CCompiler)`
to distutils' `build_ext.build_extension`, which MoRI's legacy
`Cython.Distutils.build_ext` path violates, failing the `amd_mori` wheel with
`AssertionError: run() must precede build_extension()` — while every pinned SHA
was still correct.

The Dockerfile therefore sets `PIP_CONSTRAINT` globally. That is the only
mechanism that reaches inside pip's build isolation; a plain
`pip install setuptools==X` in the image does **not** affect it.

## Known failure modes

Each of these was hit on a real MI300X cluster and is fixed in-tree. They are
recorded because the symptom is a long way from the cause in every case.

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: run() must precede build_extension()` building `amd_mori` | unpinned build toolchain (above) | `PIP_CONSTRAINT` in the Dockerfile |
| `OCI runtime create failed: ... not a directory`, then the surviving node loops on `Waiting for nodes. . .` forever | Docker creates a *directory* at a missing bind-mount source, so the first run on a node lacking an RDMA lib poisons it for all later runs; `[ -e ]` matches that directory | launchers test with `[ -f ]` |
| `LLVM ERROR: Do not know how to expand this operator's operand!` in `determine_available_memory`, `quantization_config=None` | gfx942 cannot codegen the a16w4 SiTUv2 heuristic kernel; the MoE must be requantized to int4 | all colocated cards set `AITER_SITUV2_A8W4=1` + `--quantization-config` |
| Job reports COMPLETED with `0 successful, 0 failed` and no `perf.csv` | launchers warned but returned 0, so a crash was indistinguishable from a clean run | launchers `exit 1` when no perf CSV is produced |

The RDMA one deserves emphasis: it is **per-node and self-propagating**, so it
presents as an intermittent multi-node failure. A node that has never run this
launcher works; one that has, and lacked the library, fails every time after.

## Cluster prerequisites beyond the model card

A model card cannot express these — they are site facts, and each one blocked a
run until supplied:

- **Wall time** must fit your SLURM *association* limit, which can be lower than
  the partition's. The cards declare `24:00:00`; an association capped at
  `08:00:00` leaves the job `PENDING (AssocMaxWallDurationPerJobLimit)` forever
  rather than failing. Check with
  `sacctmgr show assoc user=$USER format=Account,QOS,MaxWall`.
- **An account** may be required (`--account`). madengine emits `#SBATCH
  --account` for `slurm_multi` only after the fix in its `di_redesign` line; with
  older madengine, export `SBATCH_ACCOUNT`.
- **Docker on the compute nodes**, not just the login node.
- **A way to distribute the image.** `--build-on-compute` requires `--registry`,
  so on a registry-less cluster neither madengine nor MAD can get an image onto
  the nodes. Building once and `docker save`ing to shared storage, then
  `docker load` per node, works and needs no registry.

## Where the configuration lives

Nothing K3-specific is baked into the image, matching the shared disagg image's
design:

| What | Where |
|------|-------|
| Model serving recipe (gfx942 knobs, KV cache, cudagraph modes, MoE quant) | `scripts/vllm_dissag/models.yaml`, entry `Kimi-K3` |
| Per-variant topology (TP/PP/EP, node count, benchmark) | `models.json` `env_vars` |
| ROCm platform + MoRI/RDMA fabric env | `scripts/vllm_dissag/connectors/moriio.env` |

### gfx942 specifics

- **`VLLM_ROCM_USE_AITER_MLA=0` is required** — the AITER MLA kernel is gfx950-only.
- gfx942 has no scaled-MXFP4 MFMA and the a16w4 SiTUv2 heuristic FlyDSL kernel
  cannot codegen there, so the MoE is requantized to packed int4 and run through
  SiTUv2 (`AITER_SITUV2_A8W4=1` plus `--quantization-config
  '{"moe":{"weight":"int4_per_group_32"}}'`).
- **`--max-num-batched-tokens` stays at 2048.** 8192 corrupts generation on this
  stack, and the 16384 profiling shape crashes LLVM codegen in the heuristic kernel.
- **`KV_CACHE_MEMORY_BYTES=40e9`** gives a 2.84M-token GPU KV cache. The lower 8e9
  value used during bring-up was a `profile_run`-hang workaround, not a memory
  limit; single requests beyond ~600K tokens need the larger cache.
- **Why TP2 in the disagg recipe:** K3's replicated (attn + shared-expert) weight is
  106.5 GiB. At TP1/DP16 that is 190.7 GiB/GPU before KV cache or the 16 GiB MoRI
  heap — it does not fit 192 GB. TP2 shards it to 53.3 GiB/GPU, giving
  137.5 GiB of weights per GPU with room to spare.

## Results — single-needle NIAH (disaggregated 2P/2D)

Needle `HELIOTROPE-7492`, greedy (temp=0), depths 0.1 / 0.5 / 0.9. All PASS,
deterministic, across the full native context range.

| context | result | eval time / request |
|---------|--------|---------------------|
| 10K–200K | 3/3 PASS | 5–88 s |
| 300K | 3/3 PASS | ~150 s |
| 500K | 3/3 PASS | ~301 s |
| 750K | 3/3 PASS | ~542 s |
| **900K** | **3/3 PASS** | **~717 s** |

Scaling is sub-quadratic. Reaching the top of that range needs
`--max-model-len 1000000` (in the `Kimi-K3` models.yaml `dp:` block) and
`KV_CACHE_MEMORY_BYTES=40000000000` (in its `env:` block) — both are the defaults
here.

> **Context units.** The NIAH harness sizes its haystack in **words**
> (`NIAH_WORDS`), and words are roughly 1.3× tokens for this filler. The table
> above is in tokens; the CSV rows emitted by a `BENCHMARK_SCRIPT=niah` run are
> labelled in words. Do not compare the two columns directly.

**One-time warmup:** a fresh serve pays a single aiter MLA-kernel JIT compile
(`fmha_fwd_hd192x128`, ~15 min) on the first ≥200K-token request, cached
thereafter. The times above are warm.

**Known residual:** the stricter 10-needle stress dips to ~9/10 at ≥20K — an RDMA
write-visibility race. Single-needle retrieval is unaffected.

### The three fixes behind these numbers

All three are committed in the vLLM the image builds (pinned by SHA); nothing is
patched at runtime.

1. **4-KV-cache-group block routing.** K3's hybrid attention allocates **4** KV
   cache groups (3 KDA/mamba + 1 MLA). The stock connector hardcoded 2-group
   indices and sent MLA KV to mamba block ids, so decode read empty blocks and
   generated fluent but context-free text. Each layer is now routed by its own
   group index.
2. **Multi-chunk prefill transfer.** The final-chunk gate used block count, which
   fires after chunk 1 when a prompt fits in ≤1 padded block — so only
   `max_num_batched_tokens` of KV ever crossed, a razor cliff at 2048. The gate now
   keys on compute progress from `scheduler_output`.
3. **KDA gather made sync-free.** `gather_initial_states` ran a diagnostic
   `bool((indices >= n).any())` per KDA layer per prefill chunk, each forcing a
   device→CPU sync — ~25k full stream drains at 750K, which presented as a hang for
   contexts above ~500K. The index clamp already made the address safe, so the
   diagnostic is now behind `K3_KDA_GATHER_LOG=1` (default off). Correctness is
   unchanged; 750K and 900K went from hanging indefinitely to passing.

## Benchmarks and results reporting

Both launchers accept `BENCHMARK_SCRIPT`:

| value | script | reports |
|-------|--------|---------|
| `sweep` | `benchmark_xPyD.sh` | throughput sweep, tok/s per (isl, osl, concurrency) |
| `long_context` | `benchmark_long_context.sh` | per-shape warmup, concurrency 1 first |
| `niah` (default for these entries) | `benchmark_niah.sh` | retrieval accuracy, needles found /10 per context size |

All three land in madengine's `perf.csv` via `parse_to_csv.py`, so the NIAH numbers
above are a CI-visible metric rather than a table in a markdown file. A context size
whose request errors is recorded as a `FAILURE` row with performance 0, so a
pass→crash regression shows up instead of silently disappearing.

The launcher copies that CSV to `perf_Kimi-K3.csv` beside itself, which is the name
each entry declares as `multiple_results`. Note that madengine's `slurm_multi`
collector does **not** resolve `multiple_results` — it globs `perf*.csv` under
`slurm.results_dir`, then falls back to `/shared_inference/$USER/model_blog_logs/
$SLURM_JOB_ID/perf.csv` and a handful of other conventional paths. That is why the
site configs set `results_dir` to the launcher's own directory
(`scripts/vllm_multinode` or `scripts/vllm_dissag`): it is what makes the declared
filename findable. `multiple_results` still carries the name for the non-SLURM paths.

### What the workload reports vs. what madengine reports

The CSV is **narrow**: the benchmark reports only what it measured, and madengine
merges in the run metadata it already owns. This is the same contract the templated
launchers use, so these gfx942 multi-node rows and the gfx950 single-node rows in
[`../README.md`](../README.md) describe themselves identically in `perf.csv`.

| From the benchmark | From madengine |
|---|---|
| `model`, `performance`, `metric`, `status` | `nnodes`, `n_gpus`, `gpus_per_node`, `launcher` |
| `benchmark`, `context_words` | `docker_image`, `base_docker`, `docker_sha` |
| `tp`, `pp`, `ep_backend`, `prefill_decode` | tags, pipeline, build number, machine name |

The right-hand column used to be hand-written by `parse_to_csv.py` from `xP`/`yD`,
which is why a colocated 2-node/16-GPU run reported itself as a 1-node/8-GPU
`disagg_1P0D` on a `nixl` backend it never used — the colocated launcher sets
`xP=1 yD=0` only to keep the shared log filenames unique. A workload cannot reliably
know its own topology; madengine can, so it does.

`status` is reported explicitly because performance alone cannot express this
benchmark's failure mode: a context size whose request errors scores 0, which is a
real measurement, and deriving status from it would file the failure as a SUCCESS.

> `BENCHMARK_SCRIPT=sweep` still uses the older full-schema CSV (the disagg model
> cards that declare no `multiple_results` depend on it). Pass `--narrow` to
> `parse_to_csv.py` to put a sweep on the same contract.

## Relationship to PR #193

These recipes originate in [PR #193](https://github.com/ROCm/MAD/pull/193), which
shipped them as standalone shell scripts under `scripts/vllm/kimik3_mi300x/`. This
integration keeps the recipes and the findings and drops the duplicated machinery:

- the 2P/2D topology is expressed as `xP=2 yD=2 TP_SIZE=2` on the existing
  `scripts/vllm_dissag/` harness instead of a bespoke ssh orchestrator
- the three colocated variants share one launcher, differing only in
  `ENABLE_EP` / `ALL2ALL_BACKEND` / `AITER_SITUV2_A8W4`
- the 32 runtime patchers are gone — their fixes are in the SHA-pinned vLLM
- the NIAH probes are the harness's existing `benchmark_niah.py`

See also [`../README.md`](../README.md) for the gfx950 single-node recipes.
