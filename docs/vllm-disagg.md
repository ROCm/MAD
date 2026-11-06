# vLLM disaggregated prefill/decode: reference

This is the complete reference for [`scripts/vllm_dissag/`](../scripts/vllm_dissag/),
the launcher for **disaggregated prefill/decode (P/D) inference** with vLLM on AMD GPUs
over a SLURM cluster. One launcher, `vllm_disagg.sh`, drives every supported
configuration through two axes: the KV-transfer **connector** and the **parallelism**
mode. Per-model flags come from the `models.yaml` catalog.

New to the terms? Read [multinode-overview.md](multinode-overview.md) first; it also
holds the architecture diagrams. How to run a card through madengine or `sbatch`, logs
and failures are in [multinode-running.md](multinode-running.md). The benchmarks and
`perf.csv` are in [benchmarks-and-results.md](benchmarks-and-results.md).

## Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [The axes and the four valid combinations](#the-axes-and-the-four-valid-combinations)
- [Supported models](#supported-models)
- [Model cards](#model-cards)
- [Kimi-K3 notes](#kimi-k3-notes)
- [Prerequisites](#prerequisites)
- [Building the Docker image](#building-the-docker-image)
- [Quick start](#quick-start)
- [Run modes: examples](#run-modes-examples)
- [Adding a new model](#adding-a-new-model)
- [Per-model environment layering](#per-model-environment-layering)
- [Node topology and ports](#node-topology-and-ports)
- [Proxy options](#proxy-options)
- [Connector hooks](#connector-hooks)
- [Recipe knobs the connectors read](#recipe-knobs-the-connectors-read)
- [DeepEP options](#deepep-options)
- [Fabric, timeouts and the container](#fabric-timeouts-and-the-container)
- [Validation status](#validation-status)
- [Benchmark configuration](#benchmark-configuration)
- [Long-context retrieval (NIAH)](#long-context-retrieval-niah)
- [Agentic replay benchmark](#agentic-replay-benchmark)
- [Dry run](#dry-run)
- [Testing](#testing)
- [Caveats](#caveats)
- [File map](#file-map)

## Architecture at a glance

```
run_xPyD_models.slurm     # sbatch entry: picks nodes, plumbs env, launches the container per node
        │
        ▼
vllm_disagg.sh            # THE launcher: role branching, barrier, proxy, benchmark, cleanup
   ├── parallelism.sh     #   WIDE_EP=0 (TP)  vs  WIDE_EP=1 (wide expert-parallel)
   ├── connectors/rixl.sh #   NixlConnector   (TP, and DeepEP wideEP)
   ├── connectors/moriio.sh #  MoRIIOConnector (MoRI-EP wideEP, and MoRIIO+TP)
   └── models.yaml        #   per-model CLI flags + per-model env overrides
```

`run_xPyD_models.slurm` runs on the batch host. It:

1. finds its own directory and checks its required files are there;
2. sources [`scripts/common/cluster.sh`](../scripts/common/cluster.sh) for site
   defaults;
3. validates `MODEL_NAME` and resolves the axes, gating the model against the
   per-combination allowlists;
4. loads `connectors/<CONNECTOR>.env` and forwards each line as `docker -e`;
5. reads `EP_TP_SIZE` and `GPU_ARCHS` from the model's recipe, checks the GPU
   architecture, reports the RDMA rails, and resolves `MODEL_PATH` on every node;
6. picks the first `xP + yD` nodes and gathers their IPs;
7. on each node, stops old containers, frees the barrier ports, pulls the image, and
   runs one container whose entry point is `vllm_disagg.sh`;
8. copies `perf.csv` to `perf_<MODEL_NAME>.csv` beside itself, or exits 1 if there is
   none.

Inside each container, `vllm_disagg.sh` resolves the axes again, does the topology
math, parses `models.yaml`, sources `parallelism.sh` and the connector, waits at a
container barrier, and branches on `NODE_RANK`. The diagrams are in
[multinode-overview.md](multinode-overview.md#architecture-diagrams-vllm-disaggregated).

## The axes and the four valid combinations

Three variables select the behaviour: the KV-transfer **`CONNECTOR`**, the
**`WIDE_EP`** parallelism mode and, for wideEP only, the all-to-all **`EP_BACKEND`**,
which is validated against the connector. wideEP pairs each connector with its own
backend, so the two cross-pairs are rejected:

| # | `CONNECTOR` | `WIDE_EP` | `EP_BACKEND` | Valid | What it is |
|---|---|---|---|---|---|
| 1 | `rixl` | `0` (TP) | none | yes | NIXL + TP (dense, tensor-parallel) |
| 2 | `moriio` | `0` (TP) | none | yes | MoRIIO + TP (dense, tensor-parallel) |
| 3 | `moriio` | `1` (wideEP) | `mori` | yes | MoRI-EP (wideEP DP+EP, mori all2all) |
| 4 | `rixl` | `1` (wideEP) | `deepep` | yes | DeepEP (wideEP DP+EP, deepep all2all) |
| 5 | `moriio` | `1` (wideEP) | `deepep` | no | cross-pair, aborts |
| 6 | `rixl` | `1` (wideEP) | `mori` | no | cross-pair, aborts |

`EP_BACKEND` defaults to the connector's partner (`moriio` to `mori`, `rixl` to
`deepep`), so you rarely set it. The legacy flags map on: `RUN_MORI=1` is combo 3,
`RUN_DEEPEP=1` is combo 4, and neither is combo 1. An invalid `CONNECTOR` (not `rixl`
or `moriio`) or `WIDE_EP` (not `0` or `1`) aborts.

`PARALLEL_MODE` follows `WIDE_EP`: `WIDE_EP=1` is `dp`, `WIDE_EP=0` is `tp`. It picks
which `models.yaml` blocks apply.

## Supported models

Use these exact `MODEL_NAME` values. Each must be in `models.yaml` and in the
allowlists in `run_xPyD_models.slurm`. The **Combos** column lists which of the four
valid combinations each model may run; the gate in `run_xPyD_models.slurm` rejects any
other pairing. Dense models are TP-only. The DeepSeek family, `GLM-5.1-FP8` and the
Kimi-K3 models are wideEP-only: TP is rejected, because their recipes need the wideEP
serve path. For DeepSeek, the TP argv would double the model's own
`--compilation-config` and drop the mandatory `+quant_fp8` op.

| Model | Type | Combos | Notes |
|---|---|---|---|
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | dense | 1, 2 (TP) | Validated serving (moriio + TP). Its recipe exports `RCCL_MSCCL_ENABLE=0` because RCCL's MSCCL scratch buffers filled the GPU. |
| `Llama-3.1-405B-Instruct-FP8-KV` | dense | 1, 2 (TP) | |
| `Qwen3-32B` | dense | 1, 2 (TP) | Validated serving (MoRIIO + TP). |
| `gpt-oss-120b` | MoE | 1, 2 (TP) | |
| `Qwen3-30B-A3B` | MoE | 1, 2 (TP) | Needs a co-versioned AITER image (see [Caveats](#caveats)). |
| `DeepSeek-V3` | MoE | 3, 4 (wideEP only) | Validated serving (moriio + MoRI-EP). |
| `DeepSeek-V3-5layer` | MoE | 3, 4 (wideEP only) | Smoke and development variant. |
| `DeepSeek-R1` | MoE | 3, 4 (wideEP only) | Needs a co-versioned AITER image (see [Caveats](#caveats)). |
| `GLM-5.1-FP8` | MoE + DSA | 3 (moriio wideEP only) | Own image (`glmv5.1` Dockerfile). 2P/2D EP16 and 4P/4D EP32 validated on the current pins (NIAH about 93 to 97 percent). moriio + TP is untested. |
| `Kimi-K3` | MoE | 3 (moriio wideEP only) | MI300X gfx942, 2P/2D, TP2 x DP8 (`EP_TP_SIZE=2`). `docker/vllm_kimi_k3` image. Card `pyt_vllm_disagg_mori_kimi-k3`. |
| `Kimi-K3-MXFP4` | MoE | 3 (moriio wideEP only) | MI300X gfx942, 2P/2D, TP2 x DP8. `docker/vllm_kimi_k3` image. Card `pyt_vllm_disagg_mori_kimi-k3-mxfp4`. Its recipe serves the name `kimi-k3`. |
| `Kimi-K3-MXFP4-MI355X` | MoE | 3 (moriio wideEP only) | MI355X gfx950, bring-up (not yet run). Native MXFP4, TP1 x DP16 (`EP_TP_SIZE=1`). `docker/vllm_kimi_k3` image built for gfx950. Card `pyt_vllm_disagg_mori_kimi-k3-mxfp4_mi355x`. |

The allowlists in `run_xPyD_models.slurm`:

| List | Gates | Members |
|---|---|---|
| `VALID_MODELS` | every run | all twelve models above |
| `WIDE_EP_ONLY_MODELS` | rejects `WIDE_EP=0` | `DeepSeek-V3`, `DeepSeek-V3-5layer`, `DeepSeek-R1`, `GLM-5.1-FP8`, `Kimi-K3`, `Kimi-K3-MXFP4`, `Kimi-K3-MXFP4-MI355X` |
| `MORI_EP_VALID_MODELS` | `CONNECTOR=moriio WIDE_EP=1` | same seven |
| `DEEPEP_VALID_MODELS` | `CONNECTOR=rixl WIDE_EP=1` | `DeepSeek-V3`, `DeepSeek-V3-5layer`, `DeepSeek-R1` |

A standalone reference launcher for Kimi-K3 disagg (for debugging) lives out of tree,
in PR #241; it is not part of this framework.

## Model cards

The cards in [`models.json`](../scripts/vllm_dissag/models.json). All run
`run_xPyD_models.slurm` with the `slurm_multi` launcher.

| Card | `MODEL_NAME` | Nodes | Axes | Benchmark | Image |
|---|---|---|---|---|---|
| `pyt_vllm_disagg_nixl_deepseek-v3` | `DeepSeek-V3` | 2 | `WIDE_EP=1`, no legacy flag: rixl + deepep | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_nixl_deepseek-r1` | `DeepSeek-R1` | 2 | same | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_nixl_deepseek-v3-5layer` | `DeepSeek-V3-5layer` | 2 | same | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_nixl_llama-3.1-405b-fp8` | `Llama-3.1-405B-Instruct-FP8-KV` | 2 | no flags: rixl + TP | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_nixl_llama-3.3-70b-fp8` | `amd-Llama-3.3-70B-Instruct-FP8-KV` | 2 | rixl + TP | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_nixl_gpt-oss-120b` | `gpt-oss-120b` | 2 | rixl + TP | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_deepseek-v3` | `DeepSeek-V3` | 2 | `RUN_MORI=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_deepseek-r1` | `DeepSeek-R1` | 2 | `RUN_MORI=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_deepseek-v3-5layer` | `DeepSeek-V3-5layer` | 2 | `RUN_MORI=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_glm-5.1-fp8` | `GLM-5.1-FP8` | 2 | `RUN_MORI=1`, `EP_TP_SIZE=8` | sweep 1024/1024 | `vllm_disagg_inference.glmv5.1` |
| `pyt_vllm_disagg_mori_glm-5.1-fp8_niah` | `GLM-5.1-FP8` | 2 | `RUN_MORI=1`, `EP_TP_SIZE=8` | niah | `vllm_disagg_inference.glmv5.1` |
| `pyt_vllm_disagg_deepep_deepseek-v3` | `DeepSeek-V3` | 2 | `RUN_DEEPEP=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_deepep_deepseek-r1` | `DeepSeek-R1` | 2 | `RUN_DEEPEP=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_deepep_deepseek-v3-5layer` | `DeepSeek-V3-5layer` | 2 | `RUN_DEEPEP=1` | sweep 1024/1024 | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_agentic_deepseek-v3` | `DeepSeek-V3` | 2 | `RUN_MORI=1` | agentic, `conformance_256k` | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_agentic_deepseek-v3-5layer` | `DeepSeek-V3-5layer` | 2 | `RUN_MORI=1` | agentic, `conformance_256k` | `vllm_disagg_inference` |
| `pyt_vllm_disagg_mori_kimi-k3` | `Kimi-K3` | 4 (2P/2D) | `RUN_MORI=1`, `EP_TP_SIZE=2` | niah | `vllm_kimi_k3` |
| `pyt_vllm_disagg_mori_kimi-k3-mxfp4` | `Kimi-K3-MXFP4` | 4 (2P/2D) | `RUN_MORI=1`, `EP_TP_SIZE=2` | niah | `vllm_kimi_k3` |
| `pyt_vllm_disagg_mori_kimi-k3-mxfp4_mi355x` | `Kimi-K3-MXFP4-MI355X` | 4 (2P/2D) | `RUN_MORI=1`, `EP_TP_SIZE=1`, `MODEL_WEIGHTS_NAME=Kimi-K3-MXFP4` | niah | `vllm_kimi_k3` |

The GLM cards run TP8 within each pool (one DP rank per node) because they carry
`EP_TP_SIZE=8`. When wideEP moved from `TP_SIZE` to `EP_TP_SIZE` (default 1), they
silently became TP1 x 8 DP ranks, and decode crashed in MoRI EP setup with every GPU
holding the full non-expert weights. The Kimi-K3 cards set `REQUIRE_LOCAL_WEIGHTS=1`,
use `NIAH_WORDS=10000,50000,100000,200000`, declare `multiple_results`
(`perf_<MODEL_NAME>.csv`) and a `skip_gpu_arch`. How to add a card is in
[adding-a-model.md](adding-a-model.md).

## Kimi-K3 notes

**Docker image.** Every Kimi-K3 card builds `docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`,
for the GPU given as `MAD_SYSTEM_GPU_ARCHITECTURE` (gfx942 or gfx950; madengine and the
MAD CI pipeline pass it). It is a Kimi-K3-pinned sibling of the generalised
`docker/vllm_disagg_inference.ubuntu.amd.Dockerfile`: vLLM fork `862bfd8`
(`kimi-k3-wideep-disagg-fullsource-v3`), MoRI v1.2.2, AITER `68e42f5f` from source,
and the DP-rank vllm-router. MAD integration is the launcher, the yaml and this
Dockerfile; the vLLM fork, AITER and connector fixes are pinned in it by commit. By
hand:

```bash
# Build (once per cluster, or after a vLLM pin change), from the MAD repo root:
docker build -f docker/vllm_kimi_k3.ubuntu.amd.Dockerfile \
  --build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx942 \
  -t kimik3-wideep-disagg:latest .

# Run via the unified launcher. sbatch options, --export included, go BEFORE the
# script; anything after it is passed to the script as an argument and silently ignored.
cd scripts/vllm_dissag
sbatch -N 4 \
  --export=ALL,MODEL_NAME=Kimi-K3-MXFP4,CONNECTOR=moriio,WIDE_EP=1,xP=2,yD=2,DOCKER_IMAGE_NAME=kimik3-wideep-disagg:latest \
  run_xPyD_models.slurm
```

A hand-built tag must still reach every node (push it, or `docker load` it there).

**`EP_TP_SIZE`.** `EP_TP_SIZE=2` comes from the model's `models.yaml` recipe; export it
to override. A submit-time `EP_TP_SIZE=1` makes Kimi-K3 fall back to plain wideEP
(`-tp 1`). See
[multinode-overview.md](multinode-overview.md#tp-within-ep-ep_tp_size) for the rules.

**Weights.** `cluster_resolve_model_path` (`scripts/common/cluster.sh`) probes
`MODEL_DIR_CANDIDATES` for `MODEL_WEIGHTS_NAME` (default `MODEL_NAME`) on every node and
rejects a candidate whose `config.json` differs between nodes. If the weights live
under another directory name (for example `Kimi-K3`), set `MODEL_WEIGHTS_NAME`; an
explicit `MODEL_PATH` skips the probe. Details in
[multinode-running.md](multinode-running.md#model-weights).

**GPU architecture.** Each Kimi-K3 recipe is written for one GPU and says so in
`GPU_ARCHS` (`Kimi-K3` and `Kimi-K3-MXFP4`: gfx942; `Kimi-K3-MXFP4-MI355X`: gfx950).
The launcher detects the allocated nodes' GPU before loading weights and refuses a
mismatch (`cluster_require_gpu_arch`); `GPU_ARCH_CHECK=0` bypasses it for bring-up.
`scripts/common/check_gpu_arch_declarations.py` keeps each card's `skip_gpu_arch` in
step with its recipe.

**Network.** RDMA rails, GID index and socket interface are not part of any Kimi-K3
recipe. They come from `cluster.sh`, which detects the fabric from the adapters
present, and from `connectors/moriio.env`. Export any of them to override.

The worker taxonomy (W1 to W5) is in
[multinode-overview.md](multinode-overview.md#kimi-k3-worker-taxonomy). The model
itself is covered in [kimi-k3.md](kimi-k3.md).

## Prerequisites

- A SLURM cluster with `xP + yD` nodes (minimum 2: `xP=1`, `yD=1`), 8 GPUs per node.
- A Docker image with vLLM, the KV connectors and the NIC drivers. See below.
- Model weights on every allocated node, under `/mnt/m2m_nobackup/models_blog/` or
  `/shared_inference/models_blog/` by default, or set `MODEL_DIR`.
- A shared filesystem for logs.

The full list, including Docker on the compute nodes and the wall-time check, is in
[multinode-running.md](multinode-running.md#prerequisites).

## Building the Docker image

No public prebuilt image ships. Build your own from the full-source Dockerfile (MoRI,
AITER, vLLM, vllm-router, and optionally UCX, RIXL, rocSHMEM and DeepEP, from Git pins
on an open ROCm CI base):

```bash
cd MAD
docker build -t <your-registry>/vllm-disagg:local \
  -f docker/vllm_disagg_inference.ubuntu.amd.Dockerfile .
```

That image serves all connectors for every model in the table above except
`GLM-5.1-FP8` and the Kimi-K3 models. `WITH_NIXL=1` (the default) builds every
transport (moriio TP and wideEP, rixl NIXL TP, DeepEP wideEP). Add
`--build-arg WITH_NIXL=0` for a lean MoRI-EP-only image (moriio plus DeepEP from the
base), which builds faster.

`GLM-5.1-FP8` needs its own image, because its vLLM, AITER and MoRI pins differ from the
DeepSeek-V3/R1 stack the shared Dockerfile is pinned to:

```bash
docker build -t <your-registry>/vllm-disagg:glmv5.1 \
  -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile .
```

That Dockerfile defaults to `WITH_NIXL=0` (MoRI-EP only, which is all GLM-5.1 uses).
Its `VLLM_REF` is a hard requirement, not a preference: MAD ships no runtime patchers,
so every GLM DSA fix has to be in the source of that image. For a future model that
needs its own pins, add a `vllm_disagg_inference.<model>.ubuntu.amd.Dockerfile` rather
than repinning the shared one.

Kimi-K3 uses `docker/vllm_kimi_k3.ubuntu.amd.Dockerfile`; see
[Kimi-K3 notes](#kimi-k3-notes).

Then `export DOCKER_IMAGE_NAME=<tag>` (or a published image) before submitting. The
launcher runs `docker pull` on every node, so a local-only tag must be pushed to a
registry the nodes can reach. Without `DOCKER_IMAGE_NAME` the launcher stops with a
message that repeats the build command above.

## Quick start

To run a card through madengine or with `sbatch` directly, with a cluster's allocation
defaults, see [multinode-running.md](multinode-running.md).

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<image>

# rixl + TP (the default if no axis or legacy flag is set)
export MODEL_NAME=Qwen3-32B xP=1 yD=1
sbatch -N 2 -n 2 --nodelist=<node0,node1> run_xPyD_models.slurm
```

## Run modes: examples

Drive the launcher with the axes (`CONNECTOR`, `WIDE_EP`, `EP_BACKEND`) or the legacy
`RUN_MORI` / `RUN_DEEPEP` flags. Both resolve to the same `vllm_disagg.sh`.

```bash
# rixl + TP (NixlConnector, dense)
export CONNECTOR=rixl WIDE_EP=0            # = default if unset

# moriio + TP (MoRIIO KV transfer, no expert-parallel)
export CONNECTOR=moriio WIDE_EP=0

# MoRI-EP wide expert-parallel        (legacy: RUN_MORI=1)
export CONNECTOR=moriio WIDE_EP=1          # EP_BACKEND defaults to mori

# DeepEP wide expert-parallel         (legacy: RUN_DEEPEP=1)
export CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep
```

Each followed by:

```bash
export MODEL_NAME=<model> xP=<prefill_nodes> yD=<decode_nodes>
sbatch -N $((xP+yD)) -n $((xP+yD)) --nodelist=<nodes> run_xPyD_models.slurm
```

### Back-compat

| Legacy flag | Resolves to |
|---|---|
| `RUN_MORI=1` | `CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori` |
| `RUN_DEEPEP=1` | `CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep` |
| neither | `CONNECTOR=rixl WIDE_EP=0` |

Setting both `RUN_MORI=1` and `RUN_DEEPEP=1` exits with an error. The legacy flags apply
only when `CONNECTOR` is not set; an explicit `CONNECTOR` wins. With a legacy flag,
an explicit `WIDE_EP` or `EP_BACKEND` still wins over the value the flag implies.

## Adding a new model

No launcher edits are needed. The full walkthrough is in
[adding-a-model.md](adding-a-model.md); the `models.yaml` schema is in
[configuration.md](configuration.md).

1. Add an entry to `models.yaml`:

   ```yaml
   My-Model-Name:
     env:                                 # optional per-model ENV overrides
       VLLM_ROCM_USE_AITER: "1"
     prefill:
       tp: "--tensor-parallel-size 8 --kv-cache-dtype fp8"
       dp: ""                             # wideEP-only flags (omit if N/A)
     decode:
       tp: "--tensor-parallel-size 8 --kv-cache-dtype fp8"
       dp: ""
   ```

   - `WIDE_EP` selects the `tp` or `dp` sub-block per role. The composed string is
     `base_flags + <mode>_flags + <role>.<mode> + experimental_flags`.
   - Do **not** put connector or transfer flags, or parallelism *degree* flags
     (`--data-parallel-size`, `--enable-expert-parallel`, `--all2all-backend`,
     `--kv-transfer-config`), in the yaml. The launcher and connector own those. The
     yaml holds model-tuning flags and `env:`.
   - The `tp:` blocks do carry `--tensor-parallel-size`: the `rixl` + TP path takes its
     degree from the yaml. The `moriio` + TP path emits its own
     (`IO_TP_SIZE`, default 8) and strips any duplicate from the yaml string.
   - Flag strings are tokenised with Python `shlex`, not `eval`, so JSON values such as
     `--quantization-config '{"moe":{...}}'` survive.
   - A `--served-model-name` in the flags becomes the name clients must request
     (`SERVED_MODEL_NAME`).
2. Add the name to `VALID_MODELS` in `run_xPyD_models.slurm`, plus
   `MORI_EP_VALID_MODELS` and/or `DEEPEP_VALID_MODELS` if it should run wideEP, and
   `WIDE_EP_ONLY_MODELS` if TP must be rejected. Mirror the change in
   `tests/gate_check.sh`.
3. Make sure the weights exist on all nodes.
4. Run `bash tests/run_all.sh`.

### models.yaml keys

| Key | Applies to |
|---|---|
| `base_flags` | Always (both roles, both modes). |
| `tp_flags` / `dp_flags` | Mode-level, both roles. |
| `prefill.tp`, `prefill.dp`, `decode.tp`, `decode.dp` | One role in one mode. |
| `experimental_flags` | Optional extra CLI flags. |
| `env:` | Per-model environment. The single home for the model's serving recipe: `KV_BLOCK_SIZE`, `KV_CACHE_DTYPE`, per-role `PREFILL_/DECODE_CUDAGRAPH_MODE`, `*_MORI_BACKEND`, `KV_CACHE_MEMORY_BYTES`, `VLLM_ROCM_USE_AITER_MLA`, `EP_TP_SIZE`, `GPU_ARCHS` and so on. |

The wideEP `dp:` blocks are mostly empty today: the moriio and rixl connectors emit
the validated wideEP serve flags themselves, and the `dp:` blocks are placeholders so
per-model tuning can be added without launcher edits (both connectors append them).
The image ships none of these values; it is a clean binary artifact, so one image
serves any model and cluster.

A model missing from `models.yaml` runs with empty flags and a warning.

## Per-model environment layering

Environment precedence, lowest to highest:

1. **connector default** (`export VAR=${VAR:-default}` in `connectors/*.sh`)
2. **`models.yaml` `env:` block**, exported before the connector's `setup_env`
3. **submit-time** value (an exported variable forwarded with `docker -e`)

So a model can override a connector default (for example a `MORI_*` knob) without
editing any script, and a per-run or per-cluster value still wins. The launcher only
exports a yaml key that is not already in the environment.

The connector platform env files (`connectors/<connector>.env`) are a separate layer
forwarded at container start; a submit-time export of the same name overrides them too.
The full picture across `cluster.sh`, the card and madengine is in
[configuration.md](configuration.md).

**`GPU_MEMORY_UTILIZATION`.** If neither the yaml nor the submitter sets it, the launcher
defaults it to 0.70 when the decode EP width (`yD x GPUS_PER_NODE`) is 32 or more, to
leave room for MoRI's larger dispatch and combine buffers at high EP width, and to 0.8
otherwise. Models that need a topology-dependent value should omit the key from the
yaml and rely on the fallback; models that need a static value can set it in `env:`.
Both connectors pass the resolved value to `vllm serve`. (The rixl + TP path once passed
none, so vLLM used its built-in 0.92 and RCCL had no room for its first all-reduce.)

## Node topology and ports

```
Node 0          -> Prefill MASTER + Proxy (co-located)
Nodes 1..xP-1   -> Prefill CHILD (if xP > 1, wideEP)
Node xP         -> Decode MASTER
Nodes xP+1..end -> Decode CHILD (if yD > 1, wideEP)
```

`num_nodes = xP + yD`. The proxy or router is co-located on the prefill master
(node 0), is CPU-only, and uses a separate port from the vLLM server.

On the wideEP path, a master runs `--api-server-count` (the GPUs per node, clamped to
the DP size) and a child runs `--data-parallel-start-rank <r> --headless`. The DP
sizes are:

```
dp_per_node          = GPUS_PER_NODE / EP_TP_SIZE
PREFILL_DP_SIZE      = xP * dp_per_node
DECODE_DP_SIZE       = yD * dp_per_node
prefill start rank   = NODE_RANK * dp_per_node
decode start rank    = (NODE_RANK - xP) * dp_per_node
```

In TP mode there is no master/child split; each node is a full server and the degree
comes from the model's `tp:` flags.

**Port defaults by connector:**

| Connector | vLLM server port | Proxy port |
|---|---|---|
| `rixl` (TP or DeepEP) | 2584 | `ROUTER_PORT` (30000 by default) |
| `moriio` (MoRI-EP or TP) | 20005 | `ROUTER_PORT` (30000) with the default vllm_router; 10001 with `PROXY_TYPE=moriio_toy` |

`ROUTER_PORT` defaults to 30000 in `cluster.sh`. Other ports:

| Connector | Port | Variable | Default |
|---|---|---|---|
| moriio | vLLM server | `MORI_SERVE_PORT` | 20005 |
| moriio | RPC | `MORI_RPC_PORT` | 13345 |
| moriio | KV | `MORI_KV_PORT` | 9711 |
| moriio | toy proxy | `MORI_PROXY_PORT` | 10001 |
| moriio | proxy ping / local ping | `MORI_PROXY_PING_PORT` / `MORI_LOCAL_PING_PORT` | 36367 / 61555 |
| moriio | handshake / notify | `MORI_HANDSHAKE_PORT` / `MORI_NOTIFY_PORT` | 8405 / 61005 |
| moriio | container barrier | `BARRIER_PORT_MORI` | 2222 |
| rixl | vLLM server | fixed | 2584 |
| rixl | KV | fixed | 14600 |
| rixl | RPC (deepep) | fixed | 13345 |
| rixl | container barrier | `BARRIER_PORT` | 15000 |
| both | router HTTP | `ROUTER_PORT` (moriio also reads `VLLM_ROUTER_HTTP_PORT`) | 30000 |
| both | router Prometheus | `VLLM_ROUTER_PROMETHEUS_PORT` | 29000 |
| driver | torch distributed | `MASTER_PORT` | 39566 (set by `run_xPyD_models.slurm`) |

The rixl + TP barrier used to default to 5000, which was collision-prone: on one
cluster something the job could not kill held 5000, and the barrier "passed" on both
nodes by connecting to it. Before starting containers, the launcher frees ports 2222
and 15000 (and `BARRIER_PORT`) on every node.

## Proxy options

`PROXY_TYPE` selects the front end on rank 0:

| Connector | Values | Default |
|---|---|---|
| `rixl` | `vllm_router`, `toy_proxy` | `vllm_router` |
| `moriio` | `vllm_router`, `moriio_toy` | `vllm_router` |

- **`vllm_router`** is a Rust load balancer. On moriio it carries the DP-rank KV-notify
  fix that wideEP DP requires; the toy proxy cannot route the notify, and decode hangs
  with "remote blocks never arrived". With `EP_TP_SIZE > 1` it is started with
  `--moriio-dp-size <PREFILL_DP_SIZE>`. Before the benchmark starts, the launcher waits
  up to `ROUTER_REGISTER_TIMEOUT_S` (default 300) for the router log to show both
  `Add Prefill` and `Add Decode`. `VLLM_ROUTER_LOG_LEVEL` defaults to `info`.
- **`toy_proxy`** / **`moriio_toy`** are the connectors' test proxies, for local
  testing. For moriio, `MORIIO_TOY_PROXY` can point at the proxy script if the image
  keeps it somewhere unusual; the moriio toy proxy gets a `curl` probe once it starts.

**`ROUTER_BINARY`** (both connectors) is the path to the `vllm-router` binary. Images
that bake the router ship it on `PATH`, so this can be left unset. Runtime-only images
that do not ship a router (for example the mori121 MoRI-EP image) need
`ROUTER_BINARY=<path>`, pointing at a built `vllm-router` on a shared mount. If it is
unset and none is on `PATH`, the launcher aborts with a clear error.

The router's log is `vllm_router_NODE0.log`; the toy proxy's is `proxy_NODE0.log`.

## Connector hooks

Every connector implements the same six hooks; the driver calls them in a fixed order
(see the sequence diagram in
[multinode-overview.md](multinode-overview.md#driver-and-connector-hook-contract)).

| Hook | `rixl.sh` | `moriio.sh` |
|---|---|---|
| `connector_init` | Ports 2584 and the router; `PROXY_TYPE`; barrier 15000 | Ports 20005 and 10001; per-role MoRI backend; barrier 2222 |
| `connector_setup_env` | UCX/NIXL (TP), or rocSHMEM/UCX/NIXL plus an in-place `sed` fix from vLLM PR #39276, so the engine id uses the DP rank (deepep) | MoRI/RDMA/rocSHMEM and the kernel caches |
| `connector_runtime_patch` | No-op (deepep patches are inline in `setup_env`) | No-op (the disagg fixes are in the image's vLLM source) |
| `connector_launch_worker` | TP server, or deepep DP+EP server | MoRIIO + TP server, or MoRI-EP DP+EP server |
| `connector_wait_workers_ready` | TP: every node's server log, then a port check; deepep: the master logs | The prefill and decode master logs |
| `connector_start_proxy` | `vllm_router` or `toy_proxy` over all P/D IPs | `vllm_router` (default) or `moriio_toy`, plus a `curl` probe |

"Waits for a log" means waiting for `Application startup complete.` while watching for
fatal lines; see [multinode-running.md](multinode-running.md#how-a-run-fails-fast).

## Recipe knobs the connectors read

These normally come from the model's `models.yaml` `env:` block. Export one to
override it for a run.

| Variable | Default | Used by | Effect |
|---|---|---|---|
| `KV_BLOCK_SIZE` | `1` | wideEP (moriio, rixl deepep) | `--block-size`. DeepSeek-V3 needs 16: block 1 with the AITER MLA fp8 decode kernel GPU-faults. |
| `KV_CACHE_DTYPE` | `fp8` | wideEP | `--kv-cache-dtype` |
| `KV_CACHE_MEMORY_BYTES` | unset | wideEP | `--kv-cache-memory-bytes`. Also skips the boot profiling forward. |
| `MAX_NUM_BATCHED_TOKENS` | unset | moriio wideEP | `--max-num-batched-tokens` |
| `PREFILL_CUDAGRAPH_MODE` | `VLLM_CUDAGRAPH_MODE` | moriio | Prefill cudagraph mode. Empty or `NONE` gives `cudagraph_mode: NONE`. |
| `DECODE_CUDAGRAPH_MODE` | `VLLM_CUDAGRAPH_MODE` (moriio); `FULL_DECODE_ONLY` (rixl deepep) | both | Decode cudagraph mode. |
| `VLLM_CUDAGRAPH_MODE` | unset | both | Fallback for both roles. |
| `CUDAGRAPH_CAPTURE_SIZES` | `1 2 4 8 16 32 64 128 256` | both | `--cudagraph-capture-sizes` when cudagraphs are on. |
| `USE_INDUCTOR_GRAPH_PARTITION` | `0` | moriio | `1` adds `use_inductor_graph_partition` to the compilation config. Off by default because it would change the config for every model; GLM opts in. |
| `PREFILL_MORI_BACKEND` | `VLLM_ALL2ALL_BACKEND`, else `mori_high_throughput` | moriio wideEP | Prefill all2all. Newer images reject the bare `mori` alias. |
| `DECODE_MORI_BACKEND` | `mori_low_latency` | moriio wideEP | Decode all2all. |
| `VLLM_ROCM_USE_AITER` | `1` | both | AITER on. |
| `VLLM_ROCM_USE_AITER_MLA` | `1` | both | AITER MLA kernel. DeepSeek-V3's recipe sets 0. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `1` | both | |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `0` | rixl | |
| `VLLM_USE_AITER_TRITON_SILU_MUL` | `0` | rixl | |
| `EP_TP_SIZE` | `1` | moriio wideEP | TP inside each DP rank (`--tensor-parallel-size` instead of `-tp 1`). |
| `IO_TP_SIZE` | `8` | moriio TP | Tensor-parallel degree of the moriio + TP server. |

Notes on how the connectors apply them:

- The **rixl deepep** path honours the recipe's `VLLM_ROCM_USE_AITER_*` knobs and its
  `DECODE_CUDAGRAPH_MODE` and `CUDAGRAPH_CAPTURE_SIZES`, as moriio does. These used to
  be hardcoded, so DeepSeek-V3, whose recipe turns AITER MLA off, still ran the AITER
  MLA backend there and decode logged "Memory access fault by GPU" on every GPU even
  with block 16. The defaults are the old hardcoded values, so recipes that set none of
  them are unchanged. Shared-experts fusion (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`)
  stays off. rixl deepep prefill runs `--enforce-eager`; decode gets one
  `--compilation-config` with `+quant_fp8`.
- On **moriio**, even "no cudagraph" is expressed as `cudagraph_mode: NONE` with
  `+quant_fp8`, because an enforce-eager worker on these AITER images routes fp8 quant
  through an AITER op whose signature mismatches the build and crashes engine init.
- The moriio connector and the rixl deepep path add `--no-enable-prefix-caching` for
  clean, cache-free throughput. `BENCHMARK_SCRIPT=agentic` or `ENABLE_PREFIX_CACHE=1`
  removes it.
  `ENABLE_SERVER_METRICS=1` (automatic for agentic) exports the servers' `/metrics`
  endpoints as `AGENTIC_SERVER_METRICS`.

## DeepEP options

Optional, for `CONNECTOR=rixl WIDE_EP=1` (or `RUN_DEEPEP=1`):

| Variable | Default | Description |
|---|---|---|
| `PREFILL_DEEPEP_BACKEND` | `deepep_high_throughput` | Prefill all2all backend |
| `DECODE_DEEPEP_BACKEND` | `deepep_low_latency` | Decode all2all backend |
| `ENABLE_DBO` | `false` | Dual-batch overlap (`--enable-dbo`) |
| `DBO_COMM_SMS` | vLLM default | DBO communication SMs override |
| `ENABLE_PROFILING` | `false` | Profiling |

## Fabric, timeouts and the container

### Connector platform env files

`run_xPyD_models.slurm` reads `connectors/<CONNECTOR>.env` and forwards each
`KEY=VALUE` as `-e KEY=${KEY:-VALUE}`, so a submit-time export of the same name wins.
These must reach the container at PID 1: PyTorch reads the allocator configuration at
import, so an export inside the launcher is too late.

Why they exist: ROCm 7.2.3 cannot dmabuf-export HIP-VMM memory. With
`expandable_segments` on, the KV cache is VMM-backed, and GPU memory registered for
RDMA fails with EFAULT (errno 14): MoRI's `RegisterRdmaMemoryRegion` fails on the first
disagg write and every request returns 500 (moriio), or NIXL/UCX or DeepEP transfers
fail (rixl).

Both files set:

| Variable | Value |
|---|---|
| `PYTORCH_ALLOC_CONF` | `expandable_segments:False` |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:False` |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `0` (legacy HSA IPC off, so the dmabuf path is used) |
| `MORI_GPU_ARCHS` | `gfx942` (replaced by the detected GPU unless you set it) |
| `HSA_NO_SCRATCH_RECLAIM` | `1` |

`moriio.env` adds MoRI and RDMA fabric tuning, validated on MI300X RoCEv2. These are
cluster-specific; override per fabric by exporting the same name:

| Variable | Value |
|---|---|
| `MORI_RDMA_TC` | `41` |
| `MORI_RDMA_SL` | `0` |
| `MORI_IO_SL` | `1` |
| `MORI_IB_ENABLE_RELAXED_ORDERING` | `1` |
| `MORI_IB_GID_INDEX` | `1` |
| `MORI_NUM_QP_PER_PE` | `8` |
| `VLLM_MORIIO_QP_PER_TRANSFER` | `2` |
| `VLLM_MORIIO_NUM_WORKERS` | `4` |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` |
| `HSA_ENABLE_SDMA` | `1` |

The moriio connector's own fallbacks, used when nothing else sets a value, include
`NCCL_IB_HCA` and `MORI_RDMA_DEVICES` of `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9`,
`NCCL_IB_GID_INDEX=3`, `NCCL_NET_GDR_LEVEL=3`, `NCCL_CROSS_NIC=1`,
`MORI_SOCKET_IFNAME=eth0`, `MORI_SHMEM_HEAP_SIZE=17179869184`,
`ROCSHMEM_HEAP_SIZE=8589934592`, `ROCSHMEM_MAX_NUM_CONTEXTS=256`,
`GPU_MAX_HW_QUEUES=2` and `HIP_FORCE_DEV_KERNARG=1`. On AINIC and Thor2 fabrics,
`cluster.sh` supplies the rails instead; see
[multinode-running.md](multinode-running.md#fabric).

### Timeouts

| Variable | Default | Meaning |
|---|---|---|
| `LOG_WAIT_TIMEOUT_SECONDS` | 4000 | How long rank 0 waits for the servers to log start-up. |
| `DISTRIBUTED_TIMEOUT_SECONDS` | 7200 | `--distributed-timeout-seconds`. |
| `VLLM_RPC_TIMEOUT` | 300000 | vLLM RPC timeout (ms). |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 3600 | |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 10800 (moriio), 3600 (rixl deepep) | |
| `VLLM_HANDSHAKE_TIMEOUT_MINS` | 30 (moriio) | |
| `VLLM_MORIIO_TRANSFER_TIMEOUT_S` / `VLLM_MORIIO_DEFERRED_TIMEOUT_S` | 600 / 1800 (moriio) | |

### The container

Each node's container runs with host networking and IPC, the GPU and InfiniBand
devices, `--privileged`, `--shm-size ${DOCKER_SHM_SIZE:-256G}`, and unlimited
`memlock`. It mounts `$HOME`, `/shared_inference`, `/mnt/m2m_nobackup`, `$LOG_PATH` at
`/run_logs`, and this directory at `/opt/nixl-vllm-cookbook` (`NIXL_COOKBOOK_PATH`).
Only an explicit list of variables is forwarded with `-e`; to pass a new variable you
must add it to that list in `run_xPyD_models.slurm`.

**Host RDMA libraries.** The host's userspace RDMA libraries must match the host kernel
driver, so the launcher mounts them over the image's copies: `libibverbs` and
`librdmacm` (both the SONAME files and the versioned files), the vendor providers
(`libmlx5`, `libionic`, `libbnxt_re`, `libefa`, `libhns`), and `/etc/libibverbs.d`. It
mounts a file only if it is a regular file (`[ -f ]`), because docker creates a
directory at a missing bind-mount source and later runs then fail with
"not a directory".

**Persistent JIT cache.** The image points its AITER, Triton, vLLM and COMGR caches at
`/opt/vllm_cache`. The launcher mounts a host directory there so kernels compile once
and are reused: a cold compile is about 15 minutes for the DeepSeek MoE/FP8 GEMM kernel
set, a warm boot about 1 minute. The directory is keyed by image ID, so a new image
starts fresh and never reuses stale-ABI objects.

| Variable | Default | Meaning |
|---|---|---|
| `JIT_CACHE_PERSIST` | `1` | `0` disables the mount (empty in-container cache). |
| `JIT_CACHE_HOST` | `/mnt/m2m_nobackup/$USER/vllm_jit_cache/<image id>` | Host directory. |
| `JIT_CACHE_SPLIT_ROLE` | `1` | With `EP_TP_SIZE > 1`, separate `prefill/` and `decode/` subdirectories. |

## Validation status

The launcher assembles correct argv for all four combinations (checked offline in
`tests/`), but each combination also needs its transport built into the image:

| Combo | Live-validated | Image requirement |
|---|---|---|
| moriio + TP | yes | MoRI (mori121 image) |
| moriio + wideEP (mori) | yes, 1P/1D and 2P/2D | MoRI (mori121 image) |
| rixl + TP | needs a NIXL-validated image | NixlConnector and a working NCCL TP path |
| rixl + wideEP (deepep) | needs a DeepEP-validated image | DeepEP all2all kernels |

The `mori_ep_fullsource` Dockerfile builds the MoRI stack; the rixl and DeepEP
combinations need an image where those transports are validated (for example the
`vllm_disagg_inference` image for NIXL).

A recipe is validated on specific shapes and GPUs, and its `models.yaml` entry says
which. Run a card at one of those shapes before suspecting the recipe.

## Benchmark configuration

`BENCHMARK_SCRIPT` selects `sweep` (default), `long_context`, `niah` or `agentic`. The
sweep's main knobs:

| Variable | Default | Description |
|---|---|---|
| `BENCHMARK_ITR` | `1` | Iterations |
| `BENCHMARK_CON` | `8 16 32 64 128 256 512` | Concurrency levels |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024 1024/8192` | ISL/OSL pairs |

Parse a results log:

```bash
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log
```

Every benchmark knob, the per-cell timeout and `[STALL]`, and the `perf.csv` format are
in [benchmarks-and-results.md](benchmarks-and-results.md).

## Long-context retrieval (NIAH)

`benchmark_niah.{sh,py}` run a needle-in-a-haystack retrieval sweep (adapted from
vllm-project/vllm#47042): animal names are planted in filler text across growing
context lengths, and the model is scored on how many it retrieves. Select it in place of
the throughput sweep:

```bash
export BENCHMARK_SCRIPT=niah
export NIAH_WORDS="2000,8000,20000,35000"     # context sizes (words); optional
```

Or run `benchmark_niah.py` standalone against any live OpenAI-compatible endpoint
(`NIAH_URL`, `NIAH_MODEL`). The model name requested must equal the server's served
name; `vllm_disagg.sh` resolves `SERVED_MODEL_NAME` from the recipe's
`--served-model-name`, else `MODEL_PATH`. Details and all knobs:
[benchmarks-and-results.md](benchmarks-and-results.md#niah-long-context-retrieval).

## Agentic replay benchmark

The agentic replay benchmark replays real Claude Code agentic coding traces using the
`inferencex-agentx-mvp` scenario from
[SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf), driven against the
vLLM router or proxy. The workload is highly prefix-repetitive (each turn re-sends the
accumulated context), so it exercises the vLLM prefix cache and reports a measured
`gpu_cache_hit_rate` next to the dataset's theoretical ceiling. It shares the
endpoint-agnostic harness (`scripts/common/agentic_lib.sh` and the AgentX suite driver)
with the SGLang integration ([sglang-disagg.md](sglang-disagg.md)).

New to AgentX? [scripts/common/agentx/README.md](../scripts/common/agentx/README.md)
has the onboarding path, the config schema and copy-paste
[examples](../scripts/common/agentx/README.md#examples);
[HOW_IT_WORKS.md](../scripts/common/agentx/HOW_IT_WORKS.md) explains the replay
mechanism.

It is a drop-in alternative to the default `benchmark_xPyD.sh` sweep, selected with
`BENCHMARK_SCRIPT=agentic`. It runs through the same `run_xPyD_models.slurm` entry
point as the other benchmarks. The legacy `AGENTIC` env shorthand has been removed; use
`BENCHMARK_SCRIPT=agentic`.

### Validated agentic configurations

They are first-class, `agentic`-tagged `models.json` entries. Discover them with:

```bash
madengine run --tags agentic
```

| Backend | Model | Connector | Entry name |
|---|---|---|---|
| sglang | Qwen3-32B | mori_io | `pyt_sglang_disagg_mori_io_agentic_qwen3-32b` |
| sglang | DeepSeek-V3 | mori_dp | `pyt_sglang_disagg_mori_dp_agentic_deepseek-v3` |
| vllm | DeepSeek-V3 | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3` |
| vllm | DeepSeek-V3-5layer | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3-5layer` |

### vLLM specifics

aiperf targets `AGENTIC_PORT`, which the launcher sets to the router or proxy port
`BENCHMARK_PORT` (set by `connector_start_proxy`: `ROUTER_PORT` for `vllm_router`,
`PROXY_PORT` for the toy proxy). The vLLM PD router returns 503 on `GET /v1/models`
under MoRIIO service discovery (its HTTP worker registry is empty), while the shared
harness gates readiness and served-model resolution on `/v1/models`. So the launcher
starts a small standard-library shim, `agentic_models_shim.py`, on `BENCHMARK_PORT+1`
(or `AGENTIC_SHIM_PORT`). It answers `/v1/models`, gated on the router's `/health`, and
stream-proxies everything else to the router; `AGENTIC_PORT` then points at the shim.
Both changes are no-ops for the default sweep (guarded by `BENCHMARK_SCRIPT=agentic`).
The shim logs to `agentic_models_shim_NODE0.log`.

### Quick start (canonical DeepSeek-V3 1P/1D, MoRI-EP)

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<mori-vllm-image>
export BENCHMARK_SCRIPT=agentic                        # selects benchmark_agentic.sh
export RUN_MORI=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000

# 2 nodes with the model on local NVMe (router+prefill co-located on node 0, decode on node 1)
sbatch -N 2 -n 2 -p amd-rccl --time=90 \
  --nodelist=<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

`BENCHMARK_SCRIPT=agentic` selects `benchmark_agentic.sh`; the connectors then enable
server-side Prometheus metrics and remove `--no-enable-prefix-caching` (prefix cache
on).

### Environment variables

The full AgentX reference is at
[scripts/common/agentx/README.md#environment-variable-reference](../scripts/common/agentx/README.md#environment-variable-reference).
The table lists only the launcher-specific knobs; the shared AgentX variables
(`AGENTIC_CONC`, `AGENTIC_CACHE_WARMUP_DURATION`, `MAX_MODEL_LEN`, `AGENTIC_CONFIG`,
`AGENTIC_WORKLOAD`, `AGENTIC_STRICT_CONTEXT`, `AGENTIC_HF_ISL_TAIL` and others) are
documented there. On disaggregated serving the shim front end may not advertise
`max_model_len`, so the window is resolved from the prefill worker (the first
`host:port` in `AGENTIC_SERVER_METRICS`) via `/v1/models`; pin `MAX_MODEL_LEN` if
autodetect returns 0.

| Variable | Default | Description |
|---|---|---|
| `BENCHMARK_SCRIPT` | `sweep` | `agentic` runs the replay (`sweep`, `long_context`, `niah`, `agentic`). |
| `DURATION` | `120` | Benchmark duration in seconds. Use 900 or more for a canonical run. |
| `AGENTIC_PORT` | `${BENCHMARK_PORT}` | Endpoint aiperf targets (the router port, then the shim port). |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (automatic for agentic). |
| `ENABLE_PREFIX_CACHE` | auto | `1` forces prefix caching on, that is removes `--no-enable-prefix-caching` (automatic for agentic). |
| `SUITE_CORPUS_FORCE` | `0` | `1` regenerates a cached corpus after you edit a profile. |

`AGENTIC_SERVER_METRICS` is set by the connector to the space-separated backend
`/metrics` endpoints (`<prefill_ip>:<SERVE_PORT> <decode_ip>:<SERVE_PORT>`, for example
`:20005` for moriio, `:2584` for rixl) and consumed by aiperf's `--server-metrics`. It
is not a boolean. Its first `host:port` (the prefill worker) is also probed for the
served `max_model_len` when the front end does not advertise it.

Config schema, profile authoring and Tier 1 / Tier 2 knobs:
[AgentX core README](../scripts/common/agentx/README.md),
[profile authoring guide](../scripts/common/agentx/profiles/README.md),
[config examples](../scripts/common/agentx/README.md#examples).

### Supported models and modes

Agentic replay works for every model in the recipe allowlist, in the combinations each
model supports. The DeepSeek family is wideEP-only; dense and MoE models such as
Qwen3-32B run TP.

| Model | Combos | Notes |
|---|---|---|
| DeepSeek-V3 | 3, 4 (wideEP) | Canonical agentic target |
| DeepSeek-R1 | 3, 4 (wideEP) | Same path as DeepSeek-V3 |
| Qwen3-32B | 1, 2 (TP) | Ships a chat template; a good smoke target |
| Qwen3-30B-A3B | 1, 2 (TP) | MoE; needs a co-versioned AITER image |
| amd-Llama-3.3-70B-Instruct-FP8-KV | 1, 2 (TP) | |
| Llama-3.1-405B-Instruct-FP8-KV | 1, 2 (TP) | Needs a larger topology |

aiperf drives `/v1/chat/completions`, so the served model must ship a `chat_template`
in its `tokenizer_config.json` (Qwen2.5 and Qwen3 Instruct do; base Llama-3.1-8B does
not).

### Invocation examples (agentic suite)

`AGENTIC_CONFIG` paths must be visible inside the container. The `$HOME:$HOME` bind
mount means `$HOME/MAD/...` works if MAD is cloned at `$HOME/MAD`. Otherwise use the
in-container `NIXL_COOKBOOK_PATH` mount (`$NIXL_REPO_DIR` on the host).

**1. Multi-workload suite config**

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<mori-vllm-image>
export BENCHMARK_SCRIPT=agentic RUN_MORI=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

**2. Single preset shorthand (no config file)**

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<vllm-image>
export BENCHMARK_SCRIPT=agentic CONNECTOR=rixl WIDE_EP=0 xP=1 yD=1 MODEL_NAME=Qwen3-32B
export AGENTIC_WORKLOAD=conformance_256k   # config-less shorthand: runs just this preset

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

**3. Single entry selected from a config (smaller model, Tier 2 filter)**

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<vllm-image>
export BENCHMARK_SCRIPT=agentic CONNECTOR=rixl WIDE_EP=0 xP=1 yD=1 MODEL_NAME=Qwen3-32B
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml
export AGENTIC_WORKLOAD=inferencex_small   # preset bundles filter: {max_isl, max_turns, sample}

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

**4. DRY_RUN preview from the login node (no server)**

```bash
cd $HOME/MAD

# Prints the resolved N-workload plan, per-workload commands and context verdicts; no server needed.
DRY_RUN=1 AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml \
  bash scripts/common/benchmark_agentic_suite.sh
```

### Output files and validation

Written to `/run_logs/$SLURM_JOB_ID/` (host `LOG_PATH/<jobid>/`):

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # single-replay aggregate (validate_agentic_result.sh)
suite_summary.json                         # multi-workload suite roll-up (workloads[])
<workload>/[conc<N>/]                      # per-workload (and per-concurrency) result dirs
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
agentic_models_shim_NODE0.log              # side-port /v1/models shim log
```

The key JSON fields, the suite roll-up, the `validate_agentic_result.sh` output, its
thresholds (`AGENTIC_MAX_ERROR_RATE` 0.10, `AGENTIC_MIN_CACHE_HIT` 0.30) and the
`perf.csv` rows are described in
[benchmarks-and-results.md](benchmarks-and-results.md#agentic-replay).

## Dry run

`DRY_RUN=1` makes `vllm_disagg.sh` print each worker's assembled `vllm serve` command
instead of running it. It skips the container barrier, the proxy and the benchmark, so
it needs no GPUs, no container and no cluster. Run it for one `NODE_RANK` at a time:

```bash
cd scripts/vllm_dissag
CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori DRY_RUN=1 NODE_RANK=0 \
  MODEL_NAME=DeepSeek-V3 MODEL_PATH=/models/DeepSeek-V3 NIXL_COOKBOOK_PATH=$PWD \
  xP=2 yD=2 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 bash vllm_disagg.sh
```

Change `NODE_RANK` to see the other roles (0 prefill master, 1 prefill child, 2 decode
master, 3 decode child for 2P/2D). What you need to set:

| Variable | Why |
|---|---|
| `MODEL_PATH` | Required; the launcher refuses to start without it. |
| `MODEL_NAME` | Selects the `models.yaml` entry. |
| `NIXL_COOKBOOK_PATH` | Where the launcher finds its files; point it at this directory. |
| `NODE_RANK`, `xP`, `yD`, `IPADDRS` | The topology to render. |
| `CONNECTOR`, `WIDE_EP`, `EP_BACKEND` (or the legacy flags) | The combination. |

The output for each worker looks like this:

```
===DRYRUN backend=<moriio|rixl|deepep> log_prefix=<prefill|decode> role=<master|child> NODE_RANK=<n>===
vllm
serve
...one argument per line...
===END===
===ENV SERVED_MODEL_NAME=...
===ENV VLLM_ROCM_USE_AITER=...
```

The `===ENV` lines after `===END===` show the AITER environment the server would start
with and the served model name clients must request, so tools that read the argv up to
`===END===` are unaffected. Through `run_xPyD_models.slurm`, `DRY_RUN` is forwarded into
the containers, which print the same blocks into `pd_vllm_bench_NODE<rank>.log`. The
batch-host steps (GPU check, weights probe) still run, and because no benchmark runs,
the job then ends with the `no perf CSV` error.

## Testing

### Offline suites (no GPUs)

Run after any change to the launcher, the connectors or the yaml:

```bash
bash tests/run_all.sh             # gate_check + argv_assert; expect ALL OFFLINE SUITES PASSED
```

| Test | What it checks |
|---|---|
| `tests/gate_check.sh` | The (model x connector x `WIDE_EP` x `EP_BACKEND`) gate accepts exactly the supported combinations and rejects the rest, including the back-compat shims. It runs a hand-kept **mirror** of the lists in `run_xPyD_models.slurm`, so editing those lists without editing the mirror is not caught. |
| `tests/argv_assert.sh` | From the launcher's `DRY_RUN=1` output, that each connector x `WIDE_EP` x role cell emits the expected `vllm serve` flags and env and omits the wrong ones. |
| `tests/parse_to_csv_assert.sh` | That `parse_to_csv.py` gives one row per sweep cell and marks stalled, request-losing and zero-throughput cells `FAILURE`. Not part of `run_all.sh`; run it directly. |

`argv_assert.sh` covers, among others:

- moriio + TP (Llama-70B): `--tensor-parallel-size`, `--disable-custom-all-reduce`,
  exactly one `--compilation-config`, and no expert-parallel, all2all or DP flags.
- moriio + wideEP (DeepSeek-V3): `--data-parallel-size`, `--enable-expert-parallel`,
  `mori_high_throughput` for prefill, `--block-size 16`, one `--compilation-config`,
  and `-tp 1` instead of `--tensor-parallel-size`.
- Kimi-K3 on gfx942: the MoE requantised with `--quantization-config` and
  `int4_per_group_32`, the `kimi_k3` reasoning parser, prefill `cudagraph_mode NONE`,
  and `kv_producer` / `kv_consumer` roles; that Kimi-K3 stays gated out of TP.
- Kimi-K3-MXFP4 2P/2D TP2 x DP8 and Kimi-K3-MXFP4-MI355X TP1 x DP16.
- `EP_TP_SIZE`: a submit-time value beats the recipe; an indivisible value is
  rejected; it needs equal pools; it is dormant for non-Kimi wideEP; and
  `cluster.sh`'s `TP_SIZE` does not leak into the wideEP layout.
- The connector `.env` files carry the RDMA-fix env, and the slurm loop turns them into
  correct `-e` arguments with the override winning.
- Per-shape warmup stays opt-in.
- The GPU-architecture gate.
- GLM-5.1 1P/1D runs TP8 within each pool, as its card says.
- rixl passes the resolved `GPU_MEMORY_UTILIZATION`, and a submit-time value wins.
- Llama-70B exports `RCCL_MSCCL_ENABLE=0`, scoped to that model.
- rixl deepep honours the recipe's KV knobs (`--block-size 16`, KV cache bytes and
  dtype) and its AITER and cudagraph knobs, and a submit-time value still wins.
- The benchmark clients request the name the servers register.
- The start-up watch treats a worker's `torch.OutOfMemoryError` as fatal, and does not
  treat an ordinary memory INFO line as fatal.

The repository-wide checks (`check_srun_quotes.py`, `check_continuations.py`,
`check_gpu_arch_declarations.py`) are listed in
[multinode-running.md](multinode-running.md#checks-that-need-no-gpus).

### Live tests on an allocation

| Script | Purpose |
|---|---|
| `tests/drive_cell.sh` | Drives one test cell on an existing allocation via `srun --overlap` (no `sbatch`), one container per node, `NODE_RANK` per node. Requires `JOBID`, `NODES`, `IPS`, `MODEL_NAME`, `MODEL_PATH`, `CONNECTOR`, `WIDE_EP`, `xP`, `yD`, `TAG`, `DOCKER_IMAGE_NAME` (and `EP_BACKEND` for wideEP). Optional: `NODE_PREFIX`, `BENCHMARK_CON`, `BENCHMARK_COMBINATIONS`, `PROXY_TYPE`, `ROUTER_BINARY`, `ROUTER_PORT`, `RUN_MORI`, `RUN_DEEPEP`. |
| `tests/run_interactive.sh <NODE_RANK>` | The per-node container launch that `drive_cell.sh` uses; mirrors the `docker run` in `run_xPyD_models.slurm`. |
| `tests/harvest.sh <RUN_TAG>` | Prints prefill start-up, `RegisterRdmaMemoryRegion failed` and `died unexpectedly` counts, and per-cell concurrency, successes, failures, output tok/s, median TTFT and ITL. |

[`tests/TEST_PLAN.md`](../scripts/vllm_dissag/tests/TEST_PLAN.md) is the before/after
verification plan from when the three legacy launchers (`vllm_disagg_server.sh`,
`vllm_disagg_mori_ep.sh`, `vllm_disagg_server_deepep.sh`) were consolidated into
`vllm_disagg.sh`.

## Caveats

**MoE models need a co-versioned AITER/vLLM image.** Dense models (Llama, Qwen3-32B)
serve on the standard image. MoE models (DeepSeek-V3, Qwen3-30B-A3B) exercise AITER's
MoE GEMM path, which can fail at engine init (a `module_gemm_common` import error, or
`device_gemm ... does not support this GEMM problem`) on an image whose AITER is not
co-versioned for that path. Use the purpose-built MoRI-EP image for MoE serving. This is
an image and AITER concern, independent of the launcher.

## File map

| File | Role |
|---|---|
| `run_xPyD_models.slurm` | sbatch entry: node pick, validation, axis shim, `docker run` env plumbing |
| `vllm_disagg.sh` | The launcher: axis resolution, yaml parse, role branch, barrier, benchmark, cleanup |
| `parallelism.sh` | TP versus wideEP shared helpers |
| `connectors/rixl.sh` | NixlConnector profile (TP and DeepEP) |
| `connectors/moriio.sh` | MoRIIOConnector profile (MoRIIO + TP and MoRI-EP) |
| `connectors/{rixl,moriio}.env` | Per-connector platform env (`expandable_segments:False` and so on), forwarded via `docker -e` |
| `models.yaml` | Per-model flags and env catalog |
| `models.json` | The model cards |
| `tests/gate_check.sh` | Combo-gate unit tests (model x connector x `WIDE_EP` allow/reject) |
| `tests/argv_assert.sh` | Per-cell `vllm serve` flag and env assertions from `DRY_RUN=1` output |
| `tests/parse_to_csv_assert.sh` | `perf.csv` row and status assertions |
| `tests/run_all.sh` | Runs the offline gates (gate_check and argv_assert) |
| `tests/{drive_cell,harvest,run_interactive}.sh` | Interactive-allocation live-test drivers |
| `tests/TEST_PLAN.md` | Before/after verification plan |
| `benchmark_xPyD.sh`, `benchmark_long_context.sh`, `benchmark_niah.{sh,py}`, `benchmark_agentic.sh`, `benchmark_parser.py`, `parse_to_csv.py` | Benchmarks and parsing (NIAH is long-context retrieval, vllm#47042) |
| `agentic_models_shim.py` | The `/v1/models` shim for the agentic path |
| `socket_barrier.py`, `socket_wait.py`, `salloc_launch.sh` | Node coordination and the salloc helper |
| `ARCHITECTURE.md` | The architecture and state diagrams |
