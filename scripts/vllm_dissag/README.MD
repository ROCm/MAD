# vLLM Disaggregated P/D Inference

Scripts to launch **disaggregated prefill/decode (P/D) inference** with vLLM on AMD GPUs over a
Slurm cluster. A **single launcher** (`vllm_disagg.sh`) drives every supported configuration through
two orthogonal axes — the KV-transfer **connector** and the **parallelism** mode — with per-model
flags supplied by a `models.yaml` catalog.

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

Three axes select behavior: the KV-transfer **CONNECTOR**, the **WIDE_EP** parallelism
mode, and (for wideEP only) the all-to-all **EP_BACKEND**, which is validated against the
connector. There are exactly **4 valid combos**; wideEP pairs each connector to its own
backend, so the two cross-pairs are rejected:

| # | `CONNECTOR` | `WIDE_EP` | `EP_BACKEND` | Valid? | What it is |
|---|-------------|-----------|--------------|--------|------------|
| 1 | `rixl`   | `0` (TP)     | —        | ✅ | NIXL + TP (dense, tensor-parallel) |
| 2 | `moriio` | `0` (TP)     | —        | ✅ | MoRIIO + TP (dense, tensor-parallel) |
| 3 | `moriio` | `1` (wideEP) | `mori`   | ✅ | MoRI-EP (wideEP DP+EP, mori all2all) |
| 4 | `rixl`   | `1` (wideEP) | `deepep` | ✅ | DeepEP (wideEP DP+EP, deepep all2all) |
| 5 | `moriio` | `1` (wideEP) | `deepep` | ❌ | cross-pair — aborts |
| 6 | `rixl`   | `1` (wideEP) | `mori`   | ❌ | cross-pair — aborts |

`EP_BACKEND` defaults to the connector's partner (moriio→mori, rixl→deepep), so you rarely
set it explicitly. Legacy flags map on: `RUN_MORI=1` → combo 3, `RUN_DEEPEP=1` → combo 4,
neither → combo 1.

## Supported models

Use these exact `MODEL_NAME` values (defined in `models.yaml` + the slurm allowlists).
The **Combos** column lists which of the 4 valid combos above each model may run in; the
slurm gate rejects any other pairing. Dense models are TP-only; the DeepSeek family,
`GLM-5.1-FP8` and the four GLM-5.3-Flash gfx942 cards (`GLM-5.3-Flash-{FP8,MXFP4}-gfx942[-hyb]`) are **wideEP-only** (TP is rejected — their
recipes need the wideEP serve path).

| Model | Type | Combos (by #) | Notes |
|-------|------|---------------|-------|
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | dense | 1, 2 (TP) | validated serving (moriio+TP) |
| `Llama-3.1-405B-Instruct-FP8-KV` | dense | 1, 2 (TP) | |
| `Qwen3-32B` | dense | 1, 2 (TP) | validated serving (MoRIIO + TP) |
| `gpt-oss-120b` | MoE | 1, 2 (TP) | |
| `Qwen3-30B-A3B` | MoE | 1, 2 (TP) | needs a co-versioned AITER image (see Caveats) |
| `DeepSeek-V3` | MoE | 3, 4 (wideEP only) | validated serving (moriio+MoRI-EP) |
| `DeepSeek-V3-5layer` | MoE | 3, 4 (wideEP only) | smoke/dev variant |
| `DeepSeek-R1` | MoE | 3, 4 (wideEP only) | needs a co-versioned AITER image (see Caveats) |
| `GLM-5.1-FP8` | MoE + DSA | 3 (moriio wideEP only) | own image (`glmv5.1` Dockerfile); 2P/2D EP16 and 4P/4D EP32 validated on the current pins (NIAH ~93-97%) |
| `GLM-5.3-Flash-FP8-gfx942` | MoE + DSA + KDA | 3 (moriio wideEP only) | gfx942 (MI300X / MI325X); own image (`glmv53flash.gfx942` Dockerfile); 1P/1D EP8, MTP on both legs, FULL_AND_PIECEWISE decode; NIAH 256K / 512K / 750K |
| `GLM-5.3-Flash-MXFP4-gfx942` | MoE + DSA + KDA | 3 (moriio wideEP only) | gfx942; same image and recipe as the FP8 card, MXFP4 checkpoint (173 GB): prefill experts as exact FP8 blocks, decode experts 4-bit (+15-17% KV per GPU on MI325X) |
| `GLM-5.3-Flash-FP8-gfx942-hyb` | MoE + DSA + KDA | 3 (moriio wideEP only) | gfx942; FP8 card with hybrid TP4 x DP2 prefill and prefill prefix-cache retention: lower long-prompt TTFT, fewer concurrent cold prefills |
| `GLM-5.3-Flash-MXFP4-gfx942-hyb` | MoE + DSA + KDA | 3 (moriio wideEP only) | gfx942; MXFP4 card with the same hybrid prefill and retention |

> **GLM-5.3-Flash expert all-to-all:** the four GLM-5.3-Flash-gfx942 cards use **MoRIIO** for the
> prefill→decode KV transfer and vLLM **`allgather_reducescatter`** for the expert all-to-all (the
> cards set `VLLM_ALL2ALL_BACKEND` / `DECODE_MORI_BACKEND` = `allgather_reducescatter`, overriding
> the `mori` wideEP default of combo 3). This is the validated, SLO-meeting recipe. MoRI-EP
> (`mori_high_throughput` / `mori_low_latency`) is a tested, accurate opt-in but not the default;
> to enable it see `skills/glm53-flash-gfx942/Recipe_enable_MoriEP.md`.

## Prerequisites

- A Slurm cluster with `xP + yD` nodes (minimum 2: `xP=1`, `yD=1`), 8 GPUs/node.
- A Docker image with vLLM + the KV connector(s) + NIC drivers. See **Building the Docker image**.
- Model weights present on all allocated nodes under `/mnt/m2m_nobackup/models_blog/` or
  `/shared_inference/models_blog/` (or set `MODEL_DIR`).
- A shared filesystem for logs.

## Building the Docker image

No public prebuilt image ships — build your own from the fullsource Dockerfile (MoRI +
AITER + vLLM + vllm-router + optional UCX/RIXL/rocSHMEM/DeepEP, from Git pins on an open
ROCm ci_base):

```bash
cd MAD
docker build -t <your-registry>/vllm-disagg:local \
  -f docker/vllm_disagg_inference.ubuntu.amd.Dockerfile .
```

That image serves all connectors for every model in the table above **except**
`GLM-5.1-FP8`. `WITH_NIXL=1` (default) builds every transport (moriio TP/wideEP + rixl
NIXL TP + DeepEP wideEP); add `--build-arg WITH_NIXL=0` for a lean MoRI-EP-only image
(moriio + deepep-from-base, faster build).

`GLM-5.1-FP8` needs its own per-model image, because its vLLM/AITER/MoRI pins differ
from the DeepSeek-V3/R1 stack the shared Dockerfile is pinned to:

```bash
docker build -t <your-registry>/vllm-disagg:glmv5.1 \
  -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile .
```

That Dockerfile defaults to `WITH_NIXL=0` (MoRI-EP only, which is all GLM-5.1 uses) and
its `VLLM_REF` is a hard requirement, not a preference — MAD ships no runtime patchers,
so every GLM DSA fix has to be in-source in that image. Add a
`vllm_disagg_inference.<model>.ubuntu.amd.Dockerfile` per future model that needs its
own pins rather than repinning the shared one.

`GLM-5.3-Flash-FP8-gfx942` and `GLM-5.3-Flash-MXFP4-gfx942` (MI300X / MI325X, gfx942) share their own image. Its vLLM, AITER and
vllm-router come from `glm53-flash-gfx942` fork branches that carry every GLM-5.3 fix in-source; the
build fails if one is missing:

```bash
docker build --ulimit nofile=1048576:1048576 -t <your-registry>/vllm-disagg:glmv53flash-gfx942 \
  -f docker/vllm_disagg_inference.glmv53flash.gfx942.ubuntu.amd.Dockerfile .
```

Its `models.yaml` entry runs MoRIIO in READ mode (`MORIIO_READ_MODE=1`, an opt-in in
`connectors/moriio.sh`), MTP on both legs, and routes both legs with `consistent_hash`
(`ROUTER_PREFILL_POLICY` / `ROUTER_DECODE_POLICY`, also opt-ins; every other model keeps
`round_robin`) so an agentic session stays on one DP rank and hits its prefix cache; aiperf
sends the session header the router keys on via `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`.
`..._long_context` runs 256K/1K and 512K/1K at c=1 and c=8 with per-shape warm-ups. The checkpoint
directory is `GLM-5.3-Flash-FP8`, not the recipe key: the `models.json` cards set
`MODEL_WEIGHTS_NAME=GLM-5.3-Flash-FP8`, and a direct `sbatch` must export it too. On MI325X (Broadcom NICs) submit with `MORI_RDMA_DEVICES` and
`NCCL_IB_HCA` set to `rdma0,...,rdma7` and `MORI_SOCKET_IFNAME=eno0`. The per-NIC steps (detect the devices, what to
export, how to verify) are in [`skills/moriio-nics-thor2`](skills/moriio-nics-thor2/SKILL.md) (Broadcom Thor2) and
[`skills/moriio-nics-cx7`](skills/moriio-nics-cx7/SKILL.md) (ConnectX-7). Build, launch and per-leg verification of both GLM-5.3-Flash cards:
[`skills/glm53-flash-gfx942`](skills/glm53-flash-gfx942/SKILL.md).
`GLM-5.3-Flash-MXFP4-gfx942` serves the AMD Quark MXFP4 checkpoint (`MODEL_WEIGHTS_NAME=GLM-5.3-Flash-MXFP4`) with
the FP8 card's flags. gfx942 has no FP4 matrix math, and every MXFP4 expert of this checkpoint converts exactly to FP8
128x128 blocks with a power-of-two block scale, so the card sets `VLLM_ROCM_MXFP4_WEIGHT_ONLY=1`,
`AITER_FLYDSL_W4_FP4=1` (exact e2m1 decode in the FlyDSL W4A16 MoE kernel) and `VLLM_ROCM_MXFP4_DENSE_FP8BLOCK=1`
(shared experts as FP8 blocks) on both legs, plus `VLLM_ROCM_MXFP4_MOE_FP8BLOCK=1` on the prefill leg only through
`PREFILL_EXTRA_ENV` (`connectors/moriio.sh` exports `PREFILL_EXTRA_ENV` / `DECODE_EXTRA_ENV` on that leg; unset for
every other model). Expected load-log lines: prefill "MXFP4 MoE experts converted exactly to FP8 128x128 blocks",
decode "MXFP4 MoE experts on AITER FlyDSL W4A16 with exact e2m1 decode". Validation results (harness, MI325X and MI300X):
[`results/GLM-5.3-Flash-MXFP4-gfx942_20261023.html`](results/GLM-5.3-Flash-MXFP4-gfx942_20261023.html).
Validation results on MI325X and MI300X (accuracy, perf, AgentX):
[`results/GLM-5.3-Flash-FP8-gfx942_20260927.html`](results/GLM-5.3-Flash-FP8-gfx942_20260927.html).
#### GLM-5.3-Flash on gfx942: EP8 or hybrid prefill, and how to run each card

Four cards share one image and one decode leg (EP8, MTP, FULL_AND_PIECEWISE CUDA graphs); they differ in the
checkpoint and in how the prefill node runs.

| Card | Checkpoint (`MODEL_WEIGHTS_NAME`) | Prefill node | Pick it for |
|---|---|---|---|
| `GLM-5.3-Flash-FP8-gfx942` | `GLM-5.3-Flash-FP8` (306 GB) | EP8: 8 data-parallel engines, one per GPU | many concurrent cold long prompts (most prefill throughput) |
| `GLM-5.3-Flash-MXFP4-gfx942` | `GLM-5.3-Flash-MXFP4` (173 GB) | EP8 | the same, with more decode KV per GPU (+17% MI325X, +31% MI300X) |
| `GLM-5.3-Flash-FP8-gfx942-hyb` | `GLM-5.3-Flash-FP8` | hybrid: TP4 x DP2 (2 engines, attention over 4 GPUs, experts still EP over all 8) | lowest time to first token on long prompts with session reuse (agentic traffic) |
| `GLM-5.3-Flash-MXFP4-gfx942-hyb` | `GLM-5.3-Flash-MXFP4` | hybrid | the same, with the MXFP4 decode KV gain |

The MXFP4 cards give the same answers as the FP8 cards: every MXFP4 expert converts exactly to FP8 128x128 blocks on
prefill, and decode decodes the 4-bit codes exactly in registers. With the hybrid prefill each long prompt prefills
about twice as fast, but only two prompts prefill at once instead of eight.

The `-hyb` cards add these `models.yaml` env settings to their EP8 card (all opt-in; unset for every other card):

| Setting | Effect |
|---|---|
| `PREFILL_TP=4` | prefill `-tp 4 --data-parallel-size 2 --data-parallel-size-local 2 --api-server-count=2`; the router expands each prefill node to 2 ranks |
| `PREFILL_KV_BLOCK_SIZE=4352` | prefill `--block-size 4352` (decode keeps 4) |
| `GLM53_HETERO_TP_FANIN=1` | MoRIIO hands the KDA state of a TP4 prefill engine to the TP1 decode ranks |
| `ROUTER_DECODE_DP_PIN=0` | the router does not pin a decode DP rank to the prefill rank |
| `PREFILL_PREFIX_CACHE_RETENTION=17408` | with prefix caching on, prefill `--prefix-cache-retention-interval 17408`: keep a KDA (linear-attention) state every 17408 tokens, so the second request of a session resumes from the shared prefix instead of re-prefilling it |

All four GLM-5.3-Flash-gfx942 cards also set `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`: server keep-alive above the
router's 50 s idle-pool timeout, so the router never reuses a connection the server is closing.

Run a card (one prefill node and one decode node; MI325X with Broadcom NICs also needs the NIC exports from
[`skills/moriio-nics-thor2`](skills/moriio-nics-thor2/SKILL.md)):

```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<registry>/vllm-disagg:glmv53flash-gfx942 CONNECTOR=moriio WIDE_EP=1 xP=1 yD=1
# pick one pair
export MODEL_NAME=GLM-5.3-Flash-FP8-gfx942       MODEL_WEIGHTS_NAME=GLM-5.3-Flash-FP8
export MODEL_NAME=GLM-5.3-Flash-MXFP4-gfx942     MODEL_WEIGHTS_NAME=GLM-5.3-Flash-MXFP4
export MODEL_NAME=GLM-5.3-Flash-FP8-gfx942-hyb   MODEL_WEIGHTS_NAME=GLM-5.3-Flash-FP8
export MODEL_NAME=GLM-5.3-Flash-MXFP4-gfx942-hyb MODEL_WEIGHTS_NAME=GLM-5.3-Flash-MXFP4
export BENCHMARK_SCRIPT=agentic   # agentic (prefix caching on), niah, long_context or sweep (prefix caching off)
sbatch -N 2 -n 2 --nodelist=<prefill_node,decode_node> run_xPyD_models.slurm
```

The same cards run through `madengine` with the `models.json` names, for example
`pyt_vllm_disagg_mori_agentic_glm-5.3-flash-mxfp4_gfx942_hyb` (AgentX conformance_256k) and `..._hyb_512k`.

Check each leg after start-up (`L=$LOG_PATH/<job_id>`):

```bash
grep -h "converted exactly to FP8 128x128" $L/prefill_NODE0.log | head -1   # MXFP4 cards: prefill experts as FP8 blocks
grep -h "FlyDSL W4A16 with exact e2m1"     $L/decode_NODE*.log  | head -1   # MXFP4 cards: decode experts 4-bit
grep -ho "'tensor_parallel_size': [0-9]*"  $L/prefill_NODE0.log | head -1   # -hyb: 4 (EP8 cards log no tensor_parallel_size)
grep -ho "'prefix_cache_retention_interval': [0-9]*" $L/prefill_NODE0.log | head -1   # -hyb with agentic: 17408
```

Image pins (`docker/vllm_disagg_inference.glmv53flash.gfx942.ubuntu.amd.Dockerfile`, no other build input): base
`rocm/vllm-dev:ci_base-build-01a0d6a8...@sha256:bd7f0c9b...`, MoRI `78b7a5f3`, AITER `11a6be6e`
(raviguptaamd/aiter `glm53-flash-gfx942-fp4lut`), FlyDSL 0.2.2, vLLM `439eabe5` (raviguptaamd/vllm
`glm53-flash-gfx942-mxfp4-mad`), vllm-router `076800bf` (raviguptaamd/router `glm53-flash-gfx942-hybrid`). The build
checks 29 GLM-5.3 fix markers and the router's `ROUTER_DECODE_DP_PIN`; the `-hyb` cards need nothing beyond them.

Short prompts (under ~2K tokens) need vLLM `6a0cb9395797` or later on the fork branch: earlier pins zeroed the decode
leg's freshly allocated KV blocks after the MoRIIO READ had already filled them, so short requests looped or
answered wrongly (GSM8K 6.4%; 96.4% with the fix). Long-context results are unchanged.

Then `export DOCKER_IMAGE_NAME=<tag>` (or a published image) before submitting. The slurm
`docker pull`s the image on every node, so a local-only tag must be pushed to a registry the nodes
can reach.

## Quick start

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<image>

# rixl + TP (the default if no axis/flags are set)
export MODEL_NAME=Qwen3-32B xP=1 yD=1
sbatch -N 2 -n 2 --nodelist=<node0,node1> run_xPyD_models.slurm
```

## Run modes — examples

Drive the launcher with the axes (`CONNECTOR` / `WIDE_EP` / `EP_BACKEND`) or the legacy
`RUN_MORI` / `RUN_DEEPEP` flags — both resolve to the same `vllm_disagg.sh`.

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
|-------------|-------------|
| `RUN_MORI=1` | `CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori` |
| `RUN_DEEPEP=1` | `CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep` |
| neither | `CONNECTOR=rixl WIDE_EP=0` |

Setting both `RUN_MORI=1` and `RUN_DEEPEP=1` exits with an error.

## Adding a new model (no launcher edits)

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
   - `WIDE_EP` selects the `tp` vs `dp` sub-block per role. Composed string =
     `base_flags + <mode>_flags + <role>.<mode> + experimental_flags`.
   - **Do NOT** put connector/transfer or parallelism *degree* flags
     (`--data-parallel-size`, `--enable-expert-parallel`, `--all2all-backend`, kv-transfer-config)
     in the yaml — the launcher/connector owns those. The yaml holds model-tuning flags + `env:`.
2. Add the name to `VALID_MODELS` in `run_xPyD_models.slurm` (plus `MORI_EP_VALID_MODELS` /
   `DEEPEP_VALID_MODELS` if it should run wideEP).
3. Ensure weights exist on all nodes.

### Per-model ENV layering

Env precedence (lowest → highest): connector default → `models.yaml` `env:` block → submit-time
`-e VAR=...`. So a model can override a connector default (e.g. a `MORI_*` knob) without editing any
script, and a per-run/per-cluster `-e` still wins.

**GPU_MEMORY_UTILIZATION**: If not set in yaml or submit-time, the launcher defaults to **0.70** at
EP≥32 (to accommodate MoRI buffer scaling at high EP width) and **0.8** otherwise. Models needing
topology-dependent values should omit this key from yaml and rely on the fallback; models needing a
static override can specify it in their `env:` block.

## Node topology (all modes)

```
Node 0          -> Prefill MASTER + Proxy (co-located)
Nodes 1..xP-1   -> Prefill CHILD (if xP > 1, wideEP)
Node xP         -> Decode MASTER
Nodes xP+1..end -> Decode CHILD (if yD > 1, wideEP)
```

`num_nodes = xP + yD`. The proxy/router is co-located on the prefill master (Node 0), CPU-only, on a
separate port from the vLLM server.

**Port defaults by connector:**

| Connector | vLLM server port | Proxy port |
|-----------|------------------|------------|
| `rixl` (TP or DeepEP) | 2584 | `ROUTER_PORT` (30000 default) |
| `moriio` (MoRI-EP or TP) | 20005 | `ROUTER_PORT` (30000) with the default vllm_router; 10001 with `PROXY_TYPE=moriio_toy` |

## Proxy options

`PROXY_TYPE`: `vllm_router` (default, Rust load balancer) or the connector's toy proxy
(`toy_proxy` for rixl, `moriio_toy` for moriio) for local testing.

`ROUTER_BINARY` (both connectors): path to the `vllm-router` binary. Images that bake the
router ship it on `PATH`, so this can be left unset. Runtime-only images that do NOT ship a
router (e.g. the mori121 MoRI-EP image) require `ROUTER_BINARY=<path>` — point it at a built
`vllm-router` on a shared mount. If unset and none is on `PATH`, the launcher aborts with a
clear error.

## Validation status (image capability)

The consolidated launcher assembles correct argv for all 4 combos (offline gates in `tests/`),
but each combo also needs the connector's transport built into the image:

| Combo | Live-validated | Image requirement |
|-------|----------------|-------------------|
| moriio + TP | ✅ | MoRI (mori121 image) |
| moriio + wideEP (mori) | ✅ 1P/1D + 2P/2D | MoRI (mori121 image) |
| rixl + TP | needs a NIXL-validated image | NixlConnector + working NCCL TP path |
| rixl + wideEP (deepep) | needs a DeepEP-validated image | DeepEP all2all kernels |

The `mori_ep_fullsource` Dockerfile builds the MoRI stack; rixl/DeepEP combos need an image
where those transports are validated (e.g. develop's `vllm_disagg_inference` for NIXL).

## DeepEP options (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFILL_DEEPEP_BACKEND` | `deepep_high_throughput` | prefill all2all backend |
| `DECODE_DEEPEP_BACKEND` | `deepep_low_latency` | decode all2all backend |
| `ENABLE_DBO` | `false` | dual-batch overlap |
| `DBO_COMM_SMS` | (vLLM default) | DBO comm SMs override |
| `ENABLE_PROFILING` | `false` | profiling |

## Benchmark configuration (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_ITR` | `1` | iterations |
| `BENCHMARK_CON` | `8 16 32 64 128 256 512` | concurrency levels |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024 1024/8192` | ISL/OSL pairs |

Parse a results log:
```bash
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log
```

### Long-context retrieval (NIAH)

`benchmark_niah.{sh,py}` run a needle-in-a-haystack retrieval sweep (adapted from
vllm-project/vllm#47042): animal names are planted in filler text across growing context
lengths and the model is scored on how many it retrieves. Select it in place of the throughput
sweep via the launcher's benchmark hook:

```bash
export BENCHMARK_SCRIPT=niah
export NIAH_WORDS="2000,8000,20000,35000"     # context sizes (words); optional
```

Or run `benchmark_niah.py` standalone against any live OpenAI-compatible endpoint
(`NIAH_URL`, `NIAH_MODEL`).

## Agentic Replay Benchmark

The agentic replay benchmark replays real Claude Code agentic coding traces using the `inferencex-agentx-mvp` scenario from [SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf), driven against the vLLM router/proxy. This workload is highly prefix-repetitive (each turn re-sends the accumulated context), so it exercises the vLLM prefix cache and reports a **measured** `gpu_cache_hit_rate` alongside the dataset's theoretical ceiling. It shares the endpoint-agnostic harness (`scripts/common/agentic_lib.sh` + the AgentX suite driver) with the [sglang integration](../sglang_disagg/README.MD#agentic-replay-benchmark).

> **New to AgentX? Start here:** [scripts/common/agentx/README.md](../common/agentx/README.md) has the onboarding path, config schema, and copy-paste [examples](../common/agentx/README.md#examples); [HOW_IT_WORKS.md](../common/agentx/HOW_IT_WORKS.md) explains the replay mechanism.

It is a drop-in alternative to the default `benchmark_xPyD.sh` concurrency sweep, selected via `BENCHMARK_SCRIPT=agentic`. No GitHub Actions / CI is involved — it runs through the same `run_xPyD_models.slurm` entrypoint as the other benchmarks.

> **Breaking change:** the legacy `AGENTIC` env shorthand has been removed. Use `BENCHMARK_SCRIPT=agentic` instead.

The validated agentic configurations are declared as first-class, `agentic`-tagged `models.json` entries. Discover them with:

```bash
madengine run --tags agentic
```

#### Validated agentic configurations

| Backend | Model | Connector | Entry Name |
|---------|-------|-----------|------------|
| sglang | Qwen3-32B | mori_io | `pyt_sglang_disagg_mori_io_agentic_qwen3-32b` |
| sglang | DeepSeek-V3 | mori_dp | `pyt_sglang_disagg_mori_dp_agentic_deepseek-v3` |
| vllm | DeepSeek-V3 | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3` |
| vllm | DeepSeek-V3-5layer | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3-5layer` |
| vllm | GLM-5.3-Flash-FP8-gfx942 | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-fp8_gfx942` (conformance_256k) |
| vllm | GLM-5.3-Flash-FP8-gfx942 | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-fp8_gfx942_512k` (conformance_512k) |
| vllm | GLM-5.3-Flash-MXFP4-gfx942 | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-mxfp4_gfx942` (conformance_256k) |
| vllm | GLM-5.3-Flash-MXFP4-gfx942 | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-mxfp4_gfx942_512k` (conformance_512k) |
| vllm | GLM-5.3-Flash-FP8-gfx942-hyb | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-fp8_gfx942_hyb` (conformance_256k) |
| vllm | GLM-5.3-Flash-FP8-gfx942-hyb | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-fp8_gfx942_hyb_512k` (conformance_512k) |
| vllm | GLM-5.3-Flash-MXFP4-gfx942-hyb | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-mxfp4_gfx942_hyb` (conformance_256k) |
| vllm | GLM-5.3-Flash-MXFP4-gfx942-hyb | mori | `pyt_vllm_disagg_mori_agentic_glm-5.3-flash-mxfp4_gfx942_hyb_512k` (conformance_512k) |

**vLLM specifics.** aiperf targets `AGENTIC_PORT`, which the launcher auto-resolves to the proxy/router port `BENCHMARK_PORT` (set by `connector_start_proxy`: `ROUTER_PORT` for `vllm_router`, `PROXY_PORT` for the toy proxy). Because the vLLM PD router 503s on `GET /v1/models` under MoRIIO service discovery (empty HTTP worker registry) while the shared harness gates readiness/served-model on `/v1/models`, the launcher starts a tiny stdlib shim (`agentic_models_shim.py`) on `BENCHMARK_PORT+1` that answers `/v1/models` (gated on the router's `/health`) and stream-proxies everything else; `AGENTIC_PORT` is then pointed at the shim. Both are byte-for-byte no-ops for the default sweep (guarded by `BENCHMARK_SCRIPT=agentic`).

### Quick start (canonical DeepSeek-V3 1P/1D, MoRI-EP)

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<mori-vllm-image>
export BENCHMARK_SCRIPT=agentic                        # selects benchmark_agentic.sh
export RUN_MORI=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000

# 3 nodes with the model on local NVMe (router+prefill co-located on node 0, decode on node 1)
sbatch -N 2 -n 2 -p amd-rccl --time=90 \
  --nodelist=<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

`BENCHMARK_SCRIPT=agentic` selects `benchmark_agentic.sh`; the connectors then auto-enable server-side Prometheus metrics and strip `--no-enable-prefix-caching` (prefix cache ON).

### Environment variables

Full AgentX env reference: [scripts/common/agentx/README.md#environment-variable-reference](../common/agentx/README.md#environment-variable-reference).
The table below lists only the launcher-specific knobs; the shared AgentX
variables (`AGENTIC_CONC`, `AGENTIC_CACHE_WARMUP_DURATION`, `MAX_MODEL_LEN`,
`AGENTIC_CONFIG`, `AGENTIC_WORKLOAD`, `AGENTIC_STRICT_CONTEXT`,
`AGENTIC_HF_ISL_TAIL`, …) are documented centrally. On this disaggregated
serving the shim front-end may not advertise `max_model_len`, so the window is
resolved from the prefill worker (the first `host:port` in
`AGENTIC_SERVER_METRICS`, auto-derived by the recipe) via `/v1/models`; pin
`MAX_MODEL_LEN` if autodetect returns 0.

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_SCRIPT` | `sweep` | Set to `agentic` to run the agentic replay (`sweep`, `long_context`, `agentic`) |
| `DURATION` | `120` | Benchmark duration in seconds (use `>=900` for a canonical/valid submission) |
| `AGENTIC_PORT` | `${BENCHMARK_PORT}` | Endpoint aiperf targets (auto-resolved to the proxy/router port, then the shim port) |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (auto-on for agentic) |
| `ENABLE_PREFIX_CACHE` | auto | `1` forces prefix cache on i.e. strips `--no-enable-prefix-caching` (auto-on for agentic) |
| `SUITE_CORPUS_FORCE` | `0` | Set `1` to regenerate a cached corpus after editing a profile. |

Note: `AGENTIC_SERVER_METRICS` is set automatically by the connector to the space-separated backend server `/metrics` endpoints (`<prefill_ip>:<SERVE_PORT> <decode_ip>:<SERVE_PORT>`, e.g. `:20005` for moriio, `:2584`/`SERVER_PORT` for rixl) and consumed by aiperf's `--server-metrics`; it is not a boolean toggle. Its first `host:port` (the prefill worker) is also probed for the served `max_model_len` when the front-end doesn't advertise it.

For config schema, profile authoring, and Tier 1/Tier 2 knobs, see:
- [AgentX core README](../common/agentx/README.md)
- [Profile authoring guide](../common/agentx/profiles/README.md)
- [Config examples cookbook](../common/agentx/README.md#examples)

### Supported models and modes

Agentic replay works for every model in the recipe allowlist, in the same combos each model supports (see **Supported models** above). The DeepSeek family is wideEP-only; dense/MoE models such as Qwen3-32B run TP.

| Model | Combos | Notes |
|-------|--------|-------|
| DeepSeek-V3 | 3, 4 (wideEP) | Canonical agentic target |
| DeepSeek-R1 | 3, 4 (wideEP) | Same path as DeepSeek-V3 |
| Qwen3-32B | 1, 2 (TP) | Ships a chat_template; good smoke target |
| Qwen3-30B-A3B | 1, 2 (TP) | MoE, needs a co-versioned AITER image |
| amd-Llama-3.3-70B-Instruct-FP8-KV | 1, 2 (TP) | |
| Llama-3.1-405B-Instruct-FP8-KV | 1, 2 (TP) | Needs larger topology |

> aiperf drives `/v1/chat/completions`, so the served model must ship a `chat_template` in its `tokenizer_config.json` (Qwen2.5/Qwen3-Instruct do; base Llama-3.1-8B does not).

### Invocation examples (agentic suite)

**Note:** `AGENTIC_CONFIG` paths must be visible inside the container. The `$HOME:$HOME` bind mount means `$HOME/MAD/...` works if MAD is cloned at `$HOME/MAD`. Alternatively use the in-container `NIXL_COOKBOOK_PATH` mount (`$NIXL_REPO_DIR` on the host).

#### 1. Multi-workload suite config

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<mori-vllm-image>
export BENCHMARK_SCRIPT=agentic RUN_MORI=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

#### 2. Single preset shorthand (no config file)

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<vllm-image>
export BENCHMARK_SCRIPT=agentic CONNECTOR=rixl WIDE_EP=0 xP=1 yD=1 MODEL_NAME=Qwen3-32B
export AGENTIC_WORKLOAD=conformance_256k   # config-less shorthand: runs just this preset

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

#### 3. Single entry selected from a config (smaller model, Tier-2 filter)

```bash
cd scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<vllm-image>
export BENCHMARK_SCRIPT=agentic CONNECTOR=rixl WIDE_EP=0 xP=1 yD=1 MODEL_NAME=Qwen3-32B
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml
export AGENTIC_WORKLOAD=inferencex_small   # preset bundles filter: {max_isl, max_turns, sample}

sbatch -N 2 -n 2 -p amd-rccl --time=90 --nodelist=<2-nodes> run_xPyD_models.slurm
```

#### 4. DRY_RUN preview from the login node (no server)

```bash
cd $HOME/MAD

# Prints the resolved N-workload plan + per-workload commands + context verdicts; no server needed.
DRY_RUN=1 AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml \
  bash scripts/common/benchmark_agentic_suite.sh
```

### Output files

Written to `/run_logs/$SLURM_JOB_ID/` (host `LOG_PATH/<jobid>/`):

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # single-replay aggregate (validate_agentic_result.sh)
suite_summary.json                         # multi-workload suite roll-up (workloads[])
<workload>/[conc<N>/]                       # per-workload (and per-concurrency) result dirs
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
agentic_models_shim_NODE0.log              # side-port /v1/models shim log
```

Key fields in the aggregate JSON:
- `request_metrics.cache.theoretical_cache_hit_rate` — dataset-derived ceiling.
- `server_metrics.present` / `server_metrics.cache.gpu_cache_hit_rate` — **measured** runtime GPU cache hit rate.
- `request_accounting` — `records_profiled` vs `records_error_dropped` (error rate is computed over measured requests, excluding warmup).

The suite roll-up (`suite_summary.json`) has a `workloads[]` array; each entry carries `workload`, `result_dir`, `status`, `theoretical_cache_hit_rate`, `gpu_cache_hit_rate`, and `error_rate`. A workload that exits non-zero is marked `INVALID` (with a `RUN_INVALID.json` sidecar in its dir) and the suite **continues** to the next workload.

### Validation

`scripts/common/validate_agentic_result.sh` runs automatically at the end of the agentic benchmark and prints, e.g.:

```
[validate] requests: 70/72 ok  error_rate=2.8%
[validate] gpu_cache_hit_rate: 32.4%
[validate] throughput_per_gpu: 6247 tok/s
[validate][PASS] error_rate within 10%
```

Thresholds: `AGENTIC_MAX_ERROR_RATE` (default `0.10`, gates PASS/WARN) and `AGENTIC_MIN_CACHE_HIT` (default `0.30`, warn-only). A low cache hit rate on `xP>1` usually indicates the router is not keeping a conversation's turns on the worker holding its prefix (session-affinity).

## Testing the launcher (no GPUs)

Offline suites validate the launcher without a cluster — run after any change to the
launcher/connectors/yaml:

```bash
bash tests/run_all.sh             # gate_check + argv_assert; expect ALL OFFLINE SUITES PASSED
```

- `tests/gate_check.sh` — connector × WIDE_EP combo validation (valid/invalid pairings, back-compat shims).
- `tests/argv_assert.sh` — asserts expected `vllm serve` flags/env are present/absent per cell, from
  the launcher's `DRY_RUN=1` output (no GPUs, no container).

You can also dry-run any single cell to inspect the assembled command:

```bash
CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori DRY_RUN=1 NODE_RANK=0 \
  MODEL_NAME=DeepSeek-V3 MODEL_PATH=/models/DeepSeek-V3 NIXL_COOKBOOK_PATH=$PWD \
  xP=2 yD=2 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 bash vllm_disagg.sh
```

## Caveats

- **MoE models need a co-versioned AITER/vLLM image.** Dense models (Llama, Qwen3-32B) serve on the
  standard image. MoE models (DeepSeek-V3, Qwen3-30B-A3B) exercise AITER's MoE GEMM path, which can
  fail at engine init (`module_gemm_common` import, or `device_gemm ... does not support this GEMM
  problem`) on an image whose AITER is not co-versioned for that path. Use the purpose-built MoRI-EP
  image for MoE serving. This is an image/AITER concern, independent of the launcher.
