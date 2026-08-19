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
slurm gate rejects any other pairing. Dense models are TP-only; the DeepSeek family is
**wideEP-only** (TP is rejected — its recipe needs the wideEP serve path).

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

## Prerequisites

- A Slurm cluster with `xP + yD` nodes (minimum 2: `xP=1`, `yD=1`), 8 GPUs/node.
- A Docker image with vLLM + the KV connector(s) + NIC drivers. See **Building the Docker image**.
- Model weights present on all allocated nodes under `/mnt/m2m_nobackup/models_blog/` or
  `/shared_inference/models_blog/` (or set `MODEL_DIR`).
- A shared filesystem for logs.

## Building the Docker image

No public prebuilt image ships — build your own from the one fullsource Dockerfile (MoRI +
AITER + vLLM + vllm-router + optional UCX/RIXL/rocSHMEM/DeepEP, from Git pins on an open
ROCm ci_base):

```bash
cd MAD
docker build -t <your-registry>/vllm-disagg:local \
  -f docker/vllm_disagg_inference.ubuntu.amd.Dockerfile .
```

One image serves all connectors. `WITH_NIXL=1` (default) builds every transport (moriio
TP/wideEP + rixl NIXL TP + DeepEP wideEP); add `--build-arg WITH_NIXL=0` for a lean
MoRI-EP-only image (moriio + deepep-from-base, faster build).

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

## MI308X + Broadcom Thor2 (non-SLURM)

`run_xPyD_models.slurm` assumes SLURM and a shared filesystem. Some clusters have neither.
`launch_disagg_skyriver.sh` is the same launcher driven over an SSH + docker mesh instead:
it picks nodes, resolves fabric IPs, and runs one container per node with the identical
`-e` env plumb. `vllm_disagg.sh` and `connectors/moriio.sh` are reused **unchanged** — they
are env-driven, so nothing below is a fork of the SLURM path.

```bash
PREFILL="node0,node1" DECODE="node2,node3" \
MODEL_NAME=GLM-5.2-FP8 MODEL_PATH=/models/GLM-5.2-FP8 \
GPU_MEMORY_UTILIZATION=0.72 \
./launch_disagg_skyriver.sh                 # 2P/2D (EP16); one node each side = EP8
```

Add `DRY_RUN=1` to print the per-node `docker run` without executing — the fastest way to
confirm a knob actually reaches the container.

### `FABRIC_PROFILE` — per-fabric env overlay

MoRI's RDMA settings (traffic class, service level, QPs per PE, socket interface) are
fabric properties, not model properties, so they do not belong in `models.yaml`. The
launcher loads them as a **second** env file layered over the connector's base:

```
connectors/${CONNECTOR}.env                      # always
connectors/${CONNECTOR}.${FABRIC_PROFILE}.env    # then this, overriding
```

| `FABRIC_PROFILE` | Effect |
|---|---|
| `thor2` (default) | loads `connectors/moriio.thor2.env` — Broadcom Thor2 RoCEv2 values |
| `-` | base `connectors/moriio.env` only (the pre-existing CX7 behaviour, byte-identical) |
| anything else | loads `connectors/moriio.<name>.env`, or **exits** if that file is missing |

A missing profile file is a hard exit, deliberately. Silently falling back to the base env
would run a Thor2 fabric with ConnectX defaults and produce a slow, working, wrong result —
the worst failure mode. `Loading connector platform env:` therefore prints **twice** on a
profiled run; once is a bug.

Two keys in that file are **fallbacks that are normally overwritten**:
`MORI_RDMA_DEVICES` and `NCCL_IB_HCA` are re-derived from live sysfs at launch, per node,
because the `bnxt_re` device↔rail mapping is not stable across reboots. Different nodes
legitimately show different device orders in the same run — that is the derivation working,
not drift.

### Prerequisite a reboot wipes

On a rail-per-NIC RoCEv2 fabric, MoRI builds an unconditional full QP mesh, so EP ranks form
QPs **between different rails**. That needs two host-side things that are runtime-only:

```bash
diag/rail_routes.sh add  node0 node1 node2 node3     # per-source policy routes via .254
diag/rail_routes.sh show node0 node1 node2 node3     # expect rules=8, rp_filter_all=2
```

Without both the per-source routes *and* `rp_filter=2` on the **receiver**, cross-rail QPs
fail at `bnxt_re_resolve_eth_dmac: -110`. Strict reverse-path filtering drops the arriving
frames silently, so the sender-side symptom is misleading. **`ip rule` and the sysctl do not
survive a reboot** — re-run `add` after any node restart, before any multi-node run.
`del` restores the rp_filter values it snapshotted at `add` time.

### Pre-flight probes (`diag/`)

| Script | Answers |
|---|---|
| `diag/rail_routes.sh {add,del,show}` | is cross-rail RDMA reachable? (see above) |
| `diag/run_ep_probe.sh <image> <tag> <n0> <n1>` | does a 2-node MoRI EP handshake complete? ~40 s, instead of discovering it 40 min into a serve |
| `diag/preflight_nodes.sh <nodes...>` | is the node idle, and does `util x capacity + MORI_SHMEM_HEAP_SIZE` leave headroom? |
| `diag/kv_capacity.sh <job_id> [nodes...]` | what KV pool did a completed run actually get? |

`kv_capacity.sh` exists because KV capacity is the metric that decides topology and the one
metric the benchmark table never prints — it is emitted once per DP worker at INFO during
boot and then scrolls away.

### Memory budget on this platform

MoRI's symmetric heap (`MORI_SHMEM_HEAP_SIZE`, 32 GiB for GLM-5.2) is allocated **outside**
PyTorch's pool, so it is invisible to `gpu_memory_utilization`. On 192 GiB MI308X,
`0.80` + a 32 GiB heap leaves ~6 GiB and OOMs during cudagraph warm-up; `0.72` is the
working value at EP16. The launcher warns when the arithmetic does not clear ~20 GiB.

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
export BENCHMARK_SCRIPT_FILE=benchmark_niah.sh
export NIAH_WORDS="2000,8000,20000,35000"     # context sizes (words); optional
```

Or run `benchmark_niah.py` standalone against any live OpenAI-compatible endpoint
(`NIAH_URL`, `NIAH_MODEL`).

> **`NIAH_SEEDS` is a comma list of seed VALUES, not a count.** `NIAH_SEEDS=10` runs seed
> #10 **once**, it does not run ten seeds. At n=1 each per-length mean is an integer, so a
> fractional gate (`>= 9.3`) can never be met — the rung scores 9.0 or 10.0 and nothing
> between. Use `NIAH_SEEDS="0,1,2"` (the default) or more for any scored run; the script
> now prints `n=<count>` and warns when only one seed is given.

### Pre-flight before a scored run

```bash
GPU_MEMORY_UTILIZATION=0.72 MORI_SHMEM_HEAP_SIZE=34359738368 \
  bash diag/preflight_nodes.sh node0 node1 node2 node3
```

Two independent checks, because two unrelated things sink a run:

- **static** — `util x capacity + MORI_SHMEM_HEAP_SIZE` must leave >= 20 GiB. MoRI's
  symmetric heap is allocated *outside* PyTorch's pool, so it is invisible to
  `gpu_memory_utilization`. `0.80` with a 32 GiB heap leaves 6.39 GiB and OOMs during
  post-KV warmup — the arithmetic is knowable before launch, so check it there.
- **dynamic** — the node must actually be idle. VRAM is sampled 3x with a settle delay and
  the **minimum** is reported (a single sample taken during teardown reads ~50 % and
  invents a phantom co-tenant), and an ABORT additionally requires a **KFD PID**: a running
  container is *not* evidence it holds GPU memory.

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

- **A huge `Estimated CUDA graph memory` means the node was busy, not that your cudagraph
  config is wrong.** That number is a **measured free-memory delta**, not a computed size:
  vLLM samples free VRAM, runs the capture-shape profile, samples again, and books the
  difference against cudagraphs — then subtracts it from the KV budget. Anything else
  taking VRAM during that ~20 s window is misattributed to your graphs, and the engine
  then refuses to boot for lack of KV space. Two tells that a reading is spurious: it
  **differs per rank** for identical shards (a real graph footprint is deterministic), and
  it is wildly larger than the ~1.7–1.8 GiB this recipe normally reports (the estimator is
  otherwise accurate to ~8 %). Fix the node, not the config — run
  `diag/preflight_nodes.sh` before the launch. **Do not** set
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to make it go away: on a genuinely busy node,
  refusing to boot is the correct behaviour, and disabling the estimator converts a clean
  startup failure into an OOM later in the run.

- **MoE models need a co-versioned AITER/vLLM image.** Dense models (Llama, Qwen3-32B) serve on the
  standard image. MoE models (DeepSeek-V3, Qwen3-30B-A3B) exercise AITER's MoE GEMM path, which can
  fail at engine init (`module_gemm_common` import, or `device_gemm ... does not support this GEMM
  problem`) on an image whose AITER is not co-versioned for that path. Use the purpose-built MoRI-EP
  image for MoE serving. This is an image/AITER concern, independent of the launcher.
