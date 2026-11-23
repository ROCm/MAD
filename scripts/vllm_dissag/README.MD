# vLLM Disaggregated P/D Inference

Scripts to launch **disaggregated prefill/decode (P/D) inference** with vLLM on AMD GPUs over a
Slurm cluster. A **single launcher** (`vllm_disagg.sh`) drives every supported configuration through
two orthogonal axes — the KV-transfer **connector** and the **parallelism** mode — with per-model
flags supplied by a `models.yaml` catalog.

## Architecture at a glance

```
run_xPyD_models.slurm     # sbatch entry: validates, picks nodes, runs node.sh on each node
        │                 #   (../common/{config,topology,launcher,dispatch})
        ▼
node.sh                   # one node: mounts, topology for its rank, docker run (../common/node/node_lib.sh)
        │
        ▼
vllm_disagg.sh            # THE launcher: role branching, barrier, proxy, benchmark, cleanup
   ├── parallelism.sh     #   WIDE_EP=0 (TP)  vs  WIDE_EP=1 (wide expert-parallel)
   ├── connectors/rixl.sh #   NixlConnector   (TP, and DeepEP wideEP)
   ├── connectors/moriio.sh #  MoRIIOConnector (MoRI-EP wideEP, and MoRIIO+TP)
   └── models.yaml        #   per-model CLI flags + per-model env overrides
```

The layers around `vllm_disagg.sh`, and the file that implements each, are mapped in
[docs/multinode-architecture.md](../../docs/multinode-architecture.md).

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
slurm gate rejects any other pairing. Dense models are TP-only; `GLM-5.1-FP8` and the
Kimi-K3 models are **wideEP-only** (TP is rejected — their recipes need the wideEP serve
path). The DeepSeek family is wideEP plus moriio+TP (`MORIIO_TP_MODELS`), the TP baseline
(card `pyt_vllm_disagg_mori_tp_deepseek-v3`).

| Model | Type | Combos (by #) | Notes |
|-------|------|---------------|-------|
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | dense | 1, 2 (TP) | validated serving (moriio+TP) on the `dsv4` image; the shared image returns incorrect output |
| `Llama-3.1-405B-Instruct-FP8-KV` | dense | 1, 2 (TP) | |
| `Qwen3-32B` | dense | 1, 2 (TP) | validated serving (MoRIIO + TP) |
| `gpt-oss-120b` | MoE | 1, 2 (TP) | |
| `Qwen3-30B-A3B` | MoE | 1, 2 (TP) | needs a co-versioned AITER image (see Caveats) |
| `DeepSeek-V3` | MoE | 2 (moriio TP), 3, 4 | validated serving (moriio+MoRI-EP) |
| `DeepSeek-V3-5layer` | MoE | 2 (moriio TP), 3, 4 | smoke/dev variant |
| `DeepSeek-R1` | MoE | 2 (moriio TP), 3, 4 | needs a co-versioned AITER image (see Caveats) |
| `DeepSeek-V4-Flash-FP8` | MoE | 3 (moriio wideEP) | own image (`dsv4` Dockerfile); MoRI-EP only; submit via `run_wideep.sh` |
| `DeepSeek-V4-Pro-FP8` | MoE | 3 (moriio wideEP) | own image (`dsv4` Dockerfile); MoRI-EP only; submit via `run_wideep.sh` |
| `GLM-5.1-FP8` | MoE + DSA | 3 (moriio wideEP only) | own stack (`vllm-glm51`); 2P/2D EP16 and 4P/4D EP32 validated on the current pins (NIAH ~93-97%) |
| `Kimi-K3` | MoE | 3 (moriio wideEP only) | MI300X gfx942 2P/2D; **TP2×DP8** (`EP_TP_SIZE=2`); `vllm-kimi-k3` image; card `pyt_vllm_disagg_mori_kimi-k3` |
| `Kimi-K3-MXFP4` | MoE | 3 (moriio wideEP only) | MI300X gfx942 2P/2D; **TP2×DP8**; `vllm-kimi-k3` image; card `pyt_vllm_disagg_mori_kimi-k3-mxfp4` |
| `Kimi-K3-MXFP4-MI355X` | MoE | 3 (moriio wideEP only) | **MI355X gfx950, bring-up (not yet run)**; native MXFP4, TP1×DP16 (`EP_TP_SIZE=1`); `vllm-kimi-k3` image (built for gfx950); card `pyt_vllm_disagg_mori_kimi-k3-mxfp4_mi355x` |

Standalone reference launcher (debug / OCI): the Kimi-K3 disagg recipe in PR#241 (out of tree; not part of this framework-only branch).

See [ARCHITECTURE.md — Kimi-K3 worker taxonomy](ARCHITECTURE.md#5-kimi-k3-mxfp4-worker-taxonomy-w1w5) for W1–W5 ↔ `NODE_RANK` mapping.

**Docker image:** every Kimi-K3 card builds the `vllm-kimi-k3` image spec
([compat/images/vllm-kimi-k3.yaml](../../compat/images/vllm-kimi-k3.yaml)), for the GPU given as
`MAD_SYSTEM_GPU_ARCHITECTURE` (gfx942 or gfx950). The vLLM, AITER and connector fixes are builds
authorized by commit in [compat/rocm.yaml](../../compat/rocm.yaml), with patches on upstream.

```bash
# Build (once per cluster / after the stack moves), from the MAD repo root, then tag the
# image as kimik3-wideep-disagg:latest (or push it where the nodes can pull it):
madengine build --tags pyt_vllm_disagg_mori_kimi-k3 \
  --additional-context '{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'

# Run via unified launcher (sbatch options, --export included, go BEFORE the script;
# anything after it is passed to the script as an argument and silently ignored):
sbatch -N 4 \
  --export=ALL,MODEL_NAME=Kimi-K3-MXFP4,CONNECTOR=moriio,WIDE_EP=1,xP=2,yD=2,DOCKER_IMAGE_NAME=kimik3-wideep-disagg:latest \
  run_xPyD_models.slurm
```

`EP_TP_SIZE=2` comes from the model's `models.yaml` recipe; export it to override. The weights
directory is found by `cluster_resolve_model_path` (`scripts/common/cluster.sh`): it probes
`MODEL_DIR_CANDIDATES` for `MODEL_WEIGHTS_NAME` (default `MODEL_NAME`) on every node and rejects a
candidate whose `config.json` differs between nodes. If the weights live under another directory
name (e.g. `Kimi-K3`), set `MODEL_WEIGHTS_NAME`; an explicit `MODEL_PATH` skips the probe.

Each Kimi recipe is written for one GPU and says so in `GPU_ARCHS` (`Kimi-K3`, `Kimi-K3-MXFP4`: gfx942;
`Kimi-K3-MXFP4-MI355X`: gfx950). The launcher detects the allocated nodes' GPU before loading weights
and refuses a mismatch (`cluster_require_gpu_arch`, `scripts/common/cluster.sh`); `GPU_ARCH_CHECK=0`
bypasses it for bring-up. `scripts/common/check_gpu_arch_declarations.py` keeps each card's
`skip_gpu_arch` in step with its recipe.

Network settings (RDMA rails, GID index, socket interface) are not part of any Kimi recipe. They
come from `scripts/common/cluster.sh`, which detects the fabric from the adapters present, and from
`connectors/moriio.env`; export any of them to override.

## Prerequisites

- A Slurm cluster with `xP + yD` nodes (minimum 2: `xP=1`, `yD=1`), 8 GPUs/node.
- A Docker image with vLLM + the KV connector(s) + NIC drivers. See **Building the Docker image**.
- Model weights present on all allocated nodes under `/mnt/m2m_nobackup/models_blog/` or
  `/shared_inference/models_blog/` (or set `MODEL_DIR`).
- A shared filesystem for logs.

## Building the Docker image

No public prebuilt image ships. Every card names an image spec (a stack) and madengine
generates the Dockerfile from it:

```json
"image_spec": "${images.vllm-mori121}",
"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"},
```

| Stack | Spec | Models | Transport stack (rixl, DeepEP) |
|-------|------|--------|--------------------------------|
| `vllm-mori121` | [compat/images/vllm-mori121.yaml](../../compat/images/vllm-mori121.yaml) | every model above except `GLM-5.1-FP8` and the Kimi-K3 models | on |
| `vllm-glm51` | [compat/images/vllm-glm51.yaml](../../compat/images/vllm-glm51.yaml) | `GLM-5.1-FP8` | off (MoRI-EP only) |
| `vllm-kimi-k3` | [compat/images/vllm-kimi-k3.yaml](../../compat/images/vllm-kimi-k3.yaml) | the Kimi-K3 models | on |
| `vllm-per-gpu` | [compat/images/vllm-per-gpu.yaml](../../compat/images/vllm-per-gpu.yaml) | the end-to-end examples that run on either GPU: `vllm-mori121` on gfx942, the Kimi-K3 stack's builds on gfx950 | on |

`amd-Llama-3.3-70B-Instruct-FP8-KV` boots and benchmarks on `vllm-mori121` but answers incorrectly;
it runs on the DeepSeek-V4 image instead. The DeepSeek-V4 models are not on a stack yet: they build
from their own Dockerfile (`docker/vllm_disagg_gfx942_inference.dsv4.ubuntu.amd.Dockerfile`) and are
submitted with `run_wideep.sh`.

From the MAD root:

```bash
madengine build --tags <card> --additional-context '{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
madengine build --tags <card> --explain ...     # print the generated Dockerfile and the resolved plan, without building
```

The three specs extend [compat/images/di-vllm.yaml](../../compat/images/di-vllm.yaml) (build
prerequisites, env, cache locations, the smoke check) and select component builds authorized in
[compat/rocm.yaml](../../compat/rocm.yaml), where each build's commit, patches and notes are kept.
MAD ships no runtime patchers, so every fix a model needs is in its stack's builds. To change what
a stack builds, add the build to `compat/rocm.yaml` (or reuse one) and select it in the stack's
spec; a future model that needs other sources gets its own spec (usually a few lines extending
`di-vllm`), not a new Dockerfile.

`DeepSeek-V4-Flash-FP8` / `DeepSeek-V4-Pro-FP8` need their own image for the same
reason, and follow the same per-model naming:

```bash
docker build -t <your-registry>/vllm-disagg:dsv4 \
  -f docker/vllm_disagg_gfx942_inference.dsv4.ubuntu.amd.Dockerfile .
```

The DSV4 source patches are applied at image build, so nothing patches at job
start; `connectors/moriio.sh` only checks that `/app/versions.txt` carries
`DSV4_PATCHES_BAKED=`.

The same image is the one to use for `amd-Llama-3.3-70B-Instruct-FP8-KV` on
moriio + TP. On its newer vLLM, Llama answers the smoke curl correctly at
2P/2D, where the shared image returns repetitive, off-topic text. Submit it the
usual way, pointing at that image:

```bash
export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:dsv4
export CONNECTOR=moriio WIDE_EP=0 MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV
```

### Launching DSV4 jobs

DSV4 submits through `./run_wideep.sh <bench> <model> <topology>`, which
derives every dependent knob from those three arguments. Set the image with
`DOCKER_IMAGE_NAME` or `--image`:

```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:dsv4

# curl + NIAH ladder + concurrency sweep in one allocation (the usual first run)
./run_wideep.sh validate dsv4fls ep16

# a single bench
./run_wideep.sh niah  dsv4pro 2p2d
./run_wideep.sh smoke dsv4fls 1p1d

# print the sbatch line without submitting
./run_wideep.sh validate dsv4fls ep32 --dry-run
```

`dsv4fls` / `dsv4pro` are shorthand for the two `MODEL_NAME` values; topology is
`xPyD` (`2p2d`) or an EP width (`ep16`, `ep32`). Useful flags: `--time HH:MM:SS`,
`--exclude <nodelist>`, `--after JOBID` to queue behind a running job. Site
settings come from the environment: `PARTITION`, `ACCOUNT`, `QOS`.

Prompt defaults are handled per model and should not be set by hand — the Flash
NIAH branch primes the answer stem (`NIAH_LIST_PRIME=1.`) and the Pro branch runs
unprimed. The MoRI dispatch trim and `DSV4_ENABLE_HMA=0` are hardcoded and cannot be overridden.

Then `export DOCKER_IMAGE_NAME=<tag>` (or a published image) before submitting. The slurm
`docker pull`s the image on every node, so a local-only tag must be pushed to a registry the nodes
can reach.

## Quick start

To run a card through madengine or with `sbatch` directly, with a cluster's allocation
defaults, see [docs/multinode-running.md](../../docs/multinode-running.md).

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

### Tiered prefix caching (`KV_OFFLOAD`)

An orthogonal overlay that attaches a CPU-RAM KV tier (GPU KV → host, for prefix reuse)
*on top of* the disagg P/D connector, by composing both via vLLM's `MultiConnector`. Two
independent axes: the tier topology (`KV_OFFLOAD`) and the CPU-tier backend
(`OFFLOAD_BACKEND`).

| Variable | Default | Description |
|----------|---------|-------------|
| `KV_OFFLOAD` | `none` | `none` (off) \| `cpu` (CPU-RAM tier) \| `fs` (CPU-RAM + filesystem tier) |
| `OFFLOAD_BACKEND` | `OffloadingConnector` | `OffloadingConnector` (alias `native`) \| `SimpleCPUOffloadConnector` \| `LMCacheConnectorV1` (alias `lmcache`) |
| `OFFLOAD_CPU_BYTES_PER_NODE` | `107374182400` (100 GiB) | **Single CPU-tier budget for all backends**: total host RAM for KV offload **per node** (bytes). The connector derives the per-rank (native) / per-worker (lmcache) value from this + node topology, so one budget means the same node footprint across native TP, native EP, and lmcache. See the conversion note in `connectors/offloading.inc.sh`. |
| `OFFLOAD_DISK_PATH` | (unset) | required for `KV_OFFLOAD=fs`: base dir for the fs tier (NVMe or NFS, node-accessible). A per-host subdir is appended. |
| `KV_CACHE_MEMORY_BYTES` | per-model (`models.yaml`) | Per-GPU HBM KV-cache cap in bytes, passed as `--kv-cache-memory-bytes` (also skips the boot profiling forward). Set it explicitly with offloading: a cap makes blocks evict into the CPU tier, and an uncapped KV pool can make the connector's extra KV-registration pass exceed the init timeout. |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | `0.0` | `LMCacheConnectorV1` only: **per-worker** fs tier in GB |
| `FORCE_FRESH_IMAGE` | `0` | Set to `1` to `docker pull` the image on every node and abort if the pull fails (never run a stale cached image). |
| `DISABLE_HOST_RDMA_MOUNTS` | `0` | Set to `1` to skip host rdma-core bind-mounts (use when host and container rdma-core ABI versions differ). |
| `MORIIO_READ_MODE` | `0` | MoRIIO only: set to `1` for READ mode (decode pulls KV from prefill instead of prefill pushing). |

**Backend choice.** `OffloadingConnector` (native, default) is the general-purpose backend
and supports the fs tier via `TieringOffloadingSpec`. `SimpleCPUOffloadConnector` is a
lighter hash-driven CPU-only alternative (no fs tier). `LMCacheConnectorV1` (lmcache) is best
for prefix-heavy workloads and supports the fs tier, using the `lmcache` package bundled in
vLLM ROCm images v0.33.0 and later; it works under both rixl and MoRIIO. The `fs` tier is supported
by `OffloadingConnector` and `LMCacheConnectorV1` only, not `SimpleCPUOffloadConnector`.

Env for each recipe (submit with the same `MODEL_NAME`/`xP`/`yD`/`sbatch` command as above):

```bash
# Common to all offload runs
export BENCHMARK_SCRIPT=multiturn
export PROXY_TYPE=vllm_router ROUTER_BINARY=<path>/vllm-router
export GPU_MEMORY_UTILIZATION=0.8
export KV_CACHE_MEMORY_BYTES=25000000000          # per-GPU HBM KV cap so blocks evict into the CPU tier
export OFFLOAD_CPU_BYTES_PER_NODE=644245094400    # 600 GiB/node; connector splits per rank/worker
export CLEAN_STALE_JIT_LOCKS=1

# ── native · CPU tier (moriio wideEP, DSV3) ──────────────────────────────────
export CONNECTOR=moriio WIDE_EP=1 VLLM_ROCM_USE_AITER_MLA=1
export KV_OFFLOAD=cpu OFFLOAD_BACKEND=OffloadingConnector

# ── native · CPU tier (rixl TP, gpt-oss-120b) ────────────────────────────────
export CONNECTOR=rixl WIDE_EP=0
export KV_OFFLOAD=cpu OFFLOAD_BACKEND=OffloadingConnector

# ── native · CPU tier (moriio TP, gpt-oss-120b) ──────────────────────────────
export CONNECTOR=moriio WIDE_EP=0
export MORIIO_READ_MODE=1                         # gpt-oss has sliding-window attn; WRITE unsupported
export KV_OFFLOAD=cpu OFFLOAD_BACKEND=OffloadingConnector

# ── native · CPU + fs tier ───────────────────────────────────────────────────
export KV_OFFLOAD=fs OFFLOAD_BACKEND=OffloadingConnector
export OFFLOAD_DISK_PATH=/mnt/kv_offload          # node-accessible storage (NVMe or NFS)

# ── lmcache · CPU (+ optional fs) tier (rixl or moriio) ──────────────────────
export KV_OFFLOAD=cpu OFFLOAD_BACKEND=LMCacheConnectorV1
export PYTHONHASHSEED=0                           # fixed + identical across all vLLM procs, else lookups miss
export DISABLE_HOST_RDMA_MOUNTS=1                 # if host rdma-core ABI differs from container
```

When `KV_OFFLOAD!=none` the launcher forces `--enable-prefix-caching` (offloading is
meaningless without it). Pair it with a small `--kv-cache-memory-bytes` cap so HBM KV
actually evicts into the CPU tier. These recipes assume a `vllm/vllm-openai-rocm` image at
v0.33.0 or later (release or nightly, as in `docker/pyt_vllm.ubuntu.amd.Dockerfile`), which
ships `OffloadingConnector`, `SimpleCPUOffloadConnector`, `MultiConnector` and `lmcache`; no
custom image is needed.

LMCache (`lmcache`) CPU and fs tiers are validated on the cluster with both `CONNECTOR=rixl` and
`CONNECTOR=moriio`, alongside the native backend. Under moriio, LMCache relies on the temporary
device-binding patch below.

`CONNECTOR=moriio` + TP (`WIDE_EP=0`) with `KV_OFFLOAD` is newly wired and not yet validated end-to-end.
`LMCACHE_MAX_LOCAL_CPU_SIZE` (GB/worker), if set explicitly, still overrides the value derived from
`OFFLOAD_CPU_BYTES_PER_NODE`.

**Temporary runtime patches** (`connectors/moriio.sh`, applied per worker at launch, idempotent).
Each is non-fatal: if its anchor is missing (image already has the fix) it logs
`WARN: patch could not be applied ... continuing without it` and the launch proceeds. Remove each
once the images you use ship the upstream fix:

| Patch | Upstream PR | Already in |
|---|---|---|
| MoRIIO sideline fix (MultiConnector + offload; WRITE-mode requests stranded in `WAITING_FOR_REMOTE_KVS`) | [vLLM #57705](https://github.com/vllm-project/vllm/pull/57705), merged 2026-10-12 | vLLM v0.33.0 and later (and nightlies from 0.32.1rc1.dev361 on), so the patch logs a no-op on supported images |
| LMCache device-binding fix (`local_worker_id` -> `torch.cuda.current_device()`; every DP rank otherwise binds GPU 0) | [LMCache #5501](https://github.com/LMCache/LMCache/pull/5501), merged 2026-10-07 | LMCache v0.5.6 and later. vLLM v0.33.0 images still bundle LMCache 0.5.3, so they need it |
| LMCache MLA broadcast device fix (retrieve broadcast copied KV chunks to `cuda:{worker_id}`, which is GPU 0 on every DP rank; non-DP0 ranks GPU-fault at the first LMCache hit under DP) | [LMCache #6109](https://github.com/LMCache/LMCache/pull/6109) (open), which supersedes the stale [#5515](https://github.com/LMCache/LMCache/pull/5515) | No release yet: still present in v0.5.6 and v0.5.7rc2 |

The MLA broadcast fix only matters when `save_only_first_rank` is on (default for MLA models) and DP > 1.
Setting `LMCACHE_EXTRA_CONFIG='{"save_only_first_rank":false}'` also avoids it, as it skips that path.

**Multi-node roles (e.g. 2P2D EP16).** The router must be built from
[vllm-project/router #458](https://github.com/vllm-project/router/pull/458) (`--moriio-dp-size-local`,
`--moriio-prefill-hosts`, `--moriio-decode-hosts`). A stock `vllm-router` lacks these flags: the
launcher then warns and only the master node's DP ranks 0-7 of each role serve. The PR is open; point
`ROUTER_BINARY` at a binary built from it.

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
script, and a per-run `-e` still wins. Across every layer the order is site (`cluster.sh`) <
connector `.env` < recipe < topology spec (`MAD_TOPOLOGY`) < submit time; the launcher applies the
recipe over site and connector defaults before `docker run`, and records where each value came from
in `$LOG_PATH/<job id>/config.resolved.json`. See
[docs/configuration.md](../../docs/configuration.md#precedence).

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
| moriio + TP | ✅ | MoRI (`vllm-mori121` image; Llama-3.3-70B on the DeepSeek-V4 image, see above) |
| moriio + wideEP (mori) | ✅ 1P/1D + 2P/2D | MoRI (`vllm-mori121` image) |
| rixl + TP | needs a NIXL-validated image | NixlConnector + working NCCL TP path |
| rixl + wideEP (deepep) | needs a DeepEP-validated image | DeepEP all2all kernels |

rixl/DeepEP combos need an image whose stack builds the transport stack (`vllm-mori121`,
`vllm-kimi-k3`); the moriio paths are the validated ones.

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
| `BENCHMARK_PROMPTS_PER_CON` / `BENCHMARK_MIN_PROMPTS` | `2` / `16` | prompts per cell = concurrency x the first, at least the second |
| `PERF_LATENCY_METRICS` | unset | `1` adds latency, per-node, per-user and KV-bounded max batch rows to perf.csv |
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

The four validated agentic configurations are declared as first-class, `agentic`-tagged `models.json` entries. Discover them with:

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
| `BENCHMARK_SCRIPT` | `sweep` | Selects the benchmark: `sweep`, `long_context`, `agentic`, `niah`, `multiturn` |
| `DURATION` | `120` | Benchmark duration in seconds (use `>=900` for a canonical/valid submission) |
| `AGENTIC_PORT` | `${BENCHMARK_PORT}` | Endpoint aiperf targets (auto-resolved to the proxy/router port, then the shim port) |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (auto-on for agentic and multiturn) |
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
### Multi-turn benchmark (for `KV_OFFLOAD`)

`benchmark_multiturn.sh` is the benchmark for the offloading overlays (`export BENCHMARK_SCRIPT=multiturn`).
Each conversation has a **unique** large prefix and runs as two passes on `vllm bench serve`:
pass A primes the cache (`prefix + q1`, tiny output); pass B measures turn 2 (`prefix + q1 + q2`).
Once the primed set exceeds the HBM KV cap, evicted prefixes reload from the CPU/fs tier with
`KV_OFFLOAD` on (low turn-2 TTFT) or are recomputed with it off (high TTFT). Around each measure
pass it scrapes the prefill `/metrics` (`PREFILL_METRICS_URL`, exported
by the connector) and writes GPU prefix-cache and external (offload) hit-rate deltas to `perf.csv`.

- `MT_SWEEP="con:np:isl ..."` is an explicit point list: `con` = in-flight requests (keep low), `np` =
  distinct primed conversations (the HBM-pressure knob; raise it to cross the HBM -> CPU boundary),
  `isl` = prefix length. Without it a legacy `MT_CON_LIST` x `MT_PREFIX_LEN_LIST` grid is used.
- Other knobs: `MT_Q1_LEN`, `MT_Q2_LEN`, `MT_OSL1`, `MT_OSL2`, `MT_SEED`, `MT_DRYRUN`; see the header of
  `benchmark_multiturn.sh` for defaults.

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
