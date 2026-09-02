## List of Models - focus SGLang Disaggregated P/D inference

Dense Models
- Qwen3-32B (https://huggingface.co/Qwen/Qwen3-32B)
- meta-llama/Llama-3.1-8B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- meta-llama/Llama-3.1-70B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- amd/Llama-3.3-70B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV)
- amd/Llama-3.1-405B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)

MoE Models
- DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- DeepSeek-R1 (https://huggingface.co/deepseek-ai/DeepSeek-R1)
- Mixtral-8x7B-v0.1 (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Kimi-K2-Instruct (https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) — 1T MLA + MoE, native FP8
- Llama-4-Scout-17B-16E-Instruct (https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) — staged from the non-gated `unsloth/Llama-4-Scout-17B-16E-Instruct` mirror by default

This repository contains scripts and documentation to launch PD Disaggregation for above models. You will find setup instructions, node assignment details and benchmarking commands.

## 📝 Prerequisites

- A Slurm cluster with required Nodes -> xP + yD (minimum size 2: xP=1 and yD=1)
- Docker container with SGLang, MoRI and NIC drivers built-in. Refer to Building the Docker image section below.
- Access to a shared filesystem for log collection( cluster specific)


## Building the Docker image
Access the Dockerfile located at `docker/sglang_disagg_inference.ubuntu.amd.Dockerfile`.
It uses `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` as the base docker image.

```bash
docker build  -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .
```

### Build arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `INSTALL_MORI` | `1` | When `1`, builds and installs MoRI at `MORI_COMMIT`. Set to any other value (e.g. `0`) to skip MoRI installation entirely. |
| `MORI_COMMIT` | `158c7e8335a0b19b3f1f422ff134d7869252135e` | MoRI commit to check out and install when `INSTALL_MORI=1`. |
| `BASE_DOCKER` | `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` | Base SGLang ROCm image. |
| `GPU_ARCH` | `gfx942` | Target AMD GPU architecture. |

Examples:

```bash
# Default build: installs MoRI pinned at MORI_COMMIT
docker build -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Skip MoRI (e.g. when using Mooncake KV transfer only)
docker build --build-arg INSTALL_MORI=0 -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Pin a different MoRI commit
docker build --build-arg MORI_COMMIT=<sha> -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .
```

## Scripts

| File | Description |
|------|-------------|
| `run.sh` | **madengine entrypoint** (models.json `scripts` target). Bridges madengine's `sglang-disagg` launcher env to the launchers below |
| `run_xPyD_models.slurm` | Standalone SLURM script to launch docker containers on all nodes via `sbatch` |
| `sglang_disagg_mori_io_ep.sh` | Unified container entrypoint (MoRI EP / Mooncake / NIXL, selected via `KV_TRANSFER_BACKEND`) — starts prefill/decode servers, proxy, and benchmark |
| `models.yaml` | Model-specific CLI flags for all supported models |
| `mori_ep_env.sh` | RDMA/NCCL/Gloo environment variables |
| `benchmark_xPyD.sh` | Concurrency sweep benchmark using sglang bench_serving |
| `benchmark_parser.py` | Log parser for CONCURRENCY benchmark logs |
| `parse_to_csv.py` | Log parser for CONCURRENCY benchmark logs (emits summary CSV and madengine perf CSV) |

## Running via madengine (`madengine run`)

This workload is registered in `models.json` (e.g. `sglang-disagg-deepseek-r1-overlay`) and
runs through madengine's built-in **`sglang-disagg`** launcher. The launcher runs one
container per node and exports the cluster topology; `run.sh` maps it to the launchers
above:

| madengine env | bridged to |
|---------------|-----------|
| `SGLANG_NODE_RANK` | `NODE_RANK` (0=proxy, 1..xP=prefill, rest=decode) |
| `SGLANG_DISAGG_PREFILL_NODES` / `..._DECODE_NODES` | `xP` / `yD` |
| `SGLANG_NODE_IPS` | `IPADDRS` (+ `MASTER_ADDR` = rank-0 IP) |
| `SGLANG_TP_SIZE` | `IO_EP_TP_SIZE` / per-server `--tp-size` |

Model + transport come from deployment `env_vars` (or `args`): `MODEL_NAME` (short key
in `models.yaml`), `MODEL_PATH` (mounted weights), `RUN_MORI` (1=MoRI EP default,
0=Mooncake/NIXL), `DP_MODE` (1=DP-attention EP for DeepSeek-V3/R1 and
Kimi-K2-Instruct inter-node).

Deploy with an additional-context config carrying `slurm` + `distributed.launcher:
"sglang-disagg"` (+ `distributed.sglang_disagg.{prefill_nodes,decode_nodes}` for a
custom split) + `env_vars`; see the Confluence package for a ready 5-node EP16 manifest.

## Docker overlays (substitutable component versions)

The base image (`docker/sglang_disagg_inference.ubuntu.amd.Dockerfile`) bakes RCCL,
MoRI and Mooncake/NIXL. To test *specific* versions of these components without
rebuilding the base, use the merged full overlay.

Its default base is a `rocm/sgl-dev` tag that already carries all of them, so the
component stages are **off by default** and the image is the base plus rdma-core.
Turn a stage on with its `ENABLE_*` arg to A/B that one component against what the
base ships. The target GPU is chosen in two places only: the `BASE_DOCKER` tag
suffix (`-mi35x` = gfx950, `-mi30x` = gfx942) and `BUILD_GPU_TARGETS`, which
`MORI_GPU_ARCHS` derives from.

| Overlay Dockerfile | Substitutes | Key build args |
|--------------------|-------------|----------------|
| `docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile` | rdma-core, and optionally RCCL + MoRI + Mooncake/NIXL | `BASE_DOCKER`, `BUILD_GPU_TARGETS`; `RDMA_CORE_VERSION` (default `63.0`, built from source — the Broadcom Thor2 `bnxt_re` queue-pair fix; pass an empty value to keep the base image's rdma-core); `ENABLE_RCCL_OVERLAY=1` with `RCCL_REPO/RCCL_BRANCH/RCCL_COMMIT`; `ENABLE_MORI_OVERLAY=1` with `MORI_REPO/MORI_BRANCH/MORI_COMMIT` or `MORI_WHEEL_URL`; `ENABLE_NIXL_OVERLAY=1` with `UCX_REPO/UCX_COMMIT` and `RIXL_REPO/RIXL_COMMIT` (the AMD "nixl" backend); `ENABLE_MOONCAKE_OVERLAY=1` with `MOONCAKE_REF` (default `v0.3.12.post1`, the first release with the cross-host HIP guard) |

> **rdma-core note:** the `bnxt_re` provider in rdma-core 39.0/50.0 corrupts the
> verbs command buffer for work issued off worker threads, so `ibv_create_qp`
> fails with EFAULT and the KV-transfer backends lose their queue pairs. That is
> why `RDMA_CORE_VERSION` defaults to `63.0` rather than being opt-in, and why
> the stage also drops the vendor `bnxt_re` provider.

> **RCCL note:** a newer rocm-systems RCCL build links `librccl` without
> the `librocm_smi64.so` `DT_NEEDED` the base carried. Since the overlay is first
> on `LD_LIBRARY_PATH`, this would break `import torch`
> (`libtorch_hip.so: undefined symbol: rsmi_init`). The overlay Dockerfile
> re-adds the dependency (`patchelf --add-needed`) and sanity-checks `import
> torch` with the overlay librccl resolved first, so the full
> `base -> rccl -> mori -> kvtransfer` chain stays importable.

## Quick Start

```bash
git clone https://github.com/ROCm/MAD.git
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export xP=1
export yD=1
export MODEL_NAME=Llama-3.1-8B-Instruct
export RUN_MORI=1  # MoRI (default). Set RUN_MORI=0 for Mooncake (KV_TRANSFER_BACKEND=mooncake)
export KV_TRANSFER_BACKEND=mori  # Validated/default. Use nixl only with an image that provides nixl._api.

# num_nodes = xP + yD
sbatch -N 2 -n 2 --nodelist=<node1,node2> run_xPyD_models.slurm
```

## Current Support Status

### KV Cache Transfer Backend

The unified launcher (`sglang_disagg_mori_io_ep.sh`) selects the disaggregation KV-cache transfer backend (`--disaggregation-transfer-backend`) via `RUN_MORI`. The backend is intentionally kept out of `models.yaml` so model configs stay backend-agnostic, and it applies to all supported models.

| `RUN_MORI` | `KV_TRANSFER_BACKEND` | Backend | Status |
|------------|-----------------------|---------|--------|
| `1` | `mori` | MoRI IO | Supported |
| `0` | `mooncake` | Mooncake | Supported |

### WideEP/LargeEP Parallelism Mode (`DP_MODE`)

`PARALLEL_MODE` is derived strictly from `DP_MODE`, which selects the flag set from `models.yaml`.

| `DP_MODE` | `PARALLEL_MODE` | Flags applied | Supported models |
|-----------|-----------------|---------------|------------------|
| `0` (default) | `tp` | `base_flags` + `tp_flags` + `prefill.tp` / `decode.tp` | All models |
| `1` | `dp` | `base_flags` + `dp_flags` (`--moe-a2a-backend mori`, DP attention) + `prefill.dp` / `decode.dp` | DeepSeek-V3, DeepSeek-R1, Kimi-K2-Instruct only |

- `DP_MODE=1` enables MoRI expert parallelism (EP) with DP attention and currently requires `RUN_MORI=1` (MoRI IO).
- It is restricted by an allowlist (`MORI_DP_MODE1_ALLOWED_MODELS`) enforced in both `run_xPyD_models.slurm` and `sglang_disagg_mori_io_ep.sh`. Launching any model outside that allowlist with `DP_MODE=1` exits with an error — use `DP_MODE=0` (TP) for everything else.

## Log Files

Logs are written to `${LOG_PATH}/${SLURM_JOB_ID}/`:

| File | Description |
|------|-------------|
| `pd_sglang_bench_serving.sh_NODE<N>.log` | Main per-node log |
| `prefill_NODE<N>.log` | Prefill server log |
| `decode_NODE<N>.log` | Decode server log |
| `benchmark_*_CONCURRENCY.log` / `.csv` | Benchmark results |

## Benchmarking

Parse benchmark results:
```bash
python3 parse_to_csv.py <log_path>/benchmark_XXX_CONCURRENCY.log -o results.csv
```

Smoke test from the proxy node:
```bash
curl -X POST http://127.0.0.1:2322/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Let me tell you a story", "sampling_params": {"temperature": 0.3}}'
```

## Agentic Replay Benchmark

The agentic replay benchmark replays real Claude Code agentic coding traces using the `inferencex-agentx-mvp` scenario from [SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf), driven against the sglang router (`:2322`). This workload is highly prefix-repetitive (each turn re-sends the accumulated context), so it exercises the radix prefix cache and reports a **measured** `gpu_cache_hit_rate` alongside the dataset's theoretical ceiling.

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

### Quick start (canonical DeepSeek-V3 1P/1D)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>          # e.g. rocm/pytorch-private:sglang-...-mori-YYYYMMDD
export BENCHMARK_SCRIPT=agentic                        # selects benchmark_agentic.sh
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000

# 3 same-subnet (10.158.x) nodes with the model on local NVMe
sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

`BENCHMARK_SCRIPT=agentic` selects `benchmark_agentic.sh`; the in-container launcher then auto-enables server-side Prometheus metrics and strips `--disable-radix-cache` (cache ON).

### Environment variables

Full AgentX env reference: [scripts/common/agentx/README.md#environment-variable-reference](../common/agentx/README.md#environment-variable-reference).
The table below lists only the launcher-specific knobs; the shared AgentX
variables (`AGENTIC_CONC`, `AGENTIC_CACHE_WARMUP_DURATION`, `MAX_MODEL_LEN`,
`AGENTIC_CONFIG`, `AGENTIC_WORKLOAD`, `AGENTIC_STRICT_CONTEXT`,
`AGENTIC_HF_ISL_TAIL`, …) are documented centrally.

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_SCRIPT` | `sweep` | Set to `agentic` to run the agentic replay (`sweep` -> `benchmark_xPyD.sh`, `agentic` -> `benchmark_agentic.sh`) |
| `DURATION` | `120` | Benchmark duration in seconds (use `>=900` for a canonical/valid submission) |
| `AGENTIC_PORT` | `2322` | sglang router port aiperf targets |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (auto-on for agentic) |
| `ENABLE_RADIX_CACHE` | auto | `1` forces radix cache on i.e. strips `--disable-radix-cache` (auto-on for agentic) |
| `SUITE_CORPUS_FORCE` | `0` | Set `1` to regenerate a cached corpus after editing a profile. |

Note: `AGENTIC_SERVER_METRICS` is set automatically by the launcher to the space-separated server metrics endpoints (`<prefill_ip>:3000 <decode_ip>:3000`) and consumed by aiperf's `--server-metrics`; it is not a boolean toggle.

For config schema, profile authoring, and Tier 1/Tier 2 knobs, see:
- [AgentX core README](../common/agentx/README.md)
- [Profile authoring guide](../common/agentx/profiles/README.md)
- [Config examples cookbook](../common/agentx/README.md#examples)

### Running HF (weka) trace datasets on your own model

The agentic suite (`scripts/common/agentx/`, driven by `benchmark_agentic_suite.sh` via an `AGENTIC_CONFIG=agentic.yaml`) can replay the SemiAnalysis HF weka corpora against *any* served model, not just the 256k/1M reference targets:

- **Model-agnostic context gating.** Leave `serving.max_model_len` unset or `0` (or `MAX_MODEL_LEN=0`) and the suite resolves the served window automatically. On disaggregated serving the router front-end may not advertise `max_model_len`, so the window is resolved from the prefill worker (the first `host:port` in `AGENTIC_SERVER_METRICS`, auto-derived by the recipe) via `/v1/models` (`data[0].max_model_len`, with a `/get_server_info` fallback); a monolith front-end that does advertise it is auto-detected directly. Each hf workload gets a loader-derived ISL tail (a `_256k` loader → 262144, a full corpus → a conservative ~1M, override with `AGENTIC_HF_ISL_TAIL`). If the tail exceeds the served window the suite **WARNs** and caps `--max-context-length` at the window; `AGENTIC_STRICT_CONTEXT=1` **SKIP**s that workload instead. An explicit `max_model_len` / `MAX_MODEL_LEN` always wins.
- **Tier 1 replay knobs (per workload).** `num_dataset_entries` (how many trace sessions to pull) and `trajectory: {min, max}` (start-window ratio) are per-workload fields; unset keeps today's defaults (`393`, `0.25`/`0.75`).
- **Tier 2 local trim (fit a smaller model).** Add a `filter: {max_isl, max_turns, sample}` block to an hf workload/preset: the corpus is downloaded once, trimmed locally (drop sessions with any turn over `max_isl`, truncate to `max_turns`, randomly `sample` with seed 42), and replayed through `--input-file` — no server-side changes.
- **Reusable presets.** Drop a `source: hf` file in `scripts/common/agentx/profiles/` (see `profiles/inferencex_256k.yaml`) and reference it with `preset: <name>`; it can bundle the loader, Tier 1/Tier 2 knobs, and a concurrency sweep + duration. Entry fields override the preset; env (`AGENTIC_CONC`/`DURATION`/`MAX_MODEL_LEN`) still wins.

See `scripts/common/agentx/agentic.example.yaml` for a worked config.

### Supported models and modes

Agentic replay works for every model in the recipe allowlist. `DP_MODE=1` (MoRI wideEP) is only valid for the DeepSeek models; all others run `DP_MODE=0` (TP-only).

| Model | `DP_MODE` | Notes |
|-------|-----------|-------|
| DeepSeek-V3 | 1 (wideEP) | Canonical agentic target |
| DeepSeek-R1 | 1 (wideEP) | Same path as DeepSeek-V3 |
| Qwen3-32B | 0 (TP) | |
| Mixtral-8x7B-Instruct-v0.1 | 0 (TP) | |
| Llama-3.1-8B-Instruct | 0 (TP) | Good quick smoke |
| amd-Llama-3.3-70B-Instruct-FP8-KV | 0 (TP) | |
| Llama-3.1-405B-Instruct-FP8-KV | 0 (TP) | Needs larger topology |

Topology is `xP` prefill + `yD` decode + 1 router node (minimum 3 nodes, i.e. 1P/1D).

### Invocation examples (agentic suite)

**Note:** `AGENTIC_CONFIG` paths must be visible inside the container. The `$HOME:$HOME` bind mount means `$HOME/MAD/...` works if MAD is cloned at `$HOME/MAD`. Alternatively use `/opt/mooncake-cookbook/scripts/...` (the in-container `MOONCAKE_REPO_DIR` mount).

#### 1. Multi-workload suite config

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

#### 2. Single preset shorthand (no config file)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_WORKLOAD=conformance_256k   # config-less shorthand: runs just this preset

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

#### 3. Longer-context baseline (Case-B 512k window)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_WORKLOAD=conformance_512k
export MAX_MODEL_LEN=524288                # Case-B ISL tail needs the 512k window

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

#### 4. Single entry selected from a config (smaller model)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=0 xP=1 yD=1 MODEL_NAME=Llama-3.1-8B-Instruct
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml
export AGENTIC_WORKLOAD=inferencex_small   # preset bundles filter: {max_isl, max_turns, sample}

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

#### 5. DRY_RUN preview from the login node (no server)

```bash
cd $HOME/MAD

# Prints the resolved N-workload plan + per-workload commands + context verdicts; no server needed.
DRY_RUN=1 AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml \
  bash scripts/common/benchmark_agentic_suite.sh
```

### Output files

Written to `/run_logs/$SLURM_JOB_ID/` (host `LOG_PATH/<jobid>/`):

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # aggregate metrics (parsed by validate_agentic_result.sh)
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
```

Key fields in the aggregate JSON:
- `request_metrics.cache.theoretical_cache_hit_rate` — dataset-derived ceiling.
- `server_metrics.present` / `server_metrics.cache.gpu_cache_hit_rate` — **measured** runtime GPU cache hit rate.
- `request_accounting` — `records_profiled` vs `records_error_dropped` (error rate is computed over measured requests, excluding warmup).

### Validation

`scripts/common/validate_agentic_result.sh` runs automatically at the end of the agentic benchmark and prints, e.g.:

```
[validate] requests: 70/72 ok  error_rate=2.8%
[validate] gpu_cache_hit_rate: 32.4%
[validate] throughput_per_gpu: 6247 tok/s
[validate][PASS] error_rate within 10%
```

Thresholds: `AGENTIC_MAX_ERROR_RATE` (default `0.10`, gates PASS/WARN) and `AGENTIC_MIN_CACHE_HIT` (default `0.30`, warn-only). A low cache hit rate on `xP>1` usually indicates the router is not keeping a conversation's turns on the worker holding its prefix (session-affinity).

## Known Issues
For larger models, such as DeepSeekV3 and Llama-3.1-405B-Instruct-FP8-KV and higher concurrency(512+), errors with below signature is observed:<br>
_'<TransferEncodingError: 400, message:\n  Not enough data to satisfy transfer length header.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n '_<br>
This leads to dropping requests and lower throughput.This issue is being discussed on the SGLang forums.
