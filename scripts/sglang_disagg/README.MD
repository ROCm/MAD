## List of Models - focus SGLang Disaggregated P/D inference

Dense Models
- Qwen3-32B (https://huggingface.co/Qwen/Qwen3-32B)
- meta-llama/Llama-3.1-8B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- amd/Llama-3.3-70B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV)
- amd/Llama-3.1-405B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)

MoE Models
- DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- DeepSeek-R1 (https://huggingface.co/deepseek-ai/DeepSeek-R1)
- Mixtral-8x7B-v0.1 (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

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
| `run_xPyD_models.slurm` | SLURM script to launch docker containers on all nodes via `sbatch` |
| `sglang_disagg_mori_io_ep.sh` | Container entrypoint — starts prefill/decode servers, proxy, and benchmark |
| `models.yaml` | Model-specific CLI flags for all supported models |
| `mori_ep_env.sh` | RDMA/NCCL/Gloo environment variables |
| `benchmark_xPyD.sh` | Concurrency sweep benchmark using sglang bench_serving |
| `benchmark_parser.py` | Log parser for CONCURRENCY benchmark logs |

## Quick Start

```bash
git clone https://github.com/ROCm/MAD.git
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export xP=1
export yD=1
export MODEL_NAME=Llama-3.1-8B-Instruct
export RUN_MORI=1  # MoRI (default). Set RUN_MORI=0 for Mooncake (KV_TRANSFER_BACKEND=mooncake)

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
| `1` | `dp` | `base_flags` + `dp_flags` (`--moe-a2a-backend mori`, DP attention) + `prefill.dp` / `decode.dp` | DeepSeek-V3, DeepSeek-R1 only |

- `DP_MODE=1` enables MoRI expert parallelism (EP) with DP attention and currently requires `RUN_MORI=1` (MoRI IO).
- It is restricted by an allowlist (`MORI_DP_MODE1_ALLOWED_MODELS`) enforced in both `run_xPyD_models.slurm` and `sglang_disagg_mori_io_ep.sh`. Launching any other model with `DP_MODE=1` exits with an error — use `DP_MODE=0` (TP) for all non-DeepSeek models.

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
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log
```

Smoke test from the proxy node:
```bash
curl -X POST http://127.0.0.1:2322/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Let me tell you a story", "sampling_params": {"temperature": 0.3}}'
```

## Agentic Replay Benchmark

The agentic replay benchmark replays real Claude Code agentic coding traces using the `inferencex-agentx-mvp` scenario from [SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf), driven against the sglang router (`:2322`). This workload is highly prefix-repetitive (each turn re-sends the accumulated context), so it exercises the radix prefix cache and reports a **measured** `gpu_cache_hit_rate` alongside the dataset's theoretical ceiling.

It is a drop-in alternative to the default `benchmark_xPyD.sh` concurrency sweep, selected via `BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh` (or the `AGENTIC=1` shorthand). No GitHub Actions / CI is involved — it runs through the same `run_xPyD_models.slurm` entrypoint as the other benchmarks.

### Quick start (canonical DeepSeek-V3 1P/1D)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>          # e.g. rocm/pytorch-private:sglang-...-mori-YYYYMMDD
export AGENTIC=1                                       # shorthand: selects benchmark_agentic.sh
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000

# 3 same-subnet (10.158.x) nodes with the model on local NVMe
sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

`AGENTIC=1` sets `BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh`; the in-container launcher then auto-enables server-side Prometheus metrics and strips `--disable-radix-cache` (cache ON). You can equivalently set `BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh` directly.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTIC` | `0` | `1` = shorthand for `BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh` |
| `BENCHMARK_SCRIPT_FILE` | `benchmark_xPyD.sh` | Set to `benchmark_agentic.sh` to run the agentic replay |
| `DURATION` | `120` | Benchmark duration in seconds (use `>=900` for a canonical/valid submission) |
| `AGENTIC_CONC` | `16` | Concurrency (parallel agent chains) |
| `AGENTIC_CACHE_WARMUP_DURATION` | `300` (DeepSeek/Kimi/GLM), else `60` | Cache warmup seconds before the measured window |
| `MAX_MODEL_LEN` | model-dependent | Max context length passed to aiperf (`--max-context-length`) |
| `AGENTIC_PORT` | `2322` | sglang router port aiperf targets |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (auto-on for agentic) |
| `ENABLE_RADIX_CACHE` | auto | `1` forces radix cache on i.e. strips `--disable-radix-cache` (auto-on for agentic) |

Note: `AGENTIC_SERVER_METRICS` is set automatically by the launcher to the space-separated server metrics endpoints (`<prefill_ip>:3000 <decode_ip>:3000`) and consumed by aiperf's `--server-metrics`; it is not a boolean toggle.

### Running HF (weka) trace datasets on your own model

The agentic suite (`scripts/common/agentx/`, driven by `benchmark_agentic_suite.sh` via an `AGENTIC_CONFIG=agentic.yaml`) can replay the SemiAnalysis HF weka corpora against *any* served model, not just the 256k/1M reference targets:

- **Model-agnostic context gating.** Leave `serving.max_model_len` unset or `0` (or `MAX_MODEL_LEN=0`) and the suite auto-detects the served window from `/v1/models` (`data[0].max_model_len`, with a `/server_info` fallback). Each hf workload gets a loader-derived ISL tail (a `_256k` loader → 262144, a full corpus → a conservative ~1M, override with `AGENTIC_HF_ISL_TAIL`). If the tail exceeds the served window the suite **WARNs** and caps `--max-context-length` at the window; `AGENTIC_STRICT_CONTEXT=1` **SKIP**s that workload instead. An explicit `max_model_len` / `MAX_MODEL_LEN` always wins.
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
