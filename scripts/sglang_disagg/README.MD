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


## Build and submit
Build the ordinary image from the repository root: `docker build -t <image> -f docker/sglang_disagg_inference.ubuntu.amd.Dockerfile .`
Build the dedicated profiling image from the repository root: `docker build --build-arg ENABLE_ROCTX=1 -t <profile-image> -f docker/sglang_disagg_inference_profile.ubuntu.amd.Dockerfile .`
Submit from `scripts/sglang_disagg/`: `DOCKER_IMAGE_NAME=<image> MODEL_NAME=<model> xP=1 yD=1 USE_CX7_NICS=1 sbatch -N2 -n2 run_xPyD_models.slurm`
Use the profiling image with `RUN_PROFILE=1`. Its Dockerfile starts from the compatible public ROCm 7.2.0 SGLang base below and upgrades installed ROCm userspace packages to ROCm 7.2.3.

### Image build arguments (ordinary first; profiling from `BASE_DOCKER`)

| Arg | Default | Description |
|-----|---------|-------------|
| `INSTALL_MORI` | `1` | When `1`, builds and installs MoRI at `MORI_COMMIT`. Set to any other value (e.g. `0`) to skip MoRI installation entirely. |
| `MORI_COMMIT` | `158c7e8335a0b19b3f1f422ff134d7869252135e` | MoRI commit to check out and install when `INSTALL_MORI=1`. |
| `BASE_DOCKER` | `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718` | Compatible public ROCm 7.2.0 SGLang base; its userspace stack is upgraded to ROCm 7.2.3. |
| `GPU_ARCH` | `gfx942` | Target AMD GPU architecture. |
| `ENABLE_ROCTX` | `0` | When `1`, `MORI_ROCTX_COMMIT` and `SGLANG_ROCTX_COMMIT` select pinned instrumentation and instrumented MoRI is rebuilt regardless of `INSTALL_MORI`, including `0`. It does not control the ROCm 7.2.3 upgrade. |

The dedicated profiling build uses ROCm 7.2.3. The ordinary image remains the exact current `develop` Dockerfile. For `EAGER=0` profiling, do not substitute the stock ROCm 7.2.0 image: its CUDA Graph replay and rocprof finalization paths are known to fail.

Examples:

```bash
# Default build: installs MoRI pinned at MORI_COMMIT
docker build -t sglang_disagg_pd_image -f docker/sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Skip MoRI (e.g. when using Mooncake KV transfer only)
docker build --build-arg INSTALL_MORI=0 -t sglang_disagg_pd_image -f docker/sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Pin a different MoRI commit
docker build --build-arg MORI_COMMIT=<sha> -t sglang_disagg_pd_image -f docker/sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Enable ROCTX markers for profiling
docker build --build-arg ENABLE_ROCTX=1 -t <profile-image> -f docker/sglang_disagg_inference_profile.ubuntu.amd.Dockerfile .
```

## Scripts

| File | Description |
|------|-------------|
| `run_xPyD_models.slurm` | SLURM script to launch docker containers on all nodes via `sbatch` |
| `sglang_disagg_mori_io_ep.sh` | Container entrypoint — starts prefill/decode servers, proxy, and benchmark |
| `models.yaml` | Model-specific CLI flags for all supported models |
| `mori_ep_env.sh` | RDMA/NCCL/Gloo environment variables |
| `benchmark_xPyD.sh` | Ordinary current-`develop` concurrency sweep benchmark (used when profiling is off) |
| `benchmark_xPyD_profile.sh` | Profiling concurrency sweep with deterministic RIDs, client CSV/manifests, and multi-sweep support |
| `benchmark_parser.py` | Log parser for CONCURRENCY benchmark logs |

## Quick Start

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export xP=1
export yD=1
export MODEL_NAME=Llama-3.1-8B-Instruct
export RUN_MORI=1  # Explicit MoRI; RUN_PROFILE=1 defaults to MoRI, while normal runs default to Mooncake.
export USE_CX7_NICS=1

# num_nodes = xP + yD
# CX7 requires an appropriate same-rail node allocation.
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

- `DP_MODE=1` enables MoRI expert A2A with DP attention; `RUN_MORI` independently selects the disaggregation KV-transfer backend.
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

`RUN_PROFILE=1` selects `benchmark_xPyD_profile.sh`; profiling off selects the exact current-`develop` `benchmark_xPyD.sh`. End-to-end profile submissions default to `SKIP_WARMUP=1`; set `SKIP_WARMUP=0` to run the legacy 1024-prompt/512-concurrency warmup. `BENCHMARK_NUM_PROMPTS` overrides the generated prompt count per sweep. The image's `SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX` feature defaults to `0` (off) in this launcher; set it explicitly to `1` to opt in.

`RUN_PROFILE=1` enables MoRI by default; profiling off defaults to Mooncake unless `RUN_MORI` is explicitly set. `EAGER=0` avoids a global CUDA-graph disable and preserves supported graphs, but SGLang/model compatibility may still disable prefill graphs.

Set `RUN_PROFILE=1` when submitting with the profiling image to enable integrated profiling (rocprof capture + kernel analysis). Profile benchmark requests receive deterministic RIDs and produce client timing/manifests without duplicate probe traffic. Multi-sweep runs use space-separated `ISL/OSL` entries in `BENCHMARK_COMBINATIONS`; a separate nonempty `BENCHMARK_CON` is required for profiling. `RUN_MORI=1` adds MoRI markers and strict request/KV correlation; `RUN_MORI=0` produces SGLang-only traces and reqstats without MoRI mapping artifacts. Output artifacts land under `moriio_profiling/artifacts/pull_<jobid>/`.

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

## Known Issues
For larger models, such as DeepSeekV3 and Llama-3.1-405B-Instruct-FP8-KV and higher concurrency(512+), errors with below signature is observed:<br>
_'<TransferEncodingError: 400, message:\n  Not enough data to satisfy transfer length header.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n '_<br>
This leads to dropping requests and lower throughput.This issue is being discussed on the SGLang forums.
