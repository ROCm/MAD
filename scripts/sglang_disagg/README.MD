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
- DeepSeek-V4-Flash-FP8 (https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) — MoRI-EP wide-EP; use the FP8-E4M3 checkpoint, DP-attention mandatory (num_key_value_heads=1). EP8 1P1D and EP16 2P2D validated on MI308X/gfx942 + Broadcom Thor2 (set USE_CX7_NICS=0).

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

## Known Issues
For larger models, such as DeepSeekV3 and Llama-3.1-405B-Instruct-FP8-KV and higher concurrency(512+), errors with below signature is observed:<br>
_'<TransferEncodingError: 400, message:\n  Not enough data to satisfy transfer length header.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n '_<br>
This leads to dropping requests and lower throughput.This issue is being discussed on the SGLang forums.

## DeepSeek-V4-Flash-FP8 (MoRI-EP wide-EP, EP8 + EP16)

DeepSeek-V4-Flash is served through the MoRI-EP wide-EP path (`DP_MODE=1`). Notes specific to this model:

- **Checkpoint:** use the **FP8-E4M3 block-quant** weights (e.g. `sgl-project/DeepSeek-V4-Flash-FP8`). The model uses the dedicated `dsv4` attention backend (not `aiter`), and `num_key_value_heads=1` so attention cannot be tensor-sharded — **DP-attention is mandatory** for any EP run.
- **MoRI-EP mode:** prefill uses `--deepep-mode normal` (high-throughput), decode uses `--deepep-mode low_latency` (matches the MoRI HT-prefill / LL-decode split). Set in `models.yaml`.
- **Prefill is eager** (`--disable-cuda-graph`); decode uses cudagraph capture.

### Topologies (via `models.json`)

| Entry | xP | yD | Nodes |
|-------|----|----|-------|
| `pyt_sglang_disagg_mori_io_dsv4-flash_ep8_1p1d`  | 1 | 1 | 2 (EP8, 1 prefill + 1 decode) |
| `pyt_sglang_disagg_mori_io_dsv4-flash_ep16_2p2d` | 2 | 2 | 4 (EP16, 2 prefill + 2 decode) |

### Docker image

A prebuilt image (SGLang + MoRI on ROCm 7.2, gfx942) is available:

```bash
docker pull rocmshared/sglang-disagg-dsv4:mori-mi308-pr
```

Or build via `docker/sglang_disagg_inference.ubuntu.amd.Dockerfile` (base `lmsysorg/sglang:v0.5.15-rocm720-mi30x`, MoRI from source).

### Broadcom Thor2 (bnxt) / non-CX7 clusters

Set `USE_CX7_NICS=0`. On bnxt the image's baked `libbnxt_re` may be ABI-stale versus the host kernel; the launcher exposes the host's ABI-correct `libbnxt_re` and runs `ldconfig` in-container automatically for this path. RDMA NIC selection (`IB_DEVICES` / `MORI_RDMA_DEVICES`) should be set to the node's RoCE devices.

### Validation (MI308X/gfx942 + Broadcom Thor2)

Both topologies validated end-to-end (serve via `sglang_router`, greedy `/v1/completions`):

- **Correctness:** needle-in-a-haystack **18/18** (context 1K→200K, needle at 10/50/90% depth) on **both** EP8 1P1D and EP16 2P2D. Factual length-sweep coherent to 200K.
- **Perf** (functional, eager decode; all requests succeeded): 8K/1K and 16K/1K input/output at concurrency 16 and 32.
