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
| `run.sh` | **madengine entrypoint** (models.json `scripts` target). Bridges madengine's `sglang-disagg` launcher env to the launchers below |
| `run_xPyD_models.slurm` | Standalone SLURM script to launch docker containers on all nodes via `sbatch` |
| `sglang_disagg_mori_io_ep.sh` | Unified container entrypoint (MoRI EP / Mooncake / NIXL, selected via `KV_TRANSFER_BACKEND`) — starts prefill/decode servers, proxy, and benchmark |
| `models.yaml` | Model-specific CLI flags for all supported models |
| `mori_ep_env.sh` | RDMA/NCCL/Gloo environment variables |
| `benchmark_xPyD.sh` | Concurrency sweep benchmark using sglang bench_serving |
| `benchmark_parser.py` | Log parser for CONCURRENCY benchmark logs |
| `parse_to_csv.py` | Log parser for CONCURRENCY benchmark logs (emits summary CSV and madengine perf CSV) |

## Running via madengine (`madengine run`)

This workload is registered in `models.json` (e.g. `sglang_disagg_deepseek-r1`) and
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
0=Mooncake/NIXL), `DP_MODE` (1=DP-attention EP for DeepSeek-V3/R1 inter-node).

Deploy with an additional-context config carrying `slurm` + `distributed.launcher:
"sglang-disagg"` (+ `distributed.sglang_disagg.{prefill_nodes,decode_nodes}` for a
custom split) + `env_vars`; see the Confluence package for a ready 5-node EP16 manifest.

## Docker overlays (substitutable component versions)

The base image (`docker/sglang_disagg_inference.ubuntu.amd.Dockerfile`) bakes RCCL,
MoRI and Mooncake/NIXL. To test *specific* versions of these components without
rebuilding the base, use the merged full overlay, which layers RCCL, MoRI and the
KV-transfer (Mooncake/NIXL) steps on top of `BASE_DOCKER` in a single image
(`base -> rccl -> mori -> kvtransfer`):

| Overlay Dockerfile | Substitutes | Key build args |
|--------------------|-------------|----------------|
| `docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile` | RCCL + MoRI + Mooncake/NIXL (merged) | `RCCL_REPO/RCCL_BRANCH/RCCL_COMMIT`, `MORI_REPO/MORI_BRANCH/MORI_COMMIT` or `MORI_WHEEL_URL`, `MOONCAKE_VERSION` / `NIXL_VERSION`, `ENABLE_RDMA62` (set to `1` on hosts needing a newer rdma-core than the base image ships, to additionally build rdma-core v62 from source) |

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
python3 parse_to_csv.py <log_path>/benchmark_XXX_CONCURRENCY.log -o results.csv
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
