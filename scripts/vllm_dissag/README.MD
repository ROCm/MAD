## List of Models - focus VLLM Disaggregated P/D inference

Supported Models (use these exact `MODEL_NAME` values):
- DeepSeek-R1
- DeepSeek-V3
- DeepSeek-V3-5layer
- amd-Llama-3.3-70B-Instruct-FP8-KV
- Llama-3.1-405B-Instruct-FP8-KV
- gpt-oss-120b

This repository contains scripts and documentation to launch PD Disaggregation using the Nixl framework for above models. You will find setup instructions, node assignment details and benchmarking commands.

## Prerequisites

- A Slurm cluster with required Nodes -> xP + yD (minimum size 2: xP=1 and yD=1)
- Docker container with VLLM, Nixl and NIC drivers built-in. Refer to Building the Docker image section below.
- Access to a shared filesystem for log collection (cluster specific)


## Building the Docker image


```bash
cd MAD
docker build -t vllm_dissag_pd_image -f docker/vllm_disagg_inference.ubuntu.amd.Dockerfile .
```

## Scripts and Benchmarking

Key files:

| File | Description |
|------|-------------|
| `run_xPyD_models.slurm` | Slurm script to launch docker containers on all nodes using sbatch |
| `vllm_disagg_server.sh` | Default PD server script (NixlConnector, no expert parallel) |
| `vllm_disagg_mori_ep.sh` | MoRI EP server script (MoRIIOConnector, expert parallel) |
| `vllm_disagg_server_deepep.sh` | DeepEP server script (NixlConnector, DeepEP all2all backends) |
| `benchmark_xPyD.sh` | Benchmark script using vLLM benchmarking tool |

## Run Modes

The `run_xPyD_models.slurm` script supports three run modes, controlled by the `RUN_MORI` and `RUN_DEEPEP` environment variables. At most one of these may be set to `1`.

| Mode | Env Variable | Server Script | KV Connector | Models |
|------|-------------|---------------|--------------|--------|
| Default (NixlConnector) | Neither set | `vllm_disagg_server.sh` | NixlConnector | All VALID_MODELS |
| MoRI EP | `RUN_MORI=1` | `vllm_disagg_mori_ep.sh` | MoRIIOConnector | DeepSeek-V3, DeepSeek-V3-5layer, DeepSeek-R1 |
| DeepEP | `RUN_DEEPEP=1` | `vllm_disagg_server_deepep.sh` | NixlConnector | DeepSeek-V3, DeepSeek-V3-5layer, DeepSeek-R1 |

Setting both `RUN_MORI=1` and `RUN_DEEPEP=1` will exit with an error.

## Sbatch Run Commands

### Default mode (NixlConnector)
```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/vllm_dissag

export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export xP=1; export yD=1; export MODEL_NAME=DeepSeek-V3
sbatch -N 2 -n 2 --nodelist=<node0,node1> run_xPyD_models.slurm
```

### MoRI EP mode
```bash
export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export RUN_MORI=1
export xP=1; export yD=1; export MODEL_NAME=DeepSeek-R1
sbatch -N 2 -n 2 --nodelist=<node0,node1> run_xPyD_models.slurm
```

### DeepEP mode
```bash
export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export RUN_DEEPEP=1
export xP=1; export yD=1; export MODEL_NAME=DeepSeek-V3
sbatch -N 2 -n 2 --nodelist=<node0,node1> run_xPyD_models.slurm
```

### DeepEP environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFILL_DEEPEP_BACKEND` | `deepep_high_throughput` | All2all backend for prefill nodes |
| `DECODE_DEEPEP_BACKEND` | `deepep_low_latency` | All2all backend for decode nodes |
| `ENABLE_DBO` | `false` | Enable Dynamic Batching Optimization |
| `DBO_COMM_SMS` | (vLLM default) | DBO communication SMs override |
| `ENABLE_PROFILING` | `false` | Enable profiling |

`num_nodes = xP + yD` (proxy co-located on prefill master node)

## Node Topology (all modes)

```
Node 0          -> Prefill MASTER + Proxy (co-located)
Nodes 1..xP-1   -> Prefill CHILD (if xP > 1)
Node xP         -> Decode MASTER
Nodes xP+1..end -> Decode CHILD (if yD > 1)
```

The proxy/router runs on the same node as the Prefill master (Node 0) to save one physical node. The proxy is CPU-only and listens on a separate port from the vLLM server.

**Port defaults by mode:**

| Mode | vLLM Server Port | Proxy Port |
|------|-----------------|------------|
| Default (NixlConnector) | 2584 | 18001 (`ROUTER_PORT`) |
| DeepEP | 2584 | 18001 (`ROUTER_PORT`) |
| MoRI EP | 20005 | 10001 (fixed, MoRI-specific) |

## Proxy Server Options

The scripts support two proxy server types via the `PROXY_TYPE` environment variable:

| Proxy Type | Description |
|------------|-------------|
| `vllm_router` (default) | Production-grade Rust-based load balancer |
| `toy_proxy` | Simple Python-based proxy for testing |

### Using Toy Proxy (for testing)
```bash
export PROXY_TYPE=toy_proxy
# Then run sbatch/srun as usual
```

> **Note:** `PROXY_TYPE` and `ROUTER_PORT` apply to **Default** and **DeepEP** modes only. MoRI EP uses its own fixed proxy (`moriio_toy_proxy_server.py` on port 10001).

## Benchmark Configuration (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_ITR` | `1` | Number of benchmark iterations |
| `BENCHMARK_CON` | `8 16 32 64 128 256 512` | Space-separated concurrency levels |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024 1024/8192` | Space-separated ISL/OSL combinations |

Example:
```bash
export BENCHMARK_ITR=2
export BENCHMARK_CON="8 16 32"
export BENCHMARK_COMBINATIONS="1024/1024 8192/1024"
sbatch -N 2 -n 2 run_xPyD_models.slurm
```

## Benchmark parser (for CONCURRENCY logs) to tabulate different data

```bash
python3 benchmark_parser.py <log_path/benchmark_XXX_CONCURRENCY.log
```
