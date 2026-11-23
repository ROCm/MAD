# Large EP Microbenchmarking (MoRI-EP & DeepEP)

## Overview

These scripts run microbenchmarks for **MoRI-EP** (MoRI collective engine) and **DeepEP** (expert-parallel / MoE communication) on a Slurm cluster with AMD Instinct accelerators and InfiniBand (CX7) networking. The benchmarks validate collective performance and low-latency behavior in both intranode and internode configurations.

The Docker image is built from the MAD repository and is intended for use on clusters with **Mellanox CX7** NICs and compatible AMD GPUs (e.g., gfx942). You can adjust the Dockerfile for different NICs or GPU architectures as needed.

## Why this benchmarking?

- **Validate communication stacks** — Before running large training or inference workloads (e.g., MoE models, distributed LLMs), microbenchmarks confirm that MoRI and DeepEP collectives achieve expected throughput and latency on your cluster. Regressions in drivers, ROCm, or NIC firmware show up here first.
- **Tune and qualify hardware** — Results help tune environment variables (e.g., `ROCSHMEM_HEAP_SIZE`, `IBDEVICES`), compare NIC/GPU configurations, and qualify new nodes or fabrics (intranode vs internode, normal vs low-latency paths).
- **Regression and CI** — These scripts can be used in CI or release validation to ensure that updates to MoRI, DeepEP, rocSHMEM, or the ROCm stack do not degrade collective performance.
- **Research and development** — Microbenchmark data supports optimization of collective algorithms, FP8/low-latency paths, and scaling studies across nodes.

## MoRI and DeepEP: background and links

### MoRI (Modular RDMA Interface)

**MoRI** is a communication backend for the ROCm platform that optimizes **inter-node collective operations** on GPU clusters. It uses RDMA (Remote Direct Memory Access) for efficient GPU-to-GPU communication across nodes and supports standard collectives. MoRI is used in distributed inference (e.g., with SGLang / vLLM on AMD Instinct).

| Resource | Link |
|----------|------|
| **MoRI (GitHub)** | [ROCm/mori](https://github.com/ROCm/mori) |

### DeepEP and rocSHMEM

**DeepEP** is a high-performance communication library for **Mixture-of-Experts (MoE)** and **expert parallelism**, providing GPU **all-to-all** primitives with optional FP8/BF16 support. The AMD ROCm version uses xGMI/Infinity Fabric for intranode communication and InfiniBand/RoCE for internode communication. DeepEP builds on **rocSHMEM** (ROCm OpenSHMEM), an intra-kernel networking library that provides GPU-centric, OpenSHMEM-like APIs and a symmetric heap on GPU memory, enabling better communication–computation overlap than host-driven networking.

| Resource | Link |
|----------|------|
| **DeepEP (GitHub)** | [ROCm/DeepEP](https://github.com/ROCm/DeepEP) |
| **rocSHMEM — What is rocSHMEM? (ROCm docs)** | [rocSHMEM introduction](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/introduction.html) |
| **rocSHMEM (GitHub)** | [ROCm/ROC_SHMEM](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem) |

## Benchmarks Covered

| Scope       | Mode        | DeepEP | MoRI-EP |
|------------|-------------|--------|---------|
| Intranode  | Normal      | ✅     | ✅      |
| Intranode  | Low latency | ✅     | ✅      |
| Internode  | Normal      | ✅     | ✅      |
| Internode  | Low latency | ✅     | ✅      |

- **Intranode**: Single-node multi-process collectives.
- **Internode**: Multi-node collectives over InfiniBand.
- **Low latency**: FP8/low-latency collective paths (e.g., rocSHMEM heap and MoRI FP8).

## Prerequisites

- Slurm cluster with AMD Instinct nodes.
- **CX7** (or compatible) InfiniBand NICs; scripts assume InfiniBand devices are available.
- Docker available on compute nodes (or use a container runtime configured for your cluster).

## How to Run

### 1. Build the Docker image

Build from the **MAD repository root** (not from this script directory). The default Dockerfile targets **gfx942** and **CX7**; edit `docker/large_ep_benchmark.ubuntu.amd.Dockerfile` if your cluster uses a different GPU arch or NIC.

```bash
# From MAD repository root
docker build -f docker/large_ep_benchmark.ubuntu.amd.Dockerfile -t ep-benchmarking:latest .
```

### 2. Set environment variables

| Variable       | Description                          | Example              |
|----------------|--------------------------------------|----------------------|
| `DOCKER_IMAGE` | Docker image to run on the cluster   | `ep-benchmarking:latest` |
| `IBDEVICES`    | InfiniBand device(s) for rocSHMEM   | `mlx5_0` (default)   |
| `SKIP_DEEPEP`  | Set to `1` to skip the DeepEP tests and run MoRI-EP only | `0` (default) |

Optional:

- `LOG_PATH`: Root directory for benchmark logs (default: `/shared_inference/${USER}/model_blog_logs`). Each job writes to `$LOG_PATH/$SLURM_JOB_ID/`, so `LOG_PATH` must be on storage shared by all nodes.

>[!NOTE]
>`run_benchmark.sbatch` also bind-mounts `/shared_inference/models` into the container. If your cluster has no such path, edit the `docker run` command in the script.

### 3. Submit the Slurm job

Navigate to the scripts directory and submit the job (the current directory is mounted into the container as the script location). The script supports **1 to N nodes**: with one node it runs the intranode tests, and with more than one node it runs **only** the internode tests.

```bash
cd scripts/large-ep-benchmark

export DOCKER_IMAGE=ep-benchmarking:latest
export IBDEVICES=mlx5_0

sbatch -p <partition> -N <num-nodes> run_benchmark.sbatch
```

**Examples:**

- Single node (intranode only):
  ```bash
  sbatch -p <partition> -N 1 run_benchmark.sbatch
  ```

- Multi-node (internode only):
  ```bash
  sbatch -p <partition> -N 4 run_benchmark.sbatch
  ```

>[!NOTE]
> Ensure the partition and account allow the requested number of nodes and that compute nodes have Docker (or the configured container runtime) and access to the built image (e.g., via a registry or shared image path).

## Output and logs

- Slurm stdout/stderr: `ep_bench_slurm_job.out` and `ep_bench_slurm_job.err` in the directory where `sbatch` was run.
- Benchmark logs are written under `$LOG_PATH/$SLURM_JOB_ID/`, including:
  - `ep_bench_results.log` — main run log
  - Single node:
    - `intranode_results.log` / `low_latency_1N_results.log` — DeepEP normal and low latency
    - `mori_intranode_results.log` / `mori_1N_ll_results.log` — MoRI-EP normal and low latency (FP8)
  - Multi-node (per rank):
    - `internode_<rank>.txt` / `low-latency-<rank>.log` — DeepEP normal and low latency
    - `mori_internode_<rank>.txt` / `mori_ll_internode_<rank>.txt` — MoRI-EP normal and low latency
  - `perf.csv` — all results parsed by `parse_ep_to_csv.py`, for madengine
- DeepEP logs are absent when `SKIP_DEEPEP=1`.

### Running through MAD

`models.json` defines `pyt_large_ep_bench_{1n,2n}` (DeepEP and MoRI-EP) and `pyt_large_ep_bench_{1n,2n,4n}_mori_only`
(`SKIP_DEEPEP=1`, tag `large_ep_scaling`). They use the `slurm_multi` launcher and pass the node count as `-N <n> -n <n>`.


## Summary

| Step | Action |
|------|--------|
| 1    | Build image from MAD root: `docker build -f docker/large_ep_benchmark.ubuntu.amd.Dockerfile -t ep-benchmarking:latest .` |
| 2    | Set `DOCKER_IMAGE` and `IBDEVICES` (and optionally `SKIP_DEEPEP`, `LOG_PATH`) |
| 3    | From `scripts/large-ep-benchmark`, run: `sbatch -p <partition> -N <num-nodes> run_benchmark.sbatch` |

For different GPU architectures or NICs, update the Dockerfile and rebuild before submitting jobs.
