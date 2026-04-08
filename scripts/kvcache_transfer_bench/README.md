# KV Cache Transfer Performance Benchmark

This benchmark measures KV cache transfer throughput across multiple backends (RIXL, Mori, Mooncake) on a two-node cluster with InfiniBand (Slurm, Kubernetes, or bare-metal).

## Overview

- **Backends**: RIXL, Mori, Mooncake (ROCm KV transfer engines)
- **Environment**: 2 nodes, Docker containers, InfiniBand (`mlx5_0` by default)
- **What it produces**: A sweep from `--start-size` through `--stop-size`, with JSON per backend, a merged CSV, and an interactive HTML report that compares backends across transfer sizes

## Prerequisites

- Two nodes with network connectivity (Slurm, Kubernetes, or bare-metal)
- Docker with InfiniBand device access
- ROCm-capable GPUs (for vLLM/KV cache estimator)

## Result files

Benchmark outputs are usually under `shared/results_<job_id>/` (path may differ if you set paths manually).


| Artifact                                                          | Description                                                                      |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `results_rixl.json`, `results_mori.json`, `results_mooncake.json` | Raw throughput JSON from each backend                                            |
| `results_merged.json`                                             | Single merged file combining all backends (normalized)                           |
| `results_merged.csv`                                              | Pivot table: transfer size vs throughput per backend                             |
| `report.html`                                                     | Interactive HTML report (tables + Plotly charts)                                 |
| `kv_cache_estimator.csv`                                          | Optional; from the KV cache estimator, used to overlay model sizes on the report |


The Slurm flow runs `merge_results.py` on the initiator so `results_merged.json`, `results_merged.csv`, and `report.html` are produced automatically. To regenerate manually, run `python scripts/merge_results.py --input-dir <results_dir>`; for KV-cache overlays, use the example in section 4 (after appending models).

## Quick Start

### 1. Build Docker Image

From the `kvcache_transfer_bench` directory:

```bash
docker build --network=host -f Dockerfile -t kv-cache-unified:latest .
```

To push to Docker Hub:

```bash
docker tag kv-cache-unified:latest <your-repo>/kv-cache-transfer-bench:latest
docker push <your-repo>/kv-cache-transfer-bench:latest
```

### 2. Run the Benchmark via Slurm

Submit the launcher from the `kvcache_transfer_bench` directory (so `SLURM_SUBMIT_DIR` points at this repo). Pass [Slurm options](https://slurm.schedmd.com/sbatch.html) to `sbatch` first, then the script path, then benchmark options:

```bash
cd kvcache_transfer_bench
sbatch [sbatch options] scripts/slurm_launcher.slrum [benchmark options]
```

**Benchmark options** (passed after the script name; forwarded by the launcher to `run_node.sh`):


| Option           | Default          | Description                                                                                    |
| ---------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| `--docker-image` | *(required)*     | Image to `docker pull` and run on both nodes (e.g. `kv-cache-unified:latest` or registry path) |
| `--start-size`   | 4096             | Minimum transfer size (bytes)                                                                  |
| `--stop-size`    | 1073741824 (1GB) | Maximum transfer size (bytes)                                                                  |
| `--backends`     | all              | Comma-separated: `rixl,mori,mooncake` or `all`                                                 |
| `--ibdevice`     | mlx5_0           | InfiniBand device                                                                              |
| `--sync-port`    | 9999             | TCP port for target/initiator sync                                                             |


**Slurm options** (passed to `sbatch` before the script):


| Option       | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `--nodelist` | Comma-separated node names (e.g. `node-hostname-1,node-hostname-2`) |
| `-t`         | Time limit (e.g. `-t 08:00:00`); required on some clusters          |
| `-N 2`       | Number of nodes (default in `scripts/slurm_launcher.slrum`)         |


**Example:**

```bash
# Run on specific nodes (pass -t if your cluster requires a runtime limit)
sbatch -t 08:00:00 --nodelist=node-hostname-1,node-hostname-2 \
  scripts/slurm_launcher.slrum --docker-image <your-repo>/kv-cache-unified:latest

# Or let Slurm pick any 2 nodes
sbatch -t 08:00:00 -N 2 scripts/slurm_launcher.slrum --docker-image <your-repo>/kv-cache-unified:latest
```

**Run on already allocated nodes:** If you have an interactive allocation (e.g. via `salloc`), run the launcher directly. It will use `SLURM_NODELIST` from your current session:

```bash
# 1. Allocate 2 nodes
salloc -N 2 -n 2 --ntasks-per-node=1 --time=08:00:00

# 2. From the allocation shell, run the benchmark
cd kvcache_transfer_bench
bash scripts/slurm_launcher.slrum --docker-image <your-repo>/kv-cache-unified:latest
```

### 3. Run on Non-Slurm Clusters (e.g. Kubernetes, Bare-Metal)

On clusters without Slurm (Kubernetes, bare-metal, etc.), you can run `run_node.sh` on each node separately. Both nodes must have:

- The benchmark code (or Docker image) available
- **Shared storage** (NFS, PVC, etc.) so both nodes see the same `shared/` folder
- Network connectivity and InfiniBand between nodes
- Docker with device access (if using the container)

**Role detection:** The script compares `hostname` with `NODE1`. If they match, it runs as **target**; otherwise as **initiator**. Ensure hostnames resolve correctly between nodes.

#### Bare-Metal: run_node.sh directly

`run_node.sh` accepts command-line arguments.

**run_node.sh arguments:**


| Argument               | Required | Default                           | Description                                    |
| ---------------------- | -------- | --------------------------------- | ---------------------------------------------- |
| `--node1`              | yes      | —                                 | Hostname of the **target** node                |
| `--node2`              | yes      | —                                 | Hostname of the **initiator** node             |
| `--kv-cache-test-path` | no       | /workspace/kvcache_transfer_bench | Path to benchmark inside container             |
| `--shared-folder`      | no       | (kv-cache-test-path)              | Base path; appends `/shared/results_<JOB_ID>`  |
| `--backends`           | no       | all                               | Comma-separated: `rixl,mori,mooncake` or `all` |
| `--start-size`         | no       | 4096                              | Minimum transfer size (bytes)                  |
| `--stop-size`          | no       | 1073741824                        | Maximum transfer size (bytes)                  |
| `--ibdevice`           | no       | mlx5_0                            | InfiniBand device                              |
| `--sync-port`          | no       | 9999                              | TCP port for target/initiator sync             |


**Example** (run on each node; `JOB_ID` from env for bare-metal):

```bash
docker run --rm --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --network host --hostname $(hostname) --add-host "$(hostname):$(hostname -I | awk '{print $1}')" \
  --ipc host --group-add video --cap-add SYS_PTRACE --privileged=true \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /sys:/sys \
  -v /path/to/kvcache_transfer_bench:/workspace/kvcache_transfer_bench --shm-size 64G \
  -e JOB_ID \
  <your-docker-image> /workspace/kvcache_transfer_bench/scripts/run_node.sh \
    --node1 node-target.example.com --node2 node-initiator.example.com \
    --kv-cache-test-path /workspace/kvcache_transfer_bench \
    --shared-folder /workspace/kvcache_transfer_bench \
    --backends rixl,mori,mooncake --start-size 4096 --stop-size 1073741824 --ibdevice mlx5_0
```

### 4. Generate KV Cache Estimator Data (Optional)

To add model-specific KV cache sizes to the report, run the estimator with a YAML config. The config defines model path, concurrency, seq-lengths, tp, pp, ep, dp, and dtype.

```bash
python kv_cache_estimator.py --config <config.yaml> [--output-dir <dir>] [--verify-vllm] [--append]
```

**Config fields:**


| Field                              | Description                                  |
| ---------------------------------- | -------------------------------------------- |
| `model.name`                       | Model path (HuggingFace or local)            |
| `model.concurrency`                | Space-separated: 1 2 4 8 16 32 64 128        |
| `model.seq-length`                 | Space-separated: 1024 2048 4096 8192         |
| `model.tp`                         | Tensor parallel sizes: 1 8                   |
| `model.pp`, `model.ep`, `model.dp` | Pipeline, expert, data parallel (default: 1) |
| `model.kv_cache_dtype`             | `fp8`, `bfloat16`, or `auto`                 |


**Example config** (`llama_70b_config.yaml`):

```yaml
model:
  name: "/shared_inference/models/amd/Llama-3.3-70B-Instruct-FP8-KV"
  concurrency: 1
  tp: 8   # Tensor parallel size (number of GPUs per model shard)
  seq-length: 1024
  kv_cache_dtype: "auto"
  pp: 1
  ep: 1
  dp: 1
```

```bash
# Single model (e.g. DeepSeek-R1)
python kv_cache_estimator.py --config deepseek_r1_config.yaml --output-dir kv_cache_results_all

# Custom output directory
python kv_cache_estimator.py --config qwen3_8b_config.yaml --output-dir kv_cache_results_qwen

# Append to existing CSV (for multiple models)
python kv_cache_estimator.py --config qwen3_8b_config.yaml --output-dir kv_cache_results_all --append
```

**After appending models**, copy `kv_cache_estimator.csv` into the benchmark results directory (the same folder as `results_rixl.json`, etc., e.g. `shared/results_<job_id>/`), then regenerate the merged JSON, CSV, and HTML so the report includes every row in the CSV:

```bash
python scripts/merge_results.py --input-dir <results_dir> --kv-cache-estimator-file <results_dir>/kv_cache_estimator.csv
```

## Project Structure

```
kvcache_transfer_bench/
├── README.md
├── Dockerfile                    # Unified image (RIXL + Mori + Mooncake)
├── kv_cache_estimator.py       # KV cache size calculator & vLLM verifier
├── scripts/
│   ├── merge_results.py        # Merge JSON results, generate CSV + HTML report
│   ├── run_node.sh             # Per-node benchmark runner (target/initiator)
│   └── slurm_launcher.slrum    # Slurm batch launcher (use with sbatch)
├── backends/
│   ├── common/                 # Shared utilities (sync, helpers)
│   ├── rixl/                   # RIXL initiator/target benchmarks
│   ├── mori/                   # Mori initiator/target benchmarks
│   └── mooncake/               # Mooncake initiator/target benchmarks
└── shared/                     # Results (shared/results_<JOB_ID>/)
```

