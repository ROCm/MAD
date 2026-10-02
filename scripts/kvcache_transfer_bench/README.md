# KV Cache Transfer Performance Benchmark

This benchmark measures KV cache transfer throughput across multiple backends (NIXL, Mori, Mooncake) on a two-node cluster with InfiniBand (Slurm, Kubernetes, or bare-metal).

## Overview

- **Backends**: NIXL, Mori, Mooncake (ROCm KV transfer engines). NIXL replaces the deprecated ROCm/RIXL (same Python API); results from runs before the switch are labelled `rixl`.
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
| `results_nixl.json`, `results_mori.json`, `results_mooncake.json` | Raw throughput JSON from each backend                                            |
| `results_merged.json`                                             | Single merged file combining all backends (normalized)                           |
| `results_merged.csv`                                              | Pivot table: transfer size vs throughput per backend                             |
| `report.html`                                                     | Interactive HTML report (tables + Plotly charts)                                 |
| `kv_cache_estimator.csv`                                          | Optional; from the KV cache estimator, used to overlay model sizes on the report |


The Slurm flow runs `merge_results.py` on the initiator so `results_merged.json`, `results_merged.csv`, and `report.html` are produced automatically. To regenerate manually, run `python scripts/merge_results.py --input-dir <results_dir>`; for KV-cache overlays, use the example in section 4 (after appending models).

## Quick Start (madengine)

Run both commands from **this directory** (`scripts/kvcache_transfer_bench`):

```bash
madengine build --tags pyt_kvcache_transfer_bench_all \
                --registry <namespace>/<repo> --build-on-compute \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags pyt_kvcache_transfer_bench_all \
                --manifest-file build_manifest.json
```

`build` compiles the image in its own 1-node Slurm job and pushes it. `run`
then takes a separate 2-node allocation (nodes, GPUs and time limit come from
the `slurm` block in `models.json`, the partition from `build`), pulls the
image on both nodes, runs the sweep, and writes
`perf.csv`, one row per backend and transfer size, plus a `FAILURE` row for
any backend that produced no results.

**Notes**

- Set your cluster's Slurm partition with `--additional-context` on `build`, as
  above; the model card does not set one, and madengine otherwise uses `gpu`.
  `build` merges it with the model card's `slurm` block, whereas a `slurm`
  block passed to `run` replaces that block entirely.
- `--registry` takes a Docker Hub namespace such as `myorg/myrepo`; madengine
  prefixes `docker.io/` and tags the image
  `<registry>:pyt_kvcache_transfer_bench_all`. The registry is what carries the
  image from the build job to the run job.
- Export `MAD_DOCKERHUB_USER` and `MAD_DOCKERHUB_PASSWORD` (a Docker Hub PAT)
  before `build`, or put them in `credential.json`. An ambient `docker login`
  only reaches the build node if `$HOME` is on shared storage.
- `--build-on-compute` keeps the build off the login node. It requires
  `--registry`, and reuses `slurm.time` from the model card as its own limit,
  override with `--additional-context '{"slurm":{"time":"06:00:00"}}'` if the
  from-source build needs longer.
- The sweep doubles from `START_SIZE` (4KB) to `STOP_SIZE` (1GB), both in bytes,
  set in the `env_vars` of `models.json`. Override them for one run with e.g.
  `madengine run ... --additional-context '{"env_vars": {"STOP_SIZE": "4294967296"}}'`.
  MoRI caps its sweep at 2GB (it rejects single transfers above `UINT32_MAX`),
  so sizes above that come from NIXL and Mooncake only.

### Reusing an already-pushed image

The image takes ~20 minutes to build from source. If one has already been
pushed, skip `--build-on-compute` and point `build` at it — this only
regenerates the manifest, it does not rebuild:

```bash
madengine build --tags pyt_kvcache_transfer_bench_all \
                --use-image docker.io/<namespace>/<repo>:pyt_kvcache_transfer_bench_all \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags pyt_kvcache_transfer_bench_all \
                --manifest-file build_manifest.json
```

The benchmark scripts are bind-mounted from the checkout at runtime rather than
baked into the image, so local edits to `run_node.sh` / `backends/` take effect
without a rebuild. Rebuild only when the Dockerfile changes.

### Interpreting the results

A healthy run has each backend contributing 19 rows (4KB → 1GB). A backend that
produces no results is recorded as an explicit `FAILURE` row instead. For
Mooncake this usually means it was built with `-DUSE_HIP=ON` but without
`-DENABLE_MULTI_PROTOCOL=ON`, so cross-node writes fail with
`hipIpcOpenMemHandle failed`.

Throughput is measured in a single pass per size, so expect run-to-run noise at
the large end — individual sizes above ~16MB have been seen to swing by 50% or
more between otherwise identical runs. Compare curve shapes, not single points.

## Quick Start (manual)

### 1. Build Docker Image

From the repository root:

```bash
docker build --network=host -f docker/kvcache_transfer_bench.ubuntu.amd.Dockerfile -t kv-cache-unified:latest ./docker
```

The image is built on the SGLang ROCm 10 base `lmsysorg/sglang:v0.5.20-rocm10-mi30x` (`lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` also works, via `--build-arg BASE_DOCKER=...`), with each engine pinned by a Dockerfile `ARG` and gated by `INSTALL_MORI` / `INSTALL_MOONCAKE` / `INSTALL_NIXL` (default `1`; any other value skips that build and keeps whatever the base image ships):

| Component | Pin                                                                |
| --------- | ------------------------------------------------------------------ |
| MoRI      | `v1.2.3.post1` (`67632e80`)                                        |
| Mooncake  | `v0.3.13.post1` (`71973589`), built with `-DENABLE_MULTI_PROTOCOL=ON` |
| NIXL      | `v1.4.1` (`778edd1d`) + ROCm fix `fe3244c8` (ai-dynamo/nixl#2191), over ROCm/ucx `da3fac2a` |

To push to Docker Hub:

```bash
docker tag kv-cache-unified:latest <your-repo>/kv-cache-transfer-bench:latest
docker push <your-repo>/kv-cache-transfer-bench:latest
```

### 2. Run the Benchmark via Slurm

Submit the launcher from the `kvcache_transfer_bench` directory (so `SLURM_SUBMIT_DIR` points at this repo). Pass [Slurm options](https://slurm.schedmd.com/sbatch.html) to `sbatch` first, then the script path, then benchmark options:

```bash
cd kvcache_transfer_bench
sbatch [sbatch options] scripts/run.slurm [benchmark options]
```

**Benchmark options** (passed after the script name; forwarded by the launcher to `run_node.sh`):


| Option           | Default          | Description                                                                                    |
| ---------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| `--docker-image` | *(required)*     | Image to `docker pull` and run on both nodes (e.g. `kv-cache-unified:latest` or registry path) |
| `--start-size`   | 4096             | Minimum transfer size (bytes)                                                                  |
| `--stop-size`    | 1073741824 (1GB) | Maximum transfer size (bytes)                                                                  |
| `--backends`     | all              | Comma-separated: `nixl,mori,mooncake` or `all`                                                 |
| `--ibdevice`     | mlx5_0           | InfiniBand device                                                                              |
| `--sync-port`    | 9999             | TCP port for target/initiator sync                                                             |


**Slurm options** (passed to `sbatch` before the script):


| Option       | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `--nodelist` | Comma-separated node names (e.g. `node-hostname-1,node-hostname-2`) |
| `-t`         | Time limit (e.g. `-t 08:00:00`); required on some clusters          |
| `-N 2`       | Number of nodes; required |


**Example:**

```bash
# Run on specific nodes (pass -t if your cluster requires a runtime limit)
sbatch -t 08:00:00 --nodelist=node-hostname-1,node-hostname-2 \
  scripts/run.slurm --docker-image <your-repo>/kv-cache-unified:latest

# Or let Slurm pick any 2 nodes
sbatch -t 08:00:00 -N 2 scripts/run.slurm --docker-image <your-repo>/kv-cache-unified:latest
```

**Run on already allocated nodes:** If you have an interactive allocation (e.g. via `salloc`), run the launcher directly. It will use `SLURM_NODELIST` from your current session:

```bash
# 1. Allocate 2 nodes
salloc -N 2 -n 2 --ntasks-per-node=1 --time=08:00:00

# 2. From the allocation shell, run the benchmark
cd kvcache_transfer_bench
bash scripts/run.slurm --docker-image <your-repo>/kv-cache-unified:latest
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
| `--backends`           | no       | all                               | Comma-separated: `nixl,mori,mooncake` or `all` |
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
    --backends nixl,mori,mooncake --start-size 4096 --stop-size 1073741824 --ibdevice mlx5_0
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

**After appending models**, copy `kv_cache_estimator.csv` into the benchmark results directory (the same folder as `results_nixl.json`, etc., e.g. `shared/results_<job_id>/`), then regenerate the merged JSON, CSV, and HTML so the report includes every row in the CSV:

```bash
python scripts/merge_results.py --input-dir <results_dir> --kv-cache-estimator-file <results_dir>/kv_cache_estimator.csv
```

## Project Structure

```
kvcache_transfer_bench/
├── README.md
├── models.json                 # madengine model card (slurm_multi launcher)
├── kv_cache_estimator.py       # KV cache size calculator & vLLM verifier
├── scripts/
│   ├── merge_results.py        # Merge JSON results, generate CSV + HTML report
│   ├── to_perf_csv.py          # Convert merged results -> madengine perf.csv
│   ├── run_node.sh             # Per-node benchmark runner (target/initiator)
│   └── run.slurm               # Slurm launcher (madengine slurm_multi, or sbatch)
├── backends/
│   ├── common/                 # Shared utilities (sync, helpers)
│   ├── nixl/                   # NIXL initiator/target benchmarks
│   ├── mori/                   # Mori initiator/target benchmarks
│   └── mooncake/               # Mooncake initiator/target benchmarks
└── shared/                     # Results (shared/results_<JOB_ID>/)
```

