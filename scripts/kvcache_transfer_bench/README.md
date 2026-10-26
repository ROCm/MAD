# KV Cache Transfer Performance Benchmark

This benchmark measures KV cache transfer throughput across multiple backends (NIXL, Mori, Mooncake) on a two-node cluster with InfiniBand (Slurm, Kubernetes, or bare-metal).

## Overview

- **Backends**: NIXL, Mori, Mooncake (ROCm KV transfer engines). NIXL replaces the deprecated ROCm/RIXL (same Python API); results from runs before the switch are labelled `rixl`.
- **MoRI has two backends**: `mori` drives MoRI's Python benchmark (`tests/python/io/benchmark.py`, one `torchrun` per size), and `mori_cpp` drives MoRI-IO's C++ benchmark (`tests/cpp/io/bench_engine`, one process for the whole sweep). Both get the same flags from `backends/common/mori_settings.py`: RDMA write from GPU memory, batch size 1 with `--batch-contiguous`, the session fast path, 1 QP per transfer, 1 worker thread, and 500 timed iterations per size after 50 warmup, with the buffer sized to the sweep maximum. The Python benchmark has no warmup option, so `backends/mori/run_benchmark.py` adds it. The benchmarks still time differently: Python averages per-transfer times (submit, then a blocking `Wait()`), while C++ times the whole loop and spin-polls for completion.
- **Environment**: 2 nodes, Docker containers, RDMA NICs (each node uses the one MoRI-IO would pick for GPU 0 unless told otherwise; see [Cluster-specific settings](#cluster-specific-settings))
- **What it produces**: A sweep from `--start-size` through `--stop-size`, with JSON per backend, a merged CSV, and an interactive HTML report that compares backends across transfer sizes



## Prerequisites

- Two nodes with network connectivity (Slurm, Kubernetes, or bare-metal)
- Docker with InfiniBand device access
- ROCm-capable GPUs (for vLLM/KV cache estimator)



## Result files

Benchmark outputs are usually under `shared/results_<job_id>/` (path may differ if you set paths manually).


| Artifact                                                          | Description                                                                      |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `results_nixl.json`, `results_mori.json`, `results_mori_cpp.json`, `results_mooncake.json` | Raw throughput JSON from each backend                   |
| `results_merged.json`                                             | Single merged file combining all backends (normalized)                           |
| `results_merged.csv`                                              | Pivot table: transfer size vs throughput per backend                             |
| `report.html`                                                     | Interactive HTML report (tables + Plotly charts)                                 |
| `kv_cache_estimator.csv`                                          | Optional; from the KV cache estimator, used to overlay model sizes on the report |


The Slurm flow runs `merge_results.py` on the initiator so `results_merged.json`, `results_merged.csv`, and `report.html` are produced automatically. To regenerate manually, run `python scripts/merge_results.py --input-dir <results_dir>`; for KV-cache overlays, use the example in section 4 (after appending models).

## Quick Start (madengine)

Run both commands from the **MAD repository root**:

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

### Choosing backends

Each model card runs a fixed set of backends in one 2-node job. Select a card by
its tag (or its name) in `--tags`, on both `build` and `run`:

| Tag                | Model card                            | Backends                                |
| ------------------ | ------------------------------------- | --------------------------------------- |
| `kvcache`          | `pyt_kvcache_transfer_bench_all`      | `nixl`, `mori`, `mori_cpp`, `mooncake`  |
| `kvcache_nixl`     | `pyt_kvcache_transfer_bench_nixl`     | `nixl`                                  |
| `kvcache_mooncake` | `pyt_kvcache_transfer_bench_mooncake` | `mooncake`                              |
| `mori`             | `pyt_kvcache_transfer_bench_mori`     | `mori` and `mori_cpp`, on the same node pair |
| `mori_py`          | `pyt_kvcache_transfer_bench_mori_py`  | `mori` (Python benchmark)               |
| `mori_cpp`         | `pyt_kvcache_transfer_bench_mori_cpp` | `mori_cpp` (C++ `bench_engine`)         |

The NIXL and Mooncake tags carry a `kvcache_` prefix because `nixl` and `mooncake`
also tag the vLLM and SGLang disaggregated-inference cards elsewhere in MAD.

For example, the C++ MoRI backend alone:

```bash
madengine build --tags mori_cpp \
                --registry <namespace>/<repo> --build-on-compute \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags mori_cpp --manifest-file build_manifest.json
```

Several tags (e.g. `--tags mori_cpp,mori_py`) select several cards, and each card
runs as its own job, possibly on a different node pair; use `mori` to compare the
two MoRI backends on the same nodes. `kvcache_transfer_bench/all` selects every
card. Each card gets its own image tag, but they share one Dockerfile, so building
several in one `build` reuses the Docker layer cache.

**Notes**

- Set your cluster's Slurm partition with `--additional-context` on `build`, as
  above; the model card does not set one, and madengine otherwise uses `gpu`.
  `build` merges it with the model card's `slurm` block, whereas a `slurm`
  block passed to `run` replaces that block entirely. The node count survives
  (it also comes from the card's `distributed.nnodes`), but `time` falls back
  to madengine's multi-node default of 12 hours, which partitions with a
  shorter limit reject, so include `time` in any `slurm` block passed to `run`.
- `--registry` takes a Docker Hub namespace such as `myorg/myrepo`; madengine
prefixes `docker.io/` and tags the image
`<registry>:kvcache_transfer_bench_pyt_kvcache_transfer_bench_all` (the card's
discovered name, `kvcache_transfer_bench/pyt_kvcache_transfer_bench_all`, with
the `/` replaced). The registry is what carries the image from the build job
to the run job.
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
Both MoRI backends cap their sweep at 2GB (MoRI rejects single transfers above
`UINT32_MAX`), so sizes above that come from NIXL and Mooncake only.

### Cluster-specific settings

Nothing in the benchmark names a particular cluster's hardware. Each node
resolves its network settings when it starts, and logs the result as
`INFO: RDMA device ..., socket interface ...`:

| Variable                     | Default when unset                                                                  | Used by |
| ---------------------------- | ----------------------------------------------------------------------------------- | ------- |
| `IBDEVICE` (or `IBDEVICES`)  | The NIC MoRI-IO would assign to GPU 0: fastest active link, then on GPU 0's NUMA node, then fewest PCIe hops, then name. The NIC that carries the default route and any bond are skipped | All four backends: NIXL through `UCX_NET_DEVICES`, Mooncake directly, and both MoRI backends through `MORI_RDMA_DEVICES` |
| `GLOO_SOCKET_IFNAME`         | The interface carrying the default route                                            | MoRI's bootstrap sockets |

A single device name is used as given. A comma-separated list (for example a
cluster's backend NICs, as `sglang_disagg` takes them) narrows detection to those
devices. Auto-detection picks a NIC that works, but pin `IBDEVICE` for runs whose
numbers you intend to compare across clusters.

Set them for one run with madengine:

```bash
madengine run --tags mori_cpp --manifest-file build_manifest.json \
              --additional-context '{"env_vars": {"IBDEVICE": "rdma0"}}'
```

or keep a cluster's settings in a JSON file and pass it to `build` with
`--additional-context-file` (`-f`), which records them in the manifest for `run`:

```json
{
  "slurm": {"partition": "<your-partition>"},
  "env_vars": {"IBDEVICE": "mlx5_0", "GLOO_SOCKET_IFNAME": "eth0"},
  "docker_build_arg": {"BASE_DOCKER": "lmsysorg/sglang:v0.5.20-rocm10-mi30x"}
}
```

`BASE_DOCKER` selects the GPU family the image is compiled for (`-mi30x` or
`-mi35x`; see [Build Docker Image](#1-build-docker-image)).

Passing the same file to `run` as well replaces the manifest's `slurm` block
(see the note on `time` above). With plain `sbatch`, export the variables before
submitting, or use `--ibdevice`.



### Reusing an already-pushed image

The image takes ~20 minutes to build from source. If one has already been
pushed, skip `--build-on-compute` and point `build` at it. This only regenerates the manifest, it does not rebuild:

```bash
madengine build --tags pyt_kvcache_transfer_bench_all \
                --use-image docker.io/<namespace>/<repo>:kvcache_transfer_bench_pyt_kvcache_transfer_bench_all \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags pyt_kvcache_transfer_bench_all \
                --manifest-file build_manifest.json
```

The benchmark scripts are bind-mounted from the checkout at runtime rather than  
baked into the image, so local edits to `run_node.sh` / `backends/` take effect  
without a rebuild. Rebuild only when the Dockerfile changes.

## Quick Start (manual)



### 1. Build Docker Image

From the repository root:

```bash
docker build --network=host -f docker/kvcache_transfer_bench.ubuntu.amd.Dockerfile -t kv-cache-unified:latest ./docker
```

The image is built on the SGLang ROCm 10 base `lmsysorg/sglang:v0.5.20-rocm10-mi30x` (`lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` also works, via `--build-arg BASE_DOCKER=...`), with each engine pinned by a Dockerfile `ARG` and gated by `INSTALL_MORI` / `INSTALL_MOONCAKE` / `INSTALL_NIXL` (default `1`; any other value skips that build and keeps whatever the base image ships). `INSTALL_MORI_CPP_BENCH` (default `1`) builds MoRI-IO's `bench_engine` for the `mori_cpp` backend from the same MoRI checkout, at `$MORI_IO_BENCH_ENGINE_BIN`:


| Component | Pin (`ARG`)                                                                                 | Commit     |
| --------- | ------------------------------------------------------------------------------------------- | ---------- |
| MoRI      | `MORI_REF=v1.2.3.post1`                                                                     | `67632e80` |
| MoRI-IO `bench_engine` | same MoRI checkout, target-only CMake build in `$MORI_ROOT/build_bench`        |            |
| Mooncake  | `MOONCAKE_REF=v0.3.13.post1`, built with `-DENABLE_MULTI_PROTOCOL=ON`                       | `71973589` |
| NIXL      | `NIXL_REF=v1.4.1` + ROCm fix `NIXL_CHERRY_PICK` (ai-dynamo/nixl#2191), over ROCm/ucx `UCX_COMMIT` | `778edd1d`, `fe3244c8`, `da3fac2a` |

The `*_REF` pins are release tags (any git ref works); UCX and the NIXL fix are
pinned by commit because neither is tagged. The build log prints the commit
each tag resolved to.

MoRI and `bench_engine` are compiled for the base image's `GPU_ARCH_LIST`
(`gfx942` in the `-mi30x` images, `gfx950` in `-mi35x`), so build for MI355X by
switching the base, e.g. `--build-arg BASE_DOCKER=lmsysorg/sglang:v0.5.20-rocm10-mi35x`,
or with madengine `--additional-context '{"docker_build_arg": {"BASE_DOCKER": "..."}}'`.


To push to Docker Hub:

```bash
docker tag kv-cache-unified:latest <your-repo>/kv-cache-transfer-bench:latest
docker push <your-repo>/kv-cache-transfer-bench:latest
```



### 2. Run the Benchmark via Slurm

Submit the launcher from the `scripts/kvcache_transfer_bench` directory (so `SLURM_SUBMIT_DIR` points at this benchmark). Pass [Slurm options](https://slurm.schedmd.com/sbatch.html) to `sbatch` first, then the script path, then benchmark options:

```bash
cd scripts/kvcache_transfer_bench
sbatch [sbatch options] scripts/run.slurm [benchmark options]
```

**Benchmark options** (passed after the script name; forwarded by the launcher to `run_node.sh`):


| Option           | Default          | Description                                                                                    |
| ---------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| `--docker-image` | *(required)*     | Image to `docker pull` and run on both nodes (e.g. `kv-cache-unified:latest` or registry path) |
| `--start-size`   | 4096             | Minimum transfer size (bytes)                                                                  |
| `--stop-size`    | 1073741824 (1GB) | Maximum transfer size (bytes)                                                                  |
| `--backends`     | all              | Comma-separated: `nixl,mori,mori_cpp,mooncake` or `all`                                        |
| `--model-name`   | pyt_kvcache_transfer_bench_all | `model` column of `perf.csv` (madengine sets it from `BENCH_MODEL_NAME`)         |
| `--ibdevice`     | detected         | RDMA device, or a list to detect from (sets `IBDEVICES`; see [Cluster-specific settings](#cluster-specific-settings)) |
| `--sync-port`    | 9999             | TCP port for target/initiator sync                                                             |


**Slurm options** (passed to `sbatch` before the script): the `#SBATCH` header in `run.slurm` sets the defaults below, and any of them can be overridden on the command line. madengine's `slurm_multi` path runs the script with `bash`, so it ignores the header and uses the `slurm` block in `models.json` instead.


| Option            | Default (`run.slurm` header)                        | Description                                                         |
| ----------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `-N`              | 2                                                   | Number of nodes; the benchmark needs exactly 2 (target, initiator)  |
| `-t`              | 04:00:00                                            | Time limit                                                          |
| `-o` / `-e`       | `slurm-kvcache-transfer-bench-<job_id>.out` / `.err` in the submit directory | Log files; a directory given here must exist |
| `-p`              | cluster default                                     | Partition                                                           |
| `--nodelist`      | any                                                 | Comma-separated node names (e.g. `node-hostname-1,node-hostname-2`) |


**Example:**

```bash
# Let Slurm pick 2 nodes, with the header's defaults
sbatch scripts/run.slurm --docker-image <your-repo>/kv-cache-unified:latest

# Run on specific nodes with a longer time limit
sbatch --nodelist=node-hostname-1,node-hostname-2 -t 08:00:00 \
  scripts/run.slurm --docker-image <your-repo>/kv-cache-unified:latest
```

**Run on already allocated nodes:** If you have an interactive allocation (e.g. via `salloc`), run the launcher directly. It will use `SLURM_NODELIST` from your current session:

```bash
# 1. Allocate 2 nodes
salloc -N 2 -n 2 --ntasks-per-node=1 --time=08:00:00

# 2. From the allocation shell, run the benchmark
cd scripts/kvcache_transfer_bench
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
| `--backends`           | no       | all                               | Comma-separated: `nixl,mori,mori_cpp,mooncake` or `all` |
| `--start-size`         | no       | 4096                              | Minimum transfer size (bytes)                  |
| `--stop-size`          | no       | 1073741824                        | Maximum transfer size (bytes)                  |
| `--ibdevice`           | no       | `$IBDEVICE`, else detected        | RDMA device, or a list to detect from          |
| `--sync-port`          | no       | 9999                              | TCP port for target/initiator sync             |
| `--model-name`         | no       | pyt_kvcache_transfer_bench_all    | `model` column of `perf.csv`                   |


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
    --backends nixl,mori,mori_cpp,mooncake --start-size 4096 --stop-size 1073741824
```

Add `--ibdevice <device>` or `-e IBDEVICE -e GLOO_SOCKET_IFNAME` to pin the
network settings instead of detecting them. The Slurm launcher mounts `/sys`
read-write, and retries that mount read-only when the docker daemon requires
it. Detection only reads sysfs, so either mount works.



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
├── models.json                 # madengine model cards (slurm_multi launcher), one per backend set
├── kv_cache_estimator.py       # KV cache size calculator & vLLM verifier
├── scripts/
│   ├── merge_results.py        # Merge JSON results, generate CSV + HTML report
│   ├── to_perf_csv.py          # Convert merged results -> madengine perf.csv
│   ├── run_node.sh             # Per-node benchmark runner (target/initiator)
│   └── run.slurm               # Slurm launcher (madengine slurm_multi, or sbatch)
├── backends/
│   ├── common/                 # Shared utilities (sync, helpers)
│   ├── nixl/                   # NIXL initiator/target benchmarks
│   ├── mori/                   # Mori initiator/target benchmarks (Python benchmark.py)
│   ├── mori_cpp/               # MoRI-IO C++ bench_engine initiator/target wrappers
│   └── mooncake/               # Mooncake initiator/target benchmarks
└── shared/                     # Results (shared/results_<JOB_ID>/)
```

