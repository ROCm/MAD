# Large EP Microbenchmark (MoRI-EP)

This benchmark measures the expert-parallel (EP) communication used by Mixture-of-Experts models: **dispatch** (sending each token to the GPUs that hold its selected experts) and **combine** (sending the expert outputs back), on 1 to N nodes with AMD Instinct GPUs and RDMA NICs (Slurm).

## Overview

- **Library**: [MoRI-EP](https://github.com/ROCm/mori). [DeepEP](https://github.com/ROCm/DeepEP) (on [rocSHMEM](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem)) is deprecated in this benchmark and off by default; see [DeepEP (deprecated)](#deepep-deprecated).
- **Tests**: dispatch and combine, each with MoRI's normal and low-latency paths. One node runs the intranode tests (xGMI); two or more nodes run the internode tests (RDMA).
- **Environment**: 1, 2 or 4 nodes, one Docker container per node, RDMA NICs (MoRI matches each GPU to its own NIC; see [Cluster-specific settings](#cluster-specific-settings))
- **What it produces**: per-test logs and a `perf.csv` with the best bandwidth (RDMA between nodes, xGMI within one) and latency for each test

Run it before large MoE training or inference jobs, or after a driver, ROCm, MoRI or NIC firmware change, to confirm EP communication still performs as expected.

## Prerequisites

- Slurm cluster with AMD Instinct nodes (the default image targets gfx942)
- RDMA NICs on each node (MoRI supports Mellanox `mlx5`, Broadcom `bnxt_re` and AMD `ionic`)
- Docker on the compute nodes, with RDMA device access
- This checkout on storage every node can see (logs go to `slurm_output/` under it)

## Model cards

`models.json` defines three cards. All run MoRI only and differ in node count; together they form a scaling series (tag `large_ep_scaling`).

| Card                              | Nodes | Tests                         |
| --------------------------------- | ----- | ----------------------------- |
| `pyt_large_ep_bench_1n_mori_only` | 1     | Intranode, normal and low latency |
| `pyt_large_ep_bench_2n_mori_only` | 2     | Internode, normal and low latency |
| `pyt_large_ep_bench_4n_mori_only` | 4     | Internode, normal and low latency |

## Result files

Benchmark outputs are written to `slurm_output/<job_id>/` in this directory (override with `LOG_PATH`). `perf.csv` is also copied to `slurm_output/perf_csv/results_<job_id>.csv` in the submit directory, where madengine collects it.

| Artifact                                       | Description                                         |
| ---------------------------------------------- | --------------------------------------------------- |
| `mori_internode_<rank>.txt`                     | Internode normal path, one file per node            |
| `mori_ll_internode_<rank>.txt`                  | Internode low-latency path (`--kernel-type v1_ll`)  |
| `mori_intranode_results.log`                   | Intranode normal path (1 node)                      |
| `mori_1N_ll_results.log`                       | Intranode low-latency path, FP8 data (1 node)       |
| `ep_bench_results.log`                         | Combined console output of the run                  |
| `perf.csv`                                     | madengine perf rows from `parse_ep_to_csv.py`: bandwidth and latency per test, from MoRI's best round |

Only rank 0 prints MoRI's result tables, so the `_1` (and higher) files contain setup output only.

## Quick Start (madengine)

Run both commands from the **MAD repository root**:

```bash
madengine build --tags pyt_large_ep_bench_2n_mori_only \
                --registry <namespace>/<repo> --build-on-compute \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags pyt_large_ep_bench_2n_mori_only \
                --manifest-file build_manifest.json
```

`build` compiles the image in its own 1-node Slurm job and pushes it. `run` then takes a separate allocation (the partition from `build`; the rest from the `slurm` block in `models.json`: the card's node count, 8 GPUs per node, 4 hours), pulls the image on every node, runs the tests, and collects `perf.csv`.

**Notes**

- Set your cluster's Slurm partition with `--additional-context` on `build`, as above; the model cards do not set one, and madengine otherwise uses `gpu`.

- `--registry` takes a Docker Hub namespace such as `myorg/myrepo`; madengine
  prefixes `docker.io/` and tags the image `<registry>:<card name>`. The
  registry is what carries the image from the build job to the run job.
- Export `MAD_DOCKERHUB_USER` and `MAD_DOCKERHUB_PASSWORD` (a Docker Hub PAT)
  before `build`, or put them in `credential.json`. An ambient `docker login`
  only reaches the build node if `$HOME` is on shared storage.
- `--build-on-compute` keeps the build off the login node. It requires
  `--registry`, and reuses `slurm.time` from the model card as its own limit.
- Pass `--additional-context` to `build`, not `run`. On `build` it is merged into the card's `slurm` block; on `run` it replaces the whole block. The node count survives (it also comes from the card's `distributed.nnodes`), but the partition is lost and `time` falls back to madengine's multi-node default of 12 hours, which partitions with a shorter limit reject, so include both in any `slurm` block passed to `run`.
- If the job stays pending behind madengine's node health check, disable it at build time with `--additional-context '{"slurm": {"partition": "<your-partition>", "enable_node_check": false}}'`.

### Cluster-specific settings

Nothing in the benchmark names a particular cluster's hardware. MoRI matches each
GPU to its own NIC from the PCIe topology (fastest active link, then the GPU's NUMA
node, then fewest PCIe hops), and compiles its EP kernels on first use for the NIC
type it finds under `/sys/class/infiniband`. These settings override that, and are
forwarded into the containers only when set:

| Variable             | Default when unset                          | Effect |
| -------------------- | ------------------------------------------- | ------ |
| `MORI_RDMA_DEVICES`  | every RDMA NIC                              | Comma-separated NICs MoRI may use, or `^`-prefixed NICs to exclude |
| `MORI_DEVICE_NIC`    | detected (`mlx5`, `bnxt` or `ionic`)        | NIC type MoRI's EP kernels are compiled for |
| `GLOO_SOCKET_IFNAME` | the interface carrying the default route    | Interface for `torchrun`'s bootstrap |
| `IBDEVICES`          | rocSHMEM's choice                           | DeepEP only (`ROCSHMEM_USE_IB_HCA`) |

Set them on `build`, which records them in the manifest for `run`, either inline or
from a JSON file of a cluster's settings with `--additional-context-file` (`-f`):

```json
{
  "slurm": {"partition": "<your-partition>"},
  "env_vars": {"MORI_RDMA_DEVICES": "^mlx5_1,mlx5_6"}
}
```

With plain `sbatch`, export them before submitting.

### Reusing an already-pushed image

If an image has already been pushed, point `build` at it. This only regenerates the manifest; it does not rebuild:

```bash
madengine build --tags pyt_large_ep_bench_2n_mori_only \
                --use-image docker.io/<namespace>/<repo>:<tag> \
                --additional-context '{"slurm": {"partition": "<your-partition>"}}'
madengine run   --tags pyt_large_ep_bench_2n_mori_only \
                --manifest-file build_manifest.json
```

The benchmark scripts are bind-mounted from the checkout at runtime rather than baked into the image, so local edits to `run_ep_bench.sh` or `parse_ep_to_csv.py` take effect without a rebuild. Rebuild only when the Dockerfile changes.

## Quick Start (manual)

### 1. Build Docker image

From the repository root:

```bash
docker build --network=host -f docker/large_ep_benchmark.ubuntu.amd.Dockerfile -t ep-benchmarking:latest ./docker
```

The image is built on the SGLang ROCm 10 base `lmsysorg/sglang:v0.5.20-rocm10-mi30x` (override with `--build-arg BASE_DOCKER=...`), with each component pinned by a Dockerfile `ARG` and gated by an `INSTALL_*` flag (any value other than `1` skips that build):

| Component | Pin (`ARG`)                                               | Commit     | Flag (default)         |
| --------- | --------------------------------------------------------- | ---------- | ---------------------- |
| MoRI      | `MORI_REF=v1.2.3.post1`                                   | `67632e80` | `INSTALL_MORI` (`1`)   |
| rocSHMEM  | `ROCSHMEM_COMMIT`, `develop` as of 2026-10-01              | `62324a88` | `INSTALL_DEEPEP` (`0`) |
| DeepEP    | `DEEPEP_COMMIT`, `main` as of 2026-10-01                   | `0f63d3e9` | `INSTALL_DEEPEP` (`0`) |

MoRI is pinned by release tag (any git ref works), and the build log prints the
commit it resolved to. rocSHMEM and DeepEP are pinned by commit because neither
publishes releases.

To push to Docker Hub:

```bash
docker tag ep-benchmarking:latest <your-repo>/large-ep-benchmark:latest
docker push <your-repo>/large-ep-benchmark:latest
```

### 2. Run the benchmark via Slurm

Submit `run_benchmark.sbatch` from this directory. The node count selects the tests: `-N 1` runs the intranode tests, `-N 2` or more runs the internode tests.

```bash
cd scripts/large-ep-benchmark

export DOCKER_IMAGE=<your-repo>/large-ep-benchmark:latest
sbatch -p <partition> -N 2 run_benchmark.sbatch
```

**Environment variables:**

| Variable            | Default                                    | Description                                              |
| ------------------- | ------------------------------------------ | -------------------------------------------------------- |
| `DOCKER_IMAGE`      | `ep-benchmarking:latest`                   | Image to pull and run on every node                      |
| `DOCKER_IMAGE_NAME` | *(unset)*                                  | Takes precedence over `DOCKER_IMAGE`; set by madengine   |
| `LOG_PATH`          | `slurm_output` in this directory           | Log directory; each job writes to `<LOG_PATH>/<job_id>/` |
| `SKIP_DEEPEP`       | `1`                                        | Set to `0` to also run the deprecated DeepEP tests       |

The network settings in [Cluster-specific settings](#cluster-specific-settings) apply here too.

Slurm stdout/stderr go to `slurm-ep-benchmarking-<job_id>.out` and `.err` in the submit directory. The script's built-in time limit is 4 hours; pass `-t` to change it.

## DeepEP (deprecated)

DeepEP is off by default and no model card runs it. To run it, build the image with `--build-arg INSTALL_DEEPEP=1` (rocSHMEM and DeepEP are compiled for **CX7** NICs) and run with `SKIP_DEEPEP=0`. This adds DeepEP's intranode, internode and low-latency tests (`intranode_results.log`, `internode_<rank>.txt`, `low-latency-<rank>.log`). `parse_ep_to_csv.py` only partly understands DeepEP's output, so most DeepEP results appear in the logs but not in `perf.csv`.

## Project Structure

```
large-ep-benchmark/
├── README.md
├── models.json               # madengine model cards (slurm_multi launcher)
├── run_benchmark.sbatch      # Slurm launcher (madengine slurm_multi, or sbatch)
├── run_ep_bench.sh           # Per-node benchmark runner (inside the container)
├── parse_ep_to_csv.py        # Convert benchmark logs -> madengine perf.csv
└── utils/                    # Socket barrier helpers
```
