# Multinode inference: how a run is put together

This page maps the code a multinode run goes through, from the card to the job's verdict, and
how the image it runs is built. Each step names the file that implements it and links to the
page that describes it in detail. Read [multinode-overview.md](multinode-overview.md) first for
the concepts (prefill and decode, launchers, connectors).

## Contents

- [One run, in order](#one-run-in-order)
- [The layers](#the-layers)
- [What differs per launcher](#what-differs-per-launcher)
- [How an image is built](#how-an-image-is-built)
- [Offline checks per layer](#offline-checks-per-layer)

## One run, in order

```
card (models.json) + environment
  |  madengine (slurm_multi wrapper), plain sbatch / salloc, a job array, or a Kubernetes pod
  v
launcher  scripts/<launcher>/run_*.slurm
  |-- configuration layers      common/config/layers.sh     record what was submitted
  |-- topology spec             common/topology/apply.sh    MAD_TOPOLOGY -> xP / yD / ...
  |-- site facts                common/cluster.sh           weights, fabric, ports
  |-- recipe                    models.yaml env:            over site and connector defaults
  |-- node selection, addresses common/launcher/launcher_lib.sh
  |-- config.resolved.json      $LOG_PATH/<job id>/
  v
dispatcher  common/dispatch/dispatch.sh   (MAD_DISPATCH: slurm-srun | spur-array | k8s-indexed)
  |  once per node, as rank 0..n-1
  v
node program  scripts/<launcher>/node.sh  (+ common/node/node_lib.sh)
  |  mounts, JIT cache, this rank's topology (MAD_TOPO_*), docker run
  v
container  -> role script (vllm_disagg.sh | sglang_disagg_mori_io_ep.sh | serve_colocated.sh)
  |  prefill / decode / router / head / worker by rank; rank 0 runs the benchmark -> perf.csv
  v
verdict  launcher_verdict -> cluster_job_verdict (cluster.sh), on rank 0
```

## The layers

| Step | What happens | Implemented in | Details |
|---|---|---|---|
| Card and environment | The card names the launcher, node count, image (Dockerfile and pin set) and `env_vars`. madengine generates a wrapper that exports them and runs the launcher with `bash`; under plain `sbatch` you export them yourself. | `scripts/<launcher>/models.json` | [multinode-overview.md](multinode-overview.md#how-madengine-runs-a-launcher-slurm_multi), [configuration.md](configuration.md#1-the-model-card) |
| Launcher | One batch script per workload. It finds its own directory (also when `sbatch` runs a spooled copy), validates the model and topology, and drives the steps below. | `scripts/vllm_dissag/run_xPyD_models.slurm`, `scripts/sglang_disagg/run_xPyD_models.slurm`, `scripts/vllm_multinode/run_multinode.slurm` | [vllm-disagg.md](vllm-disagg.md#architecture-at-a-glance), [sglang-disagg.md](sglang-disagg.md#how-a-run-works) |
| Configuration layers | `config_begin` records every name set at submit time before anything else runs; each later layer is labelled as it is applied. Order, weakest first: site (`cluster.sh`) < connector `.env` < recipe (`models.yaml` `env:`) < topology spec < submit time. madengine says where each submitted value came from in `MAD_ENV_SOURCES`. | `scripts/common/config/layers.sh` | [configuration.md](configuration.md#precedence) |
| Topology spec | With `MAD_TOPOLOGY` naming a JSON spec, the launcher's own settings (`xP`, `yD`, `WIDE_EP`, `EP_TP_SIZE`, `TP_SIZE`, `PP_SIZE`, `ROUTER_NODE`, ...) are set from it before the defaults; a setting also given directly must agree. A `router` role (`ROUTER_NODE=1`) puts the router and the benchmark client on a node of their own after decode. | `scripts/common/topology/apply.sh`, `scripts/common/topology/topology.py`, examples in `scripts/common/topology/specs/` | [configuration.md](configuration.md#topology-spec-files) |
| Site facts | Weight roots and the weights probe, fabric detection, ports, timeouts, the GPU architecture check, the verdict helper. Every value is `${VAR:-default}`. | `scripts/common/cluster.sh` | [configuration.md](configuration.md#5-site-facts-scriptscommonclustersh) |
| Recipe | The model's `env:` block is applied on the batch node over site and connector defaults, under what was submitted or set by the spec. Serve flags are read in the container. | `config_apply_recipe` in `scripts/common/config/layers.sh`; `models.yaml` | [configuration.md](configuration.md#2-the-recipe-modelsyaml) |
| Node selection and addresses | The first N nodes of the allocation, in the scheduler's order (N from the topology; a shorter allocation is refused), the `SLURM_*` view narrowed to them, each node's address (optionally the one on `FABRIC_SUBNET_PREFIX`), `MASTER_ADDR`, and `ROUTER_ADDR` with `ROUTER_NODE=1`. | `launcher_select_nodes`, `launcher_node_ips` in `scripts/common/launcher/launcher_lib.sh` | [multinode-running.md](multinode-running.md#with-sbatch-directly) |
| Resolved configuration | Rank 0 writes every name in the launcher's environment, with its value and the layer that set it. | `config_write` in `layers.sh`, called by `launcher_run_nodes` | [configuration.md](configuration.md#the-resolved-environment-per-job) |
| Dispatcher | How the launcher reaches its nodes: their addresses (`dispatch_node_addresses`), running a command once per node as ranks 0..n-1 (`dispatch_run`), and where the job-level steps run (`dispatch_is_leader`). `slurm-srun` (default): one allocation, one `srun` per call. `spur-array`: one single-node task per node of a job array, meeting in a rendezvous directory on shared storage (`MAD_RENDEZVOUS_DIR`). `k8s-indexed`: one pod per node of an Indexed Job, meeting on the shared results volume. | `scripts/common/dispatch/dispatch.sh` (sourced by `cluster.sh`) | [multinode-running.md](multinode-running.md#as-a-job-array-one-task-per-node), [multinode-kubernetes.md](multinode-kubernetes.md#how-a-card-runs-on-kubernetes) |
| Node program | Runs on each node with `SLURM_PROCID` as its rank: clears this launcher's leftover containers, builds the mounts (site storage; on the two vLLM launchers also the host RDMA libraries and the persistent JIT cache, after pulling the image), resolves its part of the topology as `MAD_TOPO_*`, and runs the container; a cancelled job stops and removes it. Exits with the container's status. | `scripts/<launcher>/node.sh`; shared parts in `scripts/common/node/node_lib.sh` | [vllm-disagg.md](vllm-disagg.md#the-container) |
| Container and role script | The image's entry is the launcher directory's role script, which takes its role from `MAD_TOPO_*` (else from `NODE_RANK`), waits at a cross-node barrier, starts its server, and on rank 0 the router and the benchmark, which writes `perf.csv` under `/run_logs/<job id>/`. | `vllm_disagg.sh` (+ `connectors/`, `parallelism.sh`), `sglang_disagg_mori_io_ep.sh`, `serve_colocated.sh` | [multinode-overview.md](multinode-overview.md#architecture-diagrams-vllm-disaggregated) |
| Verdict | After every node is done and the containers are removed, the leader publishes `perf.csv` as `perf_<MODEL_NAME>.csv` (what the card's `multiple_results` names), then fails the job when a node's container exited non-zero, when there is no `perf.csv`, or when a row is `FAILURE`. Under a per-node dispatch the other nodes exit with their own status. | `launcher_verdict` in `launcher_lib.sh`; `cluster_job_verdict` in `cluster.sh` | [benchmarks-and-results.md](benchmarks-and-results.md#what-makes-a-run-pass) |

On Kubernetes the pod entry point sets up what a batch script sees (`SLURM_*`,
`MAD_DISPATCH=k8s-indexed`) and puts stand-ins for `docker` and `scontrol` first on `PATH`, so
`docker run` runs the command in the pod itself: `scripts/common/k8s/mad_k8s_node.sh`,
`scripts/common/k8s/mad_k8s.py`. `scripts/common/k8s/submit.py` submits a card without
madengine. See [multinode-kubernetes.md](multinode-kubernetes.md).

## What differs per launcher

The two disaggregated launchers use every step above. The colocated launcher
(`run_multinode.slurm`) runs on the whole allocation, so it does not narrow it with
`launcher_select_nodes`; it takes the addresses with `launcher_node_ips`, runs `node.sh` with
`dispatch_run` itself (with its own teardown), and ends with `launcher_verdict`. The SGLang
launcher applies no recipe `env:` block (SGLang recipes have none); its container defaults from
`connectors/mori.env` are recorded as `container-default`.

## How an image is built

Each engine has one Dockerfile, and a **pin set** says what it builds:

```
card: "dockerfile": "../../docker/di_vllm", "dockercontext": ".", "docker_build_arg": {"PINS": "<set>"}
  |  madengine adds the set's base image for the build's GPU architecture
  |  (by hand: python3 docker/pins/pins.py build-args <set> --arch <gfx>)
  v
docker/di_vllm.ubuntu.amd.Dockerfile | docker/di_sglang.ubuntu.amd.Dockerfile
  |  pins.py env: the set resolved for this architecture and base; stops on any other
  |  each component checked out at its pinned upstream commit, patches from docker/patches/ applied
  v
image  /app/pins.json, /app/versions.txt (every component's commit), checked against the set
```

| Pin set | Dockerfile | Used by |
|---|---|---|
| `vllm-mori121` | `di_vllm` | The vLLM disaggregated cards except GLM-5.1 and Kimi-K3 |
| `vllm-glm51` | `di_vllm` | The GLM-5.1 cards |
| `vllm-kimi-k3` | `di_vllm` | Every Kimi-K3 vLLM multinode card, disaggregated and colocated |
| `sglang-disagg` | `di_sglang` | The SGLang disaggregated cards |

Build by hand, from the MAD root:

```bash
docker build -f docker/di_vllm.ubuntu.amd.Dockerfile \
  $(python3 docker/pins/pins.py build-args vllm-mori121 --arch gfx942) -t <tag> .
```

The image carries no recipe, tuning or platform environment: those are applied at launch from
`models.yaml` and the connector `.env` files. [`docker/pins/README.md`](../docker/pins/README.md)
is the reference: what each set pins and why, the patches and the upstream changes they wait on,
the rules, and the per-model Dockerfiles the sets replace.

## Offline checks per layer

None of these need GPUs; run them from the MAD root. The full list is in
[multinode-running.md](multinode-running.md#checks-that-need-no-gpus).

| Layer | Check |
|---|---|
| Whole run, per card | `python3 scripts/common/tests/snapshot/snapshot.py`: each node's `docker run` and engine command against the goldens. `--spooled` runs each launcher from a spool copy as `sbatch` does; `--equivalence spur-array` and `--equivalence k8s-indexed` require the same per node under those dispatchers; `--spec-file` runs every card from a topology spec; `--update` rewrites the goldens. |
| Configuration layers | `bash scripts/common/tests/config_layers_assert.sh`; `python3 -m unittest discover -s scripts/common/config/tests` |
| Topology | `python3 -m unittest discover -s scripts/common/topology/tests` |
| Node selection, verdict | `bash scripts/common/tests/launcher_lib_assert.sh`, `bash scripts/common/tests/launcher_verdict_assert.sh` |
| Dispatcher | `bash scripts/common/tests/dispatch_assert.sh` |
| Node program | `bash scripts/common/tests/node_lib_assert.sh` |
| Site fabric | `bash scripts/common/tests/cluster_fabric_assert.sh` |
| Kubernetes | `python3 -m unittest discover -s scripts/common/k8s/tests` |
| Pin sets | `python3 docker/pins/pins.py validate`; `python3 -m unittest discover -s docker/pins/tests` |
