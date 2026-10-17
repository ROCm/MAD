# SGLang disaggregated prefill/decode

This page is the complete reference for [`scripts/sglang_disagg`](../scripts/sglang_disagg). It
launches SGLang with prefill and decode disaggregated (P/D disaggregation) across the nodes of a
SLURM allocation: `xP` nodes run prefill servers, `yD` nodes run decode servers, and a router sends
each request through both. The prefill servers transfer the KV cache to the decode servers over
RDMA.

For the concepts (prefill, decode, KV transfer, router, EP) see
[multinode-overview.md](multinode-overview.md). For madengine, `sbatch` and `salloc` usage in
general see [multinode-running.md](multinode-running.md). For where each setting lives see
[configuration.md](configuration.md).

## Supported models

Dense models:

- Qwen3-32B (<https://huggingface.co/Qwen/Qwen3-32B>)
- meta-llama/Llama-3.1-8B-Instruct (<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>)
- amd/Llama-3.3-70B-Instruct-FP8-KV (<https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV>)
- amd/Llama-3.1-405B-Instruct-FP8-KV (<https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV>)

MoE models:

- DeepSeek-V3 (<https://huggingface.co/deepseek-ai/DeepSeek-V3>)
- DeepSeek-R1 (<https://huggingface.co/deepseek-ai/DeepSeek-R1>)
- Mixtral-8x7B-v0.1 (<https://huggingface.co/mistralai/Mixtral-8x7B-v0.1>)

`MODEL_NAME` must be one of these exact values. The batch script rejects any other name and lists
the valid ones:

| `MODEL_NAME` | Type | `DP_MODE=1` allowed |
|---|---|---|
| `Qwen3-32B` | dense | no |
| `Llama-3.1-8B-Instruct` | dense | no |
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | dense | no |
| `Llama-3.1-405B-Instruct-FP8-KV` | dense | no |
| `Mixtral-8x7B-Instruct-v0.1` | MoE | no |
| `DeepSeek-V3` | MoE | yes |
| `DeepSeek-R1` | MoE | yes |

`MODEL_NAME` is also the directory name the launcher looks for under the weight roots, unless
`MODEL_WEIGHTS_NAME` says otherwise (see [Model weights](#model-weights)).

## Prerequisites

- A SLURM cluster with `xP + yD` nodes. The minimum is 2 (`xP=1`, `yD=1`).
- A Docker image with SGLang, MoRI and the NIC drivers built in. See
  [Building the image](#building-the-image).
- A shared filesystem for log collection. The path is cluster-specific; it defaults to
  `LOG_PATH=/shared_inference/$USER/model_blog_logs` from
  [`cluster.sh`](../scripts/common/cluster.sh).
- Docker on the compute nodes. The batch script runs `docker run` on each node under `srun`.
- The model weights on every allocated node, or on shared storage every node reads.

## Building the image

The Dockerfile is [`docker/sglang_disagg_inference.ubuntu.amd.Dockerfile`](../docker/sglang_disagg_inference.ubuntu.amd.Dockerfile).
Its base image is `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x`.

```bash
cd MAD/docker
docker build -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .
```

### Build arguments

| Arg | Default | Description |
|---|---|---|
| `INSTALL_MORI` | `1` | When `1`, builds and installs MoRI at `MORI_COMMIT`. Any other value (for example `0`) skips MoRI entirely. |
| `MORI_COMMIT` | `158c7e8335a0b19b3f1f422ff134d7869252135e` | MoRI commit to check out and install when `INSTALL_MORI=1`. |
| `BASE_DOCKER` | `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` | Base SGLang ROCm image. |
| `GPU_ARCH` | `gfx942` | Target AMD GPU architecture. |

Examples:

```bash
# Default build: installs MoRI pinned at MORI_COMMIT
docker build -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Skip MoRI (for example when using Mooncake KV transfer only)
docker build --build-arg INSTALL_MORI=0 -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .

# Pin a different MoRI commit
docker build --build-arg MORI_COMMIT=<sha> -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .
```

Through madengine, pass the same arguments as `docker_build_arg` in `--additional-context`.

The launcher expects `py-spy`, `flask` and `pyyaml` in the image. It installs any that is missing,
so an image supplied through `DOCKER_IMAGE_NAME` still works, but a MAD-built image does no work
at start-up.

## Files

| File | Description |
|---|---|
| `run_xPyD_models.slurm` | SLURM batch script. Validates the model, picks nodes, resolves the weights, and launches one container per node with `srun` and `docker run`. |
| `sglang_disagg_mori_io_ep.sh` | Container entrypoint. Starts the prefill and decode servers, the router (proxy), and the benchmark. |
| `models.yaml` | Model-specific SGLang CLI flags for all supported models. |
| `mori_ep_env.sh` | RDMA, NCCL, Gloo and MoRI environment variables. Sourced by the entrypoint. |
| `benchmark_xPyD.sh` | Concurrency sweep using `sglang.bench_serving`. |
| `benchmark_agentic.sh` | Agentic trace replay. A thin shim over `scripts/common/benchmark_agentic.sh --backend sglang` that also writes `perf.csv`. |
| `benchmark_parser.py` | Parses a `*_CONCURRENCY.log` into a table, optionally CSV. |
| `parse_to_csv.py` | Parses a `*_CONCURRENCY.log` into a CSV of the maximum total token throughput per (concurrency, ISL, OSL). Also writes madengine's `perf.csv` (`--perf-csv`, `--model-name`) and converts agentic JSON (`--agentic-json`). See [HowToParseData.md](../scripts/sglang_disagg/HowToParseData.md). |
| `socket_barrier.py`, `socket_wait.py` | Node coordination: a TCP barrier across nodes, and a wait until a remote port closes. |
| `models.json` | The model cards. See [Model cards](#model-cards). |
| `salloc_launch.sh` | Sample commands for running inside an `salloc` allocation. |
| `benchmark_xPyD_GSP.sh` | Runs the SGLang GSM8K benchmark (`benchmark/gsm8k/bench_sglang.py`). |
| `sglang_disagg_server.sh`, `set_env_vars.sh` | An earlier server launcher and its NCCL environment. `run_xPyD_models.slurm` always uses `sglang_disagg_mori_io_ep.sh`. |

## Quick start

To run a card through madengine, or with `sbatch` and a cluster's allocation defaults, see
[scripts/common/README.md](../scripts/common/README.md) and
[multinode-running.md](multinode-running.md).

With `sbatch` directly:

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<DOCKER_IMAGE_NAME>
export xP=1
export yD=1
export MODEL_NAME=Llama-3.1-8B-Instruct
export RUN_MORI=1  # MoRI. Set RUN_MORI=0 for Mooncake (KV_TRANSFER_BACKEND=mooncake)

# num_nodes = xP + yD
sbatch -N 2 -n 2 --nodelist=<node1,node2> run_xPyD_models.slurm
```

Set `RUN_MORI` explicitly. The model cards do. When it is unset, `run_xPyD_models.slurm` treats it
as `0` and uses Mooncake.

Through madengine:

```bash
madengine run --tags pyt_sglang_disagg_mori_io_llama-3.1-8b \
  --additional-context '{"slurm": {"nodes": 2, "time": "06:00:00"}}' --live-output
```

Inside an allocation you already hold (`salloc`), run the batch script with `bash`. From
[`salloc_launch.sh`](../scripts/sglang_disagg/salloc_launch.sh):

```bash
# salloc -N 5 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
export xP=2; export yD=2; export MODEL_NAME=Qwen3-32B
bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
```

`salloc_launch.sh` also lists `Qwen3-30B-A3B` and `Mixtral-8x7B-v0.1`. Neither is in the batch
script's model list, so those two lines are rejected.

## How a run works

1. `run_xPyD_models.slurm` runs on the first node of the allocation. It checks `DOCKER_IMAGE_NAME`
   is set, validates `MODEL_NAME`, and sources `scripts/common/cluster.sh`.
2. It checks same-rail placement when `USE_CX7_NICS=1` (see [NICs](#nics-and-rail-placement)).
3. It checks the GPU architecture (`cluster_require_gpu_arch`) and resolves the weights on every
   node (`cluster_resolve_model_path`).
4. It takes the first `xP + yD` nodes of the allocation, sorted by name. The first `xP` are prefill,
   the rest are decode. Node 0 also runs the router, so no extra node is needed.
5. It resolves each node's IP from `hostname -I`, preferring an address that starts with
   `FABRIC_SUBNET_PREFIX` (nodes are multi-homed, and picking the wrong subnet hangs the socket
   barrier).
6. It stops any running containers on each node and starts one container per node with
   `docker run --network host --ipc host --privileged`, the GPU and InfiniBand devices, a 64 GB
   `/dev/shm`, the logs at `/run_logs`, and this directory at `/opt/mooncake-cookbook`.
7. In each container, `sglang_disagg_mori_io_ep.sh` reads `models.yaml`, sources `mori_ep_env.sh`,
   waits at a container barrier, and starts its role.

| `NODE_RANK` | Role |
|---|---|
| `0` | First prefill server, and the router (`sglang_router.launch_router --pd-disaggregation`, port 2322) |
| `1 .. xP-1` | Remaining prefill servers |
| `xP .. xP+yD-1` | Decode servers |

Each SGLang server listens on port 3000. Node 0 waits until every server log prints
`The server is fired up and ready to roll!` (with `DP_MODE=1`, only the master prefill and master
decode), then starts the router. It then polls a real completion through the router, because
`/v1/models` answers before the workers are registered. Then it runs a curl smoke test (skip with
`SKIP_CURL_TEST=1`) and the benchmark (skip with `SKIP_BENCHMARK=1`). The other nodes wait for the
router port to open, then wait for it to close, then stop their server.

With `DP_MODE=0` the router registers every prefill and decode node. With `DP_MODE=1` only the
master nodes serve HTTP, so the router registers node 0 and node `xP`, and uses
`--load-balance-method follow_bootstrap_room` so each request stays on the DP rank that owns its
bootstrap slot.

**Failure handling.** When a server dies during start-up, the launcher prints its first error
lines, the tail of its log, and which processes hold GPU memory. It writes
`/run_logs/<job>/ABORTED`, so every other node stops at its next barrier instead of waiting for the
time limit. A cancelled job stops its containers.

**The job's verdict.** After every node's container has exited and the containers are removed,
`run_xPyD_models.slurm` fails the job (and so the madengine run) when:

- any node's container exited non-zero, which it does when its server or the benchmark failed; or
- the benchmark ran (`SKIP_BENCHMARK` is not `1`) and wrote no `/run_logs/<job>/perf.csv`.

A passing run exits 0 on every node and leaves a `perf.csv`. Each of its rows then carries its own
`SUCCESS` or `FAILURE` (see [benchmarks-and-results.md](benchmarks-and-results.md#throughput-rows)).

## Current support status

### KV cache transfer backend

The launcher selects the disaggregation KV-cache transfer backend
(`--disaggregation-transfer-backend`) from `RUN_MORI`. The backend is kept out of `models.yaml` so
the model recipes stay backend-agnostic. It applies to all supported models.

| `RUN_MORI` | `KV_TRANSFER_BACKEND` | Backend | Status |
|---|---|---|---|
| `1` | `mori` | MoRI IO | Supported |
| `0` | `mooncake` | Mooncake | Supported |

### Wide EP / large EP parallelism mode (`DP_MODE`)

`PARALLEL_MODE` is derived strictly from `DP_MODE`, which selects the flag set from `models.yaml`.

| `DP_MODE` | `PARALLEL_MODE` | Flags applied | Supported models |
|---|---|---|---|
| `0` (default) | `tp` | `base_flags` + `tp_flags` + `prefill.tp` / `decode.tp` | All models |
| `1` | `dp` | `base_flags` + `dp_flags` (`--moe-a2a-backend mori`, DP attention) + `prefill.dp` / `decode.dp` | DeepSeek-V3, DeepSeek-R1 only |

- `DP_MODE=1` enables MoRI expert parallelism (EP) with DP attention, and requires `RUN_MORI=1`
  (MoRI IO).
- It is restricted by an allowlist, `MORI_DP_MODE1_ALLOWED_MODELS`, enforced in both
  `run_xPyD_models.slurm` and `sglang_disagg_mori_io_ep.sh`. Any other model with `DP_MODE=1` exits
  with an error. Use `DP_MODE=0` (TP) for all non-DeepSeek models.

Parallel sizes the launcher passes (not the recipe):

| | `DP_MODE=0` | `DP_MODE=1` |
|---|---|---|
| `--tp-size` | `GENERIC_TP_SIZE` (default 8) on every server | `xP * GPUS_PER_NODE` for prefill, `yD * GPUS_PER_NODE` for decode |
| `--dp-size`, `--ep-size` | not passed | same as `--tp-size` per role |
| `--dist-init-addr`, `--nnodes`, `--node-rank` | not passed | first prefill (or first decode) IP, port `DIST_INIT_PORT` (5757) |

With `DP_MODE=0` and `xP>1` or `yD>1`, each node runs an independent TP server, and the router
registers all of them.

## Recipes (`models.yaml`)

[`models.yaml`](../scripts/sglang_disagg/models.yaml) holds the SGLang CLI flags per model. It is
loaded when `MODEL_NAME` is set.

| Key | Meaning |
|---|---|
| `base_flags` | Always applied, prefill and decode. |
| `tp_flags` / `dp_flags` | Mode-level flags applied to both prefill and decode. Omit when empty. |
| `prefill.<tp\|dp>`, `decode.<tp\|dp>` | Role and mode specific flags. |
| `experimental_flags` | Optional extra flags appended after the role and mode flags on both roles. For try-outs independent of the mode. Omit or `""` when unused. |

Do not set `--tp-size` or `--disaggregation-transfer-backend` here; the launcher manages them.

The composed strings are:

```
prefill: base_flags + <mode>_flags + prefill.<mode> + experimental_flags
decode:  base_flags + <mode>_flags + decode.<mode>  + experimental_flags
```

A different recipe file can be used with `MODELS_YAML=<path>`.

**Radix prefix cache.** Several `prefill` and `decode` blocks carry `--disable-radix-cache` for the
default xPyD sweeps, which measure clean, cache-free throughput. The agentic replay needs the cache
on, so the launcher strips `--disable-radix-cache` when `BENCHMARK_SCRIPT=agentic` or
`ENABLE_RADIX_CACHE=1`. Only the agentic-tagged cards exercise that. Do not remove the flag from the
recipes, or the sweeps lose their cache-free baseline.

### Every recipe

All models share `base_flags` of the form
`--attention-backend aiter --watchdog-timeout 1000000 --mem-fraction-static <f>`.

| Model | `--mem-fraction-static` | `prefill.tp` | `decode.tp` |
|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 0.73 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |
| `Qwen3-32B` | 0.73 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | 0.65 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |
| `Llama-3.1-405B-Instruct-FP8-KV` | 0.65 | `--disable-cuda-graph --disable-radix-cache` | `--disable-radix-cache --cuda-graph-bs $(seq 1 8)` |
| `Mixtral-8x7B-Instruct-v0.1` | 0.73 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |
| `DeepSeek-V3` | 0.73 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |
| `DeepSeek-R1` | 0.73 | `--disable-cuda-graph` | `--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512` |

DeepSeek-V3 and DeepSeek-R1 also have `dp` recipes:

| Key | Value |
|---|---|
| `dp_flags` | `--moe-a2a-backend mori --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --enable-dp-attention-local-control-broadcast` |
| `prefill.dp` (DeepSeek-V3) | `--max-running-requests 8192 --cuda-graph-bs $(seq 1 3) --mem-fraction-static 0.65` |
| `prefill.dp` (DeepSeek-R1) | `--max-running-requests 8192 --cuda-graph-bs $(seq 1 3)` |
| `decode.dp` | `--max-running-requests 8192 --cuda-graph-bs 1 2 4 8 16 24 32 48 64 96 128` |

`--enable-dp-attention-local-control-broadcast` is set on both roles. It forces all DP ranks to step
together, so idle ranks enter the MoRI EP collective. Because prefill and decode use the same
mode-level flags, they live in `dp_flags`; role-specific batch tuning is in `prefill.dp` and
`decode.dp`.

## RDMA and NCCL environment

[`mori_ep_env.sh`](../scripts/sglang_disagg/mori_ep_env.sh) sets the fabric environment inside the
container. Every value is `${VAR:-default}`.

### NICs and rail placement

| `USE_CX7_NICS` | NICs | Notes |
|---|---|---|
| `1` | The 8 CX7 400G rail NICs: `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9` | Requires same-rail nodes |
| `0` | `mlx5_1`, the 100G management NIC | Cross-rail safe |

`mori_ep_env.sh` defaults `USE_CX7_NICS` to `1`, but the batch script always forwards
`USE_CX7_NICS=${USE_CX7_NICS:-0}` (and `cluster.sh` defaults it to `0`), so a run that does not set
it uses `mlx5_1`. The model cards set it explicitly.

The chosen list becomes `IB_DEVICES` (passed to `--disaggregation-ib-device`), `NCCL_IB_HCA` and
`MORI_RDMA_DEVICES`. The batch script does not forward `IB_DEVICES`, `NCCL_IB_HCA` or
`MORI_RDMA_DEVICES` into the container, so select the NICs with `USE_CX7_NICS`.

**Same-rail check.** With `USE_CX7_NICS=1`, a cross-rail node pair makes NCCL time out inside
`ibv_modify_qp`. Because `base_flags` carry `--watchdog-timeout 1000000`, that timeout is a hang,
not a failure. So before any container starts, the batch script reads SLURM's topology
(`scontrol show topology`) and exits if the nodes span more than one leaf switch. The batch script
header requests `--switches=2`. To avoid the failure: resubmit, set `--switches=1` to ask for one
rail up front, set `USE_CX7_NICS=0`, or set `REQUIRE_SAME_RAIL=0` to bypass the check. The check is
only as good as the cluster's `topology.conf`.

### Variables

| Variable | Default | Meaning |
|---|---|---|
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 with IPv4 |
| `NCCL_CROSS_NIC` | `1` | Allow multiple NICs per GPU |
| `NCCL_NET_GDR_LEVEL` | `3` | GPUDirect RDMA across PCI bridges |
| `NCCL_IB_DISABLE` | `0` | Use IB transport |
| `NCCL_IB_QPS_PER_CONNECTION` | `4` | |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | `1` | |
| `NCCL_BUFFSIZE` | `8388608` (8 MB) | |
| `NCCL_IB_TIMEOUT` | `22` | |
| `NCCL_IB_RETRY_CNT` | `7` | |
| `NCCL_IB_SL` | `0` | |
| `NCCL_IB_TC` | `106` | DSCP 26 shifted left by 2 |
| `NCCL_IB_PCI_RELAXED_ORDERING` | `1` | |
| `NCCL_IB_ADAPTIVE_ROUTING` | `1` | |
| `NCCL_TOPO_DUMP_FILE` | `/tmp/nccl_topo.xml` | |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | `INFO`, `INIT,NET,GRAPH` | |
| `MORI_IB_GID_INDEX` | `3` | |
| `MORI_QPS_PER_CONNECTION` | `4` | |
| `GLOO_SOCKET_IFNAME` | the default-route interface | Control plane |
| `NCCL_SOCKET_IFNAME`, `MORI_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` | |
| `GLOO_TIMEOUT_MS` | `300000` | |
| `TORCH_DIST_INIT_BARRIER_TIMEOUT` | `300` | |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `1200` | |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | `1200` | |
| `SGLANG_USE_AITER` | `1` | |
| `SGLANG_ROUTER_STDOUT_LOGS` | `0` | |

With `DP_MODE=1` it also sets MoRI EP tuning:

| Variable | Default |
|---|---|
| `SGLANG_MORI_FP8_DISP` | `True` (`False` when `MODEL_NAME` contains `mxfp4`) |
| `SGLANG_MORI_FP4_DISP` | `False` |
| `SGLANG_MORI_FP8_COMB` | `False` |
| `MORI_MAX_DISPATCH_TOKENS_DECODE` | `160` |
| `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD` | `2 * MORI_MAX_DISPATCH_TOKENS_DECODE` |
| `MORI_SHMEM_HEAP_SIZE` | `17179869184` (16 GiB; the 4 GiB default is too small for EP 32 and above) |

Only the names the batch script forwards reach the container from your submit environment. The
fabric names it forwards are `USE_CX7_NICS`, `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` and
`NCCL_DEBUG_FILE`.

## Model weights

The batch script resolves the weights with `cluster_resolve_model_path` from
[`cluster.sh`](../scripts/common/cluster.sh). It looks for `<root>/<MODEL_WEIGHTS_NAME>` in each of
`MODEL_DIR_CANDIDATES` (local NVMe first, then shared NFS, then `MODEL_DIR` if set), and takes the
first that is present, non-empty and identical on every node. `MODEL_WEIGHTS_NAME` defaults to
`MODEL_NAME`. `MODEL_PATH` skips the probe. `REQUIRE_LOCAL_WEIGHTS=1` excludes shared storage. See
[configuration.md](configuration.md#5-site-facts-scriptscommonclustersh).

## Environment knobs of the batch script

| Variable | Default | Meaning |
|---|---|---|
| `DOCKER_IMAGE_NAME` | none (required) | Image to run on every node |
| `MODEL_NAME` | none (required) | One of the supported names |
| `xP`, `yD` | `1`, `1` | Prefill and decode node counts |
| `DP_MODE` | `0` | `1` selects the `dp` recipes (DeepSeek only) |
| `RUN_MORI` | `0` when unset | `1` MoRI IO, else Mooncake |
| `USE_CX7_NICS` | `0` | See above |
| `REQUIRE_SAME_RAIL` | `1` | `0` skips the same-rail check |
| `BENCHMARK_SCRIPT` | `sweep` | `sweep` (`benchmark_xPyD.sh`) or `agentic` (`benchmark_agentic.sh`) |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024` | ISL/OSL pairs for the sweep, space-separated |
| `BENCHMARK_ITR` | `1` | Sweep iterations |
| `SKIP_BENCHMARK`, `SKIP_CURL_TEST` | `0` | `1` skips that phase |
| `LOG_PATH` | `/shared_inference/$USER/model_blog_logs` | Host log root |
| `FABRIC_SUBNET_PREFIX` | from `cluster.sh` | Preferred subnet for node IPs |
| `GPU_ARCHS`, `GPU_ARCH_CHECK` | unset, `1` | GPU architecture check |
| `PERF_GPU_ARCH` | detected arch | Label for results |

Inside the container, the entrypoint also reads `GPUS_PER_NODE` (8), `GENERIC_TP_SIZE` (8),
`DIST_INIT_PORT` (5757), `BARRIER_PORT` (4342), `MODELS_YAML`, `ROUTER_READY_TIMEOUT_SECONDS`
(4000), `ROUTER_POLL_SLEEP_SECONDS` (10), `SEARCH_SIGNAL`, `ROUTER_HTTP_BASE` and `CURL_TEST_MODEL`.
These use their defaults unless the image or a forwarded name sets them.

## Model cards

[`models.json`](../scripts/sglang_disagg/models.json) has one card per backend, mode and model.
Every card uses `docker/sglang_disagg_inference`, launcher `run_xPyD_models.slurm`,
`distributed.launcher: slurm_multi` with `nnodes: 2`, `slurm: {nodes: 2, gpus_per_node: 8, time: 24:00:00}`,
`args: "-N 2 -n 2"`, and `env_vars` with `xP=1`, `yD=1`.

| Card | `MODEL_NAME` | `RUN_MORI` | `DP_MODE` | `USE_CX7_NICS` | Benchmark |
|---|---|---|---|---|---|
| `pyt_sglang_disagg_mori_io_llama-3.1-8b` | `Llama-3.1-8B-Instruct` | 1 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_qwen3-32b` | `Qwen3-32B` | 1 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_llama-3.3-70b-fp8` | `amd-Llama-3.3-70B-Instruct-FP8-KV` | 1 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_llama-3.1-405b-fp8` | `Llama-3.1-405B-Instruct-FP8-KV` | 1 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_mixtral-8x7b` | `Mixtral-8x7B-Instruct-v0.1` | 1 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_deepseek-v3` | `DeepSeek-V3` | 1 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_deepseek-r1` | `DeepSeek-R1` | 1 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_dp_deepseek-v3` | `DeepSeek-V3` | 1 | 1 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_dp_deepseek-r1` | `DeepSeek-R1` | 1 | 1 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_llama-3.1-8b` | `Llama-3.1-8B-Instruct` | 0 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_qwen3-32b` | `Qwen3-32B` | 0 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_llama-3.3-70b-fp8` | `amd-Llama-3.3-70B-Instruct-FP8-KV` | 0 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_llama-3.1-405b-fp8` | `Llama-3.1-405B-Instruct-FP8-KV` | 0 | 0 | 1 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_mixtral-8x7b` | `Mixtral-8x7B-Instruct-v0.1` | 0 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_deepseek-v3` | `DeepSeek-V3` | 0 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mooncake_deepseek-r1` | `DeepSeek-R1` | 0 | 0 | 0 | sweep 1024/1024 |
| `pyt_sglang_disagg_mori_io_agentic_qwen3-32b` | `Qwen3-32B` | 1 | 0 | 0 | agentic, `conformance_256k` |
| `pyt_sglang_disagg_mori_dp_agentic_deepseek-v3` | `DeepSeek-V3` | 1 | 1 | 0 | agentic, `conformance_256k` |

## Log files

Logs are written to `${LOG_PATH}/${SLURM_JOB_ID}/` on the host, which is `/run_logs/<job>/` in the
container.

| File | Description |
|---|---|
| `pd_sglang_bench_serving.sh_NODE<N>.log` | Main per-node log |
| `prefill_NODE<N>.log` | Prefill server log |
| `decode_NODE<N>.log` | Decode server log |
| `proxy_NODE0.log` | Router log |
| `curl_smoke_NODE0.log` | Smoke test response |
| `benchmark_*_CONCURRENCY.log` / `.csv` | Benchmark results |
| `perf.csv` | Results in madengine's format |
| `ABORTED` | Written by the node that gave up, with the reason |

## Benchmarking

The default benchmark, `benchmark_xPyD.sh`, runs `sglang.bench_serving` against the router at
`127.0.0.1:2322` with `--dataset-name random --random-range-ratio 1.0 --pd-separated`. It runs one
warm-up at concurrency 512 with 1024 prompts, then for each iteration, each ISL/OSL pair in
`BENCHMARK_COMBINATIONS` and each concurrency in `8 16 32 64 128 256 512`, it sends
`2 * concurrency` prompts (at least 16). It writes
`benchmark_<job>_<timestamp>_xP<n>_yD<n>_<model>_CONCURRENCY.log`, then `parse_to_csv.py` writes
the matching `.csv` and `perf.csv`.

Parse benchmark results:

```bash
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log
```

or:

```bash
python3 parse_to_csv.py <log_path>/benchmark_XXX_CONCURRENCY.log -o results.csv
```

`parse_to_csv.py` skips everything before `RUNNING: the benchserving script for iter: 1` (the
warm-up), reads ISL, OSL and concurrency from each `RUNNING: prompts ... isl X osl Y con Z` line,
and keeps the maximum `Total Token throughput (tok/s)` per configuration across iterations. Without
`-o` it writes `<log_file_name>_results.csv`. Its columns are `Concurrency`, `Input tokens`,
`Output tokens`, `Total Token throughput (tok/s)`.

Smoke test from the proxy node:

```bash
curl -X POST http://127.0.0.1:2322/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Let me tell you a story", "sampling_params": {"temperature": 0.3}}'
```

See [benchmarks-and-results.md](benchmarks-and-results.md) for how results reach madengine.

## Agentic replay benchmark

The agentic replay benchmark replays real Claude Code agentic coding traces, using the
`inferencex-agentx-mvp` scenario from [SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf),
against the SGLang router (`:2322`). The workload is highly prefix-repetitive: each turn re-sends
the accumulated context. So it exercises the radix prefix cache, and reports a measured
`gpu_cache_hit_rate` next to the dataset's theoretical ceiling.

New to AgentX? [scripts/common/agentx/README.md](../scripts/common/agentx/README.md) has the
onboarding path, config schema and copy-paste
[examples](../scripts/common/agentx/README.md#examples).
[HOW_IT_WORKS.md](../scripts/common/agentx/HOW_IT_WORKS.md) explains the replay mechanism.

It is a drop-in alternative to the default `benchmark_xPyD.sh` sweep, selected with
`BENCHMARK_SCRIPT=agentic`. No GitHub Actions or CI is involved; it runs through the same
`run_xPyD_models.slurm` entrypoint as the other benchmarks.

Breaking change: the legacy `AGENTIC` environment shorthand is removed. Use
`BENCHMARK_SCRIPT=agentic`.

The four validated agentic configurations are `agentic`-tagged `models.json` cards. List them with:

```bash
madengine run --tags agentic
```

| Backend | Model | Connector | Card |
|---|---|---|---|
| sglang | Qwen3-32B | mori_io | `pyt_sglang_disagg_mori_io_agentic_qwen3-32b` |
| sglang | DeepSeek-V3 | mori_dp | `pyt_sglang_disagg_mori_dp_agentic_deepseek-v3` |
| vllm | DeepSeek-V3 | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3` |
| vllm | DeepSeek-V3-5layer | mori | `pyt_vllm_disagg_mori_agentic_deepseek-v3-5layer` |

The vLLM side is documented in [vllm-disagg.md](vllm-disagg.md).

### Quick start (canonical DeepSeek-V3 1P/1D)

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic                        # selects benchmark_agentic.sh
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000

# 3 same-subnet nodes with the model on local NVMe
sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

`BENCHMARK_SCRIPT=agentic` selects `benchmark_agentic.sh`. The in-container launcher then turns on
server-side Prometheus metrics (`--enable-metrics --enable-metrics-for-all-schedulers`) and strips
`--disable-radix-cache` (cache on).

The README describes the topology as `xP` prefill + `yD` decode + 1 router node (minimum 3 nodes
for 1P/1D), and its examples request 3 nodes. The batch script uses only the first `xP + yD` nodes
of the allocation and runs the router on node 0, co-located with the first prefill server.

### Environment variables

The full AgentX reference is
[scripts/common/agentx/README.md#environment-variable-reference](../scripts/common/agentx/README.md#environment-variable-reference).
The table lists only the launcher-specific knobs. The shared AgentX variables (`AGENTIC_CONC`,
`AGENTIC_CACHE_WARMUP_DURATION`, `MAX_MODEL_LEN`, `AGENTIC_CONFIG`, `AGENTIC_WORKLOAD`,
`AGENTIC_STRICT_CONTEXT`, `AGENTIC_HF_ISL_TAIL`, and others) are documented there.

| Variable | Default | Description |
|---|---|---|
| `BENCHMARK_SCRIPT` | `sweep` | `agentic` runs the agentic replay (`sweep` runs `benchmark_xPyD.sh`, `agentic` runs `benchmark_agentic.sh`) |
| `DURATION` | `120` | Benchmark duration in seconds. Use 900 or more for a canonical, valid submission |
| `AGENTIC_PORT` | `2322` | The SGLang router port aiperf targets |
| `ENABLE_SERVER_METRICS` | auto | `1` forces server Prometheus metrics on (automatic for agentic) |
| `ENABLE_RADIX_CACHE` | auto | `1` forces the radix cache on, by stripping `--disable-radix-cache` (automatic for agentic) |
| `SUITE_CORPUS_FORCE` | `0` | `1` regenerates a cached corpus after you edit a profile |

`AGENTIC_SERVER_METRICS` is set by the launcher to the space-separated server metrics endpoints
(`<prefill_ip>:3000 <decode_ip>:3000`) and consumed by aiperf's `--server-metrics`. It is not a
boolean. The launcher points aiperf at the SGLang servers' `:3000/metrics` directly, because the
router exposes Prometheus on a separate port and lacks `gpu_cache_hit_rate`.

The batch script forwards these agentic names into the container when set: `AGENTIC_CONC`,
`DURATION`, `AGENTIC_PORT`, `MODEL_PREFIX`, `WEKA_LOADER_OVERRIDE`, `AIPERF_UNSAFE_OVERRIDE`,
`AGENTIC_CACHE_WARMUP_DURATION`, `MAX_MODEL_LEN`, `AGENTIC_CONFIG`, `AGENTIC_WORKLOAD`,
`SUITE_CORPUS_DIR`, `AIPERF_PIN`, `AGENTIC_UTILS_PIN`, `SUITE_CORPUS_FORCE`,
`AGENTIC_STRICT_CONTEXT`, `AGENTIC_HF_ISL_TAIL`, `HF_HOME`, `ENABLE_SERVER_METRICS`,
`ENABLE_RADIX_CACHE`, `DRY_RUN`, `AGENTIC_ALLOW_UV_INSTALL`, `AGENTIC_UV_VERSION`. A `$HOME` in
`AGENTIC_CONFIG` is expanded at submit time.

For the config schema, profile authoring, and Tier 1 and Tier 2 knobs, see:

- [AgentX core README](../scripts/common/agentx/README.md)
- [Profile authoring guide](../scripts/common/agentx/profiles/README.md)
- [Config examples cookbook](../scripts/common/agentx/README.md#examples)

### Running HF (weka) trace datasets on your own model

The agentic suite (`scripts/common/agentx/`, driven by `benchmark_agentic_suite.sh` with
`AGENTIC_CONFIG=agentic.yaml`) can replay the SemiAnalysis HF weka corpora against any served model,
not only the 256k and 1M reference targets.

- **Model-agnostic context gating.** Leave `serving.max_model_len` unset or `0` (or
  `MAX_MODEL_LEN=0`) and the suite resolves the served window itself. On disaggregated serving the
  router may not advertise `max_model_len`, so the window is read from the prefill worker (the
  first `host:port` in `AGENTIC_SERVER_METRICS`, derived by the recipe) through `/v1/models`
  (`data[0].max_model_len`, with a `/get_server_info` fallback). A monolithic front end that does
  advertise it is detected directly. Each HF workload gets a loader-derived ISL tail: a `_256k`
  loader gives 262144, a full corpus a conservative value of about 1M; override with
  `AGENTIC_HF_ISL_TAIL`. If the tail exceeds the served window, the suite warns and caps
  `--max-context-length` at the window; `AGENTIC_STRICT_CONTEXT=1` skips that workload instead. An
  explicit `max_model_len` or `MAX_MODEL_LEN` always wins.
- **Tier 1 replay knobs (per workload).** `num_dataset_entries` (how many trace sessions to pull)
  and `trajectory: {min, max}` (the start-window ratio) are per-workload fields. Unset keeps the
  defaults (`393`, and `0.25` / `0.75`).
- **Tier 2 local trim (fit a smaller model).** Add a `filter: {max_isl, max_turns, sample}` block to
  an HF workload or preset. The corpus is downloaded once and trimmed locally: sessions with any
  turn over `max_isl` are dropped, sessions are truncated to `max_turns`, and `sample` sessions are
  chosen at random with seed 42. The result is replayed through `--input-file`. No server change is
  needed.
- **Reusable presets.** Put a `source: hf` file in `scripts/common/agentx/profiles/` (see
  `profiles/inferencex_256k.yaml`) and reference it with `preset: <name>`. A preset can bundle the
  loader, Tier 1 and Tier 2 knobs, and a concurrency sweep and duration. Entry fields override the
  preset; the environment (`AGENTIC_CONC`, `DURATION`, `MAX_MODEL_LEN`) still wins.

See `scripts/common/agentx/agentic.example.yaml` for a worked config.

### Supported models and modes

Agentic replay works for every model in the recipe allowlist. `DP_MODE=1` (MoRI wide EP) is valid
only for the DeepSeek models; all others run `DP_MODE=0` (TP only).

| Model | `DP_MODE` | Notes |
|---|---|---|
| DeepSeek-V3 | 1 (wide EP) | Canonical agentic target |
| DeepSeek-R1 | 1 (wide EP) | Same path as DeepSeek-V3 |
| Qwen3-32B | 0 (TP) | |
| Mixtral-8x7B-Instruct-v0.1 | 0 (TP) | |
| Llama-3.1-8B-Instruct | 0 (TP) | Good quick smoke test |
| amd-Llama-3.3-70B-Instruct-FP8-KV | 0 (TP) | |
| Llama-3.1-405B-Instruct-FP8-KV | 0 (TP) | Needs a larger topology |

### Invocation examples (agentic suite)

`AGENTIC_CONFIG` paths must be visible inside the container. The `$HOME:$HOME` bind mount means
`$HOME/MAD/...` works if MAD is cloned at `$HOME/MAD`. Otherwise use
`/opt/mooncake-cookbook/scripts/...` (the in-container `MOONCAKE_REPO_DIR` mount).

**1. Multi-workload suite config**

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

**2. Single preset shorthand (no config file)**

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_WORKLOAD=conformance_256k   # config-less shorthand: runs just this preset

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

**3. Longer-context baseline (Case-B 512k window)**

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export AGENTIC_WORKLOAD=conformance_512k
export MAX_MODEL_LEN=524288                # the Case-B ISL tail needs the 512k window

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

**4. Single entry selected from a config (smaller model)**

```bash
cd scripts/sglang_disagg

export DOCKER_IMAGE_NAME=<mori-sglang-image>
export BENCHMARK_SCRIPT=agentic
export RUN_MORI=1 DP_MODE=0 xP=1 yD=1 MODEL_NAME=Llama-3.1-8B-Instruct
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml
export AGENTIC_WORKLOAD=inferencex_small   # preset bundles filter: {max_isl, max_turns, sample}

sbatch -N 3 -n 3 -p amd-rccl --time=90 \
  --nodelist=<router-node>,<prefill-node>,<decode-node> \
  run_xPyD_models.slurm
```

**5. DRY_RUN preview from the login node (no server)**

```bash
cd $HOME/MAD

# Prints the resolved workload plan, per-workload commands and context verdicts; no server needed.
DRY_RUN=1 AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml \
  bash scripts/common/benchmark_agentic_suite.sh
```

`salloc_launch.sh` also has a dense-model quick smoke:

```bash
export BENCHMARK_SCRIPT=agentic RUN_MORI=1 DP_MODE=0 xP=1 yD=1 MODEL_NAME=Llama-3.1-8B-Instruct
export DURATION=120 AGENTIC_CONC=8
sbatch -N 3 -n 3 -p amd-rccl --nodelist=<3-nodes> run_xPyD_models.slurm
```

### Output files

Written to `/run_logs/$SLURM_JOB_ID/` (host `LOG_PATH/<jobid>/`):

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # aggregate metrics (parsed by validate_agentic_result.sh)
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
```

`benchmark_agentic.sh` also converts each aggregate JSON into `perf.csv` with `parse_to_csv.py`,
one set of rows per workload, so madengine collects the run's metrics. It skips this under
`DRY_RUN=1`.

Key fields in the aggregate JSON:

- `request_metrics.cache.theoretical_cache_hit_rate`: the dataset-derived ceiling.
- `server_metrics.present` / `server_metrics.cache.gpu_cache_hit_rate`: the measured runtime GPU
  cache hit rate.
- `request_accounting`: `records_profiled` against `records_error_dropped`. The error rate is
  computed over measured requests, excluding warm-up.

### Validation

`scripts/common/validate_agentic_result.sh` runs at the end of the agentic benchmark and prints, for
example:

```
[validate] requests: 70/72 ok  error_rate=2.8%
[validate] gpu_cache_hit_rate: 32.4%
[validate] throughput_per_gpu: 6247 tok/s
[validate][PASS] error_rate within 10%
```

Thresholds:

| Variable | Default | Effect |
|---|---|---|
| `AGENTIC_MAX_ERROR_RATE` | `0.10` | Gates PASS / WARN |
| `AGENTIC_MIN_CACHE_HIT` | `0.30` | Warn only |

A low cache hit rate with `xP>1` usually means the router does not keep a conversation's turns on
the worker that holds its prefix (session affinity).

## Known issues

For larger models, such as DeepSeek-V3 and Llama-3.1-405B-Instruct-FP8-KV, at high concurrency
(512 and above), errors with this signature appear:

```
<TransferEncodingError: 400, message:
  Not enough data to satisfy transfer length header.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
```

Requests are dropped and throughput is lower. The issue is being discussed on the SGLang forums.

## See also

- [configuration.md](configuration.md): every configuration layer and which wins.
- [multinode-running.md](multinode-running.md): running, logs, failures, offline checks.
- [vllm-disagg.md](vllm-disagg.md): the vLLM disaggregated launcher.
- [benchmarks-and-results.md](benchmarks-and-results.md): benchmarks and `perf.csv`.
