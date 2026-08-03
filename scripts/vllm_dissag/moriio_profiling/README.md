# MoRIIO profiling for vLLM disaggregated runs

This is the operator reference for the current working-tree implementation. The
repository spelling is `scripts/vllm_dissag/` (not `vllm_disagg/`). Submit
`scripts/vllm_dissag/run_xPyD_models.slurm`; it is the normal entrypoint and
invokes `vllm_disagg.sh` inside each container.

## Execution path

`run_xPyD_models.slurm` resolves the model and topology, loads
`connectors/moriio.env`, launches one container per selected node, and records
the distributed and post-processing status. Inside each container,
`vllm_disagg.sh` sources `connectors/moriio.sh`; the connector calls
`moriio_profiling/hooks.sh` to put each prefill or decode server under
rocprofv3.

The launcher bind-mounts the checkout at `/opt/nixl-vllm-cookbook` and
`${LOG_PATH}` at `/run_logs`. Raw shards therefore persist on the host. After
the containers exit, `process_kernels.sh` runs `trace_tools.py` on a compute
node to combine CSV shards and, when enabled, extract request mappings.
TraceLens analysis is separate and optional.

## Build the profiling image

### Targets and source versions

The Dockerfile is
`docker/vllm_disagg_inference.profiling.ubuntu.amd.Dockerfile`. Its last target is
`normal`, so omitting `--target` builds the non-profile target.

- `normal` uses the shared `runtime` stage. The Dockerfile pins the base to
  `rocm/vllm-dev:ci_base-0fcd9b99cc9d63202da4c858d8ebc6582c9e2491`,
  ROCm/MoRI to `v1.2.1`, AITER to `0.1.16.post3` with `flydsl==0.2.2`,
  and Rust to `1.88.0`. Its vLLM ref
  `vllm_2p2d_wide-ep_write_shikpate_test_06_29_customer` and router ref
  `ravgupta/discovery-dp-rank-roundrobin` are branch names, not immutable
  commit pins. The build records their resolved SHAs in `/app/versions.txt`.
- `profiling` starts from the same runtime, replaces MoRI with
  `https://github.com/AakarshAMD/mori-fork.git` at immutable commit
  `ac44f1aaeb1887985dce2b9bdf82f54c19d008f9`, and initializes its
  `spdlog` and `msgpack-c` submodules.
- The profiling target also builds Perfetto at immutable commit
  `c794fceabe584dc9172e5512aaaeecc21019a635`, installs
  `/usr/local/bin/traceconv`, and sets
  `TRACECONV=/usr/local/bin/traceconv`. That commit reports Perfetto
  `v56.1-c794fceab`.
- The normal target neither copies this binary nor sets `TRACECONV`.
- `WITH_NIXL=0` is sufficient and preferred for a MoRIIO-only image.
  `WITH_NIXL=1` is needed only when the same runtime must also support the
  `rixl` connector (NIXL TP or DeepEP wideEP); it adds UCX at `da3fac2a`,
  RIXL at `f33a5599`, rocSHMEM, and DeepEP builds.

### Build from the login node through Slurm

Do not run the build on the login node. Submit it to one compute node, push
there, and use a unique tag so another node cannot reuse an older image under
the same name. This template deliberately invokes Bash because Slurm
`--wrap` otherwise uses `/bin/sh`, which does not support `pipefail`.

Run from the remote repository root:

```bash
cd /path/to/MAD

IMAGE_TAG="<your-vllm-profiling-image>"
export IMAGE_TAG

sbatch --parsable \
  --export=ALL,IMAGE_TAG \
  --job-name=moriio-profile-build \
  --partition="<compute-partition>" \
  --time=03:00:00 \
  --gres=gpu:1 \
  -N 1 -n 1 \
  --output="<shared-output-root>/moriio-build-%j.out" \
  --error="<shared-output-root>/moriio-build-%j.err" \
  --wrap="exec bash -lc 'set -euo pipefail
    cd /path/to/MAD
    DOCKER_BUILDKIT=1 docker build \
      --target profiling \
      --build-arg WITH_NIXL=0 \
      -f docker/vllm_disagg_inference.profiling.ubuntu.amd.Dockerfile \
      -t \"\$IMAGE_TAG\" .
    docker push \"\$IMAGE_TAG\"
    docker image inspect --format \"{{.Id}}\" \"\$IMAGE_TAG\"
    docker run --rm --entrypoint bash \"\$IMAGE_TAG\" -lc \
      \"cat /app/versions.txt; traceconv --version\"
  '"
```

Add the site's current bad-node exclusion to `sbatch` when required. After the
job completes, retain the tag and registry digest and verify the version output
in the compute-node build log.

## What `RUN_PROFILE` does

`RUN_PROFILE=1` is the supported master capture gate. After normal model
validation, connector defaults and legacy axis selection resolve first.
Profiling requires the effective connector to be `moriio`; `RUN_PROFILE=1`
never selects it. With no explicit connector or legacy selector, the connector
defaults to `rixl`, so `run_xPyD_models.slurm` prints
`CONNECTOR=moriio is required when RUN_PROFILE=1`; direct `vllm_disagg.sh`
invocation prints `mori must be on for profiling`. Both write to stderr and
exit `1` before profiling helper checks, sourcing, or container launch. Legacy
`RUN_MORI=1` can still resolve the effective connector to `moriio`; commands
should set `CONNECTOR=moriio` explicitly.

Only in this profile path, `run_xPyD_models.slurm` requires
`moriio_profiling/hooks.sh` and `moriio_profiling/process_kernels.sh`, sources
`hooks.sh`, and later verifies `benchmark_xPyD_profile.sh`. After the containers
exit, it sources `process_kernels.sh` for post-processing. Inside each container,
`vllm_disagg.sh` independently requires and sources `hooks.sh`. The ordinary
`RUN_PROFILE=0` path does not perform these profile-helper checks.

When both gates are enabled, the workflow:

- rejects conflicting legacy `RUN_DEEPEP=1`;
- sets `ROCPROF=1`;
- defaults `ROCPROF_FLAGS` to
  `--kernel-trace --marker-trace`;
- defaults `ROCPROF_DIR_BASE` to `/run_logs`;
- forces `MORI_ROCTX_TRANSFER=1`;
- defaults `MORIIO_REQID_MAP=0`;
- forces `ANALYZE_KERNELS=0`; and
- overrides the selected benchmark file with
  `benchmark_xPyD_profile.sh`.

Each server command is prefixed with:

```text
rocprofv3 --kernel-trace --marker-trace --disable-signal-handlers \
  --output-format pftrace csv json -d <role-dir> \
  -o %hostname%_%pid% --
```

`--disable-signal-handlers` is appended if absent so vLLM owns graceful
shutdown. Leave `RUN_PROFILE` unset or set it to `0` for the ordinary
non-profile path. Any other value is rejected.

`MORIIO_REQID_MAP` is an optional profiling subfeature and defaults to `0`.
Omit it for kernels, markers, and combined traces without request correlation.
Set it to exact `1` only when the request map is needed; this enables the
installed-vLLM patch and post-run extraction of `reqid_map.csv`.

## Models, topology, and router controls

Model names are case-sensitive.

- `DOCKER_IMAGE_NAME` is mandatory and must identify the profiling image.
- `MODEL_NAME` selects the model catalog entry.
- `DeepSeek-V3`, `DeepSeek-V3-5layer`, and `DeepSeek-R1` are wideEP-only and
  are the only model keys accepted for MoRIIO wideEP. Use `WIDE_EP=1`.
- WideEP uses local DP equal to `GPUS_PER_NODE` and global per-role DP of
  `xP * GPUS_PER_NODE` or `yD * GPUS_PER_NODE`. vLLM is launched with
  `-tp 1`; `IO_TP_SIZE` is ignored.
- `Llama-3.1-405B-Instruct-FP8-KV`,
  `amd-Llama-3.3-70B-Instruct-FP8-KV`, `gpt-oss-120b`, `Qwen3-32B`, and
  `Qwen3-30B-A3B` pass the TP model gate with `WIDE_EP=0`. Catalog acceptance
  is not proof of runtime validation; `models.yaml` records a known
  `device_gemm` failure for `Qwen3-30B-A3B` on this stack.
- In ordinary Slurm submissions, MoRIIO TP resolves to tensor-parallel size
  `8`. The connector checks `IO_TP_SIZE`, then `GENERIC_TP_SIZE`, but
  `run_xPyD_models.slurm` forwards neither as a submit-time override.
- `GPUS_PER_NODE` defaults to `8`; keep it equal to the Slurm GPU allocation.
- `xP` and `yD` each default to `1`. Allocate exactly `xP + yD` nodes and
  tasks. The proxy is co-located on prefill rank 0.
- `PROXY_TYPE` defaults to `vllm_router`. Use it for wideEP; the toy proxy
  cannot perform the required wideEP KV notification routing.
- `ROUTER_PORT` defaults to `30000`. `ROUTER_BINARY` can select an alternate
  executable. `MORIIO_TOY_PROXY` can select the toy proxy script only when
  `PROXY_TYPE=moriio_toy`.
- The model probe checks every allocated node before the launcher selects the
  first `xP + yD` nodes. Over-allocation can therefore fail on an unused node
  that lacks the node-local model.

Do not set legacy `RUN_MORI=1` for a dense Llama profile merely to correct
metadata. Outside exact profile selection it means MoRIIO wideEP, which is
unsupported for Llama. Leave `RUN_MORI` and `RUN_DEEPEP` unset or set both to
`0`, and set `CONNECTOR=moriio` explicitly; `RUN_PROFILE=1` never selects it.
The current `perf.csv` backend tag bug is documented below.

### DeepSeek wideEP template

Run from `scripts/vllm_dissag/`. Replace the placeholder with the unique tag
or digest just built.

```bash
IMAGE="<PROFILING_IMAGE_TAG_OR_DIGEST>"

RUN_PROFILE=1 \
CONNECTOR=moriio \
MORIIO_REQID_MAP=1 \
RUN_MORI=0 \
RUN_DEEPEP=0 \
DOCKER_IMAGE_NAME="$IMAGE" \
MODEL_NAME=DeepSeek-V3 \
WIDE_EP=1 \
xP=2 \
yD=2 \
GPUS_PER_NODE=8 \
PROXY_TYPE=vllm_router \
SKIP_WARMUP=1 \
BENCHMARK_SCRIPT=sweep \
BENCHMARK_ITR=1 \
BENCHMARK_NUM_PROMPTS=8 \
BENCHMARK_CON="1" \
BENCHMARK_COMBINATIONS="256/8" \
sbatch --export=ALL --partition="<compute-partition>" --time=02:00:00 --gres=gpu:8 \
  -N 4 -n 4 run_xPyD_models.slurm
```

### Llama TP template

```bash
IMAGE="<PROFILING_IMAGE_TAG_OR_DIGEST>"

RUN_PROFILE=1 \
CONNECTOR=moriio \
MORIIO_REQID_MAP=1 \
RUN_MORI=0 \
RUN_DEEPEP=0 \
DOCKER_IMAGE_NAME="$IMAGE" \
MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV \
WIDE_EP=0 \
xP=1 \
yD=1 \
GPUS_PER_NODE=8 \
PROXY_TYPE=vllm_router \
SKIP_WARMUP=1 \
BENCHMARK_SCRIPT=sweep \
BENCHMARK_ITR=1 \
BENCHMARK_NUM_PROMPTS=8 \
BENCHMARK_CON="1" \
BENCHMARK_COMBINATIONS="256/8" \
sbatch --export=ALL --partition="<compute-partition>" --time=02:00:00 --gres=gpu:8 \
  -N 2 -n 2 run_xPyD_models.slurm
```

Assignments must be in the environment of `sbatch`; keep them immediately
before it and use `--export=ALL`. Choose a site-appropriate compute partition
and add current site exclusions as needed.

## Benchmark contract

With `RUN_PROFILE=0`, `BENCHMARK_SCRIPT=sweep` selects
`benchmark_xPyD.sh`; `long_context` selects
`benchmark_long_context.sh`. With exact `RUN_PROFILE=1`, either valid selector
is subsequently replaced by `benchmark_xPyD_profile.sh`. Use `sweep` to avoid
misleading logs.

The profile benchmark receives these launcher-supported controls:

- `BENCHMARK_ITR=1`
- `BENCHMARK_CON="8 16 32 64 128 256 512"`
- `BENCHMARK_COMBINATIONS="1024/1024 8192/1024 1024/8192"`
- `SKIP_WARMUP=1`
- `BENCHMARK_NUM_PROMPTS`, unset by default

`RUN_PROFILE=1` defaults `SKIP_WARMUP=1`; keeping it skipped reserves the capture
for measured requests and avoids warmup traffic in the trace. Set `SKIP_WARMUP=0` explicitly to include warmup.

The quoted values above are defaults, not recommended profile sizes.
`BENCHMARK_CON` is a space-separated list. `BENCHMARK_COMBINATIONS` is a
space-separated list of `ISL/OSL` pairs. Quote either assignment whenever it
contains spaces. If `BENCHMARK_NUM_PROMPTS` is unset, each point uses
`max(2 * concurrency, 16)` prompts.

When warmup is not skipped, the script itself defaults to concurrency `1`,
`16` prompts, ISL `32`, and OSL `32`. It also has a `STEP_TIMEOUT` default of
`1800` seconds and scales that floor for token counts above 2048. The current
Slurm docker argument list does not forward `WARMUP_CON`, `WARMUP_PROMPTS`,
`WARMUP_ISL`, `WARMUP_OSL`, or `STEP_TIMEOUT` from submission, so ordinary
profile jobs should use `SKIP_WARMUP=1` and should not claim those names as
working `sbatch` overrides.

**For profiling, use exactly one concurrency, one ISL/OSL pair, and
`BENCHMARK_ITR=1`. Submit a separate job for every additional point.**

rocprof wraps the entire server lifetime: model load, JIT, warmup, every
benchmark point, and shutdown all append to the same per-process shards.
There is no per-point capture boundary. More points increase raw output,
three combined outputs, finalization time, merge time, and persistent storage.

The current aggregate parser keys results only by `(ISL, OSL, concurrency)`
and keeps the maximum total-token throughput across repeated iterations.
Iterations with the same key therefore collapse to one aggregate CSV and
`perf.csv` row. The raw capture remains one job-wide trace regardless of the
number of points.

The benchmark shell does not reliably propagate every non-timeout client
failure, and `parse_to_csv.py` can still exit successfully. A zero Slurm exit
or a `SUCCESS` row in `perf.csv` is not benchmark validation.

## Supported controls

### Capture paths, flags, and mapping

These values are carried through `connectors/moriio.env`, so submit-time
exports override the file defaults:

- `MORIIO_REQID_MAP=0`: exact `1` enables patching and extraction.
- `ROCPROF=0`: `RUN_PROFILE=1` sets it to `1`.
- `ROCPROF_FLAGS="--kernel-trace --marker-trace"`: keep both for the supported
  workflow; the hook appends `--disable-signal-handlers`.
- `ROCPROF_DIR_BASE=/run_logs`: keep output under the persistent bind mount.
- `MORI_ROCTX_TRANSFER=0`: `RUN_PROFILE=1` forces it to `1`.

`LOG_PATH` controls the host root mounted at `/run_logs`; set it to a shared
location such as `<shared-output-root>`. Keep `ROCPROF_DIR_BASE` under that
mount or capture can be ephemeral.

`ANALYZE_KERNELS` is host-side post-processing state. Profile mode forces it
to `0`; do not attempt to enable TraceLens during capture.

The mapping patcher supports manual CLI options `--check`, `--revert`, and
`--moriio-dir`; `--moriio-dir` is not a launcher environment variable. The
patch is idempotent, validates all source anchors before writing, and creates
`.orig_moriio_reqid_map` backups inside the installed package in the
ephemeral container.

### Shutdown and readiness

The effective profiled shutdown defaults are:

- `VLLM_PROFILING_SHUTDOWN_TIMEOUT_S=120`, passed to vLLM as
  `--shutdown-timeout`;
- `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=120`;
- `ROCPROF_FINALIZE_TIMEOUT_S=180`; and
- `ROCPROF_KILL_TIMEOUT_S=10`.

The hook sends `TERM` only to the waitable vLLM/rocprofv3 parent, waits for
vLLM to stop EngineCore and workers in order, verifies a kernel CSV, and uses
a recursive `KILL` only after the finalize timeout.

The current Slurm docker argument list does not forward submit-time values
for those four names. Ordinary `sbatch` runs therefore use the defaults above
unless the launcher or connector environment file is changed. Other
forwarded lifecycle controls are `LOG_WAIT_TIMEOUT_SECONDS` (connector
default `4000`), `VLLM_HANDSHAKE_TIMEOUT_MINS` (`30`),
`VLLM_ENGINE_READY_TIMEOUT_S` (`10800`),
`DISTRIBUTED_TIMEOUT_SECONDS` (`7200`), `VLLM_RPC_TIMEOUT` (`300000`), and
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (`3600`).

### Connector and fabric environment

`connectors/moriio.env` is the source of the current ROCm/RDMA platform
values. Every non-comment key is forwarded with submit-time environment
precedence. In particular:

- `PYTORCH_ALLOC_CONF=expandable_segments:False`
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`
- `HSA_ENABLE_IPC_MODE_LEGACY=0`
- `MORI_GPU_ARCHS=gfx942`
- `HSA_NO_SCRATCH_RECLAIM=1`
- `MORI_RDMA_TC=41`, `MORI_RDMA_SL=0`, `MORI_IO_SL=1`
- `MORI_IB_ENABLE_RELAXED_ORDERING=1`, `MORI_IB_GID_INDEX=1`
- `MORI_NUM_QP_PER_PE=8`
- `VLLM_MORIIO_QP_PER_TRANSFER=2`
- `VLLM_MORIIO_NUM_WORKERS=4`
- `HSA_FORCE_FINE_GRAIN_PCIE=1`, `HSA_ENABLE_SDMA=1`

These fabric values were chosen for the current cluster and are not portable
defaults. The launcher also supports submitted overrides for
`MORI_SOCKET_IFNAME`, `MORI_RDMA_DEVICES`, `NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX`, `NCCL_NET_GDR_LEVEL`, `NCCL_CROSS_NIC`,
`NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME`.

Current vLLM warns that `VLLM_MORIIO_QP_PER_TRANSFER` is deprecated and
ignored in favor of `qp_per_transfer` inside
`kv_connector_extra_config`. Do not treat the environment value as evidence
that the requested QP count took effect.

`ROUTER_PORT` is a working submit-time override and defaults to `30000`.
Most other connector ports are fixed at their in-container defaults because
the Slurm wrapper does not forward their environment names:
RPC `13345`, serve `20005`, KV `9711`, discovery/ping `36367`, local ping
`61555`, handshake `8405`, notify `61005`, and barrier `2222`.
`MASTER_PORT` is hard-coded to `39566`. These host-network ports can collide
with another job on the same node.

Generic launcher controls such as model recipe flags, cache locations,
memory utilization, Docker shared memory, and persistent JIT caching still
apply, but they are not profiling-specific. Consult the launcher,
`models.yaml`, and `connectors/moriio.env` rather than treating them as
profile controls.

### Analysis controls

`process_kernels.sh --kernels` supports:

- `LOG_PATH` for job lookup;
- `DOCKER_IMAGE_NAME`, which should always be set explicitly;
- `NIXL_COOKBOOK_PATH` (default `/opt/nixl-vllm-cookbook`);
- `SBATCH_PARTITION`, set to a site-appropriate compute partition such as
  `<analysis-partition>` (code fallbacks are site-specific);
- `SBATCH_GRES` (default `gpu:1`);
- `SBATCH_TIME` (default `02:00:00`); and
- `DRY_RUN=1`.

The analysis library uses `TRACELENS_REPO` (default
`https://github.com/AMD-AGI/TraceLens.git`), `VENV` (default
`moriio_profiling/external_copies/venv`), `TRACECONV`, and `TRIM_PCT`
(default `5`). The convenience `--kernels` Docker command does not pass host
values for those four names. The profiling image supplies `TRACECONV`;
custom values require an image environment or a manual `docker run -e`.

## Lifecycle

1. Resolved `CONNECTOR=moriio` plus `RUN_PROFILE=1` enables capture,
   transfer markers, mapping state, and the profile benchmark on the Slurm side.
2. Connector and model environment is forwarded to each host-network
   container.
3. If mapping is enabled, each container patches its installed vLLM package
   before its first worker launch.
4. Each role starts under rocprofv3. Model initialization, barriers, server
   readiness, router registration, and a bring-up completion request occur
   before the benchmark.
5. `benchmark_xPyD_profile.sh` writes the client log, aggregate CSV, and
   `perf.csv`.
6. The proxy stops, each vLLM parent receives `TERM`, and rocprofv3 flushes
   process shards. The hook requires at least one kernel CSV with a header
   and data row per role/node.
7. Containers are removed after the distributed `srun` returns.
8. Post-processing always attempts the three CSV-based combinations, then
   conditionally extracts request mappings. Profile mode skips TraceLens.
9. Distributed-run or post-processing failures are propagated in profile
   mode, and raw shards are retained.

Benchmark output can stop minutes before trace flush and combination finish.
Wait for terminal Slurm state and stable artifact sizes.

## Artifacts

With `LOG_PATH=<shared-output-root>`, the host job directory is:

```text
<shared-output-root>/<JOB_ID>/
```

With `LOG_PATH=/some/root`, it is `/some/root/<JOB_ID>/`. Container
`/run_logs/<JOB_ID>/` is the bind-mounted view of the same directory. Slurm
stdout and stderr remain beside it as `slurm-<JOB_ID>.out` and
`slurm-<JOB_ID>.err` under the default log root.

A 2P2D job has:

```text
rocprof_prefill_NODE0/
rocprof_prefill_NODE1/
rocprof_decode_NODE2/
rocprof_decode_NODE3/
```

Each role/node directory contains `%hostname%_%pid%` process shards:

- `*_kernel_trace.csv`: per-dispatch GPU kernel records; these are the
  required combine inputs.
- `*_marker_api_trace.csv`: ROCTx records. A file may be absent for a process
  that emitted no markers.
- `*_agent_info.csv`: rocprofiler agent metadata.
- `*_results.json`: native rocprofiler-SDK JSON. It is not a combine input.
- `*_results.pftrace`: native binary Perfetto trace. It is not a combine
  input, but optional TraceLens analysis prefers it.

Job-level logs and results include:

- `prefill_NODE*.log` and `decode_NODE*.log`: server, mapping, and shutdown
  evidence.
- `vllm_router_NODE*.log` or `proxy_NODE*.log`: routing and registration.
- `pd_vllm_bench_NODE*.log`: full per-node launcher output.
- `benchmark_<JOB_ID>_<timestamp>_xP<xP>_yD<yD>_<model>_CONCURRENCY.log` and
  `.csv`: client results and aggregate throughput.
- `perf.csv`: madengine aggregate rows. It is performance metadata, not trace
  validation.
- `reqid_map.csv`: opt-in request correlation output.
- `kernel_analysis_v2/<role_NODE>/`: strict TraceLens and first-party summaries (name is configurable with `KERNEL_ANALYSIS_OUTPUT_NAME`).

Automatic combination writes:

- `combined_all.pftrace`: every discovered prefill and decode process;
- `combined_rank0.pftrace`: exactly the manifest worker whose emitted
  `local_rank` is `0` for each role/node; and
- `combined_prefill.chrome.json`: every discovered prefill process.

All three combined files are Chrome-trace JSON assembled from CSV, including
the files whose names end in `.pftrace`. Do not pass a combined `.pftrace` to
native `traceconv`. The combiner normalizes timestamps independently to each
node's first selected kernel timestamp. Durations and ordering within a node
are useful, but lanes on different nodes do not share a synchronized global
clock. The three combined files can also have different per-node origins.

## Request and transfer correlation

With `MORIIO_REQID_MAP=1`, the patch logs mappings for `write`,
`write_single`, and `read` calls. Automatic extraction currently reads only
`prefill_NODE*.log`, which matches the current MoRIIO WRITE path.

The current CSV field order is:

```text
write_uid,direction,request_id,transfer_id,layer,role,node_rank,pid
```

MoRI WRITE markers look like:

```text
mori.rdma.kv_transfer bytes=<n> wrs=<n> merged=<n> id=<write_uid>
```

**Never join on bare `write_uid`.** UIDs are process-local and routinely
collide across workers. Build marker identity from the role/node directory,
the marker CSV `Process_Id`, the marker direction, and the marker ID. The
required join key is:

```text
(role, node_rank, pid, direction, write_uid)
```

For current WRITE markers, infer direction `write`. Hostname is unnecessary
because `node_rank` identifies the job node and `pid` identifies the process
on that node. A bare UID can appear independently in many workers while the
composite keys remain unique.

In WRITE mode, markers and extracted maps normally occur on active prefill
master workers. Prefill child nodes and decode nodes can have zero markers.
Even some prefill master workers can have zero markers when no request is
routed to them; their marker CSV may be absent. This is not data loss when
the active workers' composite keys match exactly and kernel capture is
complete. A header-only map after successful transfers is not a pass.

Manual extraction from `scripts/vllm_dissag/`:

```bash
python3 moriio_profiling/trace_tools.py extract-reqid \
  /path/to/<JOB_ID>/prefill_NODE*.log \
  -o /path/to/<JOB_ID>/reqid_map.csv
```

Include decode logs manually only when analyzing a READ-mode path.

## Automatic and optional post-processing

Automatic post-processing:

- the `RUN_PROFILE=1` workflow, which forces `ROCPROF=1`, always builds
  the three combined traces;
- extracts `reqid_map.csv` only for `MORIIO_REQID_MAP=1`;
- never needs `traceconv`, native PFTrace, or native JSON; and
- preserves raw output and returns a post-processing failure to the profile
  launcher.

Optional analysis should run only after capture validation:

```bash
SBATCH_PARTITION="<analysis-partition>" \
DOCKER_IMAGE_NAME="<your-vllm-profiling-image>" \
bash moriio_profiling/process_kernels.sh --kernels <JOB_ID>
```

The script accepts a numeric job ID or directory and starts a one-GPU `srun`
when it is not already in an allocation. Choose a site-appropriate compute
partition and do not rely on its hard-coded fallback image.

First use clones TraceLens and creates a Python venv under
`moriio_profiling/external_copies/`, then installs TraceLens editable plus
`pandas`, `plotly`, `matplotlib`, `openpyxl`, `ijson`, `perfetto`, and
`numpy`. The mounted checkout must be writable and network/package access is
needed when dependencies are not cached.

Treat `external_copies/` as third-party/generated operational content and
exclude it from source-only clone transfers. An optional executable at
`external_copies/traceconv_bin/traceconv`, when present, is only the
`trace_tools.py` fallback when `TRACECONV` is unset. It is not copied by the
Dockerfile and is never used by automatic CSV combination. Source-only
transfers may omit it. The profiling image's `/usr/local/bin/traceconv` is
authoritative in the normal analysis container.

Analysis first reconstructs `rank_manifest.json` from the archived worker
rank lines and requires one unambiguous `local_rank=0` PID for every role/node.
It prefers that PID's native PFTrace; captures without PFTrace use TraceLens's
native rocprof JSON path. Raw kernel CSV is never a TraceLens fallback (it is
used only for the separately labeled trimmed summary). Every TraceLens,
conversion, normalization, and bucketing result is checked, every node must
produce the required outputs, and any node failure makes the job fail after
other nodes finish diagnostic collection.

TraceLens is invoked with `--min_event_ns 0`, overriding its 5000 ns default.
First-party categorization puts intra-node and inter-node EpDispatch/EpCombine
payload in `MORI EP`, keeps `EP_Barrier` separate, leaves staging/sync/support
in `Communication`, and applies the corrected fmoe/opus classifications.

Converted Perfetto JSON is streamed through a narrow sanitizer before
TraceLens. It removes only duration-bearing records with `end < begin` or an
unsigned duration above signed int64, recording exact correlation IDs and
counts. The default corruption guard is both at most 100 records and at most
`0.0001` (0.01%) of duration records; explicit overrides are
`INVALID_EVENT_MAX_COUNT` and `INVALID_EVENT_MAX_FRACTION`.

`kernel_analysis_v2/<role_NODE>/` contains TraceLens JSON/reports/CSVs,
`sanitizer_report.json`, selection/window metadata, and first-party normalized,
per-kernel, category, and trimmed summaries. For each exact kernel name with at
least 20 calls, the default `TRIM_PCT=5` drops the slowest `ceil(5%)`, retains
at least one call, then recomputes the denominator and percentages. This is a
heuristic for whole-capture data, not benchmark-window filtering.
Benchmark-window filtering is applied only when boundary and per-node
clock-correlation data are available; historical results may remain
`whole_capture`.

## Full validation criteria

Slurm `COMPLETED` or exit `0:0` is necessary but not sufficient. Validate all
of the following.

1. **Benchmark:** every intended point appears once; each reports the expected
   successful request count, zero failures, no `[STALL]`, and nonzero request,
   output-token, and total-token throughput. The bring-up completion request
   is separate from the benchmark.
2. **Topology:** every expected role/node directory exists. With the current
   launcher, full validation expects `GPUS_PER_NODE` kernel process shards per
   role/node, not merely the one good shard enforced by the hook. Every kernel
   CSV must have a header and data.
3. **Raw finalization:** expected `agent_info`, native PFTrace, and native JSON
   files are nonempty when those output formats are required. Parse large JSON
   from a fresh handle after size and mtime stabilize.
4. **Combined traces:** all three files are nonempty, stable, and fully parse
   as Chrome JSON. Their process-lane and kernel-event counts must agree with
   discovered CSV inputs; zero kernel events is a failure.
5. **Markers:** require nonzero `mori.rdma.kv_transfer` markers on the workers
   that performed transfers. Do not require a marker file from every worker.
6. **Mapping:** when enabled, require the exact eight-column header, populated
   `role/node_rank/pid`, rows beyond the header, unique composite keys, and an
   exact composite-key join to WRITE markers. Mapping request count can include
   the bring-up request, so it need not equal benchmark request count.
7. **Finalization:** every role log must show readiness, orderly parent-driven
   shutdown, profiler finalization inside the timeout, and kernel verification.
   Slurm and post-processing logs must have no propagated error.
8. **Provenance:** record the command, source checkout, image tag and digest,
   node list, model path, and workload independently of `perf.csv`.

## Known warnings, hiccups, and limits

- `correlation_id.cpp:176] empty thread-local correlation id stack` appears at
  error severity for the asynchronous MoRI marker path. It does not by itself
  prove data loss. Treat it as benign only when kernels are complete,
  finalization is clean, and the composite map/marker join is exact.
- The pinned vLLM emits repeated "Unknown vLLM environment variable" warnings.
  Llama also emits the deprecated checkpoint `kv_scale` warning and the
  deprecated/ignored `VLLM_MORIIO_QP_PER_TRANSFER` warning.
- Cold AITER JIT compilation, missing tuned-GEMM fallback messages, FP8
  accuracy cautions, resource-tracker messages, and API-worker
  "port 20005 is used" messages can occur during coordinated shutdown. They
  are not a substitute for validation and should be investigated if
  readiness, serving, or finalization differs.
- `perf.csv` currently derives the backend only from legacy `RUN_MORI` and
  `RUN_DEEPEP`. A correct `CONNECTOR=moriio`, `RUN_PROFILE=1`, `RUN_MORI=0` run is therefore
  mislabeled `vllm_disagg,nixl`. The Docker image, Docker SHA, Git commit, and
  several other provenance fields are blank because the launcher does not
  pass or populate them in the benchmark container. Record provenance
  separately; do not set `RUN_MORI=1` on dense Llama to work around the tag.
- Large multi-point captures can contain stable interior corruption in native
  `*_results.json` even when CSV-derived combined traces parse. Native PFTrace,
  native JSON, and CSV are separate outputs; parse each required format and
  preserve the healthy shards when one format is bad.
- Fixed host-network ports can collide with a lingering or concurrent job,
  especially router discovery port `36367`. Inspect router bind and
  registration logs, coordinate cleanup with the owning user, or exclude the
  node. The registration timeout only warns and then proceeds.
- Broken node libraries such as `libionic.so.1` and AMD-SMI startup failures
  are allocation or node-health problems rather than profiling defects. Use
  the site's current bad-node list rather than a historical exclusion.
- The node-local model probe is allowed to fail before the shared-model probe
  succeeds. Slurm can retain a nonzero `DerivedExitCode` from that first
  `srun` even when the launcher selects `<shared-model-root>` and
  continues. Read the model-selection log, final `ExitCode`, and artifacts;
  do not classify any one field in isolation.
- Shared storage can briefly expose stale metadata. Wait for terminal job
  state and stable size/mtime, close the old reader, and reopen the canonical
  host path. Repeated failure at the same offset after stabilization is strong
  corruption evidence.
- Combined timestamps are node-relative, `combined_rank0` is manifest-selected
  local rank zero, and optional analysis covers one process per role/node. None of these artifacts
  alone is a globally synchronized, all-worker kernel analysis.
