# Integrated SGLang MoRI I/O profiling

This directory documents the supported profiling path for disaggregated SGLang.
`scripts/sglang_disagg/run_xPyD_models.slurm` launches
`sglang_disagg_mori_io_ep.sh` across `xP + yD` nodes. With `RUN_PROFILE=1`,
the prefill and decode server commands are wrapped once by rocprofv3; with
profiling off, their normal launch path is unchanged. `RUN_PROFILE=1` selects
`benchmark_xPyD_profile.sh`; profiling off selects the exact current-`develop`
`benchmark_xPyD.sh`. Profile benchmark requests receive deterministic request
IDs (RIDs), so no second profiling/probe request set is sent. After server finalization,
`moriio_profiling/process_kernels.sh` verifies capture completeness and creates
RID-selected traces, ReqTimeStats joins, optional MoRI maps, and kernel buckets.

> [!WARNING]
> Start with a **small output sequence length (OSL), preferably `8`**, for smoke
> and validation runs. Do not begin with OSL 1024 or another large value under
> full kernel + marker tracing. Trace volume, rocprof capture/finalization time,
> JSON size, NFS pressure, and analysis cost grow with generated tokens, request
> count, concurrency, and the number of sweeps. A practical first run is ISL
> `64` or `128` (or `1024` when that context is required), OSL `8`, concurrency
> `1`, 2-8 prompts, and one or a few sweeps. Increase one dimension at a time
> only after strict validation succeeds. In `BENCHMARK_COMBINATIONS="ISL/OSL"`,
> OSL is the second component. Plan disk and wall time, and monitor `/run_logs`
> plus shared artifact capacity throughout capture and finalization.

## Build and image contract

Build from the repository root on a compute node, not a disk-constrained login
node, and push the resulting image where every allocated node can pull it:

```bash
docker build --build-arg ENABLE_ROCTX=1 -t <profile-image> \
  -f docker/sglang_disagg_inference_profile.ubuntu.amd.Dockerfile .
docker push <profile-image>
```

Pass that exact image with `DOCKER_IMAGE_NAME` at submission. Rebuild when the
Dockerfile, pinned SGLang content, or either SGLang/MoRI instrumentation patch
changes. Shell/Python orchestration-only edits that are supplied by the
host-mounted repository may not require an image rebuild, but verify the mount
path and image compatibility before relying on that. Do not treat a previously
validated tag as universally applicable.

## Copy-paste submissions

Run these from `scripts/sglang_disagg/` with `DOCKER_IMAGE_NAME=<profile-image>`.
For the CX7 cluster path, set `USE_CX7_NICS=1` on appropriate same-rail nodes; `RUN_PROFILE=1` requires nonempty `BENCHMARK_CON`.
Profile submissions default to `SKIP_WARMUP=1`; set `SKIP_WARMUP=0` to run
the legacy warmup. The examples also skip the optional curl smoke test after
proxy readiness.

Minimal single sweep (Llama, MoRI, per-node TP8):

```bash
DOCKER_IMAGE_NAME=<profile-image> \
MODEL_NAME=Llama-3.1-8B-Instruct \
xP=1 yD=1 DP_MODE=0 \
RUN_PROFILE=1 RUN_MORI=1 \
USE_CX7_NICS=1 \
SKIP_CURL_TEST=1 \
BENCHMARK_ITR=1 \
BENCHMARK_COMBINATIONS="64/8" \
BENCHMARK_CON="1" \
BENCHMARK_NUM_PROMPTS=8 \
sbatch -N2 -n2 --gres=gpu:8 --partition=amd-rccl \
  run_xPyD_models.slurm
```

Multi-sweep example (Llama, MoRI, per-node TP8):

```bash
DOCKER_IMAGE_NAME=<profile-image> \
MODEL_NAME=Llama-3.1-8B-Instruct \
xP=1 yD=1 DP_MODE=0 \
RUN_PROFILE=1 RUN_MORI=1 \
USE_CX7_NICS=1 \
SKIP_CURL_TEST=1 \
BENCHMARK_ITR=2 \
BENCHMARK_COMBINATIONS="64/8 128/8" \
BENCHMARK_CON="1 2" \
BENCHMARK_NUM_PROMPTS=2 \
sbatch -N2 -n2 --gres=gpu:8 --partition=amd-rccl \
  run_xPyD_models.slurm
```

`BENCHMARK_NUM_PROMPTS` is the request count **per sweep**. When it is omitted,
profiling retains `benchmark_xPyD_profile.sh`'s derived `p_con` behavior:
`max(2 * concurrency, 16)`.

## Profiling and MoRI behavior

`RUN_PROFILE=1` enables MoRI by default when `RUN_MORI` is unset; set
`RUN_MORI=0` to opt out and use the Mooncake non-MoRI profiling backend. The
launcher applies the following behavior:

- Both `RUN_MORI=0` and `RUN_MORI=1` wrap server workers with one continuous
  rocprofv3 capture, set `SGLANG_ROCTX=1`, `REQ_TIME_STATS=1`,
  `ROCPROF_FLAGS="--kernel-trace --marker-trace"`, and
  `ROCPROF_DIR_BASE=/run_logs`, then run the benchmark and postprocessor.
- `RUN_MORI=1` additionally enables MoRI marker/transfer instrumentation and
  requires complete request-to-MoRI KV mapping.
- `RUN_MORI=0` uses the Mooncake transfer backend, omits the request/MoRI map,
  and emits reqstats with MoRI fields zeroed. SGLang traces and strict client
  RID/ReqTimeStats joins remain required.
- `REQ_TIME_STATS=1` adds `--enable-request-time-stats-logging` to both servers.
  The engine logs contain `ReqTimeStats(...)` records, and rocprof marker CSVs
  contain SGLang request-stage markers used for trace lanes and joins.
- `EAGER=1` appends `--disable-cuda-graph` to both prefill and decode server
  commands; `EAGER=0` preserves supported graphs, though SGLang/model compatibility may still disable prefill graphs.

The launcher derives topology and tensor-parallel settings. `xP + yD` is the
required node count. With `DP_MODE=0`, each node is an independent TP server
(default TP8); with allowed `DP_MODE=1` models, role TP/DP/EP sizes scale from
role node count and GPUs per node. Set `xP`, `yD`, and `DP_MODE`; do not assume an
arbitrary user-supplied TP command bypasses this derivation.

## Multi-sweep contract

Profiling supports multiple `BENCHMARK_ITR` iterations, space-separated
`BENCHMARK_COMBINATIONS`, and space-separated `BENCHMARK_CON` values. Every
iteration x shape x concurrency point receives this deterministic key:

```text
i<iteration>_isl<ISL>_osl<OSL>_c<concurrency>
```

For more than one sweep:

- Each sweep has a unique RID prefix: RIDs are
  `profile-<jobid>-<sweep_key>-NNN` (`NNN` starts at `000`).
- Each sweep has its own raw client CSV and manifest under `/run_logs/<JOBID>/`
  (host default: `/shared_inference/<user>/model_blog_logs/<JOBID>/`):
  `rocprof_probe_client_<sweep_key>.csv` and
  `rocprof_probe_manifest_<sweep_key>.json`.
- Strict per-sweep outputs are isolated under
  `artifacts/pull_<JOBID>/sweeps/<sweep_key>/`.
- The server workers remain inside one continuous rocprof capture. RID filters
  isolate trace/request analysis per sweep; kernel bucket analysis covers the
  capture by pooling all verified local worker CSVs per node before grouping and trimming; durations/counts are summed, not averaged.
- Duplicate keys, mixed fixed/keyed artifacts, orphan or empty CSV/manifest
  pairs, and duplicate prefixes among discovered pairs are rejected; a wholly absent pair is not discoverable automatically, so users/orchestration must verify the expected `iterations x shapes x concurrencies` count.

For exactly one sweep, backward-compatible names and layout remain:
`profile-<jobid>-NNN`, `rocprof_probe_client.csv`,
`rocprof_probe_manifest.json`, and outputs directly under
`artifacts/pull_<JOBID>/`.

The client CSV records per-RID send/first-token/completion timing. The manifest
records the expected RID set and benchmark window. They are not required to
produce raw server rocprof files, but the integrated postprocessor
requires both so it can enforce CSV/manifest RID equality, ReqTimeStats
completeness, and (when enabled) exact MoRI mapping.

### Correlation and trace invariants

- Filtered MoRI correlation is sweep-scoped. When `rid_rooms` is active, skip
  workers with no target RID bounds, reject mappings outside the selected rooms,
  and emit only selected rooms. Keep PID/file-scoped UID resolution and
  unfiltered behavior unchanged. This prevents cross-sweep contamination.
- In DP-attention, empty MoRI lanes can be expected: one DP owner handles a
  request and its KV transfer while other EP workers can still execute kernels.
  Track labels such as `TP<n>` identify worker/global ranks; ROCTX markers are
  CPU-process ranges, not proof that every worker owns the request. Validate
  expected versus observed bytes and transfer post/completion pairing before
  diagnosing capture loss.
- Concurrent engine ranges in one PID can limit per-request attribution when an
  engine marker has no transfer UID. Transfer posts/completions and byte
  accounting remain the authoritative correlation checks.
- Fully parse and validate every clean trace JSON. If it is malformed,
  truncated, or NUL-tailed, rebuild it from source captures; file existence
  alone is not successful validation.

## Capture, finalization, and post-processing

`hooks.sh` finalizes scheduler workers before the launch-server parent, verifies
per-node output counts, and uses a node completion barrier. Hook-level defaults are `ROCPROF_FINALIZE_TIMEOUT=1800`, `ROCPROF_STALL_LIMIT=45`, and
`ROCPROF_NODE_BARRIER_TIMEOUT=2100` seconds. If overriding them, ensure the
variables are actually propagated into the container; the normal submission
wrapper does not forward these three variables explicitly.

`roctx_finalize_workers` may report that it could not find the process-scoped
rocprof directory and use its job/topology/mtime fallback. Treat that as
informational only when all expected worker files and strict post-processing
validate. NFS reads may transiently expose incomplete-looking files; retry the
read, validate JSON/checksums, and use copy-to-temp plus atomic rename when
regenerating artifacts. Never normalize a missing file, invalid JSON,
checksum change, incomplete worker set, or failed strict join as benign.

The integrated launcher runs:

```bash
moriio_profiling/process_kernels.sh <JOBID>
```

Manual forms are:

```bash
moriio_profiling/process_kernels.sh <JOBID>              # alias for run
moriio_profiling/process_kernels.sh run <JOBID>
moriio_profiling/process_kernels.sh verify <JOBID>
moriio_profiling/process_kernels.sh trace <JOBID>
moriio_profiling/process_kernels.sh analyze <JOBID>
```

`verify` requires the same expected worker PID set across each node's
`*_kernel_trace.csv`, `*_marker_api_trace.csv`, and `*_results.json` files.
`trace` builds each RID-selected clean trace, reqstats, client copy, and optional
MoRI map in staging before finalizing required outputs. `analyze` builds
best-effort kernel buckets for each raw capture directory. `run` performs strict
`trace` followed by best-effort `analyze`; a best-effort kernel-analysis warning
does not weaken strict capture/request validation.

The default integrated artifact root is exactly:

```text
scripts/sglang_disagg/moriio_profiling/artifacts/pull_<JOBID>/
```

Single-sweep outputs live at that root; multi-sweep request outputs repeat under
`sweeps/<sweep_key>/`:

```text
roctx_mori_clean_prefill_decode_<JOBID>.json
roctx_mori_clean_probe_only_<JOBID>.json
reqstats_per_request_<JOBID>.csv
reqstats_per_request_<JOBID>_prefill.csv
reqstats_per_request_<JOBID>_decode.csv
rocprof_probe_client[_<sweep_key>].csv
rocprof_probe_manifest[_<sweep_key>].json
request_mori_map_<JOBID>.csv      # RUN_MORI=1 only
request_mori_map_<JOBID>.md       # RUN_MORI=1 only
```

The two compatibility clean traces are copied from the same RID-selected trace
and are byte-identical by design; `probe_only` does not represent
separate traffic.

Per-node continuous-capture analysis is under:

```text
analyze_phase/rocprof_<role>_NODE<n>/buckets/
```

Expected bucket files include `kernel_summary_normalized.csv`,
`kernel_summary_trimmed.csv`, `perkernel_buckets.csv`, and
`bycat_buckets.csv`. The clean traces are Perfetto/Chrome trace-event JSON, not
pftrace binaries.

## Offline CLI

`trace_tools.py` exposes:

```bash
python3 moriio_profiling/trace_tools.py build-trace --help
python3 moriio_profiling/trace_tools.py correlate --help
python3 moriio_profiling/trace_tools.py reqstats --help
python3 moriio_profiling/trace_tools.py buckets --help
python3 moriio_profiling/trace_tools.py trimmed-summary --help
python3 moriio_profiling/trace_tools.py analyze --help
python3 moriio_profiling/trace_tools.py self-test-categories
```

The integrated path uses `--rid-prefix`, worker-count validation,
`--require-data`, `--require-client`, and, for MoRI, `--require-complete`.
`--probe-only` and its time-gap behavior exist only for legacy artifacts; new
runs select normal benchmark requests by RID. The CLI has no llmscope runtime
dependency, and raw CSV analysis does not require TraceLens.

## Troubleshooting checklist

1. Confirm `DOCKER_IMAGE_NAME` pulls on every allocated node and contains the required
   instrumentation.
2. Confirm `MODEL_NAME` is allowed and one model path is available on all nodes.
3. Allocate at least `xP + yD` nodes and verify each launcher-selected rendezvous IP is reachable on the intended network/interface and matches the allocation.
4. Check prefill/decode readiness logs, router registration, and readiness
   timeouts before blaming profiling.
5. Verify every expected GPU worker produced matching kernel, marker, and
   results files; then check finalization messages and `.profile_done_NODE*`
   barrier completion.
6. For multi-sweep runs, require
   `iterations x shapes x concurrencies` sweep directories, unique RIDs, and
   the expected request count in every CSV/manifest/reqstats join.
7. If capture size or finalization is unhealthy, return to OSL `8`, concurrency
   `1`, 2-8 prompts, and one sweep before increasing one dimension at a time.
