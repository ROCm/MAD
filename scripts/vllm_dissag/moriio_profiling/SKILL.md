---
name: moriio-profiling
description: Builds, runs, validates, and analyzes short vLLM disaggregated MoRIIO profiles. Use for profiling image builds, rocprofv3 capture, request-to-transfer mapping, trace processing, artifact validation, or MoRIIO profiling troubleshooting.
---

# MoRIIO profiling

Read [README.md](README.md) before operating or changing this workflow. It is
the detailed source for pins, defaults, forwarding limits, artifacts, and
warnings. Keep this skill prescriptive and concise.

## Non-negotiable rules

1. Use the exact path `scripts/vllm_dissag/`.
2. Submit `run_xPyD_models.slurm`; it is the normal entrypoint.
3. Build on a compute node through `sbatch`, not on the login node.
4. Build with `--target profiling`. Omitting it builds the final `normal`
   target without the instrumented MoRI or image-built `traceconv`.
5. Use `--build-arg WITH_NIXL=0` for MoRIIO-only work. Use `1` only if the
   image must also run `CONNECTOR=rixl`.
6. Give every build a unique registry tag, push it, and record its digest.
7. Each profile job is one continuous server-lifetime trace. Use one small
   concurrency value, one small ISL/OSL pair, one iteration
   (`BENCHMARK_ITR=1`), and one small request count; keep OSL especially small.
   Submit separate jobs for additional points.
8. Never infer success from Slurm state, `perf.csv`, or command exit alone.

## Build

Follow the compute-node `sbatch --wrap="exec bash -lc ..."` template in the
README. The inner build must include:

```bash
DOCKER_BUILDKIT=1 docker build \
  --target profiling \
  --build-arg WITH_NIXL=0 \
  -f docker/vllm_disagg_inference.profiling.ubuntu.amd.Dockerfile \
  -t "$IMAGE_TAG" .
docker push "$IMAGE_TAG"
```

Verify `/app/versions.txt` and `traceconv --version` from the pushed image.
Do not rely on a reused tag or a node-local image cache.

## Submit a bounded capture

Run from `scripts/vllm_dissag/` and keep assignments in the environment of
`sbatch --export=ALL`. Profiling requires the effective connector to be
`moriio`; `RUN_PROFILE=1` never selects it. Without it,
`run_xPyD_models.slurm` prints
`CONNECTOR=moriio is required when RUN_PROFILE=1`; direct `vllm_disagg.sh`
invocation prints `mori must be on for profiling`. Both write to stderr and
exit `1` before profile-helper checks, sourcing, or container launch. Set
`CONNECTOR=moriio` explicitly:

```bash
RUN_PROFILE=1 \
CONNECTOR=moriio \
MORIIO_REQID_MAP=1 \
RUN_MORI=0 \
RUN_DEEPEP=0 \
DOCKER_IMAGE_NAME="<PROFILING_IMAGE_TAG_OR_DIGEST>" \
MODEL_NAME="<MODEL_NAME>" \
WIDE_EP="<0_OR_1>" \
xP="<PREFILL_NODES>" \
yD="<DECODE_NODES>" \
GPUS_PER_NODE=8 \
PROXY_TYPE=vllm_router \
SKIP_WARMUP=1 \
BENCHMARK_SCRIPT=sweep \
BENCHMARK_ITR=1 \
BENCHMARK_NUM_PROMPTS=1 \
BENCHMARK_CON="1" \
BENCHMARK_COMBINATIONS="256/8" \
sbatch --export=ALL --partition="<compute-partition>" --time=02:00:00 --gres=gpu:8 \
  -N "<P_PLUS_D>" -n "<P_PLUS_D>" run_xPyD_models.slurm
```

`RUN_PROFILE=1` defaults `SKIP_WARMUP=1`; keeping it skipped reserves the capture
for measured requests and avoids warmup traffic in the trace. Set `SKIP_WARMUP=0` explicitly to include warmup.
Choose a site-appropriate compute partition.

For `DeepSeek-V3`, `DeepSeek-V3-5layer`, or `DeepSeek-R1`, set
`WIDE_EP=1`; vLLM uses DP with `-tp 1`. For dense Llama, set `WIDE_EP=0`;
ordinary Slurm submissions use TP size `8` because `IO_TP_SIZE` and
`GENERIC_TP_SIZE` are not forwarded.

Do not set legacy `RUN_MORI=1` on dense Llama to work around the incorrect
`nixl` tag in `perf.csv`.

Request mapping defaults off (`MORIIO_REQID_MAP=0`). Set it to exact `1`
only when request correlation is needed; omit the assignment when uncorrelated
kernels and markers are sufficient.

## Wait through finalization

Benchmark completion is not job completion. Wait for:

- proxy shutdown;
- each vLLM/rocprofv3 parent to finalize;
- kernel CSV verification on every role/node;
- container cleanup;
- `combined_all.pftrace`, `combined_rank0.pftrace`, and
  `combined_prefill.chrome.json`; and
- conditional `reqid_map.csv` extraction.

Profile mode forces `ANALYZE_KERNELS=0`; TraceLens must not run during capture.

## Validate

Require all of the following:

- every intended benchmark point reports the expected successful requests,
  zero failures, no `[STALL]`, and nonzero throughput;
- every expected role/node and worker kernel CSV exists with a header and
  data;
- all required native outputs are nonempty and parse after size/mtime
  stabilizes;
- all three combined files fully parse as Chrome JSON and contain nonzero
  kernel events with the expected process lanes;
- transfer markers exist on active WRITE workers; zero-marker prefill child,
  decode, or unrouted workers are allowed;
- every role log shows readiness and clean bounded finalization; and
- image tag/digest, source, model path, topology, node list, and workload are
  recorded outside `perf.csv`.

The combined files ending in `.pftrace` are Chrome JSON, not native PFTrace.
`combined_rank0` means the manifest worker with exact `local_rank=0` per
role/node. Timestamps are normalized per node, not globally synchronized.

## Correlate mappings correctly

The current map columns are:

```text
write_uid,direction,request_id,transfer_id,layer,role,node_rank,pid
```

Never join on bare `write_uid`; it is process-local. Join mapping rows to
marker rows on:

```text
(role, node_rank, pid, direction, write_uid)
```

Take role/node from the marker shard directory, PID from marker
`Process_Id`, and current WRITE direction as `write`. Hostname is unnecessary
when `node_rank` identifies the job node.

When mapping is enabled, require populated identity columns, unique composite
keys, rows beyond the header, and an exact map/marker join for active workers.

## Run optional analysis

After full capture validation:

```bash
SBATCH_PARTITION="<analysis-partition>" \
DOCKER_IMAGE_NAME="<PROFILING_IMAGE_TAG_OR_DIGEST>" \
bash moriio_profiling/process_kernels.sh --kernels <JOB_ID>
```

Set the image explicitly. The script writes versioned TraceLens and first-party
outputs under `kernel_analysis_v2/<role_NODE>/` by default and creates or reuses
`moriio_profiling/external_copies/tracelens` and `venv`; the checkout must be
writable. Choose a site-appropriate compute partition.

Analysis requires an unambiguous emitted PID/rank manifest and selects exact
`local_rank=0`. PFTrace is preferred; TraceLens's native rocprof JSON path is
used when PFTrace was not emitted. Raw CSV is never a TraceLens fallback.
Malformed converted events are narrowly discarded under the documented count
and fraction guard. TraceLens is invoked with `--min_event_ns 0`, overriding
its 5000 ns default. Intra-node and inter-node EpDispatch/EpCombine payload is
`MORI EP`; `EP_Barrier` is separate, staging/sync/support remains
`Communication`, and corrected fmoe/opus classifications are applied.

For each exact kernel name with at least 20 calls, default `TRIM_PCT=5` drops
the slowest `ceil(5%)`, retains at least one call, then recomputes the
denominator and percentages. This is a heuristic for whole-capture data, not
benchmark-window filtering. Benchmark-window filtering is applied only when
boundary and per-node clock-correlation data are available; historical results
may remain `whole_capture`. All node failures are collected, then propagated
nonzero. Do not analyze `combined_*.pftrace` as native PFTrace.

## Triage known noise

Treat these as conditionally benign only after full validation:

- `empty thread-local correlation id stack`;
- unknown or deprecated vLLM environment warnings;
- AITER JIT and missing tuned-GEMM fallback messages;
- FP8 accuracy cautions and shutdown/resource-tracker noise.

Always investigate:

- missing or corrupt required raw JSON;
- empty kernel CSVs or combined traces;
- unmatched composite mapping keys;
- profiler finalization timeout;
- router registration/bind failures on fixed host-network ports; and
- node-library, AMD-SMI, model-path, or other allocation failures.

## Maintenance guardrails

- Treat current implementation as source of truth, not historical runs.
- Keep request mapping opt-in.
- Keep capture-time `ANALYZE_KERNELS=0`.
- Preserve raw shards on every failure.
- Keep third-party/generated `external_copies/` out of source-only transfers.
- Do not claim submit-time overrides work unless the Slurm docker invocation
  actually forwards them.
- Put detailed rationale in [README.md](README.md), not in this skill.
