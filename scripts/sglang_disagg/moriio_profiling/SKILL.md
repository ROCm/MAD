---
name: sglang-moriio-profiling
description: Runs and validates integrated MoRI I/O profiling with rocprofv3/ROCTX for disaggregated SGLang xPyD jobs. Use when building a profiling image, submitting RUN_PROFILE jobs, validating single- or multi-sweep artifacts, or troubleshooting capture and request attribution.
---

# Integrated SGLang MoRI I/O profiling

## When to use

Use this skill for the supported `RUN_PROFILE=1` path through
`scripts/sglang_disagg/run_xPyD_models.slurm`,
`sglang_disagg_mori_io_ep.sh`, `benchmark_xPyD_profile.sh`, and
`moriio_profiling/process_kernels.sh`. Profiling off uses `benchmark_xPyD.sh`,
restored byte-for-byte from current `develop`. Do not invent a separate
submit/probe/capture workflow.

Success means:

- every expected prefill/decode GPU worker has kernel, marker, and results files;
- every expected sweep has unique deterministic RIDs and complete client,
  ReqTimeStats, and optional MoRI joins;
- strict trace processing passes; and
- artifacts are copied with integrity checks without weakening validation.

> [!WARNING]
> Keep OSL small. Start with OSL `8`, concurrency `1`, 2-8 prompts, and one or a
> few sweeps. OSL is the second value in
> `BENCHMARK_COMBINATIONS="ISL/OSL"`. Do not start with OSL 1024 under full
> kernel + marker tracing: generated tokens, requests, concurrency, and sweep
> count multiply trace size, finalization time, JSON/NFS load, and analysis
> cost. Use ISL `64`/`128`, or `1024` only when required, then increase one
> dimension at a time after a clean run. Check `/run_logs` and shared artifact
> capacity before and during the job.

## Non-negotiable constraints

- Preserve normal/develop launch behavior when `RUN_PROFILE=0`.
- Do not restore duplicate profiling probe traffic. Profile the normal
  benchmark requests, which already receive deterministic RIDs.
- Do not silently weaken worker-count, RID, ReqTimeStats, client, JSON, hash, or
  MoRI completeness checks.
- Do not delete client CSV/manifest artifacts unless their strict validation
  role is replaced end-to-end.
- Do not modify unrelated files or discard another user's working-tree changes.
- Do not call a partial capture successful. Best-effort kernel analysis does
  not excuse a failed strict trace/request phase.

## Workflow

### 1. Preflight

1. Work from `scripts/sglang_disagg/` and inspect the current working tree.
2. Confirm `MODEL_NAME` is accepted and its model directory is available on all
   allocated nodes.
3. Set `xP` and `yD`; allocate at least `xP + yD` nodes. The launcher derives
   role topology and TP/DP/EP settings. For `DP_MODE=0`, each role node is an
   independent TP server (default TP8). Use `DP_MODE=1` only for allowlisted
   models.
   For the CX7 cluster path, set `USE_CX7_NICS=1` on appropriate same-rail nodes.
4. Confirm the image can be pulled on every selected node.
5. `RUN_PROFILE=1` enables MoRI by default when `RUN_MORI` is unset. Override it
   explicitly only when needed:
   - `RUN_MORI=1`: MoRI markers plus strict request/KV mapping.
   - `RUN_MORI=0`: Mooncake backend, SGLang traces/reqstats, zeroed MoRI reqstats
     columns, and no request/MoRI map.
6. Profile submissions default to `SKIP_WARMUP=1`. Set `SKIP_WARMUP=0` only
   when the legacy warmup is wanted. Decide separately whether to set
   `SKIP_CURL_TEST=1`.
7. Set nonempty `BENCHMARK_CON` (mandatory under `RUN_PROFILE=1`), then start with OSL `8`, low concurrency and prompts, and minimal sweeps.

### 2. Decide whether to rebuild

Build from the repository root on a compute node and push the image:

```bash
docker build --build-arg ENABLE_ROCTX=1 -t <profile-image> \
  -f docker/sglang_disagg_inference_profile.ubuntu.amd.Dockerfile .
docker push <profile-image>
```

Rebuild after Dockerfile, pinned SGLang, or SGLang/MoRI instrumentation-patch
changes. Host-mounted shell/Python orchestration-only changes may not require a
rebuild; verify the actual mount and image compatibility. Never hard-code a previously validated
image as universally applicable. Submit with `DOCKER_IMAGE_NAME`.

### 3. Submit

Minimal single sweep:

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

Multi-sweep example:

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

`BENCHMARK_NUM_PROMPTS` is per sweep. If omitted, profiling uses
`max(2 * concurrency, 16)` from `benchmark_xPyD_profile.sh`.

`RUN_PROFILE=1` sets SGLang ROCTX, `REQ_TIME_STATS=1`, rocprof kernel + marker
tracing, and `/run_logs` output. It wraps each server once and reuses that
continuous capture across all sweeps. `EAGER=1` is a separate opt-in that adds
`--disable-cuda-graph` to both role commands; `EAGER=0` preserves supported graphs, though SGLang/model compatibility may still disable prefill graphs.

### 4. Monitor

Monitor scheduler state, the Slurm output/error files, and
`/shared_inference/<user>/model_blog_logs/<JOBID>/` (mounted as
`/run_logs/<JOBID>`). Check, in order:

1. model validation and launcher-selected rendezvous IP reachability on the intended network/interface;
2. prefill/decode server readiness and router registration;
3. benchmark sweep progress;
4. rocprof worker finalization and `.profile_done_NODE*` barrier progress; and
5. strict `process_kernels.sh` results.

A job/topology/mtime fallback message from `roctx_finalize_workers` can be
benign only if every expected worker artifact and all later strict checks pass.
Never classify missing files, invalid JSON, changed checksums, or incomplete
joins as benign.

### 5. Validate capture and sweeps

The launcher invokes the postprocessor after profiling. For manual diagnosis:

```bash
moriio_profiling/process_kernels.sh verify <JOBID>
moriio_profiling/process_kernels.sh trace <JOBID>
moriio_profiling/process_kernels.sh analyze <JOBID>
moriio_profiling/process_kernels.sh run <JOBID>
```

`process_kernels.sh <JOBID>` is also `run`.

For every role node, require the same expected PID set across:

```text
*_kernel_trace.csv
*_marker_api_trace.csv
*_results.json
```

For multiple iterations, shapes, or concurrencies, compute the expected count:

```text
BENCHMARK_ITR * number_of_shapes * number_of_concurrencies
```

Each sweep key is `i<iteration>_isl<ISL>_osl<OSL>_c<concurrency>`. Require that
many directories under:

```text
moriio_profiling/artifacts/pull_<JOBID>/sweeps/<sweep_key>/
```

Each sweep must have its own RID prefix, client CSV, and manifest. RIDs are
`profile-<jobid>-<sweep_key>-NNN`; raw client files are
`rocprof_probe_client_<sweep_key>.csv` and
`rocprof_probe_manifest_<sweep_key>.json`. Reject duplicate keys/prefixes,
mixed fixed/keyed files, orphan/empty pairs, duplicate RIDs, or request-count mismatches among discovered pairs; a wholly absent pair is not discoverable, so users/orchestration must enforce the expected count.

Exactly one sweep retains `profile-<jobid>-NNN`, fixed client filenames, and
outputs directly under `artifacts/pull_<JOBID>/`.

Per sweep, require non-empty:

```text
roctx_mori_clean_prefill_decode_<JOBID>.json
roctx_mori_clean_probe_only_<JOBID>.json
reqstats_per_request_<JOBID>.csv
reqstats_per_request_<JOBID>_prefill.csv
reqstats_per_request_<JOBID>_decode.csv
rocprof_probe_client[_<sweep_key>].csv
rocprof_probe_manifest[_<sweep_key>].json
request_mori_map_<JOBID>.csv/.md       # only with RUN_MORI=1
```

The two compatibility JSON traces are byte-identical copies by design;
they do not represent separate traffic. The CSV/manifest are not fundamental
to raw rocprof capture, but are mandatory for the integrated strict client RID
and ReqTimeStats checks. ReqTimeStats markers must remain present.

Continuous-capture kernel outputs use one verified local-rank-0 worker per node/role and are under
`artifacts/pull_<JOBID>/analyze_phase/rocprof_<role>_NODE<n>/buckets/`, including
`kernel_summary_normalized.csv`, `kernel_summary_trimmed.csv`,
`perkernel_buckets.csv`, and `bycat_buckets.csv` when analysis succeeds.

#### Correlation and trace invariants

- With filtered MoRI correlation (`rid_rooms` active), skip workers that have no
  target RID bounds, reject mappings outside the selected rooms, and emit only
  selected rooms. Preserve PID/file-scoped UID resolution and the unfiltered
  path. This is required to prevent cross-sweep contamination.
- Under DP-attention, an empty MoRI lane can be expected: one DP owner handles a
  request and its KV transfer while other EP workers can still execute kernels.
  Track labels such as `TP<n>` are worker/global ranks, while ROCTX markers are
  CPU-process ranges. Check expected/observed bytes and transfer post/completion
  pairing before classifying an empty lane as capture loss.
- Concurrent engine ranges in the same PID can limit per-request attribution
  when an engine marker lacks a transfer UID. Transfer posts/completions and
  byte accounting remain authoritative.
- Fully parse and validate clean trace JSON. If it is malformed, truncated, or
  NUL-tailed, rebuild it from source captures instead of accepting its existence.

### 6. Copy artifacts

Copy only after strict validation. Preserve the entire job-scoped artifact root
and the raw client files. For NFS or cross-filesystem copies:

1. copy to a temporary destination;
2. compare source/destination hashes and parse JSON;
3. verify sweep and request counts again; and
4. atomically rename the temporary destination when the filesystem permits.

Transient NFS holes warrant a retry and integrity recheck, not relaxed
validation or deletion of the source.

## Failure policy

> [!IMPORTANT]
> A failed run can reflect a bad compute node or hardware/driver state rather
> than profiling code. Before modifying profiling code, inspect the first causal
> errors; validate `rocminfo`, `/dev/kfd`, GPU/RAS state,
> `torch.cuda.is_available()`, and Slurm node health; and, when practical,
> reproduce with profiling disabled or the base image on the same node. Exclude
> a confirmed bad node and notify cluster administrators instead of patching
> profiler code.

Stop and report the first failing phase with its job ID, node/role, sweep key,
expected versus observed counts, and relevant paths. Check image pulls, model
paths, xP+yD allocation, launcher-selected rendezvous IPs, server/router timeouts, per-GPU
capture triplets, and finalization barriers. For multi-sweep failures, compare
the directory count to iteration x shape x concurrency and check unique RIDs
and per-sweep request counts. Reduce to OSL `8`, concurrency `1`, 2-8 prompts,
and one sweep before scaling again. Do not edit source, suppress errors, or
manually fabricate missing artifacts to make validation pass.

See [README.md](README.md) for the full artifact and CLI reference.
