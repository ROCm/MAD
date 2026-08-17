# Launch and results

Keywords: madengine run --manifest-file --live-output, perf.csv perf_super,
multiple_results, slurm_output node_N.out, multi-node aggregation, tok/s/gpu,
TFLOPS, sbatch squeue, MODEL_DIR scripts missing

## Launch

From `<WORKDIR>/rundir`, in a shell where `mad.env` has been sourced:

```bash
cd "$WORKDIR/rundir"
source mad.env                       # same shell that runs madengine
madengine run --manifest-file run_manifest_<workload>.json --live-output
```

Useful flags (`madengine run --help` for the full list):
- `--manifest-file/-m <file>` — run an existing manifest (skips the separate
  build phase; the image is still ensured at run time, see below).
- `--live-output/-l` — stream container/job output in real time.
- `--output/-o <file>` — performance output CSV (default `perf.csv`).
- `--summary-output/-s <file>` — JSON run summary.
- `--keep-alive` — leave containers up after the run (debugging).
- `--keep-model-dir` — keep the staged model dir (debugging).
- `--verbose/-v` — verbose logging.

`madengine` renders the sbatch script from the `deployment_config` and submits
the SLURM job itself. Watch the job with `squeue -u $USER`; per-node
stdout/stderr land under `slurm_output/` (e.g. `node_0.out`, `node_1.out`).

## Where the image comes from (madengine handles it — no pre-build needed)

For a `local_image: true` manifest, the execution phase calls
`_ensure_local_image_available()` on each node (`container_runner.py`), so no
pre-build or pre-pull is needed. Per node the logic is:

1. `docker image inspect <docker_image>` — already present? Then nothing to do.
2. On the primary node (NODE_RANK 0), if the image is missing:
   - if a tar exists at `MAD_DOCKER_BUILDS/<sanitized-tag>.tar` -> `docker load` it;
   - else **build it from the manifest `dockerfile`** (`docker build -f <dockerfile> --pull ...`),
     and if that build fails, **fall back to `docker pull <docker_image>`**;
   - if `MAD_DOCKER_BUILDS` is set and the tar is absent -> `docker save` the
     image into that tar.
3. A TCP barrier syncs all nodes; then worker nodes (rank > 0) `docker load`
   the tar (when `MAD_DOCKER_BUILDS` is set) or build/pull independently.

Implications for the agent:
- **No manual build/pull step.** Just run `madengine run -m`.
- **First run is slower** when the image doesn't exist yet — rank 0 builds it
  once, then everyone reuses the tar.
- **`MAD_DOCKER_BUILDS` lives on shared FS** so rank 0's tar is visible to
  workers (otherwise every node rebuilds/pulls). This is the same gotcha as in
  `SKILL.md`.
- **The build path needs the manifest's `dockerfile` to resolve** — it lives in
  the cloned `MAD` (the dockerfile path is relative to the MAD repo).
  If the dockerfile is missing/unresolvable, madengine falls back to
  `docker pull <docker_image>`, which then requires that tag to exist in a
  reachable registry.
- Keep the manifest's `docker_image` tag meaningful: it is both the build tag
  and the pull fallback tag, and it names the tar in `MAD_DOCKER_BUILDS`.

## Where results land

- `perf.csv` (or `--output` path) in `rundir` — the aggregated performance row.
- `perf_super*.csv` / `perf_entry*.csv` — intermediate per-entry files.
- The model's `multiple_results` CSV (e.g.
  `perf_primus-megatron-Llama-3.1-8B.csv`) — per-model detail.
- `slurm_output/node_*.out` — raw logs (throughput banner, NCCL/RCCL transport
  lines, tracebacks).

## Multi-node aggregation gotcha

Throughput is frequently emitted only by the LAST global rank (often the last
node), not node 0. So node 0's local perf CSV can be empty even on a fully
healthy run. The login-node aggregated `perf.csv` in `rundir` is the source of
truth over any single node's file. A node-0-only "empty perf" check is a false
negative — the aggregation across nodes picks the non-empty result.

## Reading the number

- Primus: report tok/s/gpu and TFLOPS/gpu (from the aggregated CSV / rank-last
  `.out` throughput banner).
- sglang_disagg: request throughput / latency from the disagg benchmark CSV.
- MLPerf training: `scripts/pyt_mlperf_training/extract_perf.py` parses the
  rank-0 `:::MLLOG` stream plus the `train_step_timing` lines on node 0 and
  writes the whole metric set (`step_time_{mean,p50,p95,stdev}_s`,
  `cluster_tokens_per_s`, `per_gpu_tokens_per_s`, `per_gpu_tflops_bf16`,
  `mfu_pct`, `time_to_train_min`). The same numbers are printed human-readably in
  `slurm_output/madengine-*_<jobid>_node_0.out` — grep `Per-iteration perf`. Read
  step time and tokens/s; `mfu_pct` is only valid if
  `MLPERF_PEAK_BF16_TFLOPS` was set, and `run_stop_aborted=1` is normal for a
  fixed-step perf run (it means the run stopped at `MLPERF_MAX_STEPS`, not at the
  benchmark's convergence target). Iteration 0 is a cold outlier (tens of
  seconds) and is excluded from the aggregates by design.
- MLPerf inference: read three things from
  `perf_pyt_mlperf_inference_llama-3.1-8b.csv`. `tokens_per_s` / `samples_per_s`
  are the throughput, but they only count as a measurement if `result_valid=1`
  (loadgen's `Result is : VALID`); `result_valid=0` almost always means
  `Min duration satisfied : NO`, i.e. `MLPERF_INF_OFFLINE_TARGET_QPS` was unset or
  too low, and the throughput printed next to it is real but not a valid run.
  `accuracy_pass=1` means the four ROUGE scores cleared 99% of the upstream BF16
  targets for the selected dataset variant (and gen_len 90%, checked only when
  the whole set was served). Reference numbers for one 8x MI355X node, Offline,
  edge 5000 samples, TP=8: 38.1 samples/s, 4877 tokens/s, rouge1 39.07. Keep in
  mind this is a single TP=8 instance — the reference's shape, not a competitive
  one, since real submissions run 8 data-parallel TP=1 replicas.

## Quick failure triage

- Run dies immediately with a missing run-script / empty `scripts/` ->
  `MODEL_DIR` not exported. Re-`source mad.env` in the launching shell and
  confirm `[ -d "$MODEL_DIR/scripts" ]`.
- Workers rebuild the image or fail to find it -> `MAD_DOCKER_BUILDS` is not on
  shared storage visible to every node. Point it at shared FS and re-run.
- All ranks exit in the first minutes with `Failed to initialize any NET
  plugin` / RCCL falls back to sockets -> wrong `NCCL_IB_HCA` for the node, or
  AINIC transport vars missing in one of the two env blocks. See
  cluster-types.md.
- Data/gated-model error -> HF token missing/expired (`~/.huggingface/token`).
  A rendered `docker run` showing `MAD_SECRETS_HFTOKEN='${MAD_SECRETS_HFTOKEN}'`
  means a manifest wrongly redeclared the token; remove that key and re-source
  `mad.env` (see manifests.md "Secrets").
- "path does not exist" / an unexpectedly empty mounted dir -> swapped
  `docker_mounts`. Entries are keyed `{container_path: host_path}`, so the host
  path is the value (see manifests.md "docker_mounts direction").
- `MODEL_PATH is missing` -> wrong model host path; the model
  resolves at `$MODEL_DIR/$MODEL_NAME` inside the container.
- The launcher starts, every rank then dies on a `TypeError` about an unexpected
  keyword argument or an `ImportError` for a submodule of a framework the image
  built fine -> a version skew baked into the image, not a cluster problem. Do
  not swap the base image; find the framework's own pin (for NeMo:
  `requirements/manifest.json`) and match it. See
  [gotchas.md](gotchas.md#pyt_mlperf_training-mlperf-llama-31).
- The run reports SUCCESS rows but the `-o` CSV "not found" -> cosmetic; the
  classic SLURM path aggregates into `perf.csv` regardless of `-o` (gotchas.md).
