# CLAUDE.md

Guidance for Claude Code when working in the MAD repository.

## What MAD is

MAD (Model Automation and Dashboarding) is a curated catalog of ~142 AI model
workloads that run on AMD Instinct GPUs via the **`madengine`** CLI. Each model
is one JSON entry in `models.json`. `madengine run --tags <tag>` builds a Docker
image, runs the model in a container, parses a performance number from stdout,
and appends a row to `perf.csv`.

The four common user tasks here are: **benchmarking**, **adding a new model**,
**tuning** a model/kernel for better perf, and **development** on the repo
itself. Dedicated subagents and `/mad-*` slash commands exist for each — see
`.claude/agents/` and `.claude/commands/`.

## The performance contract (most important convention)

A model's run script MUST print exactly one line to stdout:

```
performance: <value> <unit>
```

madengine greps this line to populate the `performance` and `metric` columns of
`perf.csv`. A run that does not emit it is recorded with no performance value.
Real example (`scripts/huggingface_bert/run.sh`):

```bash
performance=$(grep -Eo "train_samples_per_second':[^,]+" log.txt | sed "s/.*: //g" | head -n 1)
echo "performance: $performance samples_per_second"
```

If a script produces many results, set `multiple_results` in `models.json` to the
CSV filename the script writes (e.g. `"multiple_results": "perf_DeepSeek-R1.csv"`).

## models.json schema

One object per model. Required fields: `name`, `url`, `dockerfile`, `scripts`,
`n_gpus`, `owner`, `training_precision`, `tags`. Optional: `data`, `timeout`,
`multiple_results`, `args`, `skip_gpu_arch`.

| Field | Notes |
|-------|-------|
| `name` | Unique. Convention: `{framework}_{project}_{workload}`, e.g. `pyt_vllm_deepseek-r1` |
| `url` | Git repo cloned into the container; `""` if none |
| `dockerfile` | Path prefix; engine appends `.ubuntu.amd.Dockerfile` → `docker/<name>.ubuntu.amd.Dockerfile` |
| `scripts` | Path to a `run.sh` (or script dir) executed inside the container |
| `n_gpus` | String. `"-1"` = all available |
| `tags` | List; `--tags` matches any tag OR the exact `name` |
| `training_precision` | e.g. `fp16`, `bf16`, `fp8`, `fp32`, or `""` |
| `timeout` | Seconds; overrides the 7200s default. `-1` = no timeout |
| `skip_gpu_arch` | Skip on these archs, e.g. `"gfx942"` |
| `args` | Extra args appended to the run script |

`models.json` must stay valid JSON — validate with `python3 -m json.tool models.json`
after editing.

## Adding a new model (4 steps)

1. **Name** it `{framework}_{project}_{workload}`.
2. **Add** an entry to `models.json` (copy the closest existing model of the same
   framework as a template).
3. **Dockerfile** `docker/<name>.ubuntu.amd.Dockerfile`. First line MUST be the
   context header, then a base image:
   ```dockerfile
   # CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
   ARG BASE_DOCKER=rocm/pytorch
   FROM $BASE_DOCKER
   ```
   Reuse an existing `docker/*.Dockerfile` of the same stack when possible rather
   than inventing a new base.
4. **Script** `scripts/<dir>/run.sh` that runs the workload and ends by echoing
   the `performance: <value> <unit>` line.

Verify (needs a GPU host): `madengine run --tags <name> --live-output`.

## madengine commands

`requirements.txt` pins madengine to `@main` (Typer CLI, v2.1.0). It exposes
five commands: `build`, `run`, `discover`, `report`, `database`. Verify exact
flags with `madengine <cmd> --help`.

- `madengine discover --tags <tag>` — list matching models (read-only, no GPU).
- `madengine build --tags <tag> [-r REGISTRY] [-a gfx942,gfx90a] [-l]` — build images, write `build_manifest.json`. `--use-image` skips the build and uses a prebuilt image.
- `madengine run --tags <tag> [-l/--live-output] [-o out.csv] [--timeout S] [--keep-alive] [--skip-model-run] [-c '<ctx>']` — full build+run (or, with `-m manifest.json`, execution-only). Writes `perf.csv`.
- `madengine report to-html --csv-file-path perf.csv` — HTML report.
- `madengine database -f perf_entry_super.json --db DB -c COLLECTION` — upload CSV/JSON to MongoDB (`-k model,timestamp` sets the unique key; `--dry-run` to preview).

`--skip-model-run` builds the image without running the workload.
`tools/run_models.py` is the **deprecated** legacy runner — prefer `madengine`.

## Deployment target (convention over configuration)

Pass deployment intent via `--additional-context` (a Python-dict/JSON string,
parsed with `ast.literal_eval` so Python dict syntax is fine). The target is
inferred from which key is present:

- `"slurm"` key → SLURM. `"k8s"`/`"kubernetes"` key → Kubernetes. Neither → local Docker.

## Profiling and tracing

Add a `tools` list to `--additional-context`:

```bash
madengine run --tags <tag> --additional-context '{"tools": [{"name": "rocprofv3_compute"}]}'
```

Common tool names: `rpd`, `rocprofv3`, `rocprofv3_compute`, `rocprofv3_memory`,
`rocprofv3_communication`, `rocm_trace_lite`, `rccl_trace`, `gpu_info_power_profiler`.
Pre-built profiling context files live in the madengine package's
`examples/profiling-configs/`.

## Outputs

- `perf.csv` — one row per run (model, n_gpus, precision, performance, metric, status, durations, git_commit, gpu_architecture, ...).
- `perf_entry_super.json` — enriched record incl. a `configs` block; this is what gets pushed to MongoDB.

## Environment variables

`MAD_SECRETS_HFTOKEN` (HuggingFace token), `MAD_MODEL_NAME`, `MAD_RUNTIME_NGPUS`,
`MAD_SYSTEM_GPU_ARCHITECTURE`, `MAD_MODEL_BATCH_SIZE`.

## Future (not yet wired)

`reference_db/mad_agent.db` (tables: `model_baselines`, `optimization_history`,
`best_configurations`, `learned_patterns`) and `knowledge_base/` are planned
for a future persistent optimization-memory layer. They are not present or
populated yet — do not assume data is available.
