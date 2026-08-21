# madengine — shared repository instructions for Copilot

These instructions are **identical in every repository in the madengine ecosystem** — both
the engine itself and the model repositories that consume it. They carry the madengine
context that all of those repos need. Determine which kind of repository you are in
before applying the role-specific sections.

## What madengine is

madengine is a Python CLI that builds and runs AI/ML model benchmarks on AMD (ROCm) and
NVIDIA (CUDA) GPUs across three execution targets: **local Docker, Kubernetes, and SLURM**.

It is a *library and tool*, not an application. The models it runs are defined in separate
**model repositories** — the reference one is [`ROCm/MAD`](https://github.com/ROCm/MAD) —
and madengine is invoked from the root of one of those repositories.

- Console entry point: `madengine.cli.app:cli_main` (script name `madengine`)
- Install: `pip install git+https://github.com/ROCm/madengine.git`

## Which repository is this?

| Signal | Role |
| --- | --- |
| `src/madengine/` exists; `pyproject.toml` declares `name = "madengine"` | **engine repo** |
| root `models.json` plus `scripts/` and `docker/`; madengine is a dependency | **model repo** |

Both roles need everything under "Shared madengine context." After that, apply
**Working in the engine repo** or **Working in a model repo**, not both.

If neither signal is present, you are somewhere else in the ecosystem (a CI superproject,
a dashboard, a wrapper). Treat the shared section as background and be explicit that the
role-specific rules may not apply.

---

# Shared madengine context

Everything in this section is true regardless of which repo you are in.

## CLI surface

`build`, `run`, `discover`, `database`, and a `report` sub-app (`to-html`, `to-email`,
`tracelens`, `tracelens-compare`).

The two that matter most: `madengine build` discovers models, builds images, and writes
`build_manifest.json`; `madengine run` reads or generates that manifest, infers the
deployment target, executes, and writes `perf.csv`.

**Exit codes are a CI contract** — `SUCCESS=0`, `FAILURE=1`, `BUILD_FAILURE=2`,
`RUN_FAILURE=3`, `INVALID_ARGS=4`.

## Deployment target inference

There is no `deploy` flag. The target is inferred from the shape of `additional_context`:

- a `k8s` or `kubernetes` key → Kubernetes
- a `slurm` key → SLURM
- neither → local Docker

Both `k8s` and `slurm` at once is an error.

## `additional_context` syntax

`--additional-context` accepts **either JSON or a Python dict literal**, because the value
is parsed with `ast.literal_eval` (with a `json.loads` attempt first at the CLI boundary).
Single-quoted keys are valid and widely used:

```bash
madengine run --tags my_model --additional-context "{'gpu_vendor':'AMD','tools':[{'name':'rocprof'}]}"
```

`--additional-context-file` takes a JSON file, applied first, then overridden by the
inline string.

## Feature axes

Nearly everything in madengine is combinatorial. Any change — engine or model — should be
considered against these:

- **deployment target**: local Docker, Kubernetes, SLURM
- **GPU vendor**: AMD (ROCm), NVIDIA (CUDA)
- **launcher**: `torchrun`, `torchtitan`, `deepspeed`, `megatron-lm`, `primus`, `vllm`,
  `sglang`, `sglang-disagg`, `slurm_multi`, plus the `docker` and `native` sentinels
- **image source**: built locally, `--use-image`, `--registry`, `--build-on-compute`,
  `--batch-manifest`, `MAD_CONTAINER_IMAGE`
- **model discovery**: root `models.json`, `scripts/<dir>/models.json`,
  `scripts/<dir>/get_models_json.py`
- **repo layout**: standalone checkout, or submodule inside a parent CI superproject
- **single-node vs multi-node**, and **single-arch vs `--target-archs`**

## The shared output contract

`perf.csv` has exactly **29 columns**, in this order:

```
model, n_gpus, nnodes, gpus_per_node, training_precision, pipeline, args, tags,
docker_file, base_docker, docker_sha, docker_image, git_commit, machine_name,
deployment_type, launcher, gpu_architecture, performance, metric, relative_change,
status, build_duration, test_duration, dataname, data_provider_type, data_size,
data_download_duration, build_number, additional_docker_run_options
```

`status` is `SUCCESS`, `FAILURE`, or `SKIPPED`. `n_gpus` is a **string** (`"-1"` means all
GPUs). `perf_entry.csv` and `perf_entry.json` are written alongside it.

---

# Working in the engine repo

Skip this section if there is no `src/madengine/`.

Python floor is **3.8** — no `list[str]`, `dict | None`, `match`, or `functools.cache`
in `src/`. Use `typing.List`, `typing.Optional`, `typing.Dict`, `functools.lru_cache`.

## Layering

A layer may call downward, never upward.

```
cli/ → orchestration/ → execution/  (local Docker)
                     └→ deployment/ (Kubernetes, SLURM)
                        ↓
                     core/, utils/, reporting/, database/
```

**`cli/`** — Typer wiring. `app.py` registers the commands; one file per command in
`cli/commands/`; `constants.py` holds `ExitCode`; `validators.py` type-checks
`--additional-context`. **This layer parses and validates only** — no build, run, or
deployment logic.

**`orchestration/`** — `BuildOrchestrator` (discover → build → write manifest) and
`RunOrchestrator` (load manifest → infer target → dispatch). `image_filtering.py` drops
images by GPU vendor and `skip_gpu_arch`.

**`execution/`** — local path. `ContainerRunner` runs models via `docker run`;
`DockerBuilder` builds and pushes; `container_runner_helpers.py` holds the pure,
heavily-tested helpers. Prefer adding logic there over growing `container_runner.py`.

**`deployment/`** — distributed path. `DeploymentFactory` maps a target to
`SlurmDeployment` or `KubernetesDeployment`. `BaseDeployment.execute()` is a Template
Method: `validate() → prepare() → deploy() → _monitor_until_complete() →
collect_results() → cleanup()`. Jinja2 templates in `deployment/templates/`; JSON
defaults in `deployment/presets/`, deep-merged by `ConfigLoader`.

**`core/`** — `Context` (detection + merged context), `Console` (shell wrapper with secret
redaction), `Docker` (lifecycle with signal-based reaping), `errors.py`, `dataprovider.py`.

**`utils/`** — `DiscoverModels`, GPU tool managers, ROCm path resolution, config parsing.

**`reporting/`** — `update_perf_csv.py`, `csv_to_html.py`, `csv_to_email.py`,
`tracelens_report.py`.

## Style and tooling

Configured in `pyproject.toml` and `.pre-commit-config.yaml`: **black** (88 cols, py38–311),
**isort** (`profile = "black"`), **flake8** (`--extend-ignore=E203,E501,W503`), **mypy**
(`python_version = "3.8"`, over `^src/madengine/` only), **bandit** (skips `B101`).
Google-style docstrings; type hints on public functions.

```bash
pip install -e .          # all deps, including k8s and dev tools
pre-commit install
pre-commit run --all-files
```

## Testing

```bash
pytest
pytest tests/unit/test_error_handling.py -v
pytest --cov=src/madengine --cov-report=html
pytest -m "not slow"
```

`tests/unit/` is fast and fully mocked; `tests/integration/` makes real Docker and system
calls; `tests/e2e/` runs full workflows; `tests/fixtures/dummy/` is a miniature model
repository used as `MODEL_DIR`.

Markers are **strict**: `unit`, `integration`, `e2e`, `slow`, `gpu`, `amd`, `nvidia`,
`cpu`, `requires_docker`, `requires_models`. Register new ones in `pyproject.toml`.
`Context` is mocked, never constructed, in unit tests — see the `amd_gpu_context`,
`nvidia_gpu_context`, and `cpu_context` fixtures in `tests/conftest.py`.

---

# Working in a model repo

Skip this section if there is no root `models.json`.

You are authoring inputs that madengine consumes. You cannot change madengine's behavior
from here — you have to conform to its contracts. The full detail is in
`.github/skills/code-review/model-repo-contract.md`; the essentials:

## Layout

```
models.json          # required at root — array of model definitions
data.json            # optional — data provider config
credential.json      # optional
docker/              # Dockerfiles; the default build context
  <name>.<os>.<vendor>.Dockerfile
  common/            # optional; exposed to builds as `--build-context tools=`
scripts/
  <model>/run.sh     # the run script
  common/            # POPULATED BY madengine at run time — do not author here
```

`scripts/common/` is filled with madengine's bundled assets during a run and cleaned up
afterwards. Never commit files there and never rely on it between runs.

## The three contracts you must satisfy

**1. A run script signals success by printing a performance line.** Exit code alone is not
enough — a run with no performance data is recorded as `FAILURE`.

```bash
echo "performance: 14164 samples_per_second"
```

The value may be an integer, decimal, or scientific notation; the metric is any token
without whitespace or commas. For multiple results, set `multiple_results` on the model
and write that CSV with at least `model,performance,metric` columns.

**2. Every Dockerfile needs a `# CONTEXT` marker in its first five lines**, holding a
Python dict literal. Missing it means the model resolves to no Dockerfiles and never
builds.

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER
```

Use `# CONTEXT {}` to match every context. Declare `ARG BASE_DOCKER` and `FROM
$BASE_DOCKER` so the base image can be overridden and its SHA recorded.

**3. `models.json` entries follow a fixed schema.** `name` is required; `dockerfile` is a
**path prefix**, not a filename (`"docker/dummy"` matches `docker/dummy.*`); `n_gpus` is a
**string**; `timeout` is an int in seconds. See the contract file for every field.

```json
{
  "name": "my_model",
  "dockerfile": "docker/my_model",
  "scripts": "scripts/my_model/run.sh",
  "n_gpus": "1",
  "owner": "you@example.com",
  "training_precision": "fp16",
  "tags": ["pyt", "fp16"],
  "args": ""
}
```

## Environment madengine gives you

A run script can rely on `MAD_MODEL_NAME`, `MAD_GPU_VENDOR`, `MAD_SYSTEM_NGPUS`,
`MAD_RUNTIME_NGPUS` (GPUs actually granted — use this, not `n_gpus`),
`MAD_SYSTEM_GPU_ARCHITECTURE`, `MAD_GUEST_OS`, `MAD_MULTI_NODE_RUNNER` (the launcher
command string), and `MAD_OUTPUT_CSV` when `multiple_results` is set. Data-backed models
also get `MAD_DATAHOME` and `MAD_DATANAME`. Distributed runs add `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, `RANK`, `NODE_RANK`, and `MAD_COLLECT_METRICS`.

## Log scanning

After a run, madengine scans the log for error substrings (`RuntimeError:`, `ValueError:`,
`Traceback (most recent call last)`, `HIP out of memory`, `FAILED`, and others). A match
fails the run **only when no performance data was extracted** — valid performance wins.
If your model legitimately prints one of those strings, add
`log_error_benign_patterns` rather than suppressing the output.

---

# Working style

Applies in every repo. The engine repo's `CLAUDE.md` is the fuller statement of the same
rules.

- **Surface assumptions and tradeoffs before implementing.** If a request has more than
  one reasonable reading, say so instead of silently picking one.
- **Simplicity first.** No speculative abstractions, no configurability that was not
  asked for, no error handling for impossible states.
- **Surgical changes.** Every changed line should trace to the request. Do not reformat,
  rename, or "improve" adjacent code. Match the surrounding style even where you would
  write it differently. Clean up orphans your own change created — nothing else.
- **Note, don't delete.** If you spot unrelated dead code or a bug outside scope, mention
  it rather than fixing it in the same change.
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `style:`,
  `perf:`, `chore:`, with a scope where it helps. Branches are `feature/<name>` or
  `fix/<name>`.

For the full PR-review protocol, including the invariant list and the cross-feature
checklist, see `.github/skills/code-review/`.
