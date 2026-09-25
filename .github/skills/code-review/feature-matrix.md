# madengine feature matrix

madengine's features are combinatorial. A change is only complete when it holds across
every cell of the axes it touches. Use this file to work out which cells a PR is on, and
then check each one.

The two questions to ask of any diff:

1. **Which axes does this sit on?**
2. **For each axis, is every cell handled — or is skipping a cell deliberate and stated?**

The axes below apply in both the engine repo and model repos — the difference is what you
check. In the engine, ask whether the *implementation* covers every cell. In a model repo,
ask whether the *model* declares itself correctly for the cells it claims to support. The
propagation maps at the end are split by role.

---

## The axes

### Deployment target

| Cell | Code path |
| --- | --- |
| local Docker | `execution/container_runner.py` |
| SLURM | `deployment/slurm.py` + `templates/slurm/job.sh.j2` |
| Kubernetes | `deployment/kubernetes.py` + `kubernetes_launcher_mixin.py` + `templates/kubernetes/*.j2` |
| SLURM self-managed | `slurm_multi` launcher — `container_runner.py::_run_self_managed`, bypasses the sbatch template |

Local and distributed are genuinely separate implementations, not one path with a flag.
Anything that changes *what a model run looks like* — an env var, a mount, a GPU flag, a
timeout, a log path, a perf row field — usually has to land in all three.

### GPU vendor

`AMD` (ROCm) and `NVIDIA` (CUDA), from `VALID_GPU_VENDORS`. They differ in `docker run`
flags (`--group-add video --cap-add=SYS_PTRACE --ipc=host` and friends for AMD versus the
NVIDIA runtime), in device visibility (`HIP_VISIBLE_DEVICES` versus `CUDA_VISIBLE_DEVICES`),
in detection (`amd-smi`/`rocm-smi` versus `nvidia-smi`), in tool managers
(`utils/rocm_tool_manager.py` versus `utils/nvidia_tool_manager.py`), and in K8s presets
(`presets/k8s/gpu-vendors/{amd,nvidia,amd-multi-gpu}.json`).

AMD is the primary target and NVIDIA support is real. A vendor-specific change that lands
only in the AMD branch needs an explicit reason.

### Launcher

`torchrun`, `torchtitan`, `deepspeed`, `megatron-lm`, `primus`, `vllm`, `sglang`,
`sglang-disagg`, `slurm_multi`, plus the `docker` and `native` non-launcher sentinels.

There are **two** dispatch sites that must stay in step:

- `deployment/slurm.py::_generate_launcher_command` → `_generate_*_command`
- `deployment/k8s_template_context.py` → `kubernetes_launcher_mixin.py::_generate_*_command`

Canonicalization is centralized (`deployment/common.py`; see invariant 8).

### Image source

| Cell | Flag / trigger |
| --- | --- |
| built from a Dockerfile | default |
| prebuilt image | `--use-image <name>` or `--use-image auto` (reads `env_vars.DOCKER_IMAGE_NAME`) |
| registry pull/push | `--registry` |
| built on a compute node | `--build-on-compute` (requires `--registry`) |
| selective batch build | `--batch-manifest` (`build_new` per model) |
| local image, no build | `MAD_CONTAINER_IMAGE` in context |

Each produces a `build_manifest.json` with slightly different fields
(`local_image`, `prebuilt`, `built_on_compute`, `registry_image`). Manifest-shape changes
must cover all six.

The `build` CLI enforces mutual exclusions: `--batch-manifest` vs `--tags`,
`--additional-context-file` vs `--additional-context`, `--use-image` vs `--registry`,
`--use-image` vs `--build-on-compute`. New flags need their own exclusion rules and
`INVALID_ARGS` handling.

### Model discovery

`utils/discover_models.py::DiscoverModels` supports three sources:

1. root `models.json`
2. `scripts/<dir>/models.json` — names are prefixed `<dir>/`, script paths rebased
3. `scripts/<dir>/get_models_json.py` — must define `list_models()` returning `CustomModel`s

Tag selection supports plain tags, scoped tags (`dummy2/model1`, `<scope>/all`), and
inline args (`model:batch_size=512`). Discovery changes need coverage of all three sources
and all three selection forms.

### Consuming repo layout

`utils/config_parser.py` resolves model config paths for both supported layouts, matching
against its known-repo list:

- standalone checkout → `./scripts/<model>/configs/`
- submodule inside a parent CI superproject → `./scripts/<repo>/<model>/configs/`

Anything that resolves a path relative to the repo root has to work in both.

### GPU architecture

`--target-archs` produces one image per arch with a `_gfxNNN` suffix;
`skip_gpu_arch` on a model card excludes archs at run time (disable with
`--disable-skip-gpu-arch`); `orchestration/image_filtering.py` filters by vendor and by
`skip_gpu_arch`; `MAD_SYSTEM_GPU_ARCHITECTURE` in context overrides detection.

### Data provider

`core/dataprovider.py`: `Local`, `NAS`, `Minio`, `AWS`, `Custom`, built by
`DataProviderFactory` from `data.json`. Perf rows carry `dataname`,
`data_provider_type`, `data_size`, `data_download_duration` — a provider change that does
not populate those leaves holes in `perf.csv`.

### Profiling tools

The rocprof family (`rocprof`, `rocprof_hip_only`, `rocprof_sys`, and the `rocprofv3_*`
variants), plus `rtl_trace`, `gpu_info_profiler`, TraceLens, and TheRock markers. Tools
wrap the model command, so they interact with every launcher; multi-node runs additionally
require MPI-aware `rocprofv3` (`deployment/common.py::is_rocprofv3_available`).

TraceLens is the one optional extra (`pip install 'madengine[tracelens]'`) — code touching
it must degrade gracefully when it is absent.

### Result shape

`reporting/update_perf_csv.py` handles `single_result` (JSON), `multiple_results` (a CSV
plus `common_info`, requiring `model,performance,metric` columns), and `exception_result`.
Multi-node runs additionally produce `perf_entry_super.json` via
`reporting/update_perf_super.py`. Status is one of `SUCCESS`, `FAILURE`, `SKIPPED`.

---

## Propagation map — engine repo

When a PR adds one of these, it must touch everything in the right-hand column. A missing
item is a blocking finding under "incomplete matrix coverage."

**A new `additional_context` key**
`cli/validators.py::validate_additional_context_structure` (type check) → `core/context.py`
(merge and defaults) → each consumer (`execution/`, `deployment/slurm.py`,
`deployment/k8s_template_context.py`) → the relevant Jinja2 template if it reaches a job
script → `docs/configuration.md` → a unit test.

**A new CLI flag**
`cli/commands/<cmd>.py` (option + mutual-exclusion checks + `INVALID_ARGS`) →
`cli/utils.py::create_args_namespace` if it crosses into the orchestrator → the
orchestrator → `docs/cli-reference.md` → `README.md` if user-facing → `CHANGELOG.md`.

**A new launcher**
`deployment/common.py::VALID_LAUNCHERS` (+ an alias entry if it has one) →
`slurm.py::_generate_launcher_command` and its `_generate_*_command` →
`k8s_template_context.py` and `kubernetes_launcher_mixin.py::_generate_*_command` →
`container_runner.py::_resolve_local_multi_node_runner_env` for the local path →
`docs/launchers.md` → tests for both distributed dispatch sites.

**A new `models.json` field**
`utils/discover_models.py::CustomModel` → `DockerBuilder` and/or `ContainerRunner` →
`build_manifest.json` via `export_build_manifest` if builds need it →
`create_run_details_dict` if it belongs in `perf.csv` → `tests/fixtures/dummy/models.json`
→ `README.md`/`docs/` → note the coordination needed with the model repositories, which
madengine does not control.

**A new `perf.csv` column**
Both header literals (invariant 4) → `create_run_details_dict` →
`update_perf_csv.py`'s three result handlers → `deployment/base.py` aggregation →
`reporting/csv_to_html.py` and `csv_to_email.py` → `database/mongodb.py` →
`tests/conftest.py::assert_perf_csv_valid` → `CHANGELOG.md` as a breaking change.

**A new deployment target**
A `BaseDeployment` subclass implementing all six abstract methods →
`factory.py::register_default_deployments` (guarded by a `try/except ImportError` if it
carries an optional dependency, matching the Kubernetes pattern) → both inference
functions (invariant 2) → `presets/` defaults and profiles → templates →
`docs/deployment.md`.

**A new GPU vendor or arch**
`cli/constants.py::VALID_GPU_VENDORS` → `utils/gpu_validator.py` detection →
a `gpu_tool_manager` implementation → `core/context.py` detection getters →
`docker run` flag construction in `container_runner.py` → `presets/k8s/gpu-vendors/` →
`orchestration/image_filtering.py` → `dockerfile_utils.py` arg parsing.

**A new preset default**
`presets/{k8s,slurm}/...` → confirm the merge order still yields the intended value
(`ConfigLoader` layers defaults → gpu-vendor → profile → user config, merging `env_vars`
and replacing lists and scalars) → a test asserting the merged result, following
`tests/unit/test_config_loader.py`.

---

## Propagation map — model repo

Field names and rules are documented in `model-repo-contract.md`.

**A new model**
`models.json` entry (`name`, `dockerfile` prefix, `scripts`, `n_gpus` as a string, `tags`,
`timeout`) → a Dockerfile per supported vendor, each with a `# CONTEXT` line → a run script
that prints `performance: <value> <metric>` → `skip_gpu_arch` for architectures it cannot
run on → a `data.json` entry if it declares `data` → tags that place it in the right CI
selection group.

**A new Dockerfile variant** (e.g. adding NVIDIA alongside AMD)
`docker/<prefix>.<os>.<vendor>.Dockerfile` with the matching `# CONTEXT` dict → confirm the
`dockerfile` prefix in `models.json` still globs both → verify the run script does not
assume ROCm-only tooling → check `skip_gpu_arch` still makes sense for the new vendor.

**A change to a run script**
Confirm the performance line still prints on every success path → confirm
`MAD_RUNTIME_NGPUS` is still respected → if the script now emits a string in
`DEFAULT_LOG_ERROR_PATTERNS`, add `log_error_benign_patterns` rather than disabling the
scan → if it writes a `multiple_results` CSV, confirm the filename still matches the model
entry and the required columns are present.

**Making a model distributed**
`distributed.launcher` set to a canonical value from `VALID_LAUNCHERS` → `nnodes` and
`nproc_per_node` → the run script reads `MAD_MULTI_NODE_RUNNER` (tolerating empty) rather
than hardcoding a launcher → `n_gpus` consistent with `gpus_per_node` → a `slurm` or `k8s`
block if it targets a cluster, which also changes the inferred deployment target.

**A new dataset**
`data.json` entry keyed by dataname, with at least one provider and a `path` → the model's
`data` field referencing it → the run script reading `MAD_DATAHOME` rather than a
hardcoded path → credentials in `credential.json`, never in `data.json`.

---

## Questions worth asking on almost any PR

**Engine**

- If this is a bug fix, does the same bug exist in the sibling deployment path? Fixes to
  `slurm.py` and `kubernetes.py` are the most frequent source of one-sided changes.
- Does the change hold when `n_gpus` is `"-1"`, when `nnodes > 1`, and when the run is on
  a non-collecting worker node?
- Does it hold when the image came from `--use-image` or `--registry` rather than a local
  build, where there is no Dockerfile and no build log?
- Does it hold on the `slurm_multi` path, which bypasses the sbatch template and manages
  its own containers?
- Does the failure path still write a `perf.csv` row? Setup failures go through
  `_create_setup_failure_perf_entry`; a silent failure with no row is invisible to CI.
- Does it change anything a model repository depends on? Those repos cannot be updated in
  lockstep.

**Model repo**

- Will this model be *discovered*? It needs a tag CI selects on, or an exact name.
- Will it *build*? Every Dockerfile needs a `# CONTEXT` line, and `dockerfile` must be a
  prefix that globs to a real file.
- Will it *run* within `timeout`, on the GPU count it declares?
- Will it *report*? The success path must print a performance line, or write the
  `multiple_results` CSV.
- Which vendors and architectures does it actually support, and does `# CONTEXT` plus
  `skip_gpu_arch` say so accurately?
