# madengine invariants

Each entry states the rule, where it is enforced, how to detect that a diff broke it, and
what breaks downstream. A violation of any of these is a **blocking** review finding.

Every entry is tagged with where it applies:

- **engine** — only matters when reviewing changes to madengine's own source
- **model repo** — only matters when reviewing model definitions, Dockerfiles, or scripts
- **both** — a cross-repo contract; the engine must not break it, and model repos must
  conform to it

---

## 1. `additional_context` is parsed with `ast.literal_eval`, not `json.loads`

**Applies to:** both

**Enforced by** `core/context.py::Context.__init__` — `ast.literal_eval(additional_context)`.

Because the parser accepts a Python literal rather than JSON, every producer passes
`repr(dict)`. Both orchestrators say so in a comment:

```python
# Use repr() instead of json.dumps() because Context uses ast.literal_eval()
```

`orchestration/build_orchestrator.py`, `orchestration/run_orchestrator.py`, and
`cli/validators.py` all sit on this contract. `cli/validators.py` tries `json.loads` first
and falls back to `ast.literal_eval`, so users may pass either JSON or single-quoted
Python-dict syntax on the command line.

**Detect (engine):** a diff that introduces `json.dumps(...)` where context is handed to
`Context`, or swaps `ast.literal_eval` for `json.loads` in `context.py`.

**Detect (model repo):** a reviewer or linter "correcting" single-quoted context in a CI
script or README to JSON on the assumption it is malformed. It is not.

**Breaks:** every user and CI job passing `--additional-context "{'k8s': {...}}"`, plus
`Context.filter()`, which also uses `ast.literal_eval` to read the Dockerfile `# CONTEXT`
marker (invariant 13).

---

## 2. Deployment target is inferred structurally, never declared

**Applies to:** both

**Enforced by** `deployment/config_loader.py::ConfigLoader.infer_and_validate_deploy_type`
(build time) and `orchestration/run_orchestrator.py::_infer_deployment_target` (run time).

- `k8s` or `kubernetes` key present → `"k8s"`
- `slurm` key present → `"slurm"`
- neither → `"local"`

`ConfigLoader` additionally raises `ValueError` when both `k8s` and `slurm` are present,
and when an explicit `deploy` value contradicts the config structure. `ConfigLoader` also
deliberately does not write a `deploy` field back into the config.

**Detect (engine):** a diff that adds a required `deploy` field, changes the key names,
adds a third inference rule to only one of the two functions, or reorders the
k8s-before-slurm precedence.

**Detect (model repo):** a config or CI invocation carrying both `k8s` and `slurm`; or one
that sets `deploy` expecting it to do something.

**Note for reviewers:** these two functions already diverge — `ConfigLoader` raises on the
k8s+slurm conflict while `_infer_deployment_target` silently prefers k8s. Any PR touching
inference should keep them consistent or narrow that gap, not widen it.

---

## 3. `build_manifest.json`: `built_images` and `built_models` share one key set

**Applies to:** engine

**Enforced by** `execution/docker_builder.py::export_build_manifest` (writer) and
`execution/container_runner.py::run_models_from_manifest` (reader, which joins the two
dicts by key). `deployment/base.py::_load_manifest` requires `built_images`,
`built_models`, and `context` to be present.

The key is the image name on the normal build path, and the model name on the prebuilt
(`--use-image`) and `--build-on-compute` paths.

**Detect:** a diff that writes to one dict without the other, changes the keying scheme on
one path only, or adds a new build path that populates only `built_images`.

**Breaks:** models silently run with empty `model_info`, producing `perf.csv` rows with
missing `tags`, `args`, and `training_precision`.

---

## 4. The `perf.csv` header is 29 columns and lives in two files

**Applies to:** engine (model repos consume the result)

**Enforced by** `reporting/update_perf_csv.py::PERF_CSV_HEADER` and
`deployment/base.py::_ensure_perf_csv_exists`, which contain byte-identical literals:

```
model, n_gpus, nnodes, gpus_per_node, training_precision, pipeline, args, tags,
docker_file, base_docker, docker_sha, docker_image, git_commit, machine_name,
deployment_type, launcher, gpu_architecture, performance, metric, relative_change,
status, build_duration, test_duration, dataname, data_provider_type, data_size,
data_download_duration, build_number, additional_docker_run_options
```

**Detect:** either literal changed without the other. Adding, removing, or reordering a
column is a schema change to a published artifact.

**Breaks:** multi-node aggregation writes rows that no longer line up with the local
runner's rows; `reporting/csv_to_html.py`, `csv_to_email.py`, the MongoDB uploader, every
model repository's CI, and downstream dashboards all read this schema.

**If a column genuinely must change:** update both literals, `reporting/`, the docs, and
`CHANGELOG.md` under Changed, and say in the PR description that it is a breaking change
for every consuming repository.

---

## 5. `n_gpus` is a string

**Applies to:** both

`models.json` declares `n_gpus` as a string (`"1"`, `"8"`, `"-1"`), `CustomModel` defaults
it to `"-1"`, and `perf.csv` carries it as `str(total_gpus)`. `"-1"` means all GPUs on the
host.

**Detect (engine):** arithmetic or comparison against an int without an explicit `int()`
cast; a diff that "fixes the type" to `int` in `CustomModel` or the perf row.

**Detect (model repo):** `"n_gpus": 8` instead of `"n_gpus": "8"`.

**Breaks:** the `models.json` contract shared with every consuming model repository, none
of which madengine controls.

---

## 6. Exit codes are a CI contract

**Applies to:** both

**Enforced by** `cli/constants.py::ExitCode`:

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | `SUCCESS` | everything succeeded |
| 1 | `FAILURE` | generic failure, including unexpected exceptions and `KeyboardInterrupt` |
| 2 | `BUILD_FAILURE` | one or more image builds failed |
| 3 | `RUN_FAILURE` | one or more model runs failed |
| 4 | `INVALID_ARGS` | mutually exclusive or malformed CLI arguments |

**Detect (engine):** a new numeric value, a reused value with different meaning, a command
path returning a bare `1` where a specific code exists, or a failure path that returns `0`.

**Detect (model repo):** CI that treats any non-zero exit as the same condition, losing
the distinction between a build break and a model regression; or CI that masks the exit
code with `|| true` and then reports success.

---

## 7. Shell inputs are quoted and secrets are redacted

**Applies to:** both

**Enforced by** `core/console.py::redact_secrets` (masks `MAD_SECRETS*=...` and `hf_`,
`sk-`, `ghp_`, `xox*` token shapes) and by `shlex.quote` throughout `core/docker.py`,
`execution/`, and `deployment/`. `core/docker.py` also validates environment variable
names, and `execution/container_runner.py` gates them on `^[A-Za-z_][A-Za-z0-9_]*$`.

**Detect (engine):**
- an f-string that interpolates a model name, path, tag, arg string, or context value
  directly into a command passed to `Console.sh`, `Docker.sh`, `subprocess`, or a Jinja2
  template that becomes a shell script
- a credential, token, registry password, or `MAD_SECRETS*` value reaching `print`,
  `logging`, a `docker run` argument list, a Jinja2-rendered sbatch script, or a K8s
  manifest body rather than a Secret
- a new log or error message that echoes a raw command line without going through the
  redaction path

**Detect (model repo):**
- a token, key, or password committed in `models.json`, `data.json`, `credential.json`, a
  Dockerfile `ENV`/`ARG` default, or a run script
- a run script that `echo`s a secret-bearing environment variable, or runs `set -x` around
  one — madengine's redaction covers its own output, not arbitrary script output
- a secret baked into an image layer rather than passed at run time

Credentials belong in `core/auth.py::load_credentials` / `login_to_registry`, in
`credential.json` (which is not committed), and in K8s in `deployment/k8s_secrets.py`.

---

## 8. Launcher names are canonicalized in exactly one place

**Applies to:** both

**Enforced by** `deployment/common.py`, which owns `VALID_LAUNCHERS`
(`torchrun`, `torchtitan`, `deepspeed`, `megatron-lm`, `primus`, `vllm`, `sglang`,
`sglang-disagg`, `slurm_multi`), the `_LAUNCHER_ALIASES` map, and the functions
`canonicalize_distributed_launcher`, `normalize_launcher`, and `is_self_managed_launcher`.

The comment on `_LAUNCHER_ALIASES` is explicit: *"Add new aliases here only; do not branch
on alternate spellings at dispatch sites."*

**Detect (engine):** a dispatch site comparing against a raw string with an alternate
spelling (`sglang_disagg` instead of `sglang-disagg`, `slurm-multi` instead of
`slurm_multi`); a second alias map; a launcher added to
`slurm.py::_generate_launcher_command` but not to `k8s_template_context.py`, or the
reverse.

**Detect (model repo):** a `distributed.launcher` value that is not in `VALID_LAUNCHERS`.
Unknown values fall back to `torchrun` on the local path with only a warning, so a typo
runs the wrong launcher instead of failing.

Defaults when no launcher is given: local → `docker`, slurm → `docker`, kubernetes →
`native`. `slurm_multi` is the one self-managed launcher — it runs the model's own
`.slurm` script on the head node and bypasses the sbatch template entirely.

---

## 9. Valid performance data outweighs a log error-pattern match

**Applies to:** both

**Enforced by** `execution/container_runner_helpers.py::resolve_run_status`, whose
docstring explains the reasoning in full.

Log scanning is post-hoc substring matching over the whole run log, so it cannot tell a
framework traceback from a model's own generated stdout — an LLM benchmark can legitimately
emit `"ValueError:"` as output text. The precedence is therefore:

1. performance present → `SUCCESS` (a pattern match is still reported, for triage)
2. no performance, pattern matched → `FAILURE`
3. no performance, worker node or deferred collection → `SUCCESS`
4. otherwise → `FAILURE`

**Detect (engine):** a diff that reorders these branches, or makes an error-pattern match
fail a run that produced performance data.

**Detect (model repo):** a model disabling the scan wholesale with
`log_error_pattern_scan: false` when the narrower `log_error_benign_patterns` would do.
Disabling the scan hides genuine failures for that model forever.

---

## 10. Containers, jobs, and directories are cleaned up on every path

**Applies to:** engine

**Enforced by** `core/docker.py` (containers labeled `{LABEL_PREFIX}.run_id` and
`{LABEL_PREFIX}.pid`, i.e. `madengine.run_id` / `madengine.pid`;
`reap_active_containers` and `_install_signal_handlers` force-remove on SIGINT, SIGTERM,
and normal exit; `cleanup()` is idempotent),
`deployment/base.py::execute` (`cleanup()` in the Template Method), and
`orchestration/run_orchestrator.py::_cleanup_model_dir_copies`.

**Detect:** a new early `return`, `raise`, or `sys.exit` between resource acquisition and
cleanup; a new container, PVC, K8s Job, or SLURM job created outside the existing cleanup
path; a `cleanup` implementation that is not idempotent.

---

## 11. `scripts/common/` is engine-owned; everything else in the repo survives a run

**Applies to:** both

**Enforced by** `orchestration/run_orchestrator.py` (`_copy_scripts` and its cleanup
counterpart). madengine writes its bundled assets into `scripts/common/` — `tools.json`,
`test_echo.sh`, `pre_scripts/`, `post_scripts/`, `tools/` — and removes exactly those
items afterwards.

**Detect (engine):** a broadened glob, an `rmtree` on `scripts/` or `scripts/common/` as a
whole, or a new asset written outside `scripts/common/`.

**Detect (model repo):** a PR that commits files under `scripts/common/`, or a run script
that writes there expecting persistence. Both are silently destroyed on the next run.

**Breaks:** deletes a model repository checkout's own model scripts. This is data loss in
a user's working tree, not just a failed run.

---

## 12. Python 3.8 is the floor

**Applies to:** engine

`requires-python = ">=3.8"`; black targets `py38`–`py311`; mypy runs at
`python_version = "3.8"` over `^src/madengine/` only.

**Detect in `src/`:** `list[str]` / `dict[str, int]` / `X | None` annotations without
`from __future__ import annotations`, `match` statements, `functools.cache`,
`typing.ParamSpec`, `dict |` merge operator, `str.removeprefix` / `removesuffix`,
walrus-heavy comprehension scoping added in later versions.

Note that mypy skips `tests/` and `src/madengine/scripts/`, so nothing catches this
automatically there.

---

## 13. Every Dockerfile carries a `# CONTEXT` marker in its first five lines

**Applies to:** both

**Enforced by** `execution/docker_builder.py::_get_dockerfiles_for_model`, which greps
`# CONTEXT ` out of the first five lines, and `core/context.py::filter`, which evaluates
the result as a Python dict literal and keeps only Dockerfiles whose keys all match the
current context.

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
```

`# CONTEXT {}` matches every context.

**Detect (model repo):** a new or renamed Dockerfile with no `# CONTEXT` line, or with the
marker pushed below line five by a license header. Also a marker that is not valid Python
dict syntax.

**Detect (engine):** a change to the grep depth, the marker string, or the `ast.literal_eval`
parse — all three are relied on by every model repository's Dockerfiles.

**Breaks:** the grep exits non-zero, the exception is swallowed, and the model resolves to
**no Dockerfiles at all**. It is skipped with no build error. This is the highest-frequency
silent failure in the ecosystem.

---

## 14. A run reports success by printing a performance line

**Applies to:** both

**Enforced by** `PERFORMANCE_LOG_PATTERN` in `deployment/base.py`, applied to run logs by
`execution/container_runner.py` locally and by `BaseDeployment._parse_performance_from_log`
for multi-node aggregation.

```
performance: <number> <metric>
```

The number accepts a sign, decimals, and scientific notation. The metric is any token with
no whitespace or comma. An optional `/unit` or comma may sit between them. A HuggingFace
Trainer fallback (`train_samples_per_second`) is tried only if the primary pattern fails.

Models with `multiple_results` write a CSV instead, needing `performance` at minimum and
`model,performance,metric` for superset reporting.

**Detect (model repo):** a run script whose success path prints no performance line; a
metric name containing a space or comma; a `multiple_results` filename that the script
never writes; a script that prints the line only on some branches.

**Detect (engine):** any change to the regex, which every existing model's output format
depends on.

**Breaks:** exit code 0 with no performance line is recorded as `FAILURE` (invariant 9,
branch 4). The model appears broken in every report even though it ran fine.
