---
name: code-review
description: Review pull requests in any repository in the madengine ecosystem — the madengine engine itself, or a model repository that consumes it. Use this for every code review in this repo. It covers madengine's layering rules, its hard invariants, the model-repository authoring contract, and the cross-cutting feature axes (deployment targets, GPU vendors, launchers, image sources, discovery methods) that a change must stay consistent with.
---

# Reviewing pull requests in the madengine ecosystem

madengine is a CLI that builds and runs AI/ML benchmarks across local Docker, Kubernetes,
and SLURM, on AMD and NVIDIA GPUs. It is consumed by separate **model repositories** that
hold model definitions, Dockerfiles, and run scripts.

This skill is installed identically in both kinds of repository. Start by working out
which one you are in, because the review focus differs sharply.

| Signal | Role | What PRs change |
| --- | --- | --- |
| `src/madengine/` exists | **engine repo** | Python source, deployment templates, presets, tests |
| root `models.json` plus `scripts/` and `docker/` | **model repo** | model definitions, Dockerfiles, run scripts, CI config |

Almost every feature in madengine is a **matrix**, not a single path. The most common
defect in the engine is not a broken function — it is a change that handles one cell and
silently diverges from the others: a fix applied to SLURM but not Kubernetes, an env var
threaded through the local runner but not the Jinja2 templates, a launcher added to one
dispatch site but not the other. In a model repo, the most common defect is a definition
that violates one of the engine's contracts and fails silently rather than loudly.

## Reference files in this skill directory

- `invariants.md` — hard rules, each tagged with which role it applies to
- `feature-matrix.md` — the cross-cutting axes and the propagation maps
- `model-repo-contract.md` — the authoring contract for model definitions, Dockerfiles,
  run scripts, `data.json`, and `additional_context`

Read `invariants.md` always. Read `feature-matrix.md` for engine PRs. Read
`model-repo-contract.md` for model-repo PRs, and also for engine PRs that change anything
a model repo depends on.

---

## Review process — engine repo

1. **Classify the change.** Which layer(s) — `cli/`, `orchestration/`, `execution/`,
   `deployment/`, `core/`, `utils/`, `reporting/`? Which feature axes?
2. **Check the invariants** in `invariants.md`. A broken invariant is always blocking.
3. **Walk the feature matrix** in `feature-matrix.md`. For each axis the change touches,
   confirm every cell is handled or that skipping one is deliberate and stated.
4. **Check the layering.** `cli/` parses and validates; `orchestration/` coordinates;
   `execution/` and `deployment/` do the work; `core/` and `utils/` are leaves. A layer
   may call downward, never upward. Business logic in `cli/commands/*.py` is a finding.
5. **Check for a downstream break.** Does the change alter anything a model repository
   depends on — the `models.json` schema, the performance line format, the `# CONTEXT`
   marker, injected `MAD_*` env vars, `perf.csv` columns, exit codes, or `data.json`?
   Model repos cannot be updated in lockstep, so any such change needs an explicit
   compatibility note in the PR. Check it against `model-repo-contract.md`.
6. **Check consistency with the existing pattern.** Find the two or three nearest
   analogues already in the tree and compare. A new deployment should mirror
   `SlurmDeployment`/`KubernetesDeployment`; a new launcher should mirror the existing
   `_generate_*_command` pair; a new preset should mirror the existing merge order. If the
   PR introduces a *different* way to do something the repo already does, that is a
   finding even when the new code works.
7. **Check tests.** New behavior needs new tests. Confirm markers are registered, that
   unit tests mock `Context` rather than constructing it, and that anything requiring a
   GPU or Docker is marked (`gpu`, `amd`, `nvidia`, `requires_docker`, `requires_models`).
8. **Check the paper trail.** Conventional commit subject; `CHANGELOG.md` entry under
   `## [Unreleased]` for user-visible changes; `docs/` updated when a flag, context key,
   `models.json` field, or `perf.csv` column changed.

## Review process — model repo

1. **Identify what was added or changed** — a model entry, a Dockerfile, a run script, a
   data definition, CI config?
2. **Validate against `model-repo-contract.md`.** The highest-yield checks, in order:
   - Does every new or changed Dockerfile have a `# CONTEXT` line in its first five
     lines? Without it the model silently never builds.
   - Does the run script print a `performance: <value> <metric>` line on the success
     path? Exit code 0 with no performance line is recorded as `FAILURE`.
   - Is `n_gpus` a string? Is `dockerfile` a path *prefix* rather than a filename?
   - Does `scripts` point at a file that exists?
   - If `multiple_results` is set, does the script actually write that CSV, with the
     required columns?
3. **Check the invariants** in `invariants.md` tagged for model repos.
4. **Check the axes that apply.** Will this model work on both GPU vendors, or is the
   Dockerfile AMD-only with no NVIDIA counterpart and no `# CONTEXT` guard? Does
   `skip_gpu_arch` cover the architectures it genuinely cannot run on? If it declares a
   launcher, is the name one of the canonical values?
5. **Check the run script for portability.** Does it read `MAD_RUNTIME_NGPUS` rather than
   hardcoding a GPU count? Does it tolerate `MAD_MULTI_NODE_RUNNER` being empty under
   `set -u`? Does it assume a package manager without checking `MAD_GUEST_OS`?
6. **Check `timeout`.** A value below the model's realistic runtime turns every run into
   a failure. A missing value means the 7200-second default.
7. **Check nothing was added under `scripts/common/`** — madengine overwrites and then
   deletes that directory on every run.

---

## Priority order for findings

Report in this order, and say which category each finding is in.

1. **Blocking — correctness and safety.** A broken invariant; a secret that can reach a
   log, a CLI argument, a committed file, or a container image layer; unquoted
   interpolation into a shell command; a container, SLURM job, or K8s resource that can
   leak on the failure path; a change to `perf.csv` columns, exit codes, or the manifest
   schema that breaks existing consumers.
2. **Blocking — silent failure.** A change that makes a model not build, not run, or not
   report, without producing an error. Missing `# CONTEXT`, a `dockerfile` prefix that
   matches nothing, a run script with no performance line, a `multiple_results` CSV that
   is never written.
3. **Blocking — incomplete matrix coverage.** One deployment target, GPU vendor, launcher,
   or image source handled and a sibling silently left behind.
4. **Important — divergence from the established pattern.** Working code that introduces
   a second way of doing an existing thing, bypasses a shared helper, or duplicates logic
   that already has a single owner.
5. **Important — missing tests or docs** for new behavior.
6. **Minor — style and clarity.** Only where automated formatters would not already catch
   it.

---

## Things specific to this ecosystem that are easy to miss

**Everywhere**

- **`n_gpus` is a string.** `"1"`, `"8"`, `"-1"` (all). Arithmetic on it without `int()`
  is a bug; converting the field to an int breaks the `models.json` contract.
- **`additional_context` is a Python dict literal or JSON**, parsed with
  `ast.literal_eval`. Single-quoted keys are correct, not a mistake.
- **The deployment target is inferred**, never declared. A `k8s`/`kubernetes` key means
  Kubernetes, a `slurm` key means SLURM, neither means local.
- **Emoji in console output is the established style** (`✅`, `⚠️`, `🚀`). Do not flag it.
  Do flag emoji added to source comments or docstrings.

**Engine repo only**

- **Python 3.8 floor.** `list[str]`, `dict | None`, `match`, and `functools.cache` are
  errors in `src/`. Use `typing.List`, `typing.Optional`, `typing.Dict`,
  `functools.lru_cache`. mypy runs at `python_version = "3.8"` but only over
  `^src/madengine/`, so tests will not catch this for you.
- **`perf.csv` has exactly 29 columns**, and the header literal exists in two files. See
  invariant 4.
- **`src/madengine/scripts/` is shipped as wheel data**, is excluded from mypy, and is
  git-ignored at the top level by design — `pyproject.toml` explains why. Do not suggest
  "fixing" the `.gitignore` or adding type annotations there.
- **`setup-page` is referenced in `README.md` and `docs/` but is not implemented in
  `src/madengine/`.** Do not treat calls to it as valid, and do not report the docs
  reference as a new defect.
- **There are no CI workflows in the engine repository.** `.github/workflows/` does not
  exist; the YAML under `examples/github-actions/` is sample config for consuming repos.
  Verify against the local gates (`pytest`, `pre-commit run --all-files`), and do not tell
  the author to "check CI".

**Model repo only**

- **`scripts/common/` is engine-owned.** It is populated at run time and deleted
  afterwards. Anything committed there is lost.
- **A model with no `tags`** can only be selected by exact name.
- **`--pull` is always passed to `docker build`**, so a base image tag that moves will
  change build results without any change in the repo.

---

## What not to comment on

Keep the signal high. Skip:

- Anything an automated formatter fixes (black, isort, flake8 in the engine repo;
  whatever the model repo configures).
- Praise, summaries of what the diff does, or restating the PR description.
- Style preferences that contradict the surrounding code.
- Pre-existing issues the PR merely moved or touched, unless the PR made them worse —
  and if you do raise one, label it explicitly as pre-existing and out of scope.
- Suggestions to add abstraction, configurability, or error handling that the change did
  not ask for. The repository instructions explicitly favor the minimum change; a review
  that pushes against that is working against the repo.

## Comment format

For each finding: the file and line, the category from the priority list, what breaks and
under which condition, and a concrete fix. Prefer a suggested diff over prose. If a
finding depends on an assumption you could not verify from the diff, say so rather than
asserting it.
