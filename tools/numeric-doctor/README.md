# Numeric Doctor

Numeric Doctor is a set of agent skills, plus the tools they call, for debugging numeric failures
in AI models on AMD GPUs. It covers:

* non-finite values (NaN, Inf);
* finite but incorrect values that affect model accuracy or convergence (magnitude spikes, wrong
  outputs).

Given the command that runs a model, it produces three things: a **mini-reproducer** (the op that
first produced the bad value, replayed without the model), a **root cause**, and a **fix or
workaround** -- the last two best effort, within a time budget.

Expected use:

* numerical issues in models or in system libraries (hipBLASLt, aiter, Triton, ...);
* PyTorch-based frameworks and models (other frameworks are refused with a warning);
* AMD Instinct GPUs, MI300X and newer;
* training and inference, on one GPU or many nodes.

Numeric Doctor is self-contained in this directory. It does not use madengine and needs no
`models.json` entry: you point it at the script that runs your model.

## 1. Get it

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD
claude          # or: codex, or open the folder in Cursor
```

There is nothing to install. The skills live in `tools/numeric-doctor/skills/` and are linked into
`.claude/skills/` (Claude Code, Cursor) and `.agents/skills/` (Codex CLI), so an agent started at
the MAD root sees them. Starting it in the package (`cd MAD/tools/numeric-doctor`, then `claude`)
works too, and paths may then be given relative to it (`examples/training/...`). The model itself runs in a container (docker, podman or apptainer) or a
Python venv, chosen by an execution profile:

| Profile (`tools/numeric-doctor/scripts/env/profiles/`) | For |
|---|---|
| `local-docker.env`, `local-podman.env`, `local-venv.env` | a machine that has the GPUs, no scheduler |
| `slurm-docker.env`, `slurm-apptainer.env` | a Slurm cluster; copy it and set your account and partition |

Numeric Doctor asks which environment to use at the start of a session and suggests one; you can
also name a profile in your request.

### 1.1 Validate with the provided examples

To check the install, run the examples against their expected outcomes. At the agent prompt:

```
> validate numeric doctor
```

or a subset, e.g. `> validate numeric doctor softmoe_train_nan`. The name `mad-numeric-doctor` works
in any prompt as well (`> validate mad-numeric-doctor`, `... with mad-numeric-doctor`). Each example's expected result card
is in `tools/numeric-doctor/validation/expected/`. The examples and the GPUs they need are listed in
`tools/numeric-doctor/examples/training/README.md` and
`tools/numeric-doctor/examples/inference/README.md`; examples that need hardware you do not have are
skipped.

## 2. Debug a job

Tell Numeric Doctor which job to debug. For the bundled Qwen3 MoE training example (8 gfx950 GPUs,
Slurm):

```
> debug tools/numeric-doctor/examples/training/qwen3_moe_gradient_nan_nohint with numeric doctor
```

On a single MI300X, try `tools/numeric-doctor/examples/training/softmoe_train_nan` instead. For your
own model, name the script that runs it: `> debug path/to/run.sh with numeric doctor`.

Numeric Doctor then works toward the three outputs. Everything the session writes goes under
`tools/numeric-doctor/artifacts/<date>-<tag>/` (for example
`artifacts/2026-11-24-qwen3_moe_gradient_nan_nohint/`), which git ignores.

1. **Mini-reproducer.** Capture the chain of ops that produced the bad value from the running
   model, replay it without the model, and package the replay as a tarball with its own README. It
   reports what was captured, whether the replay reproduces the failure, and where the tarball is.
2. **Root cause and fix.** From the reproducer, it looks for the root cause and a fix or workaround,
   tests the fix on the reproducer and on the full model, and packages it as a tarball. If it finds
   no root cause, or a cause but no fix, it says so.
3. **Wrap-up.** It writes `report.md` (the investigation report) and `result.json` (a result card),
   and prints the time spent on each step.

To call a skill by name instead: `/debug-numeric <script>` in Claude Code, `$debug-numeric <script>`
in Codex CLI; in Cursor, name the skill in your request. Validation is `validate-examples`.

**Codex CLI:** the jobs need the Docker socket and network access, which Codex's default sandbox
blocks. Approve those commands outside the sandbox, or start Codex with
`--sandbox danger-full-access` on a dedicated GPU host.

## Layout

| Path | What it holds |
|---|---|
| `skills/debug-numeric/` | the debugging skill: `SKILL.md` and one reference file per step |
| `skills/validate-examples/` | the validation skill |
| `scripts/` | capture, replay, packaging, root-cause and fix tools; `scripts/nd` runs every job |
| `scripts/env/` | the execution environment: runtimes, schedulers and site profiles |
| `examples/` | training and inference models that fail, with their run scripts |
| `validation/expected/` | the expected result card of each example |
| `tests/` | the tools' own tests |
| `mods/` | an optional Claude Code status panel for a running investigation |

## Development

GPU-free checks, from the MAD root:

```bash
python3 -m unittest discover -s tools/tests -p 'test_numeric_doctor.py'
python3 tools/validation/skill_validator.py
```

Keep this directory public-clean: no personal paths, cluster or partition names, job ids or
internal links. `tools/tests/test_numeric_doctor.py` checks for them; set
`NUMERIC_DOCTOR_DENYLIST` to a file of extra terms (one per line) to check terms that cannot be
listed in a public repo. Keep example READMEs and comments free of root causes: the validation runs
are blind.
