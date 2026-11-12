---
name: mad-debug-nan
description: Debug a NaN, inf, or loss/gradient blow-up in a MAD PyTorch model with nan-doctor — capture the op that first produced it with madengine's nan_doctor tool, replay it without the model, and package a standalone reproducer; optionally root-cause it and propose a fix. Use when a MAD model NaNs and the user wants to find out why.
argument-hint: "<tag> [--nan-doctor-home DIR] [-c '<context json>']"
disable-model-invocation: true
context: fork
agent: mad-nan-doctor
allowed-tools: Bash(madengine discover *) Bash(rocm-smi *) Bash(amd-smi *) Bash(docker images *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Bash(bash .claude/skills/mad-common/nan-doctor-run.sh *) Read Write Grep Glob
---

Debug a NaN in `$ARGUMENTS`.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`.
`--nan-doctor-home DIR` names the nan-doctor checkout (default
`$HOME/nan-doctor`). `-c` passes a madengine additional context (JSON or a
Python dict) for every run, e.g. `docker_env_vars` or `slurm`; it must not
carry `tools`.

## Pre-flight
Claude Code runs this before the skill starts. Other agents: run it first from
the repo root and stop if it fails.
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. **nan-doctor.** The pinned commit is the `NAN_DOCTOR_SHA` constant in
   `.claude/workflows/mad-nan-hunt.js` (one place to change it). If the
   checkout is missing, show the clone command from that file and ask before
   cloning. If its `git rev-parse HEAD` differs from the pin, say so and ask
   whether to continue; never move or reset the user's checkout.
2. **Resolve.** `madengine discover --tags <tag> --json`: exactly one model,
   otherwise list the matches and ask. The target is SLURM when the context
   has a `slurm` key, local otherwise; K8s is not supported yet: stop and say so.
3. **Intake.** `<ARTIFACT_DIR>` = `<NAN_DOCTOR_HOME>/artifacts/mad-<date>-<model>`
   (suffix `-2`, `-3`, ... if it exists). Ask the user only for what MAD cannot
   tell, each with a default: run the clean NaN check first (default yes, 1
   run), capsule data mode (default shapes; values when the data is public or
   synthetic), root-cause after the reproducer (default ask after step 6),
   which groups to replay (default largest). Write `<ARTIFACT_DIR>/intake.json`
   (`source: "mad"`, `env: "madengine"`, `run` = the `madengine run --tags <tag> --live-output`
   command, `additional_context` = the context without `tools`, `world_size`
   from the model's `n_gpus` and the context, the answers as `preferences`) and
   validate it with nan-doctor's `intake.py`.
4. **Investigate.** Follow nan-doctor's `debug-nan` skill from step 0 with that
   intake (`step1-intake.md`, `env-madengine.md`), within the subagent
   conventions. It ends with a reproducer (step 6); steps 7 and 8 only as the
   user chose. A fix is proposed in the report, never applied to MAD or the image.
5. **Hand over.** Copy `report.md`, `ledger.md` and the reproducer tarball into
   `nan_hunts/<artifact-dir name>/` and print where everything is.

For an unattended run across the same steps (plan by default, `--execute` to
run), use the `mad-nan-hunt` workflow.

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-nan-doctor` subagent. In Cursor, delegate
  to the `mad-nan-doctor` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_nan_doctor` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-nan-doctor.md` first for the conventions it assumes.
