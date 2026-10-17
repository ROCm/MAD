---
name: mad-validate
description: Statically validate MAD model entries (no GPU) — JSON, paths, Dockerfile header, output contract. Use after adding/editing a model or before a GPU run to lint models.json.
argument-hint: [tag-or-model | all]
context: fork
agent: mad-perf-analyst
allowed-tools: Read Grep Glob Bash(python3 *)
---

Statically validate MAD model definitions for `$ARGUMENTS` (default: all). This is a
GPU-free lint of `models.json` and the files it points at — it does NOT build or run
anything.

## Task
Run the bundled checker (uses only the stdlib; do not install packages):

```!
python3 ${CLAUDE_SKILL_DIR}/scripts/validate.py "$ARGUMENTS"
```

(An empty argument is treated as `all`.)

The checker reports two severities:
- **Errors** (non-zero exit) break a run: invalid JSON, missing/duplicate `name`,
  missing Dockerfile or missing `# CONTEXT ... AMD` header, missing scripts path, or
  no output contract (neither a `performance:` line nor `multiple_results`).
- **Warnings** are missing MAD-convention metadata (`url`, `owner`,
  `training_precision`, `tags`, `n_gpus`) — madengine defaults these, so they do not
  fail the build.

Report a compact pass/fail summary. If anything fails, list each problem with the
model name and how to fix it. This is the right check to run after `/mad-add-model`
and before any GPU run.
