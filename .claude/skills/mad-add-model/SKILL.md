---
name: mad-add-model
description: Scaffold a new MAD model (models.json entry + Dockerfile + run.sh with the performance line). Use when the user wants to add a new model or workload to MAD.
argument-hint: <framework_project_workload> [base notes / repo url]
disable-model-invocation: true
context: fork
agent: mad-model-author
allowed-tools: Bash(madengine discover *) Bash(python3 -m json.tool *) Bash(bash .claude/skills/mad-common/preflight.sh) Read Write Edit Grep Glob
---

Add a new model to MAD named `$0`. Extra context: $ARGUMENTS

## Pre-flight
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. Pick the CLOSEST existing model of the same framework (vLLM, PyTorch/Primus, JAX
   MaxText, xDiT, SGLang, Megatron, HuggingFace, ...) as a template — read its
   `models.json` entry, its `docker/<name>.ubuntu.amd.Dockerfile`, and its
   `scripts/.../run.sh`. Reuse those patterns; do not invent new ones.
2. Produce three artifacts:
   a. A `models.json` entry. Required: `name`, `url`, `dockerfile`, `scripts`,
      `n_gpus`, `owner`, `training_precision`, `tags`. Name = `{framework}_{project}_{workload}`.
      Keep `models.json` valid JSON.
   b. `docker/$0.ubuntu.amd.Dockerfile` whose first line is exactly
      `# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}`. Prefer pointing the
      `dockerfile` field at an existing same-stack Dockerfile over a near-duplicate.
   c. `scripts/<dir>/run.sh` satisfying ONE output contract:
      - single result: end with `echo "performance: $performance <unit>"`, parsed
        from the workload's log output, OR
      - multiple results: have the script WRITE its own CSV and set
        `"multiple_results": "<that-file>.csv"` in the entry. Never mix the two.
3. Validate: `python3 -m json.tool models.json` parses.
4. Confirm selectable (GPU-free): `madengine discover --tags $0` lists it.

Do NOT run `madengine run` (needs GPUs). Report the three file paths created and the
chosen template, then state the verification command for a GPU host:
`madengine run --tags $0 --live-output`. After this, run `/mad-validate $0`.
