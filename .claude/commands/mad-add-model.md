---
description: Scaffold a new MAD model (models.json entry + Dockerfile + run.sh with the performance line)
argument-hint: <framework_project_workload> [base notes / repo url]
---

Add a new model to MAD named `$1`. Extra context: $ARGUMENTS

Use the `mad-model-author` subagent. It should:
1. Pick the closest existing model of the same framework as a template.
2. Create the `models.json` entry, `docker/$1.ubuntu.amd.Dockerfile` (with the
   `# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}` header), and
   `scripts/<dir>/run.sh` ending in `echo "performance: $performance <unit>"`.
3. Validate `models.json` with `python3 -m json.tool models.json`.
4. Confirm the entry is selectable with `madengine discover --tags $1` (GPU-free).

Report the files created and the verification command
`madengine run --tags $1 --live-output` (requires a GPU host).
