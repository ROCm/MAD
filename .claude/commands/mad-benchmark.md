---
description: Build and run a madengine benchmark for the given tag/model
argument-hint: <tag-or-model> [extra options]
---

Benchmark `$ARGUMENTS` with madengine.

Use the `mad-benchmark-runner` subagent. It should:
1. Confirm the tag matches real models via `madengine discover --tags $1`.
2. Assemble the `madengine run --tags $1 --live-output` command (add `-o`,
   `--timeout`, profiling `--additional-context`, or `slurm`/`k8s` keys as the
   request implies), and list required env vars (e.g. `MAD_SECRETS_HFTOKEN`).
3. Check for GPUs first. If none, print the exact command(s) to run on a GPU
   host instead of executing. If GPUs exist, run it and report where `perf.csv`
   landed.
