---
description: Build and run a madengine benchmark for the given tag/model
argument-hint: <tag-or-model> [extra options]
---

Benchmark `$ARGUMENTS` with madengine.

Use the `mad-benchmark-runner` subagent. It should:
0. Pre-flight: check madengine is installed and cwd is the MAD repo root.
   ```bash
   if ! command -v madengine &>/dev/null; then
     if [ -f requirements.txt ] && grep -q madengine requirements.txt; then
       echo "[pre-flight] madengine not found. Installing from requirements.txt..."
       pip install -r requirements.txt
     else
       echo "[pre-flight] madengine not found and requirements.txt is missing."
       echo "  Install:  pip install git+https://github.com/ROCm/madengine.git@main"
       echo "  Or clone MAD and run from its root (which has requirements.txt)."
       exit 1
     fi
   fi
   if [ ! -f models.json ]; then
     echo "[pre-flight] Warning: models.json not found — run from the MAD repo root."
   fi
   ```
1. Confirm the tag matches real models via `madengine discover --tags $1`.
2. Assemble the `madengine run --tags $1 --live-output` command (add `-o`,
   `--timeout`, profiling `--additional-context`, or `slurm`/`k8s` keys as the
   request implies), and list required env vars (e.g. `MAD_SECRETS_HFTOKEN`).
3. Check for GPUs first. If none, print the exact command(s) to run on a GPU
   host instead of executing. If GPUs exist, run it and report where `perf.csv`
   landed.
