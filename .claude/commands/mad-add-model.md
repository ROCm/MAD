---
description: Scaffold a new MAD model (models.json entry + Dockerfile + run.sh with the performance line)
argument-hint: <framework_project_workload> [base notes / repo url]
---

Add a new model to MAD named `$1`. Extra context: $ARGUMENTS

Use the `mad-model-author` subagent. It should:
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
1. Pick the closest existing model of the same framework as a template.
2. Create the `models.json` entry, `docker/$1.ubuntu.amd.Dockerfile` (with the
   `# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}` header), and
   `scripts/<dir>/run.sh` ending in `echo "performance: $performance <unit>"`.
3. Validate `models.json` with `python3 -m json.tool models.json`.
4. Confirm the entry is selectable with `madengine discover --tags $1` (GPU-free).

Report the files created and the verification command
`madengine run --tags $1 --live-output` (requires a GPU host).
