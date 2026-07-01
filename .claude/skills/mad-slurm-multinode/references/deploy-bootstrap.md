# Deploy bootstrap (Steps 0-4)

Keywords: clone MAD madengine, miniforge conda madenv, pip install -e,
rundir, source mad.env, idempotent setup, git switch, python 3.12, docker check

Detailed, idempotent bring-up of a fresh SLURM node. Run each step only if its
result is missing — re-running a completed step stays safe. Substitute
`$WORKDIR` with the user-provided working directory. `$SKILL_DIR` is the
absolute path to this skill's directory (the folder with `SKILL.md`); export it
once because the steps below `cd` into `$WORKDIR`, so bare `scripts/...` paths
would not resolve.

## Step 0 — Preflight

```bash
bash "$SKILL_DIR/scripts/preflight.sh"
```

Hard requirements (FAIL -> stop, this is not a valid target node):
- `docker` present and `docker info` succeeds (daemon reachable, user in group).
- A SLURM client: `sinfo` and `sbatch` on PATH.
- `git` (needed to clone/switch the repos in Step 1). `preflight.sh` treats a
  missing `git` as a hard FAIL.

Soft requirements (warn, can be installed in later steps):
- Python >= 3.10 (madengine needs 3.10+; we create a 3.12 conda env anyway).
- conda/miniforge (installed in Step 2 if absent).
- A GPU SMI (`rocm-smi` for AMD, `nvidia-smi` for NVIDIA) for arch detection.
- HF token at `~/.huggingface/token` (required at run time for gated Llama-3.1).

## Step 1 — Clone repos (idempotent)

```bash
cd "$WORKDIR"

# MAD source -> MODEL_DIR. Repo URL is confirmed with the user; clone if missing.
if [ ! -d MAD ]; then
  git clone <MAD_REPO_URL> --recursive
fi
# Branch is ASKED (no default). Switch, then sync submodules.
( cd MAD && git switch "<MAD_BRANCH>" && git submodule update --init --recursive )

# madengine. Repo URL is confirmed with the user; clone if missing.
if [ ! -d madengine ]; then
  git clone <MADENGINE_REPO_URL> --recursive
fi
# Branch is ASKED (no default).
( cd madengine && git switch "<MADENGINE_BRANCH>" && git submodule update --init --recursive )
```

Confirm the repo and branch with the user before cloning or switching. No branch
name is assumed (not a default, not one inferred from a prior session); the
branch is asked per repo and then `git switch`ed to. When the dirs already
exist, the skill shows the current branch
(`git -C <repo> branch --show-current`) and asks before changing it rather than
switching silently. An existing checkout is left intact, since the user may have
local edits.

After cloning or switching, the assets the chosen manifest references resolve
under the `MAD` checkout (this becomes `$MODEL_DIR`). Confirm the
manifest's `dockerfile` and the model `scripts`/`run.sh` exist:

```bash
# example for primus_llama-3.1-8b — adjust to the chosen manifest's fields
( cd MAD \
  && [ -f docker/primus_megatron_train_rccl_overlay.ubuntu.amd.Dockerfile ] \
  && [ -f scripts/primus_scaleout/megatron-lm/run.sh ] \
  || echo "STOP: manifest dockerfile/run.sh not found under MAD" )
```

A missing path stops the run with a report rather than a silent search. The
usual causes are a wrong branch, uninitialized submodules, or a different repo
layout, so the next step is to ask the user for the correct branch.

## Step 2 — conda env (idempotent)

```bash
# an existing conda/miniforge may just be missing from PATH — source it first
if ! command -v conda >/dev/null 2>&1; then
  for cp in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" \
            "/opt/conda" "${CONDA_PREFIX:-}" "${MAMBA_ROOT_PREFIX:-}"; do
    if [ -n "$cp" ] && [ -f "$cp/etc/profile.d/conda.sh" ]; then
      source "$cp/etc/profile.d/conda.sh"; break
    fi
  done
fi

# install miniforge only if conda is still missing
if ! command -v conda >/dev/null 2>&1; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

conda env list | grep -q '^madenv ' || conda create -y -n madenv python=3.12
conda activate madenv
```

Notes:
- An already-installed conda/miniforge that is simply off PATH gets sourced from
  its common install prefix (`$HOME/miniforge3`, `$HOME/miniconda3`,
  `$HOME/mambaforge`, `/opt/conda`, `$CONDA_PREFIX`, `$MAMBA_ROOT_PREFIX`)
  rather than reinstalled, which avoids duplicate installs. `preflight.sh`
  reports the same find.
- `-b` runs the miniforge installer unattended; `-p` sets the prefix.
- If conda was just installed in this shell, `source .../conda.sh` (or open a
  new shell) before `conda activate`.

## Step 3 — Install madengine

```bash
cd "$WORKDIR"
pip install -e ./madengine
madengine --help >/dev/null && echo "madengine OK"
```

Editable install so the user's branch changes are picked up without reinstall.

## Step 4 — rundir + mad.env

```bash
cd "$WORKDIR"
mkdir -p rundir && cd rundir

# copy the archetype template (see cluster-types.md to choose):
#   CX7/Mellanox-RoCE  -> mad.env.cx7-roce.template
#   AMD-AINIC/Pollara  -> mad.env.amd-ainic.template
cp "$SKILL_DIR/assets/mad.env/mad.env.<archetype>.template" ./mad.env
# resolve every <FILL_...>, confirm node-specific values, then:
source mad.env
```

After `source mad.env`, sanity-check the critical exports in the SAME shell:

```bash
echo "MODEL_DIR=$MODEL_DIR"
echo "MAD_DOCKER_BUILDS=$MAD_DOCKER_BUILDS"
echo "MAD_SYSTEM_GPU_ARCHITECTURE=$MAD_SYSTEM_GPU_ARCHITECTURE"
[ -d "$MODEL_DIR/scripts" ] || echo "WARNING: MODEL_DIR/scripts missing — run.sh will not be found"
```

The HF token file (`~/.huggingface/token`) exists and is valid — Llama-3.1
is gated, so a missing file is created before sourcing.
