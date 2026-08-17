---
name: mad-slurm-multinode
description: Deploy and run madengine performance tests on an unprepared SLURM cluster from scratch. Use when the user wants to perf-test a model via a madengine run engine (any engine that has a template under assets/manifests/), set up a fresh node (clone MAD + madengine, conda/miniforge env, pip install), build a mad.env, pick/adapt a run manifest, and launch a multi-node SLURM run. Covers the cluster archetypes documented in references/cluster-types.md (such as CX7/Mellanox-RoCE, AMD-AINIC/Pollara, and Broadcom-Thor2-RoCE), required interactive inputs (node host, work dir, data dir, MAD_DOCKER_BUILDS, SLURM partition/account/qos/reservation/nodelist), and the cluster-specific communication-backend env vars (RCCL/NCCL over the right transport, e.g. RDMA) that decide whether the right transport is exercised.
compatibility: Requires git, docker, conda/miniforge, a SLURM CLI (sbatch/sinfo), rocm-smi or nvidia-smi, and internet access for cloning and image pulls.
metadata:
  author: mkuznet1 (mikhail.kuznetsov@amd.com)
  version: "0.1"
---

# mad-slurm-multinode

Bring up madengine on a fresh SLURM node and run a Llama-3.1 perf test
end to end: prerequisites -> clone -> conda env -> install -> mad.env ->
manifest -> `madengine run`. Covers the cluster archetypes documented in
[references/cluster-types.md](references/cluster-types.md).

> **Skill paths — read first.** When this skill activates you are given the
> absolute path to its directory (the folder containing this `SKILL.md`). All
> `scripts/`, `references/`, and `assets/` paths below are relative to THAT
> directory, not to `$WORKDIR`. The workflow `cd`s into `$WORKDIR`, so set the
> skill dir once and reference files through it:
>
> ```bash
> export SKILL_DIR="<absolute path to this skill's directory>"
> ```
>
> Then use `"$SKILL_DIR/scripts/..."`, `"$SKILL_DIR/assets/..."`, etc. Markdown
> links like `references/foo.md` are also relative to `$SKILL_DIR`.

## When this fires

- "Run a perf test with madengine on `<node>`" / "set up madengine on this cluster".
- A Llama-3.1 perf workload that has a template under `assets/manifests/`, on SLURM.
- A node that has nothing installed yet (no conda, no repos, no rundir).

Out of scope:
- RCCL build-vs-build validation delta campaigns (build image A vs B and
  compare the transport) — that is a separate workflow, out of scope here.
- Accumulating many iterations across multiple SLURM allocations (large
  multi-allocation campaigns) — out of scope here.
- Single-GPU local (non-SLURM) smoke tests -> just `madengine run --tags ...` directly.

## Responsibilities — who does what

The skill stays in its lane and follows the steps rather than re-deriving the
workload. The split is:

- **mad-slurm-multinode (this skill)** configures one run: brings up the node
  (repos, conda env), writes `mad.env` and the manifest (paths, SLURM selectors,
  transport vars), launches `madengine run`, and reads the perf CSV.
- **madengine** orchestrates: renders the sbatch script from the manifest,
  ensures the docker image per node (load tar / build / pull), runs the
  multi-node job, and aggregates results.
- **MAD** holds the Dockerfiles and the per-model `run.sh`. Inside the
  container, `run.sh` runs the actual workload **and acquires/prepares the
  dataset and model weights** (data prep, model download, etc.). The
  skill points it at the right paths and launches; staging datasets or
  downloading models by hand is `run.sh`'s job, not a pre-step the skill adds.

**Do not fix madengine or MAD code.** When a failure is rooted in the madengine
orchestrator or the MAD repo (a Dockerfile, `run.sh`, model code, an
orchestration bug), the skill reports it to the user with the evidence and
stops — it does not patch those repos unless the user explicitly asks. The
skill's own edits stay confined to `rundir/` (`mad.env`, the manifest) and the
user-provided inputs. A known per-workload workaround that only sets env/manifest
values (see [references/gotchas.md](references/gotchas.md)) is applied in the
manifest, not by editing madengine/MAD.

## Required inputs — ask before exploring

The first action of a run, before any other tool call (Shell, Read,
transcript/past-session search, or web search), is to list the required inputs
that are missing or not clearly identifiable and resolve them in a single
`AskQuestion` round. A missing or unrecognized input is a question to the user,
not a research task — inferring it from prior sessions, `agent-transcripts`,
repo history, or the web is a known failure mode of this skill and is skipped.
Tool calls begin once the required inputs are in hand.

These come from the user, and the skill asks for them rather than assuming.
The first four have no safe defaults, so the skill asks; a value already in the
conversation gets reused.

1. **Compute node hostname or IP** to run on (e.g. the login/jump node you SSH
   to) — where the bootstrap, build, and `sbatch` submission happen.
2. **Working directory** `$WORKDIR` — the root that holds the cloned repos
   (`MAD`, `madengine`), the `rundir/` (the filled `mad.env`, the
   filled manifest, `slurm_output/`, and run logs), and the perf-result CSVs.
   Picking it explicitly keeps one run's artifacts together and reusable.
3. **Cluster archetype** — one of the archetypes in
   [references/cluster-types.md](references/cluster-types.md). You may run
   `scripts/detect_cluster_env.sh` to *propose* it, but confirm with the user.
4. **Data root** (`MAD_DATAHOME` + caches) and **`MAD_DOCKER_BUILDS` dir** —
   state what each holds so the user can point them at the right place:
   - the **data root** holds datasets, tokenizer, and model weights, plus the
     `HF_HOME`/`TORCH_HOME`/pip caches the run reads and writes;
   - **`MAD_DOCKER_BUILDS`** is the shared image-tar cache: rank 0 saves the
     built image there and every worker loads the same tar, so it lives on
     shared FS visible to every node (see Gotchas).
5. **SLURM specifics, on demand**: `partition`, `account`, `qos`,
   `reservation`, `nodelist`/`exclude`, node count. These are cluster-private
   and are not stored in this skill — requested per run.
6. **Branch per repo** for `MAD` and `madengine`. No branch is
   assumed — the skill asks which branch to use for each repo and `git switch`es
   to it (see "Repo + branch rule").

The HF token file `~/.huggingface/token` provides gated Llama-3.1 access;
`mad.env` reads it into `MAD_SECRETS_HFTOKEN` and madengine forwards it to the
container (see [references/manifests.md](references/manifests.md) "Secrets"), so
confirm it exists before launch.

**Probe one node before asking about cluster shape.** The cluster-shape values
(GPU arch, HCA list, GID index, management iface) are discoverable on the node
itself, so the skill allocates one node from the user-provided partition and
runs the probe there rather than interrogating the user:

```bash
srun -p <partition> [--reservation <res>] [--nodelist <node>] -N1 \
  bash "$SKILL_DIR/scripts/detect_cluster_env.sh"
```

Only the cluster-private selectors (partition / account / qos / reservation /
nodelist) come from the user; the probe reports the rest. The same holds for
`scripts/preflight.sh`, which reflects whichever node it runs on — for
compute-node values it runs through `srun` on an allocated node.

Defaults you MAY assume unless told otherwise (state them when you do):
- conda env name `madenv`, Python 3.12.

**Repo + branch rule (every run):** the repo and the branch for both
`MAD` and `madengine` are confirmed with the user before cloning or
switching. No branch name is assumed — not a default, not one inferred from a
prior session — the branch is asked and then `git switch`ed to. An existing
checkout is left as-is rather than switched silently: when the directory already
exists, the skill shows its current branch
(`git -C <repo> branch --show-current`) and asks before changing it.

## Workflow

Track progress with this checklist:

- [ ] Step 0: preflight (`scripts/preflight.sh`)
- [ ] Step 1: clone repos (idempotent)
- [ ] Step 2: conda env (idempotent)
- [ ] Step 3: `pip install -e ./madengine`
- [ ] Step 4: rundir + mad.env (filled + sourced)
- [ ] Step 5: pick + adapt manifest
- [ ] Step 6: launch + collect results

Full step detail (commands, idempotency guards) is in
[references/deploy-bootstrap.md](references/deploy-bootstrap.md). The short
version:

Each step below ends with a **Guard** — the enumerated conditions are the only
branches to consider. If a guard matches, take its action; if none match, the
step passed and the next step follows. This keeps the path deterministic and
avoids re-analyzing a step that already succeeded.

### Run logs

Every long command (`madengine run`, docker build, conda/pip install) streams
its stdout+stderr to `<rundir>/.cursor.logs/<kind>/<UTC-timestamp>-<slug>.log`
via `tee`, which keeps a run reproducible and inspectable after the fact
(several bugs here surfaced only in these logs). Kinds map to subdirectories
(`.cursor.logs/build/`, `.cursor.logs/run/`, `.cursor.logs/install/`); a
filename carries a UTC timestamp and a short slug and holds no secrets. Example:

```bash
mkdir -p "$WORKDIR/rundir/.cursor.logs/run"
madengine run --manifest-file run_manifest_<workload>.json --live-output \
  -o perf_<workload>.csv \
  2>&1 | tee "$WORKDIR/rundir/.cursor.logs/run/$(date -u +%Y%m%dT%H%M%SZ)-primus-8b.log"
```

The log files are `*.log`, which `.gitignore` already ignores, so the
`.cursor.logs/` contents never land in git.

### Step 0 — Preflight

```bash
bash "$SKILL_DIR/scripts/preflight.sh"
```

Checks Python >= 3.10, docker (present + daemon reachable), conda/miniforge,
SLURM CLI (`sinfo`/`sbatch`), and a GPU SMI (`rocm-smi`/`nvidia-smi`). Fix
any FAIL before continuing. If docker or SLURM is missing this is not a valid
target node — stop and tell the user.

**Guard:**
- **If Step 0 reports any FAIL** → stop, report which check failed, do not continue.
- **If docker or SLURM is missing** → this node is not a valid target; tell the user and stop.

### Steps 1-3 — Bootstrap (only do what's missing)

```bash
cd "$WORKDIR"
# Repo URLs are confirmed with the user; clone only if missing.
[ -d MAD ] || git clone <MAD_REPO_URL> --recursive
[ -d madengine ]   || git clone <MADENGINE_REPO_URL> --recursive
# Branch is ASKED per repo (no default). For an existing checkout, show the
# current branch first and ask before switching:
#   git -C MAD branch --show-current
#   git -C madengine    branch --show-current
( cd MAD && git switch "<MAD_BRANCH>" && git submodule update --init --recursive )
( cd madengine    && git switch "<MADENGINE_BRANCH>"    && git submodule update --init --recursive )

# conda: install miniforge only if `conda` is absent (see deploy-bootstrap.md)
conda env list | grep -q '^madenv ' || conda create -y -n madenv python=3.12
conda activate madenv
pip install -e ./madengine
```

After cloning or switching, the chosen manifest's `dockerfile` and the model
`scripts`/`run.sh` resolve under the `MAD` checkout (later
`$MODEL_DIR`). A missing path stops the run with a report (likely a wrong
branch, uninitialized submodules, or a different layout) rather than a silent
search — ask the user for the correct branch.

**Guard:**
- **If a clone, `git switch`, or submodule init fails** → stop, report, ask the user for the correct repo/branch; do not guess another branch.
- **If `pip install -e ./madengine` fails** → stop and report; a madengine bug is not patched here (see Responsibilities).
- **If the manifest's `dockerfile`/`run.sh` do not resolve under `$MODEL_DIR`** → stop and report (likely wrong branch or uninitialized submodules); ask the user.

### Step 4 — rundir + mad.env

```bash
cd "$WORKDIR" && mkdir -p rundir && cd rundir
# copy the mad.env template for the archetype (one file per archetype under
# assets/mad.env/; see references/cluster-types.md). The template stays unedited:
cp "$SKILL_DIR/assets/mad.env/mad.env.<archetype>.template" ./mad.env
# FILL every <FILL_...> placeholder, then:
source mad.env
```

Every `<FILL_...>` placeholder is resolved before `source`. The cluster-specific
values (`MAD_SYSTEM_GPU_ARCHITECTURE`, `NCCL_SOCKET_IFNAME`,
`NCCL_IB_GID_INDEX`, and the per-manifest `NCCL_IB_HCA` list) are confirmed
against the actual node — defaults in the template are archetype-typical, not
guaranteed.
Run `bash "$SKILL_DIR/scripts/detect_cluster_env.sh"` to propose them and use
the matrix in [references/cluster-types.md](references/cluster-types.md) to
interpret.

**Guard:**
- **If a `<FILL_...>` value is unknown** → ask the user; do not infer it from a prior session, the repo, or the web.
- **If `source mad.env` errors, or `MODEL_DIR`/`MAD_DOCKER_BUILDS` come back empty** → stop and report before continuing.

### Step 5 — Manifest

Copy the matching template from `assets/manifests/` into `rundir/`, then adapt.
That directory holds one `*.template.json` per workload/size; pick the one named
for the requested workload.

```bash
cp "$SKILL_DIR/assets/manifests/<workload>.template.json" run_manifest_<workload>.json
# fill the manifest, then statically validate it (GPU-free; reads $MODEL_DIR):
bash "$SKILL_DIR/scripts/validate_manifest.sh" run_manifest_<workload>.json
```

`validate_manifest.sh` is a read-only static check: JSON validity, leftover
`<FILL_...>` placeholders, `NCCL_IB_HCA` set and equal in both env blocks (skipped
for single-node manifests, which carry no transport vars), network-interface
consistency, `slurm.nodes == distributed.nnodes` (+ nodelist
cardinality), no stray `MAD_SECRETS_HFTOKEN`, AINIC transport-var symmetry, and
(with `mad.env` sourced) that the `dockerfile`/`run.sh` resolve under
`$MODEL_DIR`. It exits non-zero on any FAIL.

What the run sets per run (cluster-private, kept out of the templates):
`deployment_config.slurm.{partition,account,qos,reservation,nodelist,exclude,nodes}`,
node count consistency (`slurm.nodes` == `distributed.nnodes`), the
`NCCL_IB_HCA` device list, and host paths. Field-by-field guidance:
[references/manifests.md](references/manifests.md).

The manifest's `dockerfile` and the model `scripts`/`run.sh` resolve under
`$MODEL_DIR` (`[ -f "$MODEL_DIR/<dockerfile>" ] && [ -f "$MODEL_DIR/<scripts>" ]`).
A missing path stops the run with a report rather than a silent search.

**Guard:**
- **If no template matches the requested engine** → ask the user; do not invent a manifest.
- **If `scripts/validate_manifest.sh` reports any FAIL** → fix it before launching (it covers JSON validity, leftover placeholders, `NCCL_IB_HCA`, iface consistency, `nodes == nnodes`, a stray HF token, and dockerfile/run.sh resolution under `$MODEL_DIR`).

### Step 6 — Launch + results

```bash
cd "$WORKDIR/rundir"
source mad.env
# -o writes the aggregated perf to a per-run CSV (instead of the default
# perf.csv), so each workload keeps its own result and parallel runs never
# clobber a shared file:
madengine run --manifest-file run_manifest_<workload>.json --live-output \
  -o perf_<workload>.csv
```

Pre-building or pulling the image is unnecessary. With `local_image: true`,
madengine ensures the image itself on each node: if it is not already present
locally it loads it from the `MAD_DOCKER_BUILDS` tar, otherwise builds it from
the manifest's `dockerfile` (and falls back to `docker pull`), then `docker
save`s it to `MAD_DOCKER_BUILDS` so workers load the same tar. Implications:
the first run is slower (it builds once on rank 0), `MAD_DOCKER_BUILDS` lives on
shared FS, and the manifest's `dockerfile` resolves (it lives in the cloned
`MAD`) or the `docker_image` tag is pullable. Details:
[references/launch-and-results.md](references/launch-and-results.md).

Perf lands in the `-o` file (`perf_<workload>.csv`) and the per-model
`multiple_results` CSV. Multi-node aggregation and how to read the result are in
the same file.

**Guard:**
- **If `madengine run` fails** → read the captured `.cursor.logs/run/...` log and triage with [references/launch-and-results.md](references/launch-and-results.md).
- **If the cause is config/input the skill owns** (mad.env, manifest, paths, transport vars) → fix it in `rundir/` and re-run.
- **If the cause is madengine or MAD code** → stop and report to the user; do not patch those repos unless asked (see Responsibilities).

## Gotchas

Cross-cutting and per-workload pitfalls — mad.env sourcing, `MAD_DOCKER_BUILDS`
on shared FS, HF-token handling, `docker_mounts` direction, `NCCL_IB_HCA`
per-cluster, AINIC transport vars, perf-CSV aggregation, and per-workload notes
(sglang_disagg and primus_megatron training) — are in
[references/gotchas.md](references/gotchas.md), read before a run.

## Examples

Sanitized end-to-end walkthroughs live in [examples/](examples/); the run
templates they build on live in `assets/` (`assets/mad.env/`,
`assets/manifests/`). New workloads and clusters are added there, not inlined
here.

<!-- Authoring note: this skill is written declaratively (statements of how it
behaves) rather than imperatively, which reads more uniformly across models and
languages. New content follows the same convention. -->

