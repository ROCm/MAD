# The execution environment: `scripts/nd`

Read in step 1. Every GPU command of the skill goes through ONE entry point, `<REPO_ROOT>/scripts/nd`
(below: `nd`), over two independent layers chosen by a **profile**:

| layer | values | what it owns |
|---|---|---|
| `RUNTIME` | `docker` `podman` `apptainer` `native` | the model's image on a node: pull / load / build, mounts, uid, devices, env, graceful stop |
| `SCHED` | `slurm` `local` | getting GPUs: submit, state, stop, the job's log |

Site profiles: `scripts/env/profiles/{local-docker,local-podman,local-venv,slurm-docker,slurm-apptainer}.env`. A new
site is one new profile file (copy the closest; the knobs are documented in the header of
`scripts/env/runtime_<RUNTIME>.sh` and `sched_<SCHED>.sh`), never a new script.

## `<ENV>` -- the investigation's profile (step 1)

```bash
<REPO_ROOT>/scripts/nd profile <site> --art <ARTIFACT_DIR> IMAGE=<image from RUN_SCRIPT> [K=V ...]
export ND_ENV=<ARTIFACT_DIR>/env.sh        # = <ENV>; every later command uses it
nd info                                     # resolved layers, image, repo + artifact paths as the runtime sees them
```

Overlay on the site profile what the run script says: `RUNTIME` (its `docker run` / `apptainer
exec` / venv), `IMAGE` (or `SIF=<file>`, or `VENV=<dir>`), `IMAGE_TAR` (an image archive the node
may need to load), `APPTAINER_FLAGS` (the run script's own flags, the whole value: e.g.
`APPTAINER_FLAGS="--pid --cleanenv"` -- only those two from a run script's `apptainer exec`: `nd`
owns `--home`, `-B`/`--bind`, `--pwd` and the mounts, and `--no-mount tmp` breaks it, below),
`EXTRA_ENV` (what the run script sets by hand, as
space-separated `VAR=VALUE` pairs: a loader path, a cache dir). `nd` already sets `USER`,
`LOGNAME`, `TORCHINDUCTOR_CACHE_DIR`, `HIP_VISIBLE_DEVICES` (below) -- don't repeat them. The K=V
overrides are appended to `env.sh` and a later line wins over the site profile's; **check the
result in `nd info`** (it prints `APPTAINER_FLAGS` / `DOCKER_ARGS` / `PODMAN_ARGS`, `EXTRA_ENV`).
Ledger the `nd profile` line. Any variable set in the calling shell overrides the file for that
one command.

## Commands

| what | command |
|---|---|
| one command, this GPU node | `nd run [--gpus all\|<id>\|<id,id>] [--env K=V]... -- '<cmd>'` -- `--gpus` is a device list (default `0`), not a count: every GPU is `all`; `nd submit --gpus N` (below) is a count |
| one command as a job | `nd submit --name <n> [--gpus 8] [--time T] [--exclusive] [--step TAG] [--node H \| --prefer-node H] [--exclude H] -- exec.sh [--gpus ..] -- '<cmd>'` (`--node`: only there; `--prefer-node`: there while idle, else anywhere; `--exclude`: a node whose launch failed) -- without `--gpus`: 1 GPU where the site needs a request; without `--time`: the job script's `#SBATCH --time` (exec.sh 30 min, capture_to_repro 2.5 h, replay_ladder 2 h) -- pass one sized to the work |
| model wave (capture/control arms, early stop) | `SPEC=<file> ... nd submit --name wave --gpus 8 --exclusive -- wave.sh` |
| capture -> 5a -> package -> 5b (step 4) | `SPEC=<file> NAN_SIGNAL=... nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh` |
| the same on a multi-node run (node-count ladder, `step4-distributed.md`) | `SPEC=<one-line all spec> ... NODE_LADDER_TRIES=<n> nd submit --name c2r --gpus 8 --nodes <NODES> --exclusive --step step4 -- capture_to_repro.sh` (`--gpus` is per node; `--nodes` > 1 needs `SCHED=slurm`) |
| 5b ladder | `CAPSULE=<pt> OUT_DIR=<dir> nd submit --name ladder_<G> --gpus 8 --exclusive -- replay_ladder.sh` |
| 5d pressure sweep | `CAPSULE=<pt> OUT_DIR=<dir> VARIANTS=<file> nd submit --name sweep --gpus 8 --exclusive -- pressure_sweep.sh` |
| step-2 rate of a multi-node model (step3: `COPIES=k`) | `CMD='<model once>' NAN_SIGNAL=... NODES=<NODES> TRIALS=8 nd submit --name rate --gpus 8 --nodes <NODES x groups> --exclusive -- rate.sh` |
| step-2 differential wave (no signal) | `CMD='<model once, inputs pinned>' [FRAMEWORK=...] nd submit --name diff --gpus 8 --exclusive --step step2 -- diff_wave.sh` (`step2-diff.md`) |
| step-6 reproducer on other nodes | `PACKAGE=<dir> nd submit --name nodes --gpus 8 --nodes 2 --exclusive --exclude <5b host> -- verify_nodes.sh` (`step6-finish.md` item 4) |
| a server as a trial (any step) | spec command `bash $ND_REPO/scripts/serve_run.sh --server '<cmd with {port}>' --requests <jsonl>` (`step1-ask.md` *vLLM* / *SGLang*) |
| step-8 model A/B (multi-rank) | `CMD='<model once>' NAN_SIGNAL=... ARMS='V=1 ...' nd submit --name ab --gpus 8 [--nodes <CAPTURE_NODES>] --exclusive -- model_ab.sh` |
| job state / wait / stop | `nd status <n>`, `nd wait <n> [--max SEC]`, `nd stop <n>` -- `<n>`: a name, or the bare id a ledger `job` line records (a resume) |
| a package's `run.sh` / `validate.sh` as a vendor runs it (step 6/8 verification) | `PACKAGE=<extracted dir> [SCRIPT=validate.sh] [PACKAGE_ENV="IMAGE=..."] nd submit --name verify --gpus 8 --exclusive --step <TAG> -- package_run.sh` -- empty environment, from the package's dir; `PACKAGE_RC=` |
| a job sbatch'ed outside nd | `nd adopt <name> <id> --step <TAG>`, then `nd wait <name> --max 240` |
| a host path as the runtime sees it | `nd path <host path>` |

**Where a job lands** (docker / podman, one node, neither `--node` nor `--prefer-node`): `nd submit` pins
an idle node that already holds the image (`IMAGE_NODE <host>: ... pinned`; every successful `nd
run` / `nd prepare` records its node in `<ART>/jobs/image_nodes`), else any node, which pulls the image
first (~2 min for 20 GB: three fresh nodes cost 4 of 18 minutes on 2026-11-19). A pinned node taken by
another job before ours starts: after `ND_PIN_WAIT` (30 s) the job is resubmitted unpinned (`PINNED ...
resubmitted unpinned`) -- never queued behind it. `ND_IMAGE_NODES=0` turns the pinning off. Two jobs
submitted together get different nodes (a pick is claimed for 2 minutes). `--prefer-node <host>` right
after this user's own job there ended waits up to `ND_PREFER_WAIT` (90 s) for the node to free up
(it reads allocated/completing for a while; without the wait every step-7/8 job of 2026-11-20 went
elsewhere and re-loaded the image); held by another user's job: unpinned at once.

Every `nd` call reads the profile from `ND_ENV`. A shell that does not keep exports between
commands (an agent's tool calls) must prefix each call: `ND_ENV=<ARTIFACT_DIR>/env.sh nd status c2r`
-- without it `nd` stops with `no artifact dir`. `nd status` prints `STATE=PENDING|RUNNING|COMPLETING|DONE|
FAILED ...` (Slurm's own states pass through: `COMPLETING`, `CONFIGURING` are not terminal) and
returns 0 for a known job, whatever its state.

Job scripts live in `scripts/jobs/` (each header lists its knobs). `--step TAG` also writes
`ledger.py job TAG <id>`. A job's log is `<ARTIFACT_DIR>/jobs/<name>_<time>.out`; its record
`<ARTIFACT_DIR>/jobs/<name>.job`. A name is free again right after `nd stop` (the stopping job's
record moves to `<name>.<id>.job`). `exec.sh ... -- <command>`: one quoted string as always, or words --
then a word with spaces or shell characters stays one word (`--contend-with '<cmd>'`).

**Paths: write commands with `$ND_REPO` and `$ND_ART`, single-quoted.** Inside every runtime they
are the repo and the artifact dir as that runtime mounts them (host paths under docker/podman/
native, `/mnt/...` under apptainer, whose read-only SIF cannot mount a host path). `PYTHONPATH`
holds `$ND_REPO/scripts`, so tools run as `python3 -m <tool>` or `python3 $ND_REPO/scripts/<tool>.py`
(`--env PYTHONPATH=<dir>` and a profile's `EXTRA_ENV` PYTHONPATH are prepended to it, never replace it).
A path the model or a capsule recorded as the runtime saw it (`groups.json`) is mapped back by the
jobs. Every command also gets `NUMERIC_DOCTOR_ROOT`, `ND_JOB_ID` (the Slurm job id; `SLURM_*` is not passed), `USER`/`LOGNAME` (dynamo's `getpass.getuser()`),
its own `TORCHINDUCTOR_CACHE_DIR`, `HIP_VISIBLE_DEVICES` from `--gpus`, `MASTER_PORT` per
first GPU, `PYTHONUNBUFFERED=1`.

**Waiting** (SKILL.md rule 9): foreground `nd wait <n> --max 240`, repeated (exit 3 = still
running; `LOG_IDLE=<s>` = the log has been silent 10+ min: a quiet phase, or a stalled node -- `nd
stop`, resubmit `--exclude <node>`), keeps the prompt cache warm; only a job with > 40 min left gets one later check
instead. `SINGLE_SESSION=1` refuses an unbounded wait or `--max` over 300.

## Images and runtime traps: `env-runtime.md`

Read it when the preflight says an image must be pulled, built or loaded, when `RUNTIME=apptainer`,
for a new site profile, or when a command or job fails in a way that looks environmental (rc
137/139/143, a bare `PermissionError`, `device_count()` 0, an all-`ERROR` wave, a job's rc that
contradicts its log). docker/podman with a published image (preflight `nothing to do now`): not
needed -- `nd` pulls or loads it per node.

## Scheduler notes

* **slurm** (`sched_slurm.sh`): `SLURM_ARGS` (account, partition), `SLURM_GPUS_FLAG` (how N GPUs
  are requested, e.g. `--gpus-per-node={n}`; empty where nodes come whole; with it set, an
  `nd submit` without `--gpus` asks for 1 GPU); `nd submit` creates
  the log dir first (a missing `-o` dir loses the output silently) and adds
  `--signal=B:TERM@60` (an all-143 wave at the limit is a walltime kill; a running job's
  `TimeLimit` cannot be raised). Timing-sensitive 1-GPU jobs (sweeps, fix validation): `--exclusive`,
  or Slurm packs other jobs onto the node and shifts the race. **Don't plan on `sbatch --test-only`
  start times** (it once estimated 25 h for jobs that started in seconds): submit, read `nd status`.
  A job script must start with `#!` -- `nd submit` wraps one that does not. Some sites reject
  batch scripts that only `sleep` to hold a node: a test needs real work.
* **local** (`sched_local.sh`): a detached process group per job, its pid recorded; `nd stop` stops
  that group and every descendant group (torchrun `setsid`s its ranks), never a pattern `pkill`
 .

## Sites

* **Slurm + apptainer, no docker** (`slurm-apptainer.env`): where a login allows ONE SSH session
  (`SINGLE_SESSION=1`), no blocking `srun`/`salloc`, no background poll loop, no parallel agents
  issuing commands -- submit, then foreground `nd wait <n> --max 240` checks. A site that accepts
  no job without a GPU request needs `SLURM_GPUS_FLAG`. A project dir that is a symlink into
  Lustre: the runtime maps both spellings. A login node's `/tmp` may be RAM. Template capture
  launchers: `examples/training/*/capture.spec`.
* **Slurm + docker on the compute nodes** (`slurm-docker.env`): nodes may come whole (no GPU
  gres: leave `SLURM_GPUS_FLAG` empty); a login node may have no docker access.
* **One GPU host, no scheduler** (`local-docker.env`, `local-podman.env`, `local-venv.env`;
  `SCHED=local`: an unbounded `nd wait` works, `nd stop` TERMs the job script, then its process
  groups); a host venv runs with `RUNTIME=native` (a library path it needs, such as
  `HIPBLASLT_TENSILE_LIBPATH`, in `EXTRA_ENV`). Shared with other users: check the GPU is idle
  before a wave, stop only your own jobs.
* **Some GPUs are someone else's** (GPU 0 busy on a shared host): set `GPU_IDS=1,2,3,4,5,6,7` in
  the profile's env file (or the environment). The runtime then sees only those devices, and
  logical ids run over them in order: `--gpus 0` is the first listed GPU (physical 1), `nd info`
  prints `GPU_IDS=... (n devices)`, and `nd run --gpus <id>` refuses an id past the subset. Use
  it instead of remapping by hand in `EXTRA_ENV`. A capture spec still names logical ids 0..n-1,
  so trim it to the devices you have.

## Job hygiene (each cost a job once)

* Never name a helper script after a stdlib module (`inspect.py`, `json.py`, `types.py`): the
  script's directory comes first on `sys.path` and `import torch` dies in a circular import.
* `chmod 777` directories a container writes when its uid may differ (the jobs do it for theirs).
* Fixed container or job names break fan-out (`step3-raise.md`); `nd` names every run uniquely.
* Never extract a step-6 tarball under `/tmp` to verify it (a daemon with `PrivateTmp` sees an
  empty bind-mount).
