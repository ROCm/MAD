# The execution environment: images and runtime traps

Read when (`env.md` points here): the preflight says an image must be pulled, built or loaded
(`IMAGE ... MISSING`, `BUILDS its image`, an archive), `RUNTIME=apptainer`, a new site profile,
or a command or job fails in a way that looks environmental (rc 137/139/143, a bare
`PermissionError`, `device_count()` 0, an all-`ERROR` wave, `Too many open files`, `LLVM ERROR:
IO failure`, a job's rc that contradicts its log).

## Images (step 1, before any job)

`capture_preflight.py <RUN_SCRIPT>` (with `ND_ENV` set) says which case applies:

| the run script | what to do |
|---|---|
| runs a published image (`IMAGE=${IMAGE:-<ref>}`, `docker run <ref>`, `docker://<ref>`) | `IMAGE=<ref>` in the profile. docker/podman: nd pulls it (`DOCKER_PULL=1`) or loads `IMAGE_TAR=` once per node -- nothing to do now. apptainer: the SIF must exist (below). |
| loads an archive (`docker load`, a `.tar`/`.tar.zst`) | `IMAGE=<the tag inside> IMAGE_TAR=<archive>`. |
| **builds** its image (`Dockerfile`, `docker build`, an apptainer `.def`) | build it ONCE, then `IMAGE=<your tag> IMAGE_TAR=<saved archive>`. If the build only adds `ENV` lines and copies the trainer, use its base image instead: nd sets the env (`EXTRA_ENV`) and mounts the repo and artifact dir. |
| apptainer-only (a `.sif`, `apptainer exec`) on a docker site | the `docker://` image it pulls, or translate its `.def` to a Dockerfile (`From:` -> `FROM`, `%post` -> `RUN`, `%environment` -> `ENV`) and build as above. Also translate the launcher: its `apptainer exec` line becomes a spec/`nd run` command. |

**Building on a docker / podman site** (in a job: login nodes may have no docker) --
`nd profile ... IMAGE=<your tag>` first, then, **without waiting for an answer** (step 1):
```bash
ND_ENV=<ART>/env.sh BUILD_DIR=<dir with the Dockerfile> [BUILD_ARGS='--build-arg BASE=...'] \
  scripts/nd submit --name build --gpus 1 --time 01:00:00 --step step1 -- build_image.sh
```
Step 1 usually ends while it builds (`ledger.py end` warns `still running`): record it again
under the step that waits on it -- `ledger.py <LEDGER> job step4 <id>` right after `start step4`
-- or the rest of the build is in no step's row.
`build_image.sh` builds with an empty `DOCKER_CONFIG` (a stored, expired registry login fails even a
public base pull: `401 personal access token is expired`) and `--network=host` (some daemons have no
default bridge), records the node in `jobs/image_nodes` (the next `nd submit` pins it while idle: no
load at all), saves `<artifacts>/image-cache/<key>/<tag>.tar.zst` (zstd -T0; ~4 min for 14 GB) and
appends `IMAGE_TAR` and `IMAGE_ID` (the build) to the profile: a node whose tag holds another build
reloads `IMAGE_TAR`, and only nodes recorded with this ID are pinned. **The build is cached across
investigations** by `<key>`, a hash of the Dockerfile, what its `COPY`/`ADD` name, `BUILD_ARGS` and
the tag: the same key again is a cache hit -- no build, no save (a second investigation's build job:
<1 s instead of 3.5-11 min), the earlier `IMAGE_ID`/`IMAGE_TAR` go into the profile, and the nodes
already holding that build are pinned. A changed byte is a new key and a new build.
`ND_IMAGE_CACHE=0` builds anyway; `ND_IMAGE_CACHE_DIR` moves the cache. Last line
`BUILD_RESULT=OK image=.. id=.. node=.. cached=0|1 key=..`: ledger it with the archive's sha256 (the
job prints it on a miss). It also keeps the build context (Dockerfile, `.dockerignore`, what its
`COPY`/`ADD` lines name) in `<ART>/image/build` and appends `IMAGE_BUILD_DIR`: the capture-to-reproducer
job ships it as the package's `image/`, and the package's `run.sh` builds the image from it where it
can be neither loaded (this site's `IMAGE_TAR` path) nor pulled -- off-site. A step-8 `fix/` package
with its own `run.sh` copies `image/` the same way.

**apptainer**: `SIF=<file>`, else `$SIF_DIR/<IMAGE with / : @ -> _>.sif`; missing -> built from
`IMAGE_TAR` by the first job (locked; ~10 min for 35 GB) or `apptainer pull docker://IMAGE` in a
job (~35 min for ~25 GB), or `apptainer build <sif> <file>.def` for a run script that builds.

## Runtime notes (the traps the layer handles; know them when something looks wrong)

* **docker** (`runtime_docker.sh`): `--init` so a stop's SIGTERM reaches the trainer (as PID 1 it
  ignores it; the capture's write-at-SIGTERM never runs); host uid + the `/dev/kfd` group
  (`DOCKER_AS_USER=1`) so NFS files stay yours; a unique name per run and `rm -f` on exit (a
  container is the daemon's child: `scancel` leaves it pinning HBM); the image pulled anonymously
  (`DOCKER_PULL=1`; an expired stored login fails even a public pull) or loaded once per node from
  `IMAGE_TAR` under a lock; the image's `ENTRYPOINT` cleared when it is a program, not a
  shell wrapper (`DOCKER_ENTRYPOINT=auto|keep|clear`; vLLM's `["vllm", "serve"]` took nd's
  `bash -c <cmd>` as server arguments; podman: `PODMAN_ENTRYPOINT`); `--ulimit nofile=524288` (docker's default 1024 kills an 8-rank
  NCCL job: `Too many open files`, `ncclSystemError`). Who else is on the node: `docker ps`. All-`ERROR` wave: `docker events
  --since <t> --filter event=die` (`exitCode=137` = an external kill). Mounts: the repo, the
  artifact dir, `$HOME` (a real one), `EXTRA_MOUNTS`; any other path is container-local and fails
  as a bare `PermissionError`.
* **podman** (`runtime_podman.sh`): rootless, `--userns=keep-id --group-add keep-groups`; without
  `/etc/subuid` ranges set `PODMAN_ARGS` to node-local `--root/--runroot` and
  `overlay.ignore_chown_errors=true`.
* **apptainer** (`runtime_apptainer.sh`): never bind `/dev/kfd` or `/dev/dri` (rootless `nodev`
  makes them unopenable: `device_count()` 0; the default `mount dev` exposes them); `PROJ` bound at
  `CONTAINER_PROJ` (`/mnt`), the repo and artifact dir must be under `PROJ`; `--home` and caches
  node-local (an NFS `flock` left by a killed process wedges imports); `LD_PRELOAD` blanked (site
  XALT injection). **`APPTAINER_FLAGS` carries the run script's own flags**: `--pid` for a capture
  (else `_gpu_procs()` counts every process on the node); `--cleanenv` when the run script
  uses it, with `EXTRA_ENV` holding what it set by hand -- without it the host environment leaks
  in and every copy can die at start-up with `LLVM ERROR: IO failure on output stream: Bad
  address` and no traceback. Never `--rocm` (host hipBLASLt over the image's), never
  `--no-mount tmp` (the run script's own `--home <dir>:/tmp` trick): `nd` puts `--home` under the
  node's `/tmp`, and with `tmp` unmounted Python segfaults at start-up (rc 139). The SIF:
  `SIF=<file>`, else `$SIF_DIR/<IMAGE with / : @ -> _>.sif`; missing -> built once from `IMAGE_TAR`
  (no `zstd` binary needed) or `apptainer pull docker://IMAGE` (~35 min for a ~25 GB image; the
  login node's `/tmp` may be RAM -- pull in a job, or set `APPTAINER_CACHEDIR`/`TMPDIR` under
  the artifact dir). Ledger file, size, digest. No `/dev/kfd` on the node -> refused (a login node).
* **native** (`runtime_native.sh`): `VENV=<dir>` (or `CONDA_ENV`), no mounts; the GPU check is
  `rocm-smi --showpids`. A package built from a venv run runs only where an equivalent venv exists.
* **Exit codes (any runtime): read the markers, not the rc.** Some private torch builds register
  an atexit handler in `torch._C` that replaces the process exit status: under
  one such image every `import torch` process exits 0 (an uncaught exception
  too), under another 1. Check once: `nd run -- 'python3 -c "import torch,sys;
  sys.exit(5)"'` must end `exit rc=5`. Under such an image `EXEC_RC=`, `TRIAL_RC=` and
  `nd run`'s rc say nothing; the jobs decide from log markers (`NAN_SIGNAL`, `LADDER_RESULT=`,
  `NATIVE_RESULT=`), so their results stand.
