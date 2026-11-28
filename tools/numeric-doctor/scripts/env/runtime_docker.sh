# RUNTIME=docker -- sourced by lib.sh.  Contract (every runtime_*.sh):
#   rt_init                   set ND_SCRATCH_RT (a writable dir inside the runtime for caches)
#   rt_path <host path>       the path as seen inside the runtime
#   rt_prepare                make the image usable on this node (idempotent, safe concurrently)
#   rt_exec <name> <gpus> <workdir> <cmd>   run cmd (bash -c) with ND_RUN_ENV, in the foreground of
#                             nd_exec_bg; returns its exit code
#   rt_stop <name> <pid>      graceful stop (SIGTERM reaches the command)
#   rt_kill <name>            hard cleanup, always called on exit
#
# Profile knobs: IMAGE (required), IMAGE_TAR (load when the node lacks IMAGE), DOCKER_PULL
# (1 = anonymous pull when absent, default 1), IMAGE_BUILD_DIR (a Dockerfile dir: build IMAGE when
# neither loads nor pulls -- a package's image/), DOCKER_AS_USER (1 = host uid, default 1),
# DOCKER_SHM (64g), DOCKER_PID_HOST (1: --pid=host for a multi-GPU trial), EXTRA_MOUNTS ("-v a:b ..." or "a:b ..."), DOCKER_ARGS (extra run flags),
# DOCKER_ENTRYPOINT (auto | keep | clear, below).
#
# Why each flag (each cost a job once):
#   --init         the early stop's SIGTERM must reach the trainer; as PID 1 it ignores it, the
#                  capture's write-at-SIGTERM never runs and docker SIGKILLs after 20 s.
#   TINI_KILL_PROCESS_GROUP=1 + the wrapper's `trap 'exit $?'`
#                  tini signals only its child, the `bash -c`: with a compound command
#                  (`python ...; echo`) bash died and the command never saw the TERM.  Now the
#                  whole group gets it and bash waits, then exits with the command's rc.
#   --user + group of /dev/kfd + USER/LOGNAME/HOME=/tmp
#                  files written to NFS stay yours (root-squash refuses root) and dynamo's
#                  getpass.getuser() finds a user.
#   unique --name + rm -f on exit
#                  a container is the DAEMON's child: scancel reaps this script and leaves it
#                  pinning HBM; a fixed name makes concurrent runs kill each other's.
#   --entrypoint=  (DOCKER_ENTRYPOINT=auto) nd runs `bash -c <cmd>` as the container's command, which an
#                  image ENTRYPOINT program takes as its arguments: vllm/vllm-openai-rocm's ["vllm",
#                  "serve"] started the OpenAI server with `bash -c ...` as arguments (2026-11-19).  A
#                  shell wrapper that execs "$@" (*.sh, tini, dumb-init) is kept; anything else cleared.
#   -v REPO:REPO   same path inside, so commands and logs need no translation (pwd -P: docker
#                  resolves symlinks host-side).

rt_init() { ND_SCRATCH_RT=/tmp; }
rt_path() { printf '%s\n' "$1"; }

_docker_anon_pull() {
  local d; d=$(mktemp -d)
  # anonymous: an expired stored registry login fails even a public pull
  DOCKER_CONFIG=$d docker pull -q "$1" >/dev/null; local rc=$?; rm -rf "$d"; return $rc
}

# _docker_have [say]: IMAGE is on this node -- and, for a built image (IMAGE_ID), this build of it
_docker_have() {
  local id; id=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) || return 1
  nd_image_stale "$id" || return 0
  [ -n "${1:-}" ] && nd_say "$IMAGE on $(hostname) is build ${id#sha256:}, not IMAGE_ID ${IMAGE_ID#sha256:}: loading IMAGE_TAR"
  return 1
}

rt_prepare() {
  command -v docker >/dev/null || nd_die "RUNTIME=docker but no docker on $(hostname)"
  : "${IMAGE:?IMAGE is required (profile or env)}"
  _docker_have && return 0
  # (only once the image check failed: no extra daemon call on the normal path)
  # No daemon access here (a login node with the docker CLI but no socket permission): say so,
  # before a pull attempt turns it into "absent ... not pulled" (2026-11-23)
  docker info >/dev/null 2>&1 || nd_die "docker is not usable on $(hostname) (no access to the daemon)$([ -e /dev/kfd ] ||
    echo "; no GPU here either -- a login node? run it as a job: nd submit --name <n> --gpus 1 -- exec.sh --gpus 0 -- '<cmd>'")"
  # One load per node per image: N copies racing `docker load` thrash local storage.
  local lock=/tmp/nd-image-$(nd_slug "$IMAGE")
  nd_lock "$lock"
  if ! _docker_have say; then
    if [ -n "${IMAGE_TAR:-}" ]; then
      [ -r "$IMAGE_TAR" ] || { nd_unlock "$lock"; nd_die "IMAGE_TAR=$IMAGE_TAR unreadable"; }
      nd_say "$IMAGE absent on $(hostname): loading $IMAGE_TAR (minutes)"
      nd_unpack "$IMAGE_TAR" | docker load >/dev/null || { nd_unlock "$lock"; nd_die "docker load failed"; }
    elif [ "${DOCKER_PULL:-1}" = 1 ] && { nd_say "$IMAGE absent on $(hostname): pulling (anonymous)"; _docker_anon_pull "$IMAGE"; }; then
      :
    elif [ -n "${IMAGE_BUILD_DIR:-}" ] && [ -f "$IMAGE_BUILD_DIR/Dockerfile" ]; then
      # a locally built image (build_image.sh) is in no registry: a package ships its build context
      # in image/ and builds it here -- off-site, the site's IMAGE_TAR does not exist (2026-11-20)
      nd_say "$IMAGE absent on $(hostname), not pullable: building it from $IMAGE_BUILD_DIR (minutes)"
      local d; d=$(mktemp -d)
      DOCKER_CONFIG=$d docker build -q --network=host -t "$IMAGE" "$IMAGE_BUILD_DIR" >/dev/null
      local rc=$?; rm -rf "$d"; [ $rc = 0 ] || { nd_unlock "$lock"; nd_die "docker build $IMAGE from $IMAGE_BUILD_DIR failed"; }
    else
      nd_unlock "$lock"; nd_die "$IMAGE absent on $(hostname) and not pulled; set IMAGE_TAR, DOCKER_PULL=1 or IMAGE_BUILD_DIR"
    fi
  fi
  nd_unlock "$lock"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || nd_die "$IMAGE still absent after load/pull (does IMAGE_TAR hold it?)"
}

_docker_mounts() {
  local m=(-v "$ND_REPO_HOST:$ND_REPO_HOST") x
  case "${ND_ART_HOST:-}" in ""|"$ND_REPO_HOST"/*) ;; *) m+=(-v "$ND_ART_HOST:$ND_ART_HOST") ;; esac
  # $HOME only when it is a real per-user dir outside the repo (mounting /root over the
  # image's own /root loses its shell profile).
  case "${HOME:-}" in ""|/|/root|"$ND_REPO_HOST"|"$ND_REPO_HOST"/*) ;; *) [ -d "$HOME" ] && m+=(-v "$HOME:$HOME") ;; esac
  # shellcheck disable=SC2086
  for x in ${EXTRA_MOUNTS:-}; do [ "$x" = -v ] && continue; m+=(-v "$x"); done
  ND_MOUNTS=("${m[@]}")
}

# ND_EP: the --entrypoint flag for IMAGE (empty = run through the image's own), per DOCKER_ENTRYPOINT.
_docker_entrypoint() {
  local ep; ND_EP=()
  case "${DOCKER_ENTRYPOINT:-auto}" in keep) return ;; clear) ND_EP=(--entrypoint=); return ;; esac
  ep=$(docker image inspect -f '{{json .Config.Entrypoint}}' "$IMAGE" 2>/dev/null)
  case "$ep" in ""|null|"[]") return ;; *.sh\"*|*tini*|*dumb-init*) return ;; esac
  ND_EP=(--entrypoint=)
  nd_say "image entrypoint $ep cleared (it would take the command as its arguments; DOCKER_ENTRYPOINT=keep to run through it)"
}

rt_exec() {
  local name=$1 gpus=$2 workdir=$3 cmd=$4 envs=() kv user=()
  _docker_mounts
  _docker_entrypoint
  for kv in "${ND_RUN_ENV[@]}"; do envs+=(-e "$kv"); done
  local mask="" pidns=()
  # A trial over several GPUs (TP / DP ranks: RCCL) shares the host's IPC namespace (--ipc=host) with every other
  # trial on the node, but its container numbers its processes from 1 -- two such trials' ranks share PIDs, and
  # RCCL's intra-node setup collides: ncclCommInitRank "unhandled cuda error" in 3 of 4 SGLang TP-2 trials, all 4
  # with --pid=host (2026-11-24); the apptainer --pid case, nd_multinode's note. DOCKER_PID_HOST=0 turns it off.
  case $gpus in *,*|all) [ "${DOCKER_PID_HOST:-1}" = 1 ] && pidns=(--pid=host) ;; esac
  if [ "${DOCKER_AS_USER:-1}" = 1 ]; then
    user=(--user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd 2>/dev/null || echo video)" -e HOME=/tmp)
  else
    mask="umask 0000; "    # as root: leave what it writes in the repo / artifact dir deletable by the user
  fi
  # a container of this name left from an earlier attempt of the same job (a headless server srun killed: only the
  # docker client died, 2026-11-24) would refuse this one -- names are job-unique, so it is ours to remove
  docker rm -f "$name" >/dev/null 2>&1
  # shellcheck disable=SC2086
  nd_exec_bg "$name" docker run --rm --init --name "$name" \
    --network=host --ipc=host "${pidns[@]}" --shm-size="${DOCKER_SHM:-64g}" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ulimit nofile=524288:524288 \
    --security-opt seccomp=unconfined \
    "${user[@]}" "${envs[@]}" "${ND_MOUNTS[@]}" "${ND_EP[@]}" ${DOCKER_ARGS:-} \
    -e TINI_KILL_PROCESS_GROUP=1 -w "$workdir" "$IMAGE" bash -c "${mask}trap 'exit \$?' TERM INT; $cmd"
}

rt_stop() { docker stop -t 20 "$1" >/dev/null 2>&1 || true; }
rt_kill() { docker rm -f "$1" >/dev/null 2>&1 || true; }
