# RUNTIME=podman -- sourced by lib.sh; contract in runtime_docker.sh.
#
# Rootless podman.  Profile knobs: IMAGE, IMAGE_TAR, PODMAN_ARGS (global flags, e.g. on a
# cluster without /etc/subuid ranges: "--root <node-local> --runroot <node-local>
# --storage-opt overlay.ignore_chown_errors=true" -- one uid mapped), PODMAN_SHM (64g),
# EXTRA_MOUNTS.  --userns=keep-id keeps the host uid (files stay yours), keep-groups keeps
# the render/video groups that open /dev/kfd.  Images live in node-local storage: like
# docker, every new node pulls or loads once.  PODMAN_ENTRYPOINT (auto | keep | clear): an image
# ENTRYPOINT program is cleared, as DOCKER_ENTRYPOINT in runtime_docker.sh.

rt_init() { ND_SCRATCH_RT=/tmp; }
rt_path() { printf '%s\n' "$1"; }
_pm() { # shellcheck disable=SC2086
  podman ${PODMAN_ARGS:-} "$@"; }

# _pm_have [say]: IMAGE is here -- and, for a built image (IMAGE_ID), this build (runtime_docker.sh)
_pm_have() {
  local id; id=$(_pm image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) || return 1
  nd_image_stale "$id" || return 0
  [ -n "${1:-}" ] && nd_say "$IMAGE is build ${id#sha256:}, not IMAGE_ID ${IMAGE_ID#sha256:}: loading IMAGE_TAR"
  return 1
}

rt_prepare() {
  command -v podman >/dev/null || nd_die "RUNTIME=podman but no podman on $(hostname)"
  : "${IMAGE:?IMAGE is required (profile or env)}"
  _pm_have && return 0
  local lock=/tmp/nd-image-$(nd_slug "$IMAGE")
  nd_lock "$lock"
  if ! _pm_have say; then
    if [ -n "${IMAGE_TAR:-}" ]; then
      nd_say "$IMAGE absent: loading $IMAGE_TAR"
      nd_unpack "$IMAGE_TAR" | _pm load >/dev/null || { nd_unlock "$lock"; nd_die "podman load failed"; }
    elif nd_say "$IMAGE absent: pulling" && _pm pull -q "$IMAGE" >/dev/null; then
      :
    elif [ -n "${IMAGE_BUILD_DIR:-}" ] && [ -f "$IMAGE_BUILD_DIR/Dockerfile" ]; then
      nd_say "$IMAGE not pullable: building it from $IMAGE_BUILD_DIR (runtime_docker.sh: IMAGE_BUILD_DIR)"
      _pm build -q --network=host -t "$IMAGE" "$IMAGE_BUILD_DIR" >/dev/null ||
        { nd_unlock "$lock"; nd_die "podman build $IMAGE from $IMAGE_BUILD_DIR failed"; }
    else
      nd_unlock "$lock"; nd_die "podman pull $IMAGE failed (IMAGE_TAR or IMAGE_BUILD_DIR to load or build it)"
    fi
  fi
  nd_unlock "$lock"
}

# ND_EP: the --entrypoint flag for IMAGE, per PODMAN_ENTRYPOINT (runtime_docker.sh: _docker_entrypoint).
_pm_entrypoint() {
  local ep; ND_EP=()
  case "${PODMAN_ENTRYPOINT:-auto}" in keep) return ;; clear) ND_EP=(--entrypoint=); return ;; esac
  ep=$(_pm image inspect -f '{{json .Config.Entrypoint}}' "$IMAGE" 2>/dev/null)
  case "$ep" in ""|null|"[]") return ;; *.sh\"*|*tini*|*dumb-init*) return ;; esac
  ND_EP=(--entrypoint=)
  nd_say "image entrypoint $ep cleared (it would take the command as its arguments; PODMAN_ENTRYPOINT=keep to run through it)"
}

rt_exec() {
  local name=$1 gpus=$2 workdir=$3 cmd=$4 envs=() kv m=(-v "$ND_REPO_HOST:$ND_REPO_HOST") x
  _pm_entrypoint
  case "${ND_ART_HOST:-}" in ""|"$ND_REPO_HOST"/*) ;; *) m+=(-v "$ND_ART_HOST:$ND_ART_HOST") ;; esac
  # shellcheck disable=SC2086
  for x in ${EXTRA_MOUNTS:-}; do [ "$x" = -v ] && continue; m+=(-v "$x"); done
  for kv in "${ND_RUN_ENV[@]}"; do envs+=(-e "$kv"); done
  nd_exec_bg "$name" podman ${PODMAN_ARGS:-} run --rm --init --name "$name" \
    --userns=keep-id --group-add keep-groups --network=host --ipc=host --security-opt label=disable \
    --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size="${PODMAN_SHM:-64g}" \
    -e HOME=/tmp "${envs[@]}" "${m[@]}" "${ND_EP[@]}" -w "$workdir" "$IMAGE" bash -c "$ND_TERM_WRAP" nd-wrap "$cmd"
}

rt_stop() { _pm stop -t 20 "$1" >/dev/null 2>&1 || true; }
rt_kill() { _pm rm -f "$1" >/dev/null 2>&1 || true; }
