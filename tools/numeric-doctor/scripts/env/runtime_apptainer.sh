# RUNTIME=apptainer -- sourced by lib.sh; contract in runtime_docker.sh.
#
# Profile knobs:
#   PROJ             host directory bound into the container; the repo and the artifact dir must
#                    be under it (default: the repo itself)
#   CONTAINER_PROJ   where PROJ appears inside (default /mnt) -- see "paths" below
#   SIF              the image file; default $SIF_DIR/<IMAGE with / : @ -> _>.sif
#   SIF_DIR          default $PROJ/sif
#   IMAGE            docker reference: `apptainer pull docker://IMAGE` when the SIF is missing
#   IMAGE_TAR        a `docker save` archive (.tar/.tar.gz/.tar.zst): built into the SIF instead
#   APPTAINER_FLAGS  extra exec flags.  --pid when capturing (the capture counts /dev/kfd holders
#                    in /proc; without a PID namespace it sees every process on the node);
#                    --cleanenv when the model's run script uses it -- then EXTRA_ENV must
#                    carry what that script set by hand.  NEVER --rocm: it binds the host's ROCm
#                    libraries over the image's hipBLASLt, the library under test.
#   NODE_SCRATCH     node-local cache dir (default /tmp/nd-cache-${SLURM_JOB_ID:-$$})
#   EXTRA_MOUNTS     extra binds, "src:dst ..."
#
# Flag translation from docker, and why:
#   paths            a SIF is read-only and cannot create a mountpoint for the host path, so a
#                    self-bind ($PROJ:$PROJ) SILENTLY does nothing and every path inside is
#                    "No such file or directory".  Bind PROJ at CONTAINER_PROJ; rt_path translates.
#   /dev/kfd, /dev/dri   DELIBERATELY not bound: rootless apptainer forces `nodev` on binds it
#                    creates, the device becomes unopenable and torch.cuda.device_count() is 0.
#                    The default `mount dev = yes` already exposes them.  Do not "fix" a GPU
#                    problem by binding /dev/*.
#   --ipc            apptainer's --ipc CREATES a namespace (the inverse of docker's --ipc=host):
#                    not passed; host IPC and the host's /dev/shm are the default.
#   HOME             set with --home (APPTAINERENV_HOME is refused and the host NFS $HOME kept).
#   caches           node-local: an flock() over NFS left by a killed process wedges every later
#                    import (Triton, MIOpen, aiter) -- ~60 node-hours once.
#   LD_PRELOAD       blanked: a site may inject XALT tracking whose library does not exist
#                    inside; every process then errors at load.
#   cleanup trap     not needed: `apptainer exec` is an ordinary child in the job's cgroup.

_apt_proj() { echo "${PROJ:-$ND_REPO_HOST}"; }
_apt_cproj() { echo "${CONTAINER_PROJ:-/mnt}"; }

rt_init() {
  ND_APT_CACHE=${NODE_SCRATCH:-/tmp/nd-cache-${SLURM_JOB_ID:-$$}}
  mkdir -p "$ND_APT_CACHE"
  ND_SCRATCH_RT=$ND_APT_CACHE        # the host path, bound as HOME and mounted as itself (/tmp is shared)
}

# PROJ may be a symlink (e.g. /home/... -> /lustre/...): a physical path (pwd -P) must map
# too, or the container gets a path it cannot see and every copy dies on FileNotFoundError.
rt_path() {
  local p=$1 proj cp phys; proj=$(_apt_proj); cp=$(_apt_cproj)
  phys=$(cd -- "$proj" 2>/dev/null && pwd -P || echo "$proj")
  case "$p" in
    "$proj"|"$phys") printf '%s\n' "$cp" ;;
    "$proj"/*) printf '%s\n' "$cp/${p#"$proj"/}" ;;
    "$phys"/*) printf '%s\n' "$cp/${p#"$phys"/}" ;;
    *) printf '%s\n' "$p" ;;
  esac
}
rt_hostpath() {
  local p=$1 cp; cp=$(_apt_cproj)
  case "$p" in "$cp") _apt_proj ;; "$cp"/*) printf '%s\n' "$(_apt_proj)/${p#"$cp"/}" ;; *) printf '%s\n' "$p" ;; esac
}

_apt_sif() {
  if [ -n "${SIF:-}" ]; then echo "$SIF"; return; fi
  : "${IMAGE:?IMAGE or SIF is required (profile or env)}"
  echo "${SIF_DIR:-$(_apt_proj)/sif}/$(nd_slug "$IMAGE").sif"
}

# Build the SIF from IMAGE_TAR, once: the SIF is one file on shared storage, so every later
# node just opens it (docker re-reads the image into node-local storage on every node).
_apt_build_from_tar() {
  local sif=$1 work tar t
  work=$ND_APT_CACHE/sif-build; mkdir -p "$work"
  export APPTAINER_TMPDIR=$work/tmp APPTAINER_CACHEDIR=$work/cache
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$sif")"
  case $IMAGE_TAR in
    *.tar) tar=$IMAGE_TAR ;;
    *) tar=$work/image.tar; nd_say "decompressing $IMAGE_TAR (minutes)"
       nd_unpack "$IMAGE_TAR" > "$tar.partial" && mv -f "$tar.partial" "$tar" || { rm -f "$tar.partial"; return 1; } ;;
  esac
  for t in docker-archive oci-archive; do      # `docker save` archives are docker-archive; try both
    nd_say "apptainer build $t:$tar -> $sif"
    if apptainer build -F "$sif.building" "$t:$tar" > "$work/build.log" 2>&1; then
      mv -f "$sif.building" "$sif"; [ "$tar" = "$IMAGE_TAR" ] || rm -f "$tar"; return 0
    fi
    rm -f "$sif.building"
  done
  tail -20 "$work/build.log" >&2
  return 1
}

rt_prepare() {
  command -v apptainer >/dev/null || nd_die "RUNTIME=apptainer but no apptainer on $(hostname)"
  # A node without GPUs (a login node) starts the container, finds zero devices and gives a
  # result that looks like a container bug.  Refuse unless asked.
  [ -e /dev/kfd ] || [ "${ND_ALLOW_NO_GPU:-0}" = 1 ] ||
    nd_die "no /dev/kfd on $(hostname): not a GPU node (submit with nd submit, or ND_ALLOW_NO_GPU=1)"
  local sif; sif=$(_apt_sif)
  [ -s "$sif" ] && return 0
  nd_lock "$sif"
  if [ ! -s "$sif" ]; then
    if [ -n "${IMAGE_TAR:-}" ]; then
      _apt_build_from_tar "$sif" || { nd_unlock "$sif"; nd_die "cannot build $sif from $IMAGE_TAR"; }
    else
      : "${IMAGE:?no SIF at $sif and no IMAGE to pull}"
      nd_say "pulling docker://$IMAGE -> $sif (a ~25 GB image: ~35 min)"
      mkdir -p "$(dirname "$sif")"
      APPTAINER_TMPDIR=$ND_APT_CACHE APPTAINER_CACHEDIR=$ND_APT_CACHE/cache \
        apptainer pull -F "$sif.building" "docker://$IMAGE" && mv -f "$sif.building" "$sif" ||
        { rm -f "$sif.building"; nd_unlock "$sif"; nd_die "apptainer pull $IMAGE failed"; }
    fi
  fi
  nd_unlock "$sif"
}

rt_exec() {
  local name=$1 gpus=$2 workdir=$3 cmd=$4 sif flags=() kv b proj
  sif=$(_apt_sif); proj=$(_apt_proj)
  [ "$(rt_path "$ND_REPO_HOST")" != "$ND_REPO_HOST" ] || nd_die "the repo $ND_REPO_HOST is not under PROJ=$proj"
  [ -z "${ND_ART_HOST:-}" ] || [ "$(rt_path "$ND_ART_HOST")" != "$ND_ART_HOST" ] ||
    nd_die "the artifact dir $ND_ART_HOST is not under PROJ=$proj"
  # memlock for RDMA, stack for deep kernels: raise to the hard limit (unlimited fails for a user)
  ulimit -l unlimited 2>/dev/null || ulimit -l "$(ulimit -Hl)" 2>/dev/null || true
  ulimit -s 65536 2>/dev/null || true
  flags=(--home "$ND_APT_CACHE" --bind "$proj:$(_apt_cproj)" --pwd "$(rt_path "$workdir")")
  # shellcheck disable=SC2086
  for b in ${EXTRA_MOUNTS:-}; do flags+=(--bind "$b"); done
  # shellcheck disable=SC2086
  for b in ${APPTAINER_FLAGS:-}; do flags+=("$b"); done
  for kv in "${ND_RUN_ENV[@]}"; do flags+=(--env "$kv"); done
  flags+=(--env "TRITON_CACHE_DIR=$ND_APT_CACHE/triton" --env "MIOPEN_USER_DB_PATH=$ND_APT_CACHE/miopen"
          --env "MIOPEN_CUSTOM_CACHE_DIR=$ND_APT_CACHE/miopen")
  export APPTAINERENV_LD_PRELOAD=""
  unset APPTAINERENV_LD_LIBRARY_PATH
  nd_exec_bg "$name" apptainer exec "${flags[@]}" "$sif" bash -lc "$ND_TERM_WRAP" nd-wrap "$cmd"
}

rt_stop() { kill -TERM "$2" 2>/dev/null || true; }
rt_kill() { :; }
