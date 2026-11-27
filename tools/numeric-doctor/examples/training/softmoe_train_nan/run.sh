#!/bin/bash
#SBATCH --job-name=softmoe-train-nan
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
# 20-layer soft-MoE transformer training, layers 3-18 torch.compile'd, 1-2 and 19-20 eager. Run from this folder.
#
#   ./run.sh           train on GPU 0 and print loss / grad_norm per step and a verdict
#                      (on a node without GPUs but with sbatch: submit itself, ./run.sh status for the log)
#   ./run.sh pull      only prepare the image (docker/podman: pull or load; apptainer: SIF in SIF_DIR)
#   ./run.sh status    show the newest job log
#
# Environment (optional): GPUS=1 (one trainer per GPU) STEPS=100 COMPILE_LAYERS=3-18 (none: all eager)
#                         TRAIN_ARGS=<more train.py arguments> OUT_DIR=./out/<date>
#                         IMAGE=<base image> RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         IMAGE_TAR=<`docker save` archive, .tar, .tar.gz or .tar.zst>: load it instead of pulling;
#                           the image is the one it holds (or IMAGE, if set: then a node that has it skips the load)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif   ACCOUNT= PARTITION= (cluster defaults) TIME=00:30:00 (self-submission)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths:
# Apptainer cannot bind to a path that runs through a symlink.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f train.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

if [ "${1:-}" = status ]; then
  f=$(ls -t "$HERE"/logs/*.out 2>/dev/null | head -1); [ -n "$f" ] || { echo "no job logs in logs/"; exit 1; }
  squeue -u "$(id -un)" -n softmoe-train-nan 2>/dev/null; echo "== $f"; tail -40 "$f"; exit 0
fi

# AMD's public PyTorch 2.10.0 / ROCm 7.2 / Python 3.12 release image, pinned by digest (Apptainer and Podman
# do not accept tag@digest): rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
IMAGE_SET=${IMAGE:+1} IMAGE_TAR=${IMAGE_TAR:-}
IMAGE=${IMAGE:-docker.io/rocm/pytorch@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4}
GPUS=${GPUS:-1} STEPS=${STEPS:-100} COMPILE_LAYERS=${COMPILE_LAYERS:-3-18} TRAIN_ARGS=${TRAIN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

if [ -z "${SLURM_JOB_ID:-}" ] && [ ! -e /dev/kfd ] && [ "${1:-}" != pull ]; then  # login node: submit
  command -v sbatch >/dev/null || { echo "no GPUs here (/dev/kfd) and no sbatch"; exit 2; }
  mkdir -p "$HERE/logs"
  job=$(sbatch --parsable ${ACCOUNT:+-A "$ACCOUNT"} ${PARTITION:+-p "$PARTITION"} -t "${TIME:-00:30:00}" \
        --chdir="$HERE" -o "$HERE/logs/%x_%j.out" -e "$HERE/logs/%x_%j.out" "$HERE/run.sh" "$@") || exit 1
  echo "submitted job $job; log: $HERE/logs/softmoe-train-nan_$job.out  (./run.sh status)"
  exit 0
fi

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
SIF=$SIF_DIR/$(tr '/:' '__' <<< "$IMAGE").sif
[ -n "$IMAGE_TAR" ] && [ -z "$IMAGE_SET" ] && SIF=$SIF_DIR/$(basename "${IMAGE_TAR%.tar*}").sif
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/softmoe-train-nan-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

NAME=softmoe-train-nan-$$
cleanup() {
  case $RUNTIME in
    docker) docker ps -q --filter "name=$NAME-" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 ;;
    podman) "${PODMAN[@]}" ps -q --filter "name=$NAME-" 2>/dev/null | xargs -r "${PODMAN[@]}" rm -f >/dev/null 2>&1 ;;
    apptainer) jobs -p | xargs -r kill 2>/dev/null; wait ;;
  esac
  [ -n "$KEEP_SCRATCH" ] && return
  [ "$RUNTIME" = podman ] && "${PODMAN[@]}" system reset -f >/dev/null 2>&1
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT; trap 'exit 130' TERM INT

# The image is public: docker pulls it anonymously (an empty DOCKER_CONFIG), because a stored but expired
# Docker Hub login makes the pull fail even for public images.
PULL_HINT="if a stored registry login has expired, log out of docker.io (or log in again)"
ctr() { if [ "$RUNTIME" = podman ]; then "${PODMAN[@]}" "$@"; else docker "$@"; fi; }
unpack() { case $1 in *.zst|*.zstd) zstd -dcq "$1" ;; *.gz) zcat "$1" ;; *) cat "$1" ;; esac; }
tar_ready() { # load IMAGE_TAR; without IMAGE, IMAGE becomes the image it holds
  local out t
  [ -r "$IMAGE_TAR" ] || { echo "cannot read IMAGE_TAR=$IMAGE_TAR"; return 1; }
  case $RUNTIME in
    docker|podman)
      [ -n "$IMAGE_SET" ] && ctr image inspect "$IMAGE" >/dev/null 2>&1 && return 0
      echo "loading $IMAGE_TAR"
      out=$(unpack "$IMAGE_TAR" | ctr load) || { echo "cannot load $IMAGE_TAR"; return 1; }
      # "Loaded image: <name:tag>" or, for an untagged archive, "Loaded image ID: sha256:..."
      [ -n "$IMAGE_SET" ] || IMAGE=$(sed -n 's/^Loaded image[^:]*: *//p' <<< "$out" | head -1 | cut -d, -f1)
      ctr image inspect "$IMAGE" >/dev/null 2>&1 || { echo "$IMAGE_TAR does not hold ${IMAGE:-an image}"; return 1; }
      [ -n "$IMAGE_SET" ] || echo "loaded $IMAGE (with IMAGE=$IMAGE, a node that has it skips the load)" ;;
    apptainer)
      [ -s "$SIF" ] && return 0
      t=$IMAGE_TAR; mkdir -p "$SIF_DIR" || return 1
      case $t in *.tar) ;; *) t=$SCRATCH_DIR/image.tar; echo "unpacking $IMAGE_TAR"
                             unpack "$IMAGE_TAR" > "$t" || { echo "cannot unpack $IMAGE_TAR"; return 1; } ;; esac
      echo "building $SIF from $IMAGE_TAR"
      apptainer build -F "$SIF" "docker-archive:$t" > "$SCRATCH_DIR/build.log" 2>&1 ||
        { tail -20 "$SCRATCH_DIR/build.log"; rm -f "$SIF"; echo "cannot build $SIF"; return 1; }
      [ "$t" = "$IMAGE_TAR" ] || rm -f "$t" ;;
  esac
}
image_ready() {
  [ -n "$IMAGE_TAR" ] && { tar_ready; return; }
  case $RUNTIME in
    docker) docker image inspect "$IMAGE" >/dev/null 2>&1 || { mkdir -p "$SCRATCH_DIR/docker-anon" &&
              echo "pulling $IMAGE" && DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker pull -q "$IMAGE" >/dev/null; } ||
              { echo "cannot pull $IMAGE; $PULL_HINT"; return 1; } ;;
    podman) "${PODMAN[@]}" image exists "$IMAGE" || "${PODMAN[@]}" pull -q "$IMAGE" >/dev/null ||
              { echo "cannot pull $IMAGE; $PULL_HINT"; return 1; } ;;
    apptainer) [ -s "$SIF" ] || { echo "pulling $IMAGE to $SIF"; mkdir -p "$SIF_DIR" &&
              apptainer pull -q "$SIF" "docker://$IMAGE"; } || { echo "cannot pull $IMAGE; $PULL_HINT"; return 1; } ;;
  esac
}
# The image has no passwd entry for the host uid, and dynamo calls getpass.getuser(): without USER/LOGNAME the
# trainer dies at the first compiled layer. Running as the host user keeps the output writable on NFS.
drun() { # drun <name> <gpu> <command...>
  local name=$1 gpu=$2 kv flag=-e envs=() t; shift 2
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
            HIP_VISIBLE_DEVICES="$gpu"; do envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" "${envs[@]}" "$IMAGE" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default; each container gets its own /tmp.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" -B "$OUT_DIR:$OUT_DIR" --pwd "$HERE" "${envs[@]}" "$SIF" "$@"
              rm -rf "$t" ;;
  esac
}

image_ready || exit 1
[ "${1:-}" = pull ] && exit 0
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "training: GPUS=$GPUS steps=$STEPS compile_layers=$COMPILE_LAYERS host=$(hostname -s) runtime=$RUNTIME image=$IMAGE" \
     "output=$OUT_DIR${TRAIN_ARGS:+ train_args=$TRAIN_ARGS}"
for g in $(seq 0 $((GPUS - 1))); do
  # shellcheck disable=SC2086
  drun "g$g" "$g" python3 "$HERE/train.py" --steps "$STEPS" --compile-layers "$COMPILE_LAYERS" $TRAIN_ARGS \
    > "$OUT_DIR/train_g$g.log" 2>&1 &
done
wait

grep -h -m3 -E '^(ENV|MODEL|DATA)' "$OUT_DIR/train_g0.log"
grep -h -E '^step:' "$OUT_DIR/train_g0.log" | tee "$OUT_DIR/steps.txt" | tail -8
tot=0 bad=0
for g in $(seq 0 $((GPUS - 1))); do
  r=$(grep -h '^RESULT' "$OUT_DIR/train_g$g.log")
  if [ -z "$r" ]; then echo "g$g: no RESULT line; end of its log:"; tail -20 "$OUT_DIR/train_g$g.log"; continue; fi
  echo "g$g: $r"; tot=$((tot + 1))
  grep -q 'nan_step=[0-9]' <<< "$r" && bad=$((bad + 1))
done
if [ "$tot" -lt "$GPUS" ]; then echo "VERDICT INCOMPLETE ($tot of $GPUS trainers finished)"
elif [ "$bad" -gt 0 ]; then echo "VERDICT NAN ($bad of $tot trainers: Loss is not finite)"
else echo "VERDICT CLEAN ($tot trainers, $STEPS steps)"; fi
