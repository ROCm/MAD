#!/bin/bash
#SBATCH --job-name=sparse-mla-decode
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# Greedy decode of a DeepSeek-V3.2 / GLM-5-shaped sparse-MLA model (one TP8 rank) on vLLM's ROCm image with
# aiter. Run from this folder; only the public vLLM ROCm image is downloaded.
#
#   ./run.sh               run the decode on one GPU and print the result
#   ./run.sh pull          only prepare the image (docker/podman: build; apptainer: SIF)
#   sbatch -p <partition> run.sh              same, on one exclusive SLURM node
#
# Environment (optional): PROMPT=8192 NEW=128 LAYERS=4 REQUESTS=1 SEED=0
#                         OUT_DIR=./out/<date> IMAGE=<image name> BASE=<base image>
#                         GEN_ARGS=<extra sparse_mla_decode_serve.py arguments>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME, such as ~/proj.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f sparse_mla_decode_serve.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on vLLM's public ROCm release image, pinned by digest (Apptainer and
# Podman do not accept tag@digest): vllm/vllm-openai-rocm:v0.33.0-rocm100 (torch 2.13.0+rocm10.0.0, HIP
# 7.15.26333, amd-aiter 0.1.26), with aiter's mla.py from aiter commit 95abba475 (./aiter_mla.py).
IMAGE=${IMAGE:-sparse-mla-decode:v1}
BASE=${BASE:-docker.io/vllm/vllm-openai-rocm@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba}
PROMPT=${PROMPT:-8192} NEW=${NEW:-128} LAYERS=${LAYERS:-4} REQUESTS=${REQUESTS:-1}
SEED=${SEED:-0}
GEN_ARGS=${GEN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}
# aiter's mla.py in the image (apptainer runs the base image and mounts ./aiter_mla.py there).
MLA_PY=/usr/local/lib/python3.12/dist-packages/aiter/mla.py

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
# Podman's storage and Apptainer's layer cache default to $HOME, which is too small for a ROCm image.
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sparse-mla-decode-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped, so image files cannot keep
# their owners, and host networking avoids the netns sysctls that need more uids.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

# The scripts only print (run.sh writes the logs on the host), so the containers run as the host user and
# mount this folder read-only.
NAME=sparse-mla-decode-$$
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
SIF=$SIF_DIR/$(tr '/:@' '___' <<< "$BASE").sif
# The base is public; a stored but expired Docker Hub login still makes the pull fail.
PULL_HINT="if a stored registry login has expired, log out of docker.io (or log in again)"
image_ready() { # build $IMAGE from ./Dockerfile (apptainer: convert the base)
  case $RUNTIME in
    # An empty DOCKER_CONFIG pulls the public base anonymously: a stored but expired Docker Hub login makes the
    # pull fail even for public images. --network=host: some daemons have no default bridge network.
    docker) docker image inspect "$IMAGE" >/dev/null 2>&1 ||
              { echo "building $IMAGE from $BASE"; mkdir -p "$SCRATCH_DIR/docker-anon" &&
                DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker build -q --network=host -t "$IMAGE" \
                  --build-arg BASE="$BASE" "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $(docker image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    podman) "${PODMAN[@]}" image exists "$IMAGE" ||
              { echo "building $IMAGE from $BASE"; "${PODMAN[@]}" build -q -t "$IMAGE" --build-arg BASE="$BASE" \
                  "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $("${PODMAN[@]}" image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    # Apptainer runs the base itself and cannot build the Dockerfile; the scripts and aiter_mla.py are read from
    # this folder.
    apptainer) [ -s "$SIF" ] || { echo "pulling $BASE to $SIF"; mkdir -p "$SIF_DIR" &&
                                  apptainer pull -q "$SIF" "docker://$BASE"; } ||
              { echo "cannot pull $BASE; $PULL_HINT"; return 1; }
            echo "IMAGE $SIF $(stat -L -c '%s bytes, pulled %y' "$SIF")" ;;
  esac
}
drun() { # drun <name> <gpu> <command...>
  local name=$1 gpu=$2 kv flag=-e envs=() mnt=() t; shift 2
  [ "$RUNTIME" = apptainer ] && { flag=--env; mnt=(-B "$HERE/aiter_mla.py:$MLA_PY:ro"); }
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" HIP_VISIBLE_DEVICES="$gpu" AITER_LOG_LEVEL=WARNING; do
    envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" "${mnt[@]}" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" "${mnt[@]}" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default. Not --rocm: it binds the host's ROCm libraries into
    # the container. Each container gets its own /tmp.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" "${mnt[@]}" --pwd "$HERE" "${envs[@]}" "$SIF" "$@"
              rm -rf "$t" ;;
  esac
}
case "${1:-run}" in
  run) ;;
  pull) image_ready; exit $? ;;
  *) sed -n '7,18p' "$HERE/run.sh"; exit 2 ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
echo "=== prompt=$PROMPT new=$NEW layers=$LAYERS requests=$REQUESTS seed=$SEED" \
     "host=$(hostname -s) runtime=$RUNTIME${GEN_ARGS:+ gen_args=$GEN_ARGS}"
image_ready || exit 1
drun env 0 python3 -c 'import torch
print("ENV torch", torch.__version__, "hip", torch.version.hip, "arch", torch.cuda.get_device_properties(0).gcnArchName)' \
  2>&1 | grep '^ENV'
LOG=$OUT_DIR/decode.log
# shellcheck disable=SC2086
drun decode 0 python3 "$HERE/sparse_mla_decode_serve.py" $GEN_ARGS --prompt-len "$PROMPT" --new-tokens "$NEW" \
  --layers "$LAYERS" --requests "$REQUESTS" --seed "$SEED" > "$LOG" 2>&1 &
wait
grep -h '^Logits are not finite' "$LOG"

echo
echo "=== seed $SEED, host $(hostname -s) ==="
res=$(grep -h '^RESULT' "$LOG" | tail -1)
wrong=$(grep -o 'wrong=[^ ]*' <<< "$res" | cut -d= -f2)
printf '%-8s wrong=%-4s %s\n' decode "${wrong:-NO_RESULT}" "$(sed -E 's/^RESULT wrong=[^ ]* *//' <<< "$res")"
if [ "$wrong" = yes ]; then verdict=REPRODUCED
elif [ "$wrong" = no ]; then verdict=NOT_REPRODUCED
else verdict=INCOMPLETE; echo "  no usable RESULT line; last lines of $LOG:"; tail -5 "$LOG"
fi
echo "VERDICT $verdict"
