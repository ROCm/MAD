#!/bin/bash
#SBATCH --job-name=qwen-sglang-serve
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# Qwen2.5-0.5B served by SGLang (sglang.Engine, SGLang's defaults). Run from this folder; only the public SGLang
# ROCm image is downloaded.
#
#   ./run.sh               run the workload on GPU 0 and print a verdict
#   ./run.sh pull          only prepare the image (docker/podman: build; apptainer: SIF)
#   sbatch -p <partition> run.sh              same, on one exclusive SLURM node
#
# Environment (optional): REQUESTS=200 PROMPT=256 NEW=8 SEED=0 OUT_DIR=./out/<date>
#                         IMAGE=<image name> BASE=<base image>
#                         GEN_ARGS=<extra qwen_sglang_serve.py arguments>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME, such as ~/proj.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f qwen_sglang_serve.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on SGLang's public ROCm release image for MI350X/MI355X, pinned by digest
# (Apptainer and Podman do not accept tag@digest): lmsysorg/sglang:v0.5.23-rocm10-mi35x.
IMAGE=${IMAGE:-qwen-sglang-serve:v0.5.23}
BASE=${BASE:-docker.io/lmsysorg/sglang@sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5}
REQUESTS=${REQUESTS:-200} PROMPT=${PROMPT:-256} NEW=${NEW:-8} SEED=${SEED:-0}
GEN_ARGS=${GEN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

#     arm    gpu
ARMS="serve  0"

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
# Podman's storage and Apptainer's layer cache default to $HOME, which is too small for a ROCm image.
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/qwen-sglang-serve-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped, so image files cannot keep
# their owners, and host networking avoids the netns sysctls that need more uids.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

# The image's aiter expects root: it writes into root-owned directories of the image (its JIT directory,
# /tmp/aiter_configs). Docker and Podman run the container as root (rootless Podman's root is the host user);
# the container writes nothing to the host, the logs are redirected here.
NAME=qwen-sglang-serve-$$
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
    # pull fail even for public images. --network=host: some daemons have no default bridge network, and the
    # build fetches crates from crates.io.
    docker) docker image inspect "$IMAGE" >/dev/null 2>&1 ||
              { echo "building $IMAGE from $BASE"; mkdir -p "$SCRATCH_DIR/docker-anon" &&
                DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker build -q --network=host -t "$IMAGE" \
                  --build-arg BASE="$BASE" "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $(docker image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    podman) "${PODMAN[@]}" image exists "$IMAGE" ||
              { echo "building $IMAGE from $BASE"; "${PODMAN[@]}" build -q --network=host -t "$IMAGE" \
                  --build-arg BASE="$BASE" "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $("${PODMAN[@]}" image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    # Apptainer runs the base itself and cannot build the Dockerfile; the base has everything, and the script
    # is read from this folder. SGLang then builds its Rust radix tree on the first request (network needed).
    apptainer) [ -s "$SIF" ] || { echo "pulling $BASE to $SIF"; mkdir -p "$SIF_DIR" &&
                                  apptainer pull -q "$SIF" "docker://$BASE"; } ||
              { echo "cannot pull $BASE; $PULL_HINT"; return 1; }
            echo "IMAGE $SIF $(stat -L -c '%s bytes, pulled %y' "$SIF")" ;;
  esac
}
drun() { # drun <name> <gpu> <command...>
  local name=$1 gpu=$2 kv flag=-e envs=() t; shift 2
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in HF_HUB_OFFLINE=1 HIP_VISIBLE_DEVICES="$gpu"; do envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" --network=host \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default. Not --rocm: it binds the host's ROCm libraries into
    # the container. Each container gets its own /tmp. The script mirrors aiter's JIT directory there: the
    # host user cannot write the image's.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" --pwd "$HERE" "${envs[@]}" --env USER="$(id -un)" --env LOGNAME="$(id -un)" \
                "$SIF" "$@"
              rm -rf "$t" ;;
  esac
}

case "${1:-all}" in
  all|serve) want=serve ;;
  pull) image_ready; exit $? ;;
  *) sed -n '7,19p' "$HERE/run.sh"; exit 2 ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
echo "=== requests=$REQUESTS prompt=$PROMPT new=$NEW seed=$SEED host=$(hostname -s)" \
     "runtime=$RUNTIME${GEN_ARGS:+ gen_args=$GEN_ARGS}"
image_ready || exit 1
drun env 0 python3 -c 'import torch
from sglang.version import __version__
print("ENV torch", torch.__version__, "hip", torch.version.hip, "arch",
      torch.cuda.get_device_properties(0).gcnArchName, "sglang", __version__)' 2>&1 | grep '^ENV'
for arm in $want; do
  read -r name gpu <<< "$(grep -w "^ *$arm" <<< "$ARMS")"
  # shellcheck disable=SC2086
  drun "$name" "$gpu" python3 "$HERE/qwen_sglang_serve.py" $GEN_ARGS --requests "$REQUESTS" \
    --prompt-len "$PROMPT" --new-tokens "$NEW" --seed "$SEED" --log-every 50 > "$OUT_DIR/$name.log" 2>&1 &
done
wait

echo
echo "=== result: seed $SEED, host $(hostname -s) ==="
verdict=REPRODUCED
for arm in $want; do
  read -r name _ <<< "$(grep -w "^ *$arm" <<< "$ARMS")"
  res=$(grep -h '^RESULT' "$OUT_DIR/$name.log")
  nan=$(grep -o 'nan_request=[^ ]*' <<< "$res" | cut -d= -f2)
  grep -h '^Logprobs are not finite' "$OUT_DIR/$name.log"
  printf '%-12s nan_request=%s\n' "$name" "${nan:-NO_RESULT}"
  if [ -z "$res" ]; then
    verdict=INCOMPLETE; echo "  no RESULT line; last lines of $OUT_DIR/$name.log:"; tail -5 "$OUT_DIR/$name.log"
  elif [ "$nan" = - ]; then
    verdict=NOT_REPRODUCED
  fi
done
echo "VERDICT $verdict"
