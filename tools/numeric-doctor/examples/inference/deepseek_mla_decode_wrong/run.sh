#!/bin/bash
#SBATCH --job-name=deepseek-mla-decode
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# DeepSeek-V3-shaped MLA batched greedy decode: some decode attention calls return wrong finite values (fp32
# parity check fails) and the greedy tokens differ between identical passes. Run from this folder; only the
# public vLLM ROCm image is downloaded.
#
#   ./run.sh               run the decode workload (GPU 0) and print a verdict
#   ./run.sh pull          only prepare the image (docker/podman: build; apptainer: SIF)
#   sbatch -p <partition> run.sh              same, on one exclusive SLURM node
#
# Environment (optional): REQUESTS=512 NEW=128 PASSES=10 LAYERS=2 TOL=0.03 SEED=0
#                         OUT_DIR=./out/<date> IMAGE=<image name> BASE=<base image>
#                         GEN_ARGS=<extra mla_decode_serve.py arguments>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME, such as ~/proj.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f mla_decode_serve.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on vLLM's public ROCm release image, pinned by digest (Apptainer and
# Podman do not accept tag@digest): vllm/vllm-openai-rocm:v0.33.0-rocm100 (torch 2.13.0+rocm10.0.0, HIP
# 7.15.26333, amd-aiter 0.1.26).
IMAGE=${IMAGE:-deepseek-mla-decode:v0.33.0}
BASE=${BASE:-docker.io/vllm/vllm-openai-rocm@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba}
REQUESTS=${REQUESTS:-512} NEW=${NEW:-128} PASSES=${PASSES:-10} LAYERS=${LAYERS:-2} TOL=${TOL:-0.03}
SEED=${SEED:-0}
GEN_ARGS=${GEN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

#     arm     gpu  expected
ARMS="decode  0    wrong"

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
# Podman's storage and Apptainer's layer cache default to $HOME, which is too small for a ROCm image.
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/deepseek-mla-decode-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped, so image files cannot keep
# their owners, and host networking avoids the netns sysctls that need more uids.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

# The scripts only print (run.sh writes the logs on the host), so the containers run as the host user and
# mount this folder read-only.
NAME=deepseek-mla-decode-$$
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
    # Apptainer runs the base itself and cannot build the Dockerfile; the base has everything, and the scripts
    # are read from this folder.
    apptainer) [ -s "$SIF" ] || { echo "pulling $BASE to $SIF"; mkdir -p "$SIF_DIR" &&
                                  apptainer pull -q "$SIF" "docker://$BASE"; } ||
              { echo "cannot pull $BASE; $PULL_HINT"; return 1; }
            echo "IMAGE $SIF $(stat -L -c '%s bytes, pulled %y' "$SIF")" ;;
  esac
}
drun() { # drun <name> <gpu> <command...>
  local name=$1 gpu=$2 kv flag=-e envs=() t; shift 2
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" HIP_VISIBLE_DEVICES="$gpu" AITER_LOG_LEVEL=WARNING; do
    envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default. Not --rocm: it binds the host's ROCm libraries into
    # the container. Each container gets its own /tmp.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" --pwd "$HERE" "${envs[@]}" "$SIF" "$@"
              rm -rf "$t" ;;
  esac
}

case "${1:-all}" in
  all) want="decode" ;;
  pull) image_ready; exit $? ;;
  *) want="$*"
     for arm in $want; do grep -qw "^ *$arm" <<< "$ARMS" || { sed -n '6,20p' "$HERE/run.sh"; exit 2; }; done ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
echo "=== requests=$REQUESTS new=$NEW passes=$PASSES layers=$LAYERS tol=$TOL seed=$SEED" \
     "host=$(hostname -s) runtime=$RUNTIME${GEN_ARGS:+ gen_args=$GEN_ARGS}"
image_ready || exit 1
drun env 0 python3 -c 'import torch, importlib.metadata as m
print("ENV torch", torch.__version__, "hip", torch.version.hip, "arch",
      torch.cuda.get_device_properties(0).gcnArchName, "amd-aiter", m.version("amd-aiter"))' 2>&1 | grep '^ENV'
for arm in $want; do
  read -r name gpu _ <<< "$(grep -w "^ *$arm" <<< "$ARMS")"
  # shellcheck disable=SC2086
  drun "$name" "$gpu" python3 "$HERE/mla_decode_serve.py" $GEN_ARGS --requests "$REQUESTS" \
    --new-tokens "$NEW" --passes "$PASSES" --layers "$LAYERS" --tol "$TOL" --seed "$SEED" \
    > "$OUT_DIR/$name.log" 2>&1 &
done
wait

echo
echo "=== matrix: seed $SEED, host $(hostname -s) ==="
verdict=REPRODUCED
for arm in $want; do
  read -r name gpu expect <<< "$(grep -w "^ *$arm" <<< "$ARMS")"
  res=$(grep -h '^RESULT' "$OUT_DIR/$name.log")
  wrong=$(grep -o 'wrong=[^ ]*' <<< "$res" | cut -d= -f2)
  printf '%-15s expect=%-5s wrong=%-4s %s\n' "$name" "$expect" "${wrong:-NO_RESULT}" \
    "$(sed -E 's/^RESULT wrong=[^ ]* *//' <<< "$res")"
  if [ -z "$res" ] || { [ "$wrong" != yes ] && [ "$wrong" != no ]; }; then
    verdict=INCOMPLETE; echo "  no usable RESULT line; last lines of $OUT_DIR/$name.log:"; tail -5 "$OUT_DIR/$name.log"
  elif [ "$expect" = wrong ] && [ "$wrong" != yes ]; then
    [ "$verdict" = INCOMPLETE ] || verdict=NOT_REPRODUCED
  fi
done
echo "VERDICT $verdict"
