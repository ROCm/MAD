#!/bin/bash
#SBATCH --job-name=qwen-vllm-tp2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
# vLLM TP2 offline serving of a Qwen2.5-1.5B-shaped model: NaN logprobs. Run from this folder; only the public
# vLLM ROCm image is downloaded. Needs 2 GPUs.
#
#   ./run.sh               serve the requests and print RESULT and a verdict
#   ./run.sh pull          only prepare the image (docker/podman: build; apptainer: SIF)
#   sbatch -p <partition> run.sh     same, on one exclusive SLURM node
#
# Environment (optional): REQUESTS=40 LONG_EVERY=10 SHORT=256 LONG=1024 NEW=8 SEED=0 GPUS=0,1 OUT_DIR=./out/<date>
#                         IMAGE=<image name> BASE=<base image>
#                         GEN_ARGS=<extra qwen_vllm_serve.py arguments>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME, such as ~/proj.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f qwen_vllm_serve.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on vLLM's public ROCm 7.2 release image, pinned by digest (Apptainer and
# Podman do not accept tag@digest): vllm/vllm-openai-rocm:v0.33.0-rocm72.
IMAGE=${IMAGE:-qwen-vllm-tp2:v0.33.0-rocm72}
BASE=${BASE:-docker.io/vllm/vllm-openai-rocm@sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182}
REQUESTS=${REQUESTS:-40} LONG_EVERY=${LONG_EVERY:-10} SHORT=${SHORT:-256} LONG=${LONG:-1024} NEW=${NEW:-8}
SEED=${SEED:-0} GPUS=${GPUS:-0,1}
GEN_ARGS=${GEN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
# Podman's storage and Apptainer's layer cache default to $HOME, which is too small for a ROCm image.
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/qwen-vllm-tp2-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped, so image files cannot keep
# their owners, and host networking avoids the netns sysctls that need more uids.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

# The image has no passwd entry for the host uid, and getpass.getuser() fails without USER/LOGNAME. Running
# as the host user keeps the output writable on NFS, which squashes root.
NAME=qwen-vllm-tp2-$$
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
drun() { # drun <name> <gpus> <KEY=VALUE ...> -- <command...>
  local name=$1 gpus=$2 kv flag=-e envs=() t; shift 2
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
            VLLM_ALLOW_INSECURE_SERIALIZATION=1 HIP_VISIBLE_DEVICES="$gpus"; do envs+=("$flag" "$kv"); done
  while [ "$1" != -- ]; do envs+=("$flag" "$1"); shift; done; shift
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$IMAGE" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default. Not --rocm: it binds the host's ROCm libraries into
    # the container. Each container gets its own /tmp.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" -B "$OUT_DIR:$OUT_DIR" --pwd "$HERE" "${envs[@]}" "$SIF" "$@"
              rm -rf "$t" ;;
  esac
}
case "${1:-run}" in
  run) ;;
  pull) image_ready; exit $? ;;
  *) sed -n '7,19p' "$HERE/run.sh"; exit 2 ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
echo "=== requests=$REQUESTS long_every=$LONG_EVERY short=$SHORT long=$LONG new=$NEW seed=$SEED gpus=$GPUS" \
     "host=$(hostname -s) runtime=$RUNTIME${GEN_ARGS:+ gen_args=$GEN_ARGS}"
image_ready || exit 1
drun env 0 -- python3 -c 'import torch, vllm
print("ENV torch", torch.__version__, "hip", torch.version.hip, "arch",
      torch.cuda.get_device_properties(0).gcnArchName, "vllm", vllm.__version__)' 2>&1 | grep '^ENV'
LOG=$OUT_DIR/serve.log
# shellcheck disable=SC2086
drun serve "$GPUS" VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 -- python3 "$HERE/qwen_vllm_serve.py" $GEN_ARGS \
  --tp 2 --requests "$REQUESTS" --long-every "$LONG_EVERY" --short-len "$SHORT" --long-len "$LONG" \
  --new-tokens "$NEW" --seed "$SEED" --log-every 5 > "$LOG" 2>&1

echo
grep -h '^Logprobs are not finite' "$LOG"
res=$(grep -h '^RESULT' "$LOG")
nf=$(grep -o 'nonfinite=[^ ]*' <<< "$res" | head -1 | cut -d= -f2)
if [ -z "$res" ]; then
  verdict=INCOMPLETE; echo "no RESULT line; last lines of $LOG:"; tail -5 "$LOG"
else
  echo "$res"
  if [ "$nf" = yes ]; then verdict=REPRODUCED; else verdict=NOT_REPRODUCED; fi
fi
echo "VERDICT $verdict"
